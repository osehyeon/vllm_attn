# NOTES — vllm_unified

Qwen3-0.6B × 커스텀 Triton attention. **Unified 커널** 단계 — vLLM v1 스타일.
한 개의 커널이 prefill / decode / chunked-prefill 을 모두 처리.

> **이전**: `../vllm_multiseq/` — 커널 2개 분리, backend 가 split-dispatch (v0 스타일)
> 다음은 split-k 최적화 / fp8 KV / flat grid 등 선택적 고도화

## 1. 이 프로젝트에서 사라진 것과 추가된 것

### 사라진 것
- **커널 2개 분리** → 1개로 통합 (`_fwd_kernel_unified`)
- **Backend 의 split-dispatch 로직** (is_prefill / is_decode 분류, prefill 그룹 / decode 그룹 각각 호출)
- **chunked prefill assert** — 이제 q_len 이 뭐든 받음

### 추가된 것
- 절대 위치 기반 causal mask 가 "구별 자체를 없앰" — prefill/decode/chunked 가 수학적으로 같은 경로
- 한 번의 커널 호출로 혼합 배치 처리
- Chunked prefill 이 자동 지원 (backend 변경 없이)

## 2. 핵심 아이디어 — "구별을 없애는" 수학

세 가지 배치 상태:

| 상태 | q_len, s_len 관계 | 각 query token 의 절대 위치 |
|---|---|---|
| Pure prefill | q_len == s_len | `q_abs = 0, 1, ..., s_len - 1` |
| Decode | q_len == 1 | `q_abs = [s_len - 1]` |
| Chunked prefill | 1 < q_len < s_len | `q_abs = [s_len - q_len, ..., s_len - 1]` |

이 모두가 **`q_abs = s_len - q_len + local_offset`** 공식으로 통합.
Causal mask 도 `q_abs[:, None] >= k_pos[None, :]` 한 줄로 모든 경우 처리.
구별이 없으니 분기도 없고, 한 커널이면 충분.

## 3. Backend 의 단순함

```python
# 전체 batch 를 한 번에 처리 — 분류·분할 없음
o_flat = triton_attention_unified(
    query[:N],                                 # flat-packed
    key_cache, value_cache,
    block_table, seq_lens, query_start_loc,
    scale=self.scale,
)
output[:N].copy_(o_flat)
```

Multiseq 의 prefill 그룹 추출 + decode 그룹 추출 + 각각 커널 호출 + output scatter 같은
복잡성이 한 줄로 줄어듦.

## 4. 왜 vLLM v1 이 unified 로 갔는가 (chunked prefill default ON)

| 이유 | 설명 |
|---|---|
| **Chunked 비용 소실** | Unified 커널이 chunked 를 추가 launch 없이 처리 → 항상 켜도 throughput 손실 미미 |
| **Latency SLO** | 긴 prompt 가 다른 decode 를 막지 않음 (head-of-line blocking 해결) |
| **GPU utilization** | prefill (compute-bound) + decode (memory-bound) 혼합 batch 로 SM·HBM 동시 활용 (Sarathi-Serve 계열 연구) |
| **정책** | v1 이 "최신 서빙 모범 사례를 default 로" 라는 방향성 |

v0 도 multi-seq + chunked 를 할 수 있었지만 (두 커널 분리 호출 오버헤드로) `enable_chunked_prefill=False` 가 default 였음. v1 이 unified 로 전환하며 **chunked 를 켜는 게 거의 항상 이득** 인 상태로 만들었고, 그래서 default ON.

## 5. 교육용으로 단순화한 부분

- **Grid 전략**: seq-first `(num_seqs * Hq, max_q_blocks)` — vLLM 실제는 token-flat `(total_q_blocks, num_kv_heads)` + `find_seq_idx` binary search. 이 프로젝트에서는 단순성을 위해 seq-first 선택. 낭비 있지만 (idle programs) 학습 포인트는 동일
- **Split-k 미사용**: 긴 컨텍스트 decode 에서 parallelism 확보를 위한 reduction 분할 (stage1 + stage2) 은 제외
- **Attention 보조 기능**: sliding_window / alibi / logits_soft_cap / MLA / sparse 등은 전부 assert 로 거부

## 6. 검증 (kernel 단독 smoke test)

**10/10 조합 PASS**:
- all prefill (q_len == s_len, 여러 길이)
- all decode (q_len == 1, 여러 s_len)
- **mixed** (prefill + decode 한 배치)
- chunked prefill (1 < q_len < s_len)
- **ultimate mix** (prefill + chunked + decode 한 배치)

모든 경우에 동일한 `triton_attention_unified` 호출로 SDPA reference 대비 `max_abs_err < 1e-2` (fp16) / `< 3e-2` (bf16).

## 7. E2E 실측 (Qwen3-0.6B, max_num_seqs=4, 4 prompts)

`max_num_batched_tokens=64` + 긴 prompt 조합으로 chunked prefill 도 실제로 trigger:

```
fired (unified) num_seqs=1 prefill-like=1 decode=0 chunked=0 max_q_len=5  tokens=5    ← 첫 prompt 의 초기 prefill
fired (unified) num_seqs=2 prefill-like=1 decode=1 chunked=0 max_q_len=63 tokens=64   ← 긴 prompt 의 첫 chunk (prefill-like) + 이미 decode 중인 seq
fired (unified) num_seqs=4 prefill-like=3 decode=1 chunked=1 max_q_len=12 tokens=22   ← 핵심: prefill + decode + **chunked** 한 forward 에 공존
```

**세 번째 줄의 의미**:
- `prefill-like=3` — 새로 들어온 짧은 prompts 의 prefill
- `decode=1` — 이미 decode 중이던 시퀀스
- **`chunked=1`** — 긴 prompt 가 64-token chunk 로 쪼개진 중간 단계 (`1 < q_len < s_len`)
- **v1 continuous batching + chunked prefill 의 정확한 모습** — 긴 prompt 가 다른 decode 를 막지 않고 한 forward 에서 공존

## 8. 파일

- `pyproject.toml` — name `my-unified-backend-block`, entry point `my_unified_backend_block`
- `triton_attn.py` — **1개 커널** (`_fwd_kernel_unified`) + 래퍼 1개
- `triton_attention_backend.py` — split-dispatch 제거, 단일 커널 호출
- `qwen3_triton_attention.ipynb` — 15셀, chunked 관찰 포인트 포함

## 9. 알려진 한계

- **seq-first grid** — vLLM v1 의 flat grid 와 다름 (idle program 낭비)
- **Split-k 미적용** — 긴 컨텍스트 decode 에서 vLLM 대비 낮은 utilization
- dtype fp16/bf16, head\_dim 2의 거듭제곱
- cudagraph 미지원 (`enforce_eager=True`)
- sliding\_window · alibi · logits\_soft\_cap · kv\_sharing · MLA · sparse 미지원
- **vLLM 0.19.1 정확히 고정**

## 10. 전체 로드맵 완주

```
vllm_padded_decode    prefill 커널 1개 + zero-pad decode   (O(s²))     교육용 장난감
       ▼
vllm_split            prefill + decode 전용 커널            (O(s))     Python gather 우회
       ▼
vllm_paged            두 커널 유지, 커널이 paged KV 직접 읽음 (single-seq)  vLLM v0 의 엔진부
       ▼
vllm_multiseq         두 커널이 각각 multi-seq batch 처리              vLLM v0 완성형 (chunked 빼고)
       ▼
vllm_unified (← 여기)  커널 1개로 prefill+decode+chunked 혼합 배치 통합   vLLM v1 스타일
```

각 단계가 직전 단계에서 **한 가지 변화** 만 추가 — paged 인덱싱, multi-seq dispatch, unified math. 이 연속된 변화를 따라가면 "왜 vLLM 이 지금의 아키텍처로 수렴했나" 가 체감된다.

---

## Block-pointer conversion notes

### 무엇이 바뀌었는가

이 `vllm_attn/block_ptr` 변형은 `tl.make_block_ptr` 기반 tile I/O 를 사용한다.
원본 `vllm_unified/triton_attn.py` 대비 변경점:

1. **Q load (Snippet C)**: `q_ptr_base + offs_q_global[:,None]*stride_q_t + ...` → `tl.make_block_ptr(base=Q+q_head_idx*stride_q_h, shape=(total_q_tokens,D), strides=(stride_q_t,stride_q_d), offsets=(q_start+pid_m*BLOCK_M,0), ...)`. `mask=` 인자 제거, `boundary_check=(0,)` 로 대체.

2. **K load (Snippet B)**: `tl.trans(k)` 제거. K_block_ptr `shape=(BLOCK_D,BLOCK_SIZE), order=(0,1)` — virtual transpose 로 `[BLOCK_D, BLOCK_N]` 형태로 직접 로드. `mask=` 제거, `boundary_check=(1,)`.

3. **V load (Snippet B)**: `shape=(BLOCK_SIZE,BLOCK_D), order=(1,0)`. `mask=` 제거, `boundary_check=(0,)`.

4. **O store (Snippet F)**: `tl.store(..., mask=q_mask[:,None])` → `tl.store(O_block_ptr, ..., boundary_check=(0,))`. O_block_ptr 는 Q_block_ptr 와 대칭 (`stride_o_t, stride_o_d`).

### 무엇이 안 바뀌었는가

- **Wrapper 함수 시그니처** — `triton_attention_unified` 인자 완전 동일. Backend 수정 불필요.
- **Online softmax 수학** — `m_i`, `l_i`, `alpha`, `p`, `acc` 누적 흐름 불변.
- **Causal mask** — `q_abs[:, None] >= offs_n[None, :]` (`q_abs = S - q_len + offs_m_local`) 불변.
- **GQA** — `kv_head_idx = q_head_idx // num_queries_per_kv` 불변.
- **Grid 토폴로지** — `(num_seqs*Hq, max_q_blocks)` 불변.
- **Scalar loads** — `block_table`, `seq_lens`, `query_start_loc` 는 raw pointer arithmetic 유지.
- **q_mask / kv_mask outer-product** — `tl.where(q_mask[:,None] & kv_mask[None,:] & causal, ...)` 불변. boundary_check 는 배열 경계 padding 만 zero-fill.

### BLOCK_N == BLOCK_SIZE 제약

paged KV 에서 각 반복이 하나의 물리적 블록에 해당한다. `physical_block_idx` 를 스칼라로 로드하고 그 블록의 base 주소에 `make_block_ptr` 를 생성하므로, `BLOCK_N` 이 `BLOCK_SIZE` 와 같아야 한다. wrapper 에서 `BLOCK_N=BLOCK_SIZE` 로 launch.

### total_q_tokens 추가 인자

`tl.make_block_ptr` 의 `shape` 파라미터에 flat Q 의 첫 축 크기 `total_q_tokens = q.shape[0]` 이 필요하다. 커널 함수에 인자 1개가 추가됨. wrapper 에서 `total_q` (= `q.shape[0]`) 를 전달. public Python wrapper signature `triton_attention_unified(q, key_cache, ...)` 는 불변.

### 정직한 Caveat

- **vLLM v1 본가는 raw pointer 사용**: `vllm/v1/attention/ops/triton_prefill_attention.py` 는 `tl.make_block_ptr` 대신 raw arithmetic 을 쓴다. Triton 컴파일러가 raw ptr 도 유사하게 최적화하므로 실제 성능 차이는 미미하다.
- **arxiv:2511.11581 / IBM vllm-triton-lib** 에서 `make_block_ptr` 기반 패턴이 소개됨. 이 구현은 그 패턴을 학습 목적으로 적용.
- boundary_check 는 배열 끝 경계에서 zero-fill 한다. 시퀀스 내 causal mask 와 q/kv padding 처리는 `tl.where` 에서 담당.
