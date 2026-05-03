# NOTES — vllm_paged

Qwen3-0.6B × 커스텀 Triton attention. **커널이 paged KV 를 직접 읽는** 단계.
vLLM v0 의 실제 구조 (prefill 은 dense + KV cache write / decode 는 PagedAttention) 와 가장 가까움.

> **이전**: `../vllm_split/` — 커널 2개지만 Python 에서 `_gather_kv` 로 paged KV 를 contiguous 화해 커널에 전달
> **다음**: `vllm_unified` (미래) — `(query_start_loc, seq_lens)` 메타로 multi-seq + prefill/decode 혼합 배치

## 1. 이 프로젝트에서 추가된 단 한 가지

**커널 내부에서 `block_table` 로 paged KV 를 간접 인덱싱**. Python 쪽의 `_gather_kv` 를 완전히 제거.

```
커널이 받는 것:
  K_cache, V_cache    : [num_blocks, block_size, num_kv_heads, head_dim]
  block_table         : [num_seqs, max_blocks]   (num_seqs == 1)
  seq_lens            : [num_seqs]
  BLOCK_SIZE (constexpr)

커널 내부 K 로드 (key position `k` 에 대해):
  logical_block   = k // BLOCK_SIZE
  slot_in_block   = k %  BLOCK_SIZE
  physical_block  = block_table[seq_idx, logical_block]
  k_vec           = K_cache[physical_block, slot_in_block, kv_head_idx, :]
```

GQA 도 커널 안에서 `kv_head = q_head // n_rep` 로 처리 (split 처럼 래퍼가 `repeat_interleave` 하지 않음).

## 2. 구조

```
MyTritonMetadata         — per-forward 메타 dataclass (block_table, seq_lens 포함)
MyTritonMetadataBuilder  — vLLM 이 매 forward 마다 호출
MyTritonImpl             — AttentionImpl. forward() 에서 prefill/decode 분기, 둘 다 paged wrapper 호출
MyTritonBackend          — AttentionBackend. 기본 선언 상속
register()               — register_backend(AttentionBackendEnum.CUSTOM, "...")
```

커널 (`triton_attn.py`):
- `_fwd_kernel_prefill_paged` — q\_len == kv\_len, causal mask, paged read
- `_fwd_kernel_decode_paged`  — q\_len == 1, no causal mask, paged read

래퍼:
- `triton_attention_prefill_paged(q, key_cache, value_cache, block_table, seq_lens, scale)`
- `triton_attention_decode_paged(q, key_cache, value_cache, block_table, seq_lens, scale)`

## 3. 설계 결정

**(1) Prefill 도 paged 에서 읽음 (decode 와 일관).**
`reshape_and_cache_flash` 로 cache 에 막 쓴 KV 를 prefill 커널이 다시 cache 에서 읽는다.
결과는 input K/V 를 직접 쓰는 것과 동일하지만 "paged 를 커널 안에서 다룬다" 학습 포인트가
두 커널 모두에서 드러난다.

**(2) GQA 는 커널 내부 `kv_head = q_head // n_rep`.**
split 은 래퍼에서 `repeat_interleave(n_rep, dim=1)` 로 KV 를 복제했지만, paged cache 는
KV head 기준으로 저장되어 있어 실시간 복제 불가능. 커널이 직접 인덱싱하는 편이 자연스럽다.

**(3) Block size 는 cache 의 속성을 그대로 사용.**
래퍼에서 `BLOCK_SIZE = key_cache.shape[1]` 로 추출해 constexpr 전달.
BLOCK_N (tile 크기) 과는 별개 축 — BLOCK_N 은 attention tile, BLOCK_SIZE 는 paged block.

## 4. 배운 함정 (paged 고유)

| # | 내용 | 해결 |
|---|---|---|
| 1 | Block table 인덱싱에서 mask out-of-range → 잘못된 block 주소 로드 가능 | `tl.load(block_table + ..., mask=kv_mask, other=0)` — mask 아웃된 위치는 임의 block 0 을 읽지만 어차피 `qk` 에서 `-inf` 로 마스킹됨 |
| 2 | Cache dtype vs query dtype 불일치 | `_common_check` 에서 `key_cache.dtype == q.dtype` 강제 |
| 3 | Prefill 에서 cache 쓰기 전에 커널 호출하면 garbage | `triton_reshape_and_cache_flash` 를 커널 호출 **전에** 실행 |

## 5. 검증 (kernel 단독 smoke test)

**20/20 조합 PASS**:
- prefill × (fp16/bf16) × D ∈ {64, 128, 256}, BS=16
- decode × (fp16/bf16) × kv\_len ∈ {32, 128, 1024}, BS=16
- 다양한 block size 에서 prefill/decode (BS ∈ {8, 16, 32, 64}), fp16, S=128, D=128

| 단계 | 방법 | 실측 |
|---|---|---|
| 커널 정확성 (prefill + decode, paged) | `_pack_to_paged` 로 paged cache 구성 → SDPA 와 비교 | **PASS** 전부 (fp16: ~0.0005, bf16: ~0.004) |
| Plugin 탐지 | `entry_points(group="vllm.general_plugins")` | **PASS** (`my_paged_backend`) |
| CUSTOM 슬롯 | `AttentionBackendEnum.CUSTOM.get_path()` | **PASS** |
| 백엔드 실행 증거 | 엔진 코어 stderr 의 `fired (prefill paged) / (decode paged)` | **PASS** |
| 출력 품질 | Qwen3-0.6B, "The capital of France is" | **PASS** |

## 6. 파일

- `pyproject.toml` — name `my-paged-backend`, entry point `my_paged_backend`
- `triton_attn.py` — paged 커널 2개 + 래퍼 2개 + `_pack_to_paged` 헬퍼 (smoke test 용)
- `triton_attention_backend.py` — `_gather_kv` 제거, paged wrapper 직접 호출
- `qwen3_triton_attention.ipynb`

## 7. 알려진 한계

- `max_num_seqs=1` — multi-seq continuous batching 은 `vllm_unified` 의 몫
- dtype fp16/bf16, head\_dim 2의 거듭제곱
- cudagraph 미지원 (`enforce_eager=True`)
- sliding\_window · alibi · logits\_soft\_cap · kv\_sharing · MLA · sparse 미지원
- **vLLM 0.19.1 정확히 고정**

## 8. 이 프로젝트의 위치

```
vllm_padded_decode    단일 prefill 커널 + zero-pad decode (O(s²))
       ▼
vllm_split            prefill + decode 전용 커널 (Python gather, O(s))
       ▼
vllm_paged (← 여기)   두 커널 유지, 커널이 paged KV 를 직접 읽음. vLLM v0 스타일
       ▼
vllm_unified (미래)   max_num_seqs>1 + prefill/decode 혼합 배치 단일 커널. vLLM v1 스타일
```

---

## Block-pointer conversion notes

### 무엇이 바뀌었는가

**`_fwd_kernel_prefill_paged`**:

1. **Q load** (Snippet D): raw pointer + mask → `tl.make_block_ptr(shape=(S,BLOCK_D), order=(1,0))` + `boundary_check=(0,)`.
2. **K load** (Snippet B): 원본은 `tl.load(K_cache + cache_base, mask=kv_mask[:,None])` (per-element 2-D 주소 계산) →
   `physical_block_idx = tl.load(block_table + ...) # scalar` 후 `kv_block_base` 계산,
   `tl.make_block_ptr(base=K_cache+kv_block_base, shape=(BLOCK_D,BLOCK_SIZE), strides=(stride_cache_d, stride_cache_slot), order=(0,1))`.
   Virtual transpose via `order=(0,1)`. `tl.trans()` 제거. 매 iteration 마다 fresh `make_block_ptr` — `tl.advance` 불가 (physical block 주소가 매 iter 변경).
3. **V load** (Snippet B): `tl.make_block_ptr(shape=(BLOCK_SIZE,BLOCK_D), order=(1,0))`, 매 iter fresh.
4. **O store** (Snippet F): raw pointer + mask → `tl.make_block_ptr` + `boundary_check=(0,)`.

**`_fwd_kernel_decode_paged`**:

1. **Q load** (Snippet E): raw pointer → 1-D `tl.make_block_ptr(shape=(BLOCK_D,), order=(0,))`. boundary_check 불필요.
2. **K load** (Snippet B 변형): K 는 transpose 없이 `(BLOCK_SIZE, BLOCK_D)` 로 로드. `tl.sum(q[None,:]*k, axis=1)` reduction. 매 iter fresh `make_block_ptr`.
3. **V load**: 동일하게 `(BLOCK_SIZE, BLOCK_D)` fresh block_ptr.
4. **O store**: 1-D block_ptr.

**Wrapper 변경 — `BLOCK_N=BLOCK_SIZE` 강제**:

- `triton_attention_prefill_paged`: `BLOCK_N=BLOCK_SIZE` 로 launch (원본은 `BLOCK_N=BLOCK`).
- `triton_attention_decode_paged`: `BLOCK_N=BLOCK_SIZE` 로 launch (원본은 `BLOCK_N=BLOCK_N`).

이 변경이 block_ptr idiom 의 핵심: 매 iteration 이 정확히 하나의 physical block 을 처리하므로
`block_table[seq_idx, n]` 스칼라 로드 1번으로 해당 iteration 의 전체 K/V tile 주소가 결정된다.

### 무엇이 안 바뀌었는가

- Online softmax 수학 — 완전 불변.
- Causal mask + `kv_mask` 의 `tl.where` — 불변. (`boundary_check` 는 load-time out-of-range 처리, tl.where 는 attention mask 처리 — 역할이 다름.)
- GQA (`kv_head_idx = q_head_idx // num_queries_per_kv`) — 불변.
- Wrapper 함수 시그니처 — 불변.
- Smoke test 코드 (`_pack_to_paged`, `_check_prefill`, `_check_decode`) — 불변.
- `block_table` / `seq_lens` 스칼라 로드는 raw pointer arithmetic 그대로 유지. block_ptr 은 tile I/O 전용.

### BLOCK_N == BLOCK_SIZE 강제의 의미 및 caveat

- **필수 조건**: `tl.make_block_ptr` 으로 paged KV 를 읽으려면 매 iteration 이 정확히 하나의 physical block 에 해당해야 한다. 여러 physical block 에 걸친 tile 은 연속 메모리가 보장되지 않으므로 `advance` 를 쓸 수 없다.
- **처리량 영향**: `BLOCK_SIZE` (vLLM 기본 16) 가 원본의 `BLOCK_N` (dtype 휴리스틱, 최대 128) 보다 작은 경우 tile 이 작아져 warp occupancy 가 감소할 수 있다. 이는 block_ptr idiom 의 구조적 비용이다.
- **정확성 영향 없음**: 마지막 partial block 은 `boundary_check=(0,)` or `(1,)` 가 zero-pad 로 처리하고, `kv_mask[None,:]` 의 `tl.where` 가 `-inf` masking 을 보장한다.
- **vLLM v1 본가**: raw pointer arithmetic 사용. 이 구현은 `arxiv:2511.11581` / IBM `vllm-triton-lib` 패턴을 학습 목적으로 적용.
