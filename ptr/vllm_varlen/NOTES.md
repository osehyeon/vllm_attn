# NOTES — vllm_varlen

Qwen3-0.6B × 커스텀 Triton attention. **Varlen launch topology** — vLLM v1 의 `kernel_unified_attention_2d` 와 동일한 grid 구조.

> **이전**: `../vllm_unified/` — 수학은 통합, 하지만 grid 는 seq-first (num_seqs × Hq × max_q_blocks)
> **차이**: grid 를 token-flat 으로 재구성, `find_seq_idx` binary search 도입

## 1. 이 프로젝트에서 바뀐 것

| 레이어 | unified | varlen |
|---|---|---|
| Q tensor | `[total_q_tokens, Hq, D]` flat | **동일** |
| Causal math | 절대 위치 `q_abs = s_len - q_len + local` | **동일** |
| **Grid** | `(num_seqs × Hq, max_q_blocks)` **seq-first** | **`(total_q_blocks_rounded, Hq)` seq-aligned flat** |
| **Seq idx** | grid 에서 자동 | **`find_seq_idx` binary search** |
| 짧은 seq 의 idle programs | max_q_blocks 만큼 많이 | 해당 seq 의 필요 개수만 |

## 2. 핵심 메커니즘

### Seq-aligned flat grid

각 시퀀스의 q 토큰을 BLOCK_Q 단위로 **round-up** 해서 flat 펼침:

```
q_lens           = [5, 1, 1]      (3개 seq)
BLOCK_Q          = 4
blocks_per_seq   = [cdiv(5,4), cdiv(1,4), cdiv(1,4)] = [2, 1, 1]
cum_q_blocks     = [0, 2, 3, 4]   # cumulative prefix (kernel 에 전달)
total_q_blocks   = 4              # grid 의 첫 축 크기

Grid programs (pid_block, q_head) 로 분배:
  pid_block=0 → seq 0, block 0 (tokens 0..3)
  pid_block=1 → seq 0, block 1 (tokens 4, 나머지 3은 padding)
  pid_block=2 → seq 1, block 0 (token 0 + padding 3)
  pid_block=3 → seq 2, block 0 (token 0 + padding 3)
```

### `find_seq_idx` — Triton 안의 binary search

```python
@triton.jit
def _find_seq_idx(cum_q_blocks_ptr, q_block_idx, num_seqs):
    lo = 0; hi = num_seqs
    while lo < hi:
        mid = (lo + hi) // 2
        start = tl.load(cum_q_blocks_ptr + mid)
        if start <= q_block_idx:
            lo = mid + 1
        else:
            hi = mid
    return lo - 1
```

`O(log num_seqs)` lookup. 각 프로그램이 자기 담당 seq 를 진입 시 1회 결정하고, 이후 scalar 로 사용.

### 한 BLOCK_Q 안에는 **항상 한 seq 만**

위 round-up 덕분에 한 block 안에 다른 시퀀스가 섞이지 않음. 그래서 프로그램 내부 로직은
unified 와 같이 **한 seq 기준** (scalar seq_idx + scalar block_table row + scalar s_len) 으로 동작 → `tl.dot` 유지, per-token gather 불필요.

## 3. 왜 이게 "실질적 flat" 인가

진정한 token-level flat (한 BLOCK_Q 안에 여러 seq 가능) 은 per-token `block_table`, per-token `s_len` 등 3D tensor gather 가 필요해 Triton 구현이 매우 복잡해지고 `tl.dot` 도 포기해야 함. vLLM v1 도 이런 극단적 packing 을 피하고 **seq 경계에서 BLOCK_Q 를 round-up** 하는 전략을 선택했음 (`kernel_unified_attention_2d` 의 `use_q_block_mode=True` 경로).

따라서 "seq-aligned flat" = vLLM v1 의 실제 launch 토폴로지.

## 4. Program 수 비교 (decode-heavy 예시)

배치: `q_lens = [32, 1, 1, 1, 1]`, BLOCK_Q = 64, Hq = 16

| 방식 | grid 공식 | program 수 |
|---|---|---|
| seq-first (unified) | `num_seqs × Hq × max_q_blocks` | 5 × 16 × 1 = **80** |
| varlen | `total_q_blocks × Hq` | `cdiv(32,64) + 4×cdiv(1,64)` × 16 = 5 × 16 = **80** |

이 예시에서는 동일 (BLOCK_Q ≥ max q_len 이라 각 seq 가 1 block). 진짜 차이가 나는 경우:

`q_lens = [128, 1, 1, 1, 1]`, BLOCK_Q = 64:
| | |
|---|---|
| seq-first | 5 × 16 × 2 = **160** (max_q_blocks=2, 하지만 4개 seq 는 1 block 만 필요 → 1/2 idle) |
| varlen | (2 + 1 + 1 + 1 + 1) × 16 = **96** (**40% 감소**) |

긴 prefill 하나 + 여러 decode 가 섞인 전형적 continuous batching 시나리오에서 효과가 드러남.

## 5. 검증 (kernel 단독 smoke test)

**10/10 PASS** — unified 와 동일한 매트릭스 (all prefill / all decode / mixed / chunked / ultimate mix) × (fp16, bf16). 같은 수학이니 오차도 동일.

## 6. E2E 실측 (Qwen3-0.6B, max_num_seqs=4, 4 prompts 동시)

`max_num_batched_tokens=64` + 긴 prompt 조합으로 chunked prefill 도 실측:

```
fired (varlen) num_seqs=1 prefill-like=1 decode=0 chunked=0 max_q_len=5  tokens=5
fired (varlen) num_seqs=2 prefill-like=1 decode=1 chunked=0 max_q_len=63 tokens=64    ← 첫 chunk (prefill-like)
fired (varlen) num_seqs=4 prefill-like=3 decode=1 chunked=1 max_q_len=12 tokens=22    ← **chunked=1** 등장
[0]  Paris. The capital of Italy is Rome. ...
[1]  the key to solving the problem. ...  (긴 prompt 의 생성)
[2]  "Macbeth" in 1606. ...
[3]  a group of developers who wanted to create ...
```

세 번째 fired 로그 = v1 continuous batching + chunked prefill 의 정확한 모습:
- 3개 짧은 prompt prefill + 1 decode + 긴 prompt 의 chunked 중간 단계가 한 varlen forward 에 공존
- Unified 와 출력·fired 카운트 모두 동일. 차이는 **grid 구성만** — varlen 은 `(total_q_blocks, Hq)` flat grid 사용

## 7. 파일

- `pyproject.toml` — `my-varlen-backend` / `my_varlen_backend`
- `triton_attn.py` — `_fwd_kernel_varlen` + `_find_seq_idx` binary search + 래퍼
- `triton_attention_backend.py` — unified 와 동일 (한 줄 호출). 래퍼 내부만 달라짐
- `qwen3_triton_attention.ipynb` — 15셀

## 8. 알려진 한계

- **GQA 를 Hkv grid 축으로 접지 않음** — 한 프로그램이 1 Q head 담당. vLLM v1 실제 방식은 `(total_q_blocks, Hkv)` + 프로그램 내부 `n_rep` 개 Q head 를 `BLOCK_M = BLOCK_Q × n_rep` 로 묶음. 이 최적화는 추가 단계
- **Split-k 미적용** — 긴 컨텍스트 decode 에서 vLLM 대비 낮은 utilization
- 나머지 제약은 unified 와 동일

## 9. 전체 로드맵 완주

```
vllm_padded_decode    prefill 커널 1개 + zero-pad decode           (O(s²))
       ▼
vllm_split            prefill + decode 전용 커널, Python gather    (O(s))
       ▼
vllm_paged            커널이 paged KV 직접 읽음                   (max_num_seqs=1)
       ▼
vllm_multiseq         두 커널 multi-seq batch, backend split       (vLLM v0 스타일)
       ▼
vllm_unified          커널 1개로 prefill+decode+chunked 통합        (수학은 v1, grid 는 seq-first)
       ▼
vllm_varlen (← 여기)   grid 도 varlen (seq-aligned flat + find_seq_idx)  (vLLM v1 실제 방식)
```

각 단계가 **정확히 한 가지만** 바꿉니다:
- decode 전용 커널 → paged read → multi-seq dispatch → 커널 통합 → grid 통합

이 연속된 변화를 따라가면 vLLM 이 지금의 아키텍처로 수렴한 과정이 코드 레벨에서 체감됩니다.
