# multiseq → unified — 어떻게 2개 커널을 1개로 통합했나

`vllm_multiseq` 와 `vllm_unified` 의 `triton_attn.py` 를 줄 단위로 대조해서, **prefill 커널 + decode 커널 → 단일 unified 커널** 로 합쳐진 메커니즘을 정리한다. 핵심 통찰은 두 가지:

1. **하나의 절대 위치 공식**(`q_abs = seq_len - q_len + local`)이 prefill / decode / chunked 의 구별을 수학적으로 없앤다.
2. `**tl.dot` 이 decode 에서도 그대로 돈다** — decode 가 단순 matvec 인데 매트릭스 곱셈 명령(MMA, 즉 텐서코어)을 쓴다. 이게 "통합"의 진짜 비용이자 이득.

---

## 0. TL;DR — 한 표로 비교


| 항목               | `vllm_multiseq` (v0 스타일)                                                                     | `vllm_unified` (v1 스타일)                   |
| ---------------- | -------------------------------------------------------------------------------------------- | ----------------------------------------- |
| 커널 수             | **2개** (`_fwd_kernel_prefill_multiseq`, `_fwd_kernel_decode_multiseq`)                       | **1개** (`_fwd_kernel_unified`)            |
| Q shape          | prefill: `[total_q_tokens, Hq, D]` flat decode: `[num_decode_seqs, Hq, D]` stacked           | `**[total_q_tokens, Hq, D]` flat 하나로 통일** |
| Causal mask      | prefill: 절대 위치 decode: 없음 (q 토큰 1개라 항상 유효)                                                   | **절대 위치 1줄로 통일**                          |
| Q×K 연산           | prefill: `tl.dot(q, k_t)` (MMA, 텐서코어) decode: `tl.sum(q[None,:] * k, axis=1)` (FMA, CUDA 코어) | `**tl.dot(q, k_t)` 단일 — decode 도 텐서코어**   |
| P×V 연산           | prefill: `tl.dot(p, v)` decode: `tl.sum(p[:,None] * v, axis=0)`                              | `**tl.dot(p, v)` 단일**                     |
| Backend dispatch | `is_prefill` / `is_decode` 분류 → 그룹별 커널 호출 → output scatter                                   | **분류·분할 없음. 한 번 호출**                      |
| Chunked prefill  | Backend `assert` 로 거부                                                                        | **자동 지원** (math 동일)                       |
| Grid             | prefill: `(num_seqs * Hq, max_q_blocks)` decode: `(num_seqs * Hq,)`                          | `**(num_seqs * Hq, max_q_blocks)` 단일**    |


---

## 1. 통합의 수학적 토대 — `q_abs = seq_len - q_len + local`

`vllm_multiseq` 의 prefill 커널에 이미 등장하는 공식이다 (파일 92행):

```python
q_abs = S - q_len + offs_m_local                      # [BLOCK_M]
```

이게 unified 의 86행에서 그대로 쓰인다 — **공식을 바꾼 게 아니라, 이 공식이 모든 경우를 다룬다는 사실을 이용했다**:


| 배치 상태           | `q_len`, `seq_len` 관계 | 각 query token 의 절대 위치           |
| --------------- | --------------------- | ------------------------------- |
| Pure prefill    | `q_len == S`          | `q_abs = 0, 1, …, S - 1`        |
| Pure decode     | `q_len == 1`          | `q_abs = [S - 1]`               |
| Chunked prefill | `1 < q_len < S`       | `q_abs = [S - q_len, …, S - 1]` |


세 경우 모두 동일한 공식이 옳은 절대 위치를 만들어내고, causal mask `q_abs[:, None] >= offs_n[None, :]` 한 줄이 모든 경우를 정확히 처리한다. **분기가 사라진 것은 코드를 합쳤기 때문이 아니라, 수학이 원래 같았기 때문**이다.

multiseq 의 decode 커널에는 이런 mask 라인 자체가 없다 — q 토큰이 1개라 모든 K 슬롯이 유효(KV mask 만 적용). unified 에서는 이 1개 토큰도 BLOCK_M 행렬의 한 행으로 들어가고, 나머지 BLOCK_M-1 행은 `q_mask` 가 false 가 되어 자동으로 무효화된다.

---

## 2. Q 텐서 shape 통일 — flat-packed 로 일원화

multiseq 에서는 두 커널의 Q shape 이 달랐다:

```
prefill:  Q [total_q_tokens, Hq, D]  +  query_start_loc [num_seqs+1]
decode:   Q [num_decode_seqs, Hq, D]                            ← shape 자체가 다름
```

decode 는 "각 seq 가 정확히 q 토큰 1개"라서 `batch_idx` 로 직접 인덱싱하면 됐다(187행).

unified 는 decode 도 prefill 과 같은 flat-packed 형태로 받는다:

```
모든 경우:  Q [total_q_tokens, Hq, D]  +  query_start_loc [num_seqs+1]
```

decode-only 배치라면 `total_q_tokens == num_decode_seqs` 이고 `query_start_loc = [0, 1, 2, …, num_seqs]` 이 된다. shape 만 달라 보이지 본질은 같다 — 한 seq 의 q 토큰이 1개인 특수 케이스.

이 통일이 backend 코드를 극적으로 줄인다. multiseq backend 가 하던 일:

```python
# multiseq backend (개념)
q_lens = query_start_loc[1:] - query_start_loc[:-1]
is_prefill = q_lens == seq_lens
is_decode  = q_lens == 1
if is_prefill.any():
    triton_attention_prefill_multiseq(q[prefill_idx], …)   # gather
if is_decode.any():
    triton_attention_decode_multiseq(last_q_tokens, …)     # 별도 stack
# 두 결과를 원래 위치로 scatter
```

unified backend:

```python
o_flat = triton_attention_unified(query[:N], key_cache, value_cache,
                                  block_table, seq_lens, query_start_loc,
                                  scale=self.scale)
output[:N].copy_(o_flat)
```

분류·gather·이중 호출·scatter 가 통째로 사라진다.

---

## 3. 핵심 — `tl.dot` 으로 decode 가 텐서코어를 타게 된 사실

여기가 사용자가 짚은 지점이다. 두 decode 경로의 Q×K 연산을 직접 비교하자.

### multiseq decode — element-wise 곱 + scalar reduction (CUDA 코어)

`vllm_multiseq/triton_attn.py:213`:

```python
q = tl.load(... offs_d * stride_q_d).to(tl.float32)   # q: [D]   1D 벡터
...
qk = tl.sum(q[None, :] * k.to(tl.float32), axis=1) * sm_scale
#         q[None,:]:[1,D] × k:[BLOCK_N,D] → element-wise [BLOCK_N,D] → sum axis=1 → [BLOCK_N]
```

P×V 도 마찬가지(223행):

```python
acc = acc + tl.sum(p[:, None] * v.to(tl.float32), axis=0)
#        p[:,None]:[BLOCK_N,1] × v:[BLOCK_N,D] → sum axis=0 → [D]
```

연산 형태: **벡터 × 행렬 → 벡터**. Triton 은 이걸 일반 FMA 루프로 컴파일한다. **CUDA 코어**(FP32 ALU) 가 돌고, 텐서코어는 안 쓴다. decode 가 memory-bound 라 어차피 ALU 가 병목이 아니므로 v0 시절 합리적인 선택이었다.

### unified — `tl.dot` 으로 매트릭스 곱셈 (텐서코어)

`vllm_unified/triton_attn.py:103`:

```python
q = tl.load(q_ptr_base + offs_q_global[:, None] * stride_q_t
                       + offs_d[None, :] * stride_q_d, ...)   # q: [BLOCK_M, D]   2D 매트릭스
...
qk = tl.dot(q, k_t) * sm_scale
#    q:[BLOCK_M,D] × k_t:[D,BLOCK_N] → [BLOCK_M, BLOCK_N]   (mma 명령)
```

P×V (146행):

```python
acc = acc + tl.dot(p.to(io_dtype), v, out_dtype=tl.float32)
#    p:[BLOCK_M,BLOCK_N] × v:[BLOCK_N,D] → [BLOCK_M, D]   (mma 명령)
```

`tl.dot` 은 SM80+ 에서 `**mma.sync` (텐서코어 명령)** 으로 내려간다. dtype 이 fp16/bf16 이면 텐서코어 fp16/bf16 파이프, accumulator 는 fp32. 즉 **decode 도 텐서코어를 탄다**.

### 그런데 decode 는 사실상 matvec 인데, 행렬 명령으로 쓰면 낭비 아닌가?

맞다. 그리고 이게 통합의 **숨은 비용**이다:

```
unified 에서 pure decode batch 처리 시:
  q_len = 1   →   q_mask 가 BLOCK_M 행 중 1행만 true
                   나머지 BLOCK_M-1 행은 0 으로 padding 되어 dot 에 들어감
                   → MMA 가 동작은 하지만 (BLOCK_M-1)/BLOCK_M 슬롯이 헛돈다
```

BLOCK_M=64 라면 **MMA tile 의 1/64 만 유의미한 work**. 표면적으론 텐서코어 utilization 이 끔찍해 보인다.

**그런데 왜 이게 받아들일 만한가** — 세 가지 이유:

1. **Decode 는 memory-bound**. 병목은 KV cache 로딩(HBM bandwidth)이지 ALU/MMA 처리량이 아니다. MMA 가 1/64 만 일해도, 어차피 KV 를 가져오는 동안 놀고 있을 텐서코어다. 컴퓨트 낭비가 latency 손해로 전환되지 않는다.
2. **Mixed batch 에서는 손실이 사라진다**. v1 의 continuous batching 은 prefill / chunked / decode 가 한 batch 에 섞여 있고, prefill 이나 chunked seq 의 `q_len > 1` 토큰이 BLOCK_M 슬롯을 채워준다. 통합 커널의 진짜 가치는 이 혼합 batch 에서 단일 launch 로 모두를 처리하는 데 있다.
3. **Pure decode batch 조차도 vLLM v1 의 실제 grid 는 token-flat** (`(total_q_blocks, num_kv_heads)` + `find_seq_idx`) 이라서, BLOCK_M 슬롯을 **여러 seq 의 decode 토큰들로 채운다**. 64-way decode batch 면 BLOCK_M=64 가 거의 빈틈없이 찬다. 이 프로젝트의 `vllm_unified` 는 교육용이라 seq-first grid 를 쓰므로 이 효과를 못 보지만, 진짜 v1 에서는 decode 도 텐서코어 utilization 이 합리적이다.

요약하면: **decode 가 텐서코어를 타는 건 의도된 설계**다. 단독 decode 에서는 utilization 이 낮지만, memory-bound 라 손해가 아니고, 혼합 batch + flat grid 에서는 자연스럽게 회복된다.

---

## 4. Causal mask — 상대 → 절대 위치 일반화

이건 사실 multiseq prefill 이 이미 절대 위치를 쓰고 있어서 새로 추가된 것은 아니다. 하지만 multiseq 단계에서 의도적으로 깔아둔 포석이다.

multiseq prefill (131행):

```python
causal = q_abs[:, None] >= offs_n[None, :]
```

multiseq decode 에는 causal mask 가 없다 (`q_mask` 도 없음 — q 가 1개니 무조건 유효).

unified (136행):

```python
causal = q_abs[:, None] >= offs_n[None, :]   # 동일
qk = tl.where(q_mask[:, None] & kv_mask[None, :] & causal, qk, float("-inf"))
```

decode 케이스(`q_len=1`)에서 `q_abs = [S-1]` 이므로 모든 KV 슬롯에 대해 causal 이 true (마지막 토큰이라 항상 모두 attend). decode 만 따로 떼어보면 mask 가 무의미하지만, **분기 없이 같은 식으로 처리**된다는 게 핵심.

이 라인이 multiseq 의 prefill 커널에 일찌감치 들어가 있었던 이유: **chunked prefill / unified 로의 확장을 미리 염두에 둔 것**. multiseq 의 NOTES.md 에도 "절대 위치가 더 일반적, chunked 확장 여지" 라고 적혀 있다.

---

## 5. Grid 구조의 자연스러운 일반화

multiseq:

```
prefill grid: (num_seqs * Hq, max_q_blocks)
decode  grid: (num_seqs * Hq,)              ← 1차원
```

unified:

```
grid: (num_seqs * Hq, max_q_blocks)         ← prefill 의 grid 를 모든 경우에 적용
```

decode 만 있는 배치에서 `max_q_blocks = cdiv(1, BLOCK_M) = 1` 이므로 grid 의 두 번째 축이 1 이 되어 multiseq decode grid 와 동등해진다. unified 코드(205행):

```python
max_q_len = int((query_start_loc[1:] - query_start_loc[:-1]).max().item())
max_q_blocks = triton.cdiv(max(max_q_len, 1), BLOCK)
```

`max(max_q_len, 1)` 은 `total_q_tokens=0` 같은 엣지 케이스 대비 (이론상 발생하지 않지만 방어).

---

## 6. Decode 커널의 "사라진" 부분들

multiseq decode 에는 있고 unified 에는 없는 것들:


| multiseq decode 의 코드             | unified 에서의 처리                                                     |
| -------------------------------- | ------------------------------------------------------------------ |
| `q.to(tl.float32)` 로 fp32 캐스트    | dtype 그대로 `tl.dot` 입력, accumulator 만 fp32 (`out_dtype=tl.float32`) |
| `m_i: scalar` (`float("-inf")`)  | `m_i: [BLOCK_M]` (`tl.full(..., float("-inf"))`)                   |
| `l_i: scalar` (`0.0`)            | `l_i: [BLOCK_M]` (`tl.zeros(...)`)                                 |
| `acc: [BLOCK_D]`                 | `acc: [BLOCK_M, BLOCK_D]`                                          |
| `tl.max(qk, axis=0)` (1D)        | `tl.max(qk, axis=1)` (2D, per-row)                                 |
| `tl.sum(p[:, None] * v, axis=0)` | `tl.dot(p.to(io_dtype), v, ...)`                                   |


**결국 차이는 "벡터 1개 처리" → "행렬 BLOCK_M 행 동시 처리"** 의 차이다. decode 가 1행만 유효하더라도 코드는 BLOCK_M 행을 처리하는 형태로 통일된다.

---

## 7. 시각화 — 데이터 흐름 비교

```
[multiseq backend]                      [unified backend]
                                      
  forward() inputs                        forward() inputs
       │                                       │
       ▼                                       ▼
  ┌─────────────────────┐                 ┌──────────────────────┐
  │ classify per-seq    │                 │  (no classification) │
  │ is_prefill / decode │                 │                      │
  └─────────────────────┘                 │  triton_attention_   │
       │                                  │     unified(...)     │
       │ split                            │                      │
       ▼                                  └──────────┬───────────┘
  ┌──────────┐  ┌─────────┐                          │
  │ prefill  │  │ decode  │                          ▼
  │ kernel   │  │ kernel  │                ┌──────────────────────┐
  │ (tl.dot) │  │ (FMA)   │                │  _fwd_kernel_unified │
  └────┬─────┘  └────┬────┘                │       (tl.dot)       │
       │             │                     │  prefill OR decode   │
       │  scatter    │                     │  OR chunked — same   │
       ▼             ▼                     │  math path           │
       output buffer                       └──────────┬───────────┘
                                                      ▼
                                                output buffer
```

multiseq 가 backend 에서 분기하던 일을, unified 는 수학으로 흡수하고 단일 커널이 모두 처리한다.

---

## 8. 실측 — chunked 가 실제로 자동 발생

`vllm_unified/NOTES.md` 의 E2E 로그(Qwen3-0.6B, max_num_seqs=4, `max_num_batched_tokens=64`):

```
fired (unified) num_seqs=1 prefill-like=1 decode=0 chunked=0  max_q_len=5  tokens=5
fired (unified) num_seqs=2 prefill-like=1 decode=1 chunked=0  max_q_len=63 tokens=64
fired (unified) num_seqs=4 prefill-like=3 decode=1 chunked=1  max_q_len=12 tokens=22
                              ▲           ▲         ▲
                              │           │         └ 긴 prompt 의 64-token chunk 중간 단계
                              │           └ 이미 decode 중이던 seq
                              └ 새 prompts 의 prefill
```

세 번째 줄이 핵심: **prefill + decode + chunked 가 동일 forward 에 공존**, 단일 unified 커널 호출로 처리. multiseq 라면 두 번 호출 + chunked 거부였을 것이다.

---

## 9. 트레이드오프 — 통합이 공짜는 아니다


| 비용                          | 설명                                                                                                                                                 |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Pure decode batch 의 MMA 헛돌이 | seq-first grid 에서 BLOCK_M 의 (BLOCK_M-1)/BLOCK_M 슬롯이 idle. 단 memory-bound 라 latency 손해는 작음.                                                         |
| Decode 의 fp32 정밀도 미세 차이     | multiseq decode 는 fp32 캐스트 후 FMA. unified 는 fp16/bf16 입력에 fp32 accumulator. 보통 동등하지만 누적 순서가 다를 수 있어 비트 단위 동일은 아님.                                  |
| Idle program 낭비             | `(num_seqs * Hq, max_q_blocks)` grid 에서 짧은 seq 는 max_q_blocks 만큼 program 을 띄우지만 대부분 q_mask 로 작업 없음. 진짜 v1 은 token-flat grid + `find_seq_idx` 로 해결. |
| 코드의 "어떤 케이스를 위한 거지?" 질문 사라짐 | 같은 코드가 세 케이스를 다 처리한다는 사실을 읽는 사람이 따로 추론해야 함. 주석으로 보완.                                                                                               |


이 비용들이 **혼합 배치 + chunked default ON** 의 이득보다 작다는 게 v1 의 판단이고, vLLM 이 v0 의 split-dispatch 를 버린 이유다.

---

## 10. 한 문장 요약

> **prefill 커널이 이미 가지고 있던 "절대 위치 causal mask" 라는 일반화 도구를 decode 에 적용**하면, decode 는 그저 `q_len == 1` 인 prefill 의 특수 케이스가 되어 동일한 `tl.dot` 기반 코드 한 벌로 모두 처리된다. 텐서코어 utilization 손실은 memory-bound 특성과 mixed batch 효과에 흡수되고, 그 대가로 backend 의 split-dispatch 가 통째로 사라지며 chunked prefill 이 무료로 떨어진다.

