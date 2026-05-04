# NOTES — block_ptr/vllm_split_v2 (split-KV decode, block_ptr 표현)

`ptr/vllm_split_v2` 의 메모리 접근을 **`tl.make_block_ptr`** 표현으로 옮긴 변형. 알고리즘은 동일 (4 커널 + log-sum-exp 결합), **메모리 접근 idiom 만 다름**.

## 0. 출처

- 알고리즘: Tri Dao et al. *Flash-Decoding for long-context inference* (2023.10).
- 메모리 접근 표현: arxiv:2511.11581 / IBM `vllm-triton-lib` 의 block_ptr idiom.
- 동일 단계 ptr 변형: `ptr/vllm_split_v2/` (smoke test, 알고리즘, 출력 모두 동등).

## 1. 무엇이 바뀌었나 (block_ptr/vllm_split → block_ptr/vllm_split_v2)

| 항목 | block_ptr/vllm_split | block_ptr/vllm_split_v2 |
|---|---|---|
| Prefill 커널 | `_fwd_kernel_prefill` | **동일** |
| Decode 커널 수 | 1 (`_fwd_kernel_decode`) | **3** (`_simple` + `_partial` + `_reduce`) |
| Decode grid | `(B*Hq,)` 1D | `(B*Hq, KV_SPLITS)` 2D + reduce `(B*Hq,)` 1D |
| 임시 buffer | 없음 | partial m / l / acc (fp32) |
| Wrapper dispatch | 단일 경로 | KV_SPLITS=1 fast path / KV_SPLITS≥2 partial+reduce |

알고리즘 (math) 변화 0, **schedule 변화만**. ptr/vllm_split_v2 와 동일.

## 2. ptr/vllm_split_v2 와 block_ptr/vllm_split_v2 의 차이

알고리즘 동일, **메모리 접근 idiom 만 다름**:

| 위치 | ptr 변형 | block_ptr 변형 |
|---|---|---|
| **Prefill K transpose** | `tl.trans(k)` 명시 호출 | `make_block_ptr(shape=(D,S), strides=(sd,ss), order=(0,1))` virtual transpose |
| **Decode Q load** | `tl.load(Q + pid_bh*sb + offs_d*sd)` | 1-D `make_block_ptr(block_shape=(BLOCK_D,), order=(0,))` |
| **Decode K/V load (loop)** | 매 iter offset 재계산 + `tl.load(K + offs_n[:,None]*ss + offs_d[None,:]*sd, mask=...)` | 2-D `make_block_ptr` 한 번 + 매 iter `tl.advance(K, (BLOCK_N, 0))` |
| **Decode boundary** | `mask=` 인자에 통합 | `boundary_check=(0,)` + algorithmic mask 는 `tl.where` 분리 |
| **KV chunk 시작 (split-KV)** | inline 산수로 `n_start * BLOCK_N` 더해 base 변경 | `make_block_ptr(offsets=(n_start * BLOCK_N, 0))` 한 줄 |
| **chunk 경계 mask** | `tl.where(mask_n, ...)` | **동일** (block_ptr 의 boundary_check 가 chunk 경계 모름 — global S 만 check) |
| **Partial scalar store (m, l)** | `tl.store(PartialM + pid_bh*sb + pid_kv*sk, m_i)` | **동일** (block_ptr 은 vector tile 전용, scalar 는 raw pointer 유지) |
| **Partial vector store (acc)** | `tl.store(PartialAcc + ... + offs_d*sd, acc)` | 1-D `make_block_ptr(block_shape=(BLOCK_D,), order=(0,))` |
| **Reduce kernel partial load** | `tl.arange` 로 직접 인덱싱 | 1-D / 2-D `make_block_ptr` 로 일괄 |

## 3. Block-pointer conversion notes (split-KV 고유)

### 3.1 KV chunk 시작점 정렬 — `n_start * BLOCK_N`

가장 큰 신경거리. ptr 변형은 `kv_start = pid_kv * chunk` 를 매 iter 의 offset 계산에 인라인:

```python
# ptr 변형
n_start = kv_start // BLOCK_N
n_end = tl.cdiv(kv_end, BLOCK_N)
for n in range(n_start, n_end):
    offs_n = n * BLOCK_N + tl.arange(0, BLOCK_N)
    k = tl.load(K + offs_n[:,None]*ss + ...)
```

block_ptr 에선 **시작점을 한 번만 정해두고 advance**:

```python
# block_ptr 변형
n_start = kv_start // BLOCK_N
K_block_ptr = tl.make_block_ptr(
    base=K + pid_bh * stride_kb,
    shape=(S, BLOCK_D), strides=(stride_ks, stride_kd),
    offsets=(n_start * BLOCK_N, 0),    # ← 여기에 chunk 시작점
    block_shape=(BLOCK_N, BLOCK_D),
    order=(1, 0),
)
for n in range(n_start, n_end):
    k = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
    ...
    K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
```

장점: offset 산수가 **한 곳에 격리**, 루프 안엔 `advance` 만. 가독성 개선.

### 3.2 chunk 경계 mask 는 `tl.where` 로 유지

block_ptr 의 `boundary_check=(0,)` 은 `shape` 의 첫 차원 (= S) 와 비교만 함. **chunk 의 kv_end 는 모름**:

```python
mask_n = (offs_n >= kv_start) & (offs_n < kv_end)
qk = tl.where(mask_n, qk, float("-inf"))
```

이 줄은 ptr 변형과 **byte-for-byte 동일**. block_ptr 가 처리해주는 건 "OOB 메모리 접근 안전" 이고, "softmax 에서 -inf 로 마스킹" 은 algorithmic mask 라서 별도.

(이건 BLOCK_PTR_MIGRATION.md 의 일반 원칙 — boundary_check 는 load-time 만, algorithmic mask 는 `tl.where` 로.)

### 3.3 Partial scalar (m, l) 는 raw pointer 유지

`m_i`, `l_i` 는 program 당 fp32 스칼라 한 개:

```python
# block_ptr 으로도 가능하지만 과도함
PartialM_block_ptr = tl.make_block_ptr(
    base=PartialM + pid_bh * stride_pm_b,
    shape=(KV_SPLITS,), strides=(stride_pm_k,),
    offsets=(pid_kv,), block_shape=(1,),
    order=(0,),
)

# 그냥 raw pointer 로 한 줄
tl.store(PartialM + pid_bh * stride_pm_b + pid_kv * stride_pm_k, m_i)
```

후자가 명료. block_ptr 의 가치는 **2D tile 의 boundary 처리 자동화** 인데, scalar 는 boundary 가 없음.

(BLOCK_PTR_MIGRATION.md §"block_table 등 스칼라 로드는 raw pointer 유지" 와 같은 원칙.)

### 3.4 Reduce kernel — partial 통째 load

reduce kernel 은 KV_SPLITS 개 partial 을 한꺼번에 load 해서 결합:

```python
# block_ptr 의 깔끔한 표현
PartialAcc_block_ptr = tl.make_block_ptr(
    base=PartialAcc + pid_bh * stride_pa_b,
    shape=(KV_SPLITS, BLOCK_D),
    strides=(stride_pa_k, stride_pa_d),
    offsets=(0, 0),
    block_shape=(KV_SPLITS, BLOCK_D),
    order=(1, 0),
)
pa = tl.load(PartialAcc_block_ptr)        # [KV_SPLITS, BLOCK_D]
```

ptr 변형의 동일 동작:
```python
offs_k = tl.arange(0, KV_SPLITS)
offs_d = tl.arange(0, BLOCK_D)
pa = tl.load(
    PartialAcc + pid_bh * stride_pa_b
    + offs_k[:, None] * stride_pa_k
    + offs_d[None, :] * stride_pa_d,
)
```

둘 다 같은 데이터를 같은 layout 으로 적재. 출력 PTX 도 거의 동일 (Triton 컴파일러가 양쪽 다 contiguous load 로 최적화).

## 4. 핵심 설계 결정

**(1) prefill 은 그대로.** block_ptr/vllm_split 의 prefill 과 byte-for-byte 동일 — split-KV 가 prefill 에 의미 없음.

**(2) decode_simple 도 그대로.** block_ptr/vllm_split 의 decode 와 byte-for-byte 동일 — KV_SPLITS=1 fast path 로 vllm_split 의 동작 보존.

**(3) KV_SPLITS 가 항상 power of 2.** Reduce kernel 의 `tl.arange(0, KV_SPLITS)` 와 `make_block_ptr(block_shape=(KV_SPLITS,), ...)` 모두 power-of-2 length 만 컴파일됨. Wrapper 가 `_pow2_floor` 로 강제. Heuristic 결과나 user 지정 모두 적용.

**(4) chunk 시작점은 BLOCK_N-aligned floor.** `kv_start` 가 BLOCK_N 의 배수가 아닐 때 `kv_start // BLOCK_N * BLOCK_N` 부터 iter 시작 → 일부 token 이 두 chunk 양쪽에서 load 되지만 mask 로 걸러져 결과 동일. block_ptr 의 `offsets` 가 BLOCK_N 정렬을 자동 가정하지 않으므로 명시.

## 5. 검증

| 단계 | 방법 | 기대값 |
|---|---|---|
| 알고리즘 정확성 (vs ptr 변형) | 같은 입력으로 두 변형 실행 | 동일 출력 (rounding 오차 내) |
| Decode 정확성 (KV_SPLITS sweep) | smoke test, 각 ns 별 | SDPA 와 max_err < 1e-2 (fp16) / 3e-2 (bf16) |
| Plugin 탐지 | `entry_points` | `my_block_ptr_split_v2_backend` 존재 |
| 실제 호출 증거 | engine core stderr | `MyTritonImpl[v2/block_ptr].forward fired (decode, split-KV)` |

RTX 5090 + Triton 3.6.0 실측 (smoke test):
```
prefill (q_len == kv_len, causal)              6/6  PASS
decode (q_len=1, split-KV, block_ptr)         40/40 PASS
ALL PASS
```

ptr/vllm_split_v2 와 동일 max_abs_err 패턴 — 두 표현이 수치적으로 동등.

## 6. 알려진 한계

block_ptr/vllm_split 와 동일 + split-KV 추가 한도:

- Dense KV 가정 (paged 아님). production 으론 `_gather_kv` 로 매 호출 dense 화 필요.
- `max_num_seqs=1` 한정.
- KV_SPLITS power of 2 강제. 휴리스틱이 14 를 골라도 8 로 round-down.
- GQA head packing 미적용 (Tensor Core 활용 ✗ — vllm v1 production 의 추가 최적화).
- vLLM 0.19.1 정확히 고정.

## 7. 파일

```
block_ptr/vllm_split_v2/
├── triton_attn.py                  # block_ptr idiom 으로 4 커널 (split-KV)
├── triton_attention_backend.py     # ptr/vllm_split_v2 와 동일, [v2/block_ptr] 로그 태그
├── pyproject.toml                  # name=my-block-ptr-split-v2-backend
├── qwen3_triton_attention.ipynb    # 15셀 데모 (block_ptr 변형)
└── NOTES.md                        # 본 문서
```

## 8. 학습자가 봐야 할 것

```
diff -u ../../ptr/vllm_split_v2/triton_attn.py triton_attn.py
```

핵심 변화:
- 모든 `tl.load(P + offs[:,None]*sa + offs_d[None,:]*sd, mask=m)` → `tl.load(make_block_ptr(...), boundary_check=...)`
- chunk 시작점 산수가 inline 에서 `make_block_ptr` 의 `offsets=` 한 인자로 정리
- `tl.advance` 가 inner loop 의 offset 갱신을 대체
- algorithmic mask (`tl.where(mask_n, qk, -inf)`) 는 양쪽 변형에서 **동일** — block_ptr 가 대체하지 않는 부분

알고리즘과 grid topology, partial buffer layout, log-sum-exp 결합 식 — 모두 ptr 변형과 byte-for-byte 동일. **block_ptr 은 *어떻게 메모리를 만지는지* 의 표현만** 바꾼 것.
