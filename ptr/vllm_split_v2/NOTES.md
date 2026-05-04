# NOTES — vllm_split_v2 (split-KV decode)

vllm_split 위에 **decode 경로의 split-KV (FlashDecoding)** 를 추가한 변형. main 6단계 로드맵의 일부가 아니라 **별도 side project** — split-KV 라는 schedule 기법을 격리해서 가르치는 데모.

## 0. 출처

- Tri Dao, Daniel Haziza, Francisco Massa, Grigory Sizov. *Flash-Decoding for long-context inference* (블로그 포스트, 2023.10).
- 본 구현은 그 idea 의 가장 짧은 형태 — dense KV (paged 아님), Q=1 decode 한정, 정적 heuristic 으로 KV_SPLITS 결정.

## 1. 무엇이 바뀌었나 (vllm_split → vllm_split_v2)

| 항목 | vllm_split | vllm_split_v2 |
|---|---|---|
| Prefill 커널 | `_fwd_kernel_prefill` | **동일** (1:1) |
| Decode 커널 수 | 1 (`_fwd_kernel_decode`) | **3** (`_simple` + `_partial` + `_reduce`) |
| Decode grid | `(B*Hq,)` 1D | **`(B*Hq, KV_SPLITS)` 2D** + reduce `(B*Hq,)` 1D |
| Decode 임시 buffer | 없음 | partial m / l / acc (fp32, ≈ B*Hq × KV_SPLITS × (D+2) × 4 B) |
| Wrapper dispatch | 단일 경로 | KV_SPLITS=1 fast path / KV_SPLITS≥2 partial+reduce |
| Backend forward | 변경 없음 | 변경 없음 (wrapper 시그니처 동일) |

알고리즘 (online softmax) 는 그대로. **schedule layer 만 추가** — 이게 FlashDecoding 의 본질.

## 2. 왜 decode 만 건드렸나

prefill 의 grid = B × Hq × ⌈S/BLOCK⌉. Qwen3 (Hq=16) prefill 에서 S=512 면 grid block 수 ≈ 16 × 8 = 128 → A100 의 108 SM 거의 채움. **이미 SM 포화** 상태라 split-KV 가 줄 이득 없음 (오히려 reduction 오버헤드만 추가).

decode 는 grid = B × Hq = 1 × 16 = 16 block. SM 의 ~15% 만 점유 → **여기에만 추가 parallelism 필요**.

## 3. 두 단계 split-KV 의 산수

### Phase 1 (`_fwd_kernel_decode_partial`)

각 program `(pid_bh, pid_kv)` 가 자기 KV chunk 만 처리:

```
chunk_size = ⌈S / KV_SPLITS⌉
kv_start = pid_kv × chunk_size
kv_end   = min(kv_start + chunk_size, S)

for n in range(kv_start//BLOCK_N, ⌈kv_end/BLOCK_N⌉):
    # 일반 online softmax (vllm_split 와 동일)
    ...

# Final 결과 대신 partial m/l/acc 저장 (l_i 로 나누지 않음)
PartialM[pid_bh, pid_kv] = m_i
PartialL[pid_bh, pid_kv] = l_i
PartialAcc[pid_bh, pid_kv, :] = acc
```

KV_SPLITS=1 인 경우는 fast path (`_fwd_kernel_decode_simple`) 로 분기 — vllm_split 의 decode 와 동일.

### Phase 2 (`_fwd_kernel_decode_reduce`)

각 program `(pid_bh,)` 가 KV_SPLITS 개 partial 을 통합. **log-sum-exp 안정 결합**이 핵심:

```
m_global  = max_k(PartialM[pid_bh, k])                          # 전역 max
scale[k]  = exp(PartialM[pid_bh, k] - m_global)                 # 각 split 의 가중치
l_global  = Σ_k (PartialL[pid_bh, k] × scale[k])                # 정규화 상수
acc_global = Σ_k (PartialAcc[pid_bh, k, :] × scale[k]) / l_global
Out[pid_bh, :] = acc_global
```

**왜 log-sum-exp?** 단순 합산 (`acc_global = Σ acc_k`) 은:
1. 각 split 이 자체 softmax 를 한 partial 결과 → 그대로 더하면 softmax invariant 깨짐
2. partial m 이 split 마다 달라 scale 이 안 맞음
3. fp16/bf16 에서 overflow / underflow 가능

수학적 등가: 한 통째 softmax 한 결과와 partial 결합한 결과가 (rounding 오차 내에서) 동일.

### KV_SPLITS=0 인 split 의 처리

KV_SPLITS > S 인 degenerate 케이스에서 일부 split 의 chunk 가 비어있을 수 있음. 그 경우:
- inner loop 가 0 회 실행
- partial m = -inf, partial l = 0, partial acc = 0
- reduce 단계: `exp(-inf - m_global) = 0` → 그 split 의 기여 0
- **자동으로 무시됨** (별도 분기 불필요)

## 4. KV_SPLITS 선택 휴리스틱 (`_choose_kv_splits`)

```python
if S < 512:
    return 1                        # 짧은 컨텍스트: 오버헤드 > 이득
num_sms = device.multi_processor_count
target = num_sms × 2                # SM 당 2 block resident 목표
if grid_bh ≥ target:
    return 1                        # 이미 grid 충분
needed = ⌈target / grid_bh⌉
return clamp(needed, 1, MAX_KV_SPLITS=16)
```

Qwen3 + A100 의 실제 결정:
- grid_bh = B × Hq = 16
- target = 108 × 2 = 216
- needed = ⌈216 / 16⌉ = 14
- 결과: **KV_SPLITS = 14 (S ≥ 512 일 때)**

→ A100 에서 14 × 16 = 224 block 이 resident 가능 → 거의 100% SM 활용.

H100 (132 SM) 이면 needed = ⌈264/16⌉ = 17 → MAX 에 걸려 16.

## 5. 핵심 설계 결정

**(1) Fast path 분기 명시.** KV_SPLITS=1 일 때 partial+reduce 로 강제로 보내지 않고 별도 simple 커널 launch. 이유:
- 짧은 KV (decode 초반 generation) 에선 split-KV 오버헤드 (extra kernel launch + partial buffer alloc) 가 이득보다 큼
- vllm_split 의 decode 와 byte-for-byte 동일한 동작을 보존 → fallback 동등성 유지
- 학습자가 "split-KV ≠ 항상 빠름" 을 명확히 보게 됨

**(2) Partial buffer 는 fp32.** decode 가 호출될 때마다 새 buffer alloc. 작아서 (≤ 16 KB × num_layers) 무해.

**(3) heuristic 은 grid_bh × num_SMs 기반, S 만으로 안 결정.** decode 의 batch_size 가 클 때 (예: 8-way batch) 는 grid 가 이미 충분해서 split-KV 가 불필요. heuristic 이 이걸 자동으로 인식.

**(4) Mask 두 겹.** `mask_n = (offs_n >= kv_start) & (offs_n < kv_end)` — chunk 경계와 sequence 경계를 동시에 체크. BLOCK_N 이 chunk 경계와 안 맞을 수 있어서 (kv_start 와 kv_end 가 BLOCK_N 의 배수가 아닐 수 있음) 둘 다 필요.

## 6. 검증 방법 & 실측

| 단계 | 방법 | 기대값 |
|---|---|---|
| Decode 정확성 (KV_SPLITS=1) | smoke test, KV_SPLITS=1 강제 | vllm_split 과 동일 max_err |
| Decode 정확성 (KV_SPLITS=2,4,8) | smoke test, 각 ns 별 강제 | SDPA 와 max_err < 1e-2 (fp16) / 3e-2 (bf16) |
| Decode 정확성 (auto) | smoke test, KV_SPLITS=None | 휴리스틱이 잡은 값으로 동일 통과 |
| 휴리스틱 합리성 | `_choose_kv_splits` 직접 호출 | S<512 → 1; S≥512 + grid 작음 → ≥2 |
| Plugin 탐지 | `entry_points(group="vllm.general_plugins")` | `my_split_v2_backend` 존재 |
| 실제 호출 증거 | engine core stderr | `MyTritonImpl[v2].forward fired (decode, split-KV)` |

## 7. 알려진 한계

- **Dense KV 가정**. 실제 vLLM production 은 paged KV — `_gather_kv` 로 매 decode 호출마다 dense 화 (느림). 학습용 단순화.
- `max_num_seqs=1` 한정. multi-seq batch 지원하려면 backend 의 q_len 분기와 `_gather_kv` 를 batch 화 필요.
- KV_SPLITS 휴리스틱은 정적. 실제론 KV 분포 (warm/cold cache) 와 layer 별 latency 측정에 따라 동적 결정이 더 좋음.
- Reduction 커널이 `KV_SPLITS` 를 constexpr 로 받음 → 동일 launch 안에 다양한 split 깊이 못 섞음 (다행히 한 forward 의 모든 layer 가 같은 ns 를 쓰므로 무해).
- GQA KV-head grid fold (vLLM v1 의 `BLOCK_M = BLOCK_Q × n_rep` 최적화) 미적용.
- **vLLM 0.19.1 정확히 고정** — main 로드맵과 동일 제약.

## 8. 파일

```
vllm_split_v2/
├── triton_attn.py                  # prefill (그대로) + decode 3 커널 (split-KV)
├── triton_attention_backend.py     # vllm_split 동일, 단 fired 로그에 [v2] 태그
├── pyproject.toml                  # name=my-split-v2-backend, entry=my_split_v2_backend
├── qwen3_triton_attention.ipynb    # 15셀 데모 (split-KV 구간 강조)
└── NOTES.md                        # 본 문서
```

**사용법**: 새 venv 에 `pip install -e .` → `LLM(..., attention_backend=AttentionBackendEnum.CUSTOM)`. 같은 venv 에 vllm_split / 다른 stage 와 동시 설치 금지.

## 9. 학습자가 vllm_split 와 비교해서 봐야 할 것

```
diff -u ../vllm_split/triton_attn.py triton_attn.py
```

핵심 변화:
- `_fwd_kernel_decode` (vllm_split) ≡ `_fwd_kernel_decode_simple` (v2). 같은 코드.
- `_fwd_kernel_decode_partial` 는 위 코드에 KV chunk 경계 mask + partial 저장만 추가.
- `_fwd_kernel_decode_reduce` 는 신규. log-sum-exp 결합 식만 들어있는 작은 커널.
- Wrapper 가 KV_SPLITS 결정 + dispatch 로직을 더 가짐 (이게 schedule layer).

알고리즘 (math) 의 변화는 0, schedule 의 변화만. **이 diff 가 정확히 "FA → FlashDecoding" 의 한 줄 요약**.
