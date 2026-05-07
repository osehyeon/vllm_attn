# NOTES — vllm_split_v3 (chunked-prefill-capable kernel)

`vllm_split` 위에 **chunked prefill 을 prefill 커널 자체의 일반화로** 얹은 변형. 메인 6단계 로드맵의 일부가 아니라 **별도 side project** — "KV cache table 없이도 chunked prefill 은 커널만의 변경으로 가능하다" 는 사실을 격리해서 보여주는 데모.

> **이전**: `../vllm_split/` — prefill 은 `q_len == kv_len` 만 받음
> **참고**: `../vllm_unified/` — chunked + paged + multi-seq 까지 통합한 v1 스타일

## 0. 핵심 명제

> **Chunked prefill 은 attention 수식의 일반화일 뿐이다. KV-cache table / paged 인덱싱과 직교한 개념.**

증명: `vllm_split` 의 prefill 커널 (dense KV, paged 아님) 의 causal mask 한 줄을 절대 위치 (`q_abs = kv_len - q_len + offs_m`) 로 바꾸기만 하면 chunked prefill 이 곧바로 작동한다. 외부에 **paged KV cache, block_table, query_start_loc 등 어떤 indirection 도 등장하지 않는다**. FA2 의 forward 가 q_len ≠ kv_len 을 native 로 받는 것과 동일한 사실.

## 1. 무엇이 바뀌었나 (vllm_split → vllm_split_v3)

| 항목 | vllm_split | vllm_split_v3 |
|---|---|---|
| Prefill 커널 시그니처 | `S` (단일 길이) | **`Q_LEN`, `KV_LEN` 분리** |
| Prefill causal mask | `offs_m >= offs_n` (상대 위치) | **`q_abs >= offs_n` (절대 위치)** |
| Prefill q\_mask | `offs_m < S` | `offs_m < Q_LEN` |
| Prefill kv 순회 | `cdiv(S, BLOCK_N)` | `cdiv(KV_LEN, BLOCK_N)` |
| Prefill grid | `(B*Hq, cdiv(S, BLOCK))` | `(B*Hq, cdiv(Q_LEN, BLOCK))` |
| Wrapper 시그니처 | `q, k, v` (모두 같은 S) | `q [B,Hq,q_len,D]`, `k/v [B,Hkv,kv_len,D]`, **`q_len <= kv_len` 허용** |
| Decode 커널 | 그대로 (1:1) | **동일** (1:1) |
| Backend 분기 | 2-way (`q_len == s_len` / else) | **3-way** (pure prefill / chunked / decode) |
| Backend 의 chunked path | 거부 (else 절이 항상 decode 가정) | `_gather_kv` 로 paged KV 가져온 뒤 prefill 커널 호출 |

알고리즘 (online softmax) 은 그대로. **causal mask 한 줄 + grid 2 줄 + wrapper shape 분리** — 그게 전부.

## 2. 왜 prefill 커널만 건드리고 decode 는 그대로 두나

decode 는 본질적으로 `q_len == 1` 의 특수 케이스고 — chunked prefill 의 일반화 안에 자연스럽게 포함되긴 하지만 — **vllm_unified** 가 이미 보여주듯 통합하면 BLOCK_M 의 (BLOCK_M-1)/BLOCK_M 슬롯이 헛돈다. 본 프로젝트는 "schedule layer 분리" 라는 v0 의 정신을 유지하면서 **chunked 만 prefill 쪽으로 흡수** — decode 는 전용 1D-program 커널이 그대로 빠르다.

즉:
- `vllm_unified`: 한 커널로 prefill / chunked / decode 통합 (BLOCK_M=64 가 decode 에 낭비되는 비용)
- `vllm_split_v3`: prefill 커널이 prefill / chunked 통합, decode 는 별도 (양쪽 다 효율)

## 3. 절대 위치 causal mask — 한 줄로 모든 경우 처리

```python
# v3 prefill kernel
q_abs = KV_LEN - Q_LEN + offs_m            # [BLOCK_M]
...
causal_mask = q_abs[:, None] >= offs_n[None, :]
```

세 케이스가 모두 **같은 식**으로 떨어진다:

| 상태 | `Q_LEN`, `KV_LEN` 관계 | `q_abs` 값 | 그 의미 |
|---|---|---|---|
| Pure prefill | `Q_LEN == KV_LEN` | `[0, 1, ..., KV_LEN - 1]` | 시퀀스 첫 처리 |
| Chunked prefill | `1 < Q_LEN < KV_LEN` | `[KV_LEN - Q_LEN, ..., KV_LEN - 1]` | 긴 prompt 의 중간 chunk |
| Single-token | `Q_LEN == 1` | `[KV_LEN - 1]` | 사실상 decode (전용 커널이 더 빠름) |

`Q_LEN == KV_LEN` 일 때 `q_abs == offs_m` 이므로 vllm_split 의 상대 위치 식 (`offs_m >= offs_n`) 과 정확히 동치 — **기존 prefill 동작이 깨지지 않는 일반화**.

## 4. Backend 의 chunked path — `_gather_kv` 재활용

vllm_split 에서 decode 전용으로만 쓰던 `_gather_kv` 헬퍼가 chunked 경로에서도 그대로 쓰인다. KV write 가 `forward()` 첫 단계에서 이미 끝났으므로, gather 시점에 paged cache 에는 **이전 chunk + 현재 chunk** 의 KV 가 모두 들어있다 → s\_len 길이의 dense KV 텐서가 나옴.

```python
# v3 backend forward, 3-way
if q_len == s_len:                    # pure prefill (cold start)
    triton_attention_prefill(q_new, k_new, v_new)        # KV 도 새 토큰만
elif q_len == 1:                      # decode
    k_full, v_full = _gather_kv(...)                     # 누적 s_len
    triton_attention_decode(q_last, k_full, v_full)
else:                                 # chunked prefill
    k_full, v_full = _gather_kv(...)                     # 누적 s_len 포함 새 chunk
    triton_attention_prefill(q_chunk, k_full, v_full)    # ← 같은 prefill 커널!
```

**chunked 분기에서 `triton_attention_prefill` 을 그대로 호출**한다는 점이 핵심. 새 함수도, 새 커널도 없다.

## 5. 한계 — 학습용으로 단순화한 것들

| 항목 | 본 프로젝트 | vLLM v1 production |
|---|---|---|
| KV 인덱싱 | dense (gather 비용 큼) | paged 직접 인덱싱 (커널 안에서 block_table 조회) |
| Multi-seq | 1 seq 한정 (`max_num_seqs=1`) | varlen multi-seq, `query_start_loc` 로 분리 |
| Q tensor 레이아웃 | `[B, Hq, q_len, D]` | flat-packed `[total_q_tokens, Hq, D]` |
| Cudagraph | NEVER | 지원 |
| 시그니처 일관성 | prefill / decode 두 함수 | 단일 unified 커널 (vllm\_unified) |

이 항목들은 **chunked prefill 의 본질과 무관** 하다는 게 본 프로젝트의 메시지. 본질은 §3 의 한 줄 식. 나머지는 production 운영 효율을 위한 별개 axis.

## 6. 검증 (kernel 단독 smoke test)

```bash
cd vllm_split_v3
python triton_attn.py
```

3 종류 × dtype 2 × 케이스 다수 = 22 조합 PASS 기대:

- **prefill** (q\_len == kv\_len): `D ∈ {64, 128, 256}` × fp16/bf16 = 6
- **chunked** (1 < q\_len < kv\_len): `(q,kv,D)` 5 조합 × fp16/bf16 = 10
  - (32, 128, 128), (16, 128, 128), (64, 256, 128), (31, 97, 64), (8, 64, 256)
  - 마지막 두 케이스가 BLOCK 경계와 어긋나서 mask 버그를 잡아냄
- **decode** (q\_len == 1): `kv_len ∈ {32, 128, 1024}` × fp16/bf16 = 6

각 케이스가 SDPA reference 의 마지막 q\_len 행과 비교 → `max_abs_err < 1e-2` (fp16) / `< 3e-2` (bf16).

## 7. E2E (vLLM 에서 chunked 가 실제 발동되는지)

`vllm_unified/NOTES.md` §7 의 관찰처럼 chunked 가 vLLM 스케줄러에서 발동되려면 다음 조건이 모두 필요:

```python
LLM(
    model="Qwen/Qwen3-0.6B",
    attention_backend=AttentionBackendEnum.CUSTOM,
    enable_chunked_prefill=True,                # v0 default OFF
    max_num_batched_tokens=64,                  # 작게 잡아야 chunk 발동
    max_num_seqs=1,                             # 본 프로젝트 한정
    enforce_eager=True,
)
# 그리고 prompt 길이가 max_num_batched_tokens 보다 길어야 chunked 분할
```

위 조건 하에서 `_logger.warning("...fired (chunked prefill) q_len=%d s_len=%d", ...)` 로그가 engine core stderr 에 찍히면 backend 의 chunked 분기가 실제로 호출된 증거. (`max_num_seqs=1` 한정이라 v0 의 multi-seq 효과는 안 나오지만, "chunked prefill 한 시퀀스" 자체는 관찰 가능.)

## 8. 파일

```
vllm_split_v3/
├── triton_attn.py                  # prefill 커널 일반화 (Q_LEN/KV_LEN 분리, q_abs causal) + decode 그대로
├── triton_attention_backend.py     # 3-way 분기 (pure / chunked / decode), FIRE_COUNTER 에 chunked 키 추가
├── pyproject.toml                  # name=my-split-v3-backend, entry=my_split_v3_backend
├── qwen3_triton_attention.ipynb    # vllm_split 와 동일 (시그니처 backwards-compatible)
└── NOTES.md                        # 본 문서
```

**사용법**: 새 venv 에 `pip install -e .` → `LLM(..., attention_backend=AttentionBackendEnum.CUSTOM, enable_chunked_prefill=True, max_num_batched_tokens=64, ...)`. vllm\_split / 다른 stage 와 동시 설치 금지.

## 9. 학습자가 vllm_split 와 비교해서 봐야 할 것

```bash
diff -u ../vllm_split/triton_attn.py triton_attn.py
diff -u ../vllm_split/triton_attention_backend.py triton_attention_backend.py
```

핵심 변화는 ~10 줄:
1. 커널 인자: `S` → `Q_LEN, KV_LEN` (2 줄)
2. `q_abs = KV_LEN - Q_LEN + offs_m` 추가 (1 줄)
3. causal mask: `offs_m >= offs_n` → `q_abs >= offs_n` (1 줄)
4. q\_mask / kv\_mask / grid M 축이 `Q_LEN` 또는 `KV_LEN` 로 분리 (~3 줄)
5. wrapper: `q.shape[2]` 와 `k.shape[2]` 분리 + `assert q_len <= kv_len` (~2 줄)
6. backend: 2-way → 3-way 분기 (chunked elif 추가, ~10 줄)

알고리즘 변화 0, **수식 일반화 한 줄이 전부**. 이 diff 가 정확히 "**FA forward 의 self-attn (q\_len==kv\_len) 한정 가정을 푼다**" 의 실제 의미.

## 10. 알려진 한계

- `max_num_seqs=1` 한정 (vllm\_split 와 동일).
- Dense KV gather (`_gather_kv`) — chunked path 에서도 dense 화 비용 발생. paged 직접 인덱싱은 vllm\_paged 이상에서만.
- Cudagraph 미지원, sliding\_window · alibi · logits\_soft\_cap · kv\_sharing · MLA · sparse 미지원.
- **vLLM 0.19.1 정확히 고정**.

## 11. 본 프로젝트의 위치 (side branch)

```
            vllm_padded_decode → vllm_split → vllm_paged → vllm_multiseq → vllm_unified   (메인 로드맵)
                                    │
                                    ├──→ vllm_split_v2  (decode 에 split-KV 추가)
                                    │
                                    └──→ vllm_split_v3  (← 여기, prefill 에 chunked 일반화)
```

두 v2/v3 는 메인 로드맵의 변형이고, **각자 독립적인 한 가지 schedule/math 변경**을 격리해서 가르친다. v2 가 schedule layer (KV split + reduce), v3 가 math layer (causal mask 일반화) — 둘이 직교라 따로 공부할 가치가 있다.
