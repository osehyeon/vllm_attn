# NOTES — vllm_padded_decode

Qwen3-0.6B × 커스텀 Triton attention 을 vLLM 위에서 돌린 **초기 단계** 구현.
한 개의 prefill 커널로 prefill 과 decode 를 모두 처리한다 (decode 는 q zero-pad 트릭).

> **다음 단계**: `../vllm_split/` — decode 전용 커널을 추가해 `O(s²) → O(s)` 로 개선.

## 1. 목표

Qwen3-0.6B (fp16, Q=16 / KV=8 / head\_dim=128) 의 attention 계산을
단일 Triton 커널로 교체한 상태로 vLLM 을 실행. 학습 포인트는 세 가지:

1. vLLM v1 `AttentionBackend` 계약 따라 custom backend 작성
2. `vllm.general_plugins` entry point 로 자동 등록
3. **Prefill 커널로 decode 까지 처리하는 zero-pad 트릭** — causal mask 의 속성을 이용

## 2. Zero-pad 트릭이 핵심 아이디어

**커널 (`_fwd_kernel_prefill`)** 은 `q_len == kv_len == S` 를 전제로 작성된 causal attention:

```
offs_m = [0, S) 쿼리 축
offs_n = [0, S) 키 축
causal: offs_m >= offs_n
```

즉 커널이 `S` 하나로 mask 를 만들 수 있으려면 **q 와 kv 의 길이가 같아야** 한다.

**그런데 decode 단계에서는** 실제로 q\_len=1, kv\_len=s\_len 으로 길이가 다르다.
이걸 해결하는 방법이 **q zero-pad**:

```
실제 q:   [              q_0]       ← 길이 1
실제 kv:  [k_0, k_1, ..., k_{s-1}]  ← 길이 s_len

         ↓ backend 가 커널 호출 전 q 를 zero-pad
q_pad:   [0, 0, ..., q_0]            ← 길이 s_len, 마지막 위치만 실제 q
kernel 호출: triton_attention(q_pad, k_full, v_full)
         → 출력 [0..s_len-1] 중 마지막 위치만 의미 있음
```

**왜 마지막 위치의 결과가 올바른가?**
Causal mask 덕분에 `q_pad[s_len-1]` 은 **모든 과거 KV** 를 attend 한다.
즉 `softmax(q_pad[-1] · Kᵀ / √d) · V` = 진짜 decode 결과.
앞의 `q_pad[0..s_len-2]` 는 garbage 지만 출력을 버리므로 무관.

## 3. 구조

```
MyTritonMetadata         — per-forward 메타 dataclass
MyTritonMetadataBuilder  — vLLM 이 매 forward 마다 호출
MyTritonImpl             — AttentionImpl. forward() 에서 prefill/decode 분기
MyTritonBackend          — AttentionBackend. is_mla/is_sparse 기본값 상속
register()               — register_backend(AttentionBackendEnum.CUSTOM, "...")
```

Backend 의 prefill/decode 분기 (`MyTritonImpl.forward`):

```python
if q_len == s_len:
    # prefill: q/k/v 같은 길이 → triton_attention 그대로
    o = triton_attention(q, k, v, scale=self.scale)
else:
    # decode: q 를 kv_len 까지 zero-pad → 같은 커널 호출 → 마지막 출력만 사용
    q_pad = torch.zeros(1, Hq, s_len, D, ...)
    q_pad[:, :, -1, :] = q[:, :, 0, :]
    o_pad = triton_attention(q_pad, k_full, v_full, scale=self.scale)
    output[:1].copy_(o_pad[:, :, -1, :])
```

## 4. 장단점

| | |
|---|---|
| **장점** | 커널 1개만 작성 → 학습 난이도 낮음. causal mask 의 속성을 구체적으로 체감 |
| **단점** | Decode 비용 `O(s_len²)` — zero-pad 쿼리들이 garbage 계산을 하느라 실제 `O(s_len)` 대비 s\_len 배 느림. 긴 컨텍스트에서 급격히 느려짐 |

> 이 단점을 해결하려면 decode 전용 커널이 필요 → **`vllm_split` 프로젝트** 에서 구현.

## 5. 배운 함정

| # | 내용 | 해결 |
|---|---|---|
| 1 | `get_name()` 이 `AttentionBackendEnum` 에 없는 이름을 반환하면 ValueError | `"CUSTOM"` 반환 |
| 2 | 엔진 코어가 별도 subprocess 로 spawn 되어 메인에서 부른 `register()` 가 전달 안 됨 → CUSTOM 이 조용히 fallback | `vllm.general_plugins` entry point 로 등록 → `pip install -e .` 후 모든 프로세스에서 자동 `register()` |
| 3 | `AttentionImpl.__init__` 에 vLLM 이 새 kwarg 추가 가능성 | `**kwargs` 로 흡수 + `vllm==0.19.1` 고정 |
| 4 | fp32 누산 의도 모호 (`tl.dot(fp16, fp16)` 기본 출력은 fp16) | `out_dtype=tl.float32` 명시 |

## 6. 검증 방법 & 실측

| 단계 | 방법 | 실측 |
|---|---|---|
| 커널 정확성 (prefill) | 랜덤 Q/K/V → SDPA 와 max\_abs\_err 비교 | **PASS** (fp16 × D ∈ {64,128,256}: ~0.0005 / bf16: 0.002~0.008) |
| Plugin 탐지 | `entry_points(group="vllm.general_plugins")` | **PASS** (`my_padded_decode_backend -> triton_attention_backend:register`) |
| CUSTOM 슬롯 점유 | `AttentionBackendEnum.CUSTOM.get_path()` | **PASS** |
| 백엔드 실행 증거 | 엔진 코어 stderr 의 `MyTritonImpl.forward fired` 로그 | **PASS** (prefill / decode 두 경로 모두 호출 확인) |
| 출력 품질 | `"The capital of France is"` + max\_tokens=32 | **PASS** |

## 7. 파일

- `pyproject.toml` — `vllm==0.19.1` 고정 + `vllm.general_plugins` entry point
- `triton_attn.py` — 단일 prefill 커널 + `triton_attention(q, k, v, scale=None)` 래퍼
- `triton_attention_backend.py` — AttentionBackend 4개 클래스 + `register()` + decode 분기의 zero-pad
- `qwen3_triton_attention.ipynb` — 15셀 튜토리얼

**사용법**: `pip install -e .` 한 번. 이후 `register()` 수동 호출·multiproc 플래그 불필요.
`vllm_split` 과 동시 설치 금지 (같은 모듈 이름) — 한 venv 에 하나씩.

## 8. 알려진 한계

- **Decode 가 `O(s_len²)`** — zero-pad 트릭의 구조적 비용. 긴 컨텍스트에서 급격히 느려짐
- dtype fp16/bf16 지원, head\_dim 은 2의 거듭제곱 (Triton `tl.arange` 요구)
- `max_num_seqs=1`, `enforce_eager=True` 한정 (배치/cudagraph 미지원)
- sliding\_window · alibi · logits\_soft\_cap · kv\_sharing · MLA · sparse 미지원
- **vLLM 0.19.1 정확히 고정** — 내부 API drift 로 다른 버전에서 깨질 수 있음

## 9. 이 프로젝트의 위치

```
vllm_padded_decode (← 여기)    단일 prefill 커널 + zero-pad decode (O(s²))
       │ decode 성능 개선
       ▼
vllm_split                     prefill + decode 전용 커널 분리 (O(s))
       │ varlen / multi-seq / cudagraph
       ▼
vllm_unified                   (미래) 한 커널로 prefill+decode 혼합 배치 처리
```
