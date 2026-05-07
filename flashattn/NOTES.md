# NOTES — flashattn

Qwen3-0.6B × **FlashAttention**-backed attention. vllm_attn 시리즈의 세 번째
사이드 트랙 — `flash_attn_varlen_func` (vLLM 번들) 을 백엔드로 사용할 때
인터페이스가 어떻게 단순해지는지 보는 프로젝트.

> **자매 프로젝트**:
> - `../ptr/vllm_unified/` — 같은 인터페이스, 직접 작성한 Triton 커널 1개
> - `../flashinfer/` — 외부 라이브러리 FlashInfer, CSR 변환 + plan/run
>
> 세 프로젝트의 `*_attention_backend.py` 를 나란히 보면 "커널 본체 자리만 다르다"
> 는 게 한 눈에 들어온다. 그리고 **이 프로젝트의 빌더가 가장 짧다**.

## 1. 무엇인가

`flash_attn_varlen_func` (= `vllm.vllm_flash_attn.flash_attn_varlen_func`) 는
vLLM 의 블록 테이블 형식을 **그대로** 받는다:

- `cu_seqlens_q` = vLLM 의 `query_start_loc`
- `seqused_k` = vLLM 의 `seq_lens`
- `block_table` = vLLM 의 `block_table_tensor`

따라서 `MetadataBuilder.build()` 는 필드 이름만 바꾸는 pass-through (~15줄) 다.
변환도, `plan()` 도, 워크스페이스 버퍼도 없다.

## 2. 같은 것 / 다른 것 (vs flashinfer, vs vllm_unified)

같은 것:
- Backend / Impl / Metadata / MetadataBuilder 4단 구조
- `AttentionBackendEnum.CUSTOM` 슬롯에 `register_backend` 로 등록
- KV write 는 동일하게 `triton_reshape_and_cache_flash` 사용
- `accept_output_buffer = True` / `forward_includes_kv_cache_update = True`
- Single launch (no split-dispatch) — vllm_unified 와 같은 구조적 통합

다른 것:

| 항목 | vllm_unified (Triton) | flashinfer | **flashattn (이 폴더)** |
|---|---|---|---|
| 커널 | 직접 작성한 Triton 1개 | FlashInfer 2 wrapper | `flash_attn_varlen_func` 1회 |
| 빌더 비용 | pass-through | CSR 변환 + plan() | **pass-through** (최소) |
| 워크스페이스 | 없음 | 128 MiB | **없음** |
| Split-dispatch | 없음 | prefill / decode 분리 | **없음** |
| KV layout | `(num_blocks, 2, ...)` unbind(1) | 동일 unbind(1) | **`(2, num_blocks, ...)` unbind(0)** |
| 외부 의존 | Triton (vLLM 번들) | flashinfer-python 휠 별도 | vLLM 번들 FA2 |

## 3. varlen 인터페이스 — 필드가 곧 인자다

`flash_attn_varlen_func` 가 받는 paged-KV 인자들과 `CommonAttentionMetadata` 필드가
1:1 대응한다:

```python
# CommonAttentionMetadata           flash_attn_varlen_func 인자
query_start_loc   (B+1,) int32  →  cu_seqlens_q
seq_lens          (B,)   int32  →  seqused_k
block_table_tensor (B, M) int32 →  block_table
max_query_len     int           →  max_seqlen_q
max_seq_len       int           →  max_seqlen_k
num_actual_tokens int           →  query[:N] / output[:N] 슬라이스
```

변환이 없으니 빌더는 단순 래핑만 한다. 이 단순성이 이 프로젝트의 핵심 메시지:
"라이브러리가 vLLM 의 메타데이터 형식을 직접 받는다면 빌더는 공짜다."

## 4. KV layout 비대칭 — 가르침 포인트

이 시리즈에서 KV cache 텐서의 K/V 분리 축이 다르다:

| 프로젝트 | `get_kv_cache_shape` | K/V 분리 방법 |
|---|---|---|
| vllm_unified | `(num_blocks, 2, block_size, Hkv, D)` | `unbind(1)` |
| flashinfer | `(num_blocks, 2, block_size, Hkv, D)` | `unbind(1)` |
| **flashattn (이 폴더)** | `(2, num_blocks, block_size, Hkv, D)` | **`unbind(0)`** |

vLLM 의 `AttentionBackend` 추상은 각 backend 가 `get_kv_cache_shape()` 로 자신의
레이아웃을 선언하므로, 모든 backend 가 같은 모양을 써야 할 이유가 없다. 이 프로젝트는
의도적으로 stock `FlashAttentionBackend` 와 같은 `(2, num_blocks, ...)` 를 골라서
"같은 인터페이스 안에서도 레이아웃이 백엔드별로 다를 수 있다" 는 점을 보여 준다.

> **주의**: `kv_cache.unbind(0)` vs `unbind(1)` 를 혼동하면 K/V 가 뒤바뀌어
> 조용히 틀린 결과가 나온다. 항상 `get_kv_cache_shape` 의 dim-0 이 무엇인지
> 확인하고 `unbind` 축을 결정한다.

## 5. 왜 plan 도, workspace 도 없는가

FlashInfer 가 workspace 를 필요로 하는 이유: split-K 리덕션 과정에서 중간 결과를
GPU 메모리에 쓰고, 그 공간을 미리 잡아 두어야 메모리 프로파일러가 정확하게 계정할 수
있기 때문이다. `flash_attn_varlen_func` 는 내부적으로 CUDA 커널이 동적 shared memory
를 쓰거나 커널 자체가 리덕션을 통합 처리하므로 별도의 외부 버퍼가 불필요하다.

결과: `profile_run` → `build()` → `forward()` 에서 발생하는 GPU 메모리 할당이 없다
— 따라서 OOM 패턴 (flashinfer 의 "workspace-in-forward" 함정) 이 구조적으로 불가능.

## 6. 알려진 한계 (이 프로젝트의 교육적 배제)

- `cudagraph_support = NEVER` — CUDA Graph 지원 불포함
- `common_prefix_len > 0` (cascade) — `NotImplementedError`
- alibi / sliding window / logits_soft_cap — `assert` 거부
- kv_sharing — `assert` 거부
- fp8 KV cache on sm_120 — sm_120 (RTX 5090) 에서 FA2 fp8 wheel 없음 → 거부
- head_dim: `% 8 == 0` and `<= 256` (FA2 라이브러리 제약)
- block_size: `% 16 == 0` (FA2 라이브러리 제약, vLLM 기본 16 OK)
- `fa_version=2` 고정 — FA3 (sm_90+ 전용) 는 이 교육 백엔드 범위 밖

## 7. 의존성

```
vllm == 0.19.1
torch
```

`flash_attn_varlen_func` 는 `vllm.vllm_flash_attn` 에서 import 한다.
vLLM 0.19.1 에 번들된 버전으로, PyPI 의 `flash-attn` 패키지를 별도로 설치하지
않는다.

> **중요**: PyPI `flash-attn` 은 sm_120 (RTX 5090 / Blackwell) 용 wheel 이
> 아직 없다. `pip install flash-attn` 하면 설치되더라도 런타임에 틀린
> CUDA 커널이 선택될 수 있다. 반드시 vLLM 번들 `vllm.vllm_flash_attn` 을 사용.

## 8. 파일

- `pyproject.toml` — name `my-flashattn-backend`, entry point
  `my_flashattn_backend = "flash_attn_attention_backend:register"`
- `flash_attn_attention_backend.py` — 4단 구조 (Metadata / Builder / Impl /
  Backend) + register helper. 빌더는 pass-through, Impl 은 단일 varlen 호출.
- `NOTES.md` — 이 파일
- `qwen3_flashattn_attention.ipynb` — 커스텀 백엔드 실행 노트북
- `qwen3_flashattn_reference.ipynb` — stock `FLASH_ATTN` 베이스라인 노트북

## 9. 4-way 비교표

| 항목 | vllm_unified | flashinfer | **flashattn (이 폴더)** | flexattn |
|---|---|---|---|---|
| 커널 출처 | 직접 작성 Triton | FlashInfer 외부 wheel | vLLM 번들 FA2 | flexattn — see ../flexattn/NOTES.md |
| 빌더 복잡도 | pass-through | CSR 변환 + plan() | **pass-through** | — |
| Workspace | 없음 | 128 MiB (builder) | **없음** | — |
| Split-dispatch | 없음 (단일 커널) | prefill / decode 분리 | **없음 (단일 varlen)** | — |
| KV layout dim-0 | num_blocks | num_blocks | **K 또는 V** | — |
| unbind 축 | 1 | 1 | **0** | — |
| 외부 pip 의존 | 없음 | flashinfer-python | **없음** | — |
| fp8 KV (sm_120) | N/A | N/A | 불가 | — |
| head_dim 제약 | 2의 거듭제곱 | {64,128,256} | **% 8 == 0, <= 256** | — |

## 10. 학습 포인트 요약

이 프로젝트가 보여주는 핵심:

1. **라이브러리 API 디자인이 빌더 복잡도를 결정한다**. FA2 가 vLLM 형식을 그대로
   받도록 설계되었기 때문에 빌더가 공짜. FlashInfer 가 CSR 을 요구하기 때문에
   빌더가 비싸다. 인터페이스는 동일한데 비용이 다른 이유가 바로 여기 있다.

2. **KV layout 은 backend 의 선택**. `get_kv_cache_shape` 가 이를 선언한다.
   형제 백엔드끼리도 다를 수 있다 — `unbind(0)` vs `unbind(1)` 는 반드시 확인.

3. **single-launch vs split-dispatch 는 라이브러리가 강제한다**. FA2 varlen 은
   단일 호출이 가능하도록 설계; FlashInfer 는 wrapper 클래스가 갈라져 있어서
   불가능. vllm_unified 와 flashattn 이 같은 "1 launch" 구조를 갖지만, 하나는
   직접 작성 커널이고 다른 하나는 외부 라이브러리다.
