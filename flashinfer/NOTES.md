# NOTES — flashinfer

Qwen3-0.6B × **FlashInfer**-backed attention. vllm_attn 시리즈의 사이드 트랙 —
Triton 커널 자리를 외부 라이브러리(FlashInfer)로 갈아끼웠을 때 vLLM v1 의
`AttentionBackend` 인터페이스가 어떻게 적응하는지 보는 프로젝트.

> **자매 프로젝트**: `../ptr/vllm_unified/` — 같은 인터페이스, 직접 작성한
> Triton 커널 1개로 prefill/decode/chunked 통합. 두 프로젝트의
> `*_attention_backend.py` 를 나란히 두고 보면 "커널 본체 자리"만 다르다는
> 게 한 눈에 들어온다.

## 1. 무엇이 같고 무엇이 다른가 (vs vllm_unified)

같은 것:
- Backend / Impl / Metadata / MetadataBuilder 4단 구조
- `AttentionBackendEnum.CUSTOM` 슬롯에 `register_backend` 로 등록
- KV cache shape `(num_blocks, 2, block_size, Hkv, D)` (NHD)
- KV write 는 동일하게 `triton_reshape_and_cache_flash` 사용
- `accept_output_buffer = True` / `forward_includes_kv_cache_update = True`

다른 것:

| 항목 | vllm_unified (Triton) | flashinfer (외부 라이브러리) |
|---|---|---|
| 커널 호출 | `triton_attention_unified(q, kc, vc, block_table, seq_lens, qsl)` 1회 | prefill: `wrapper.plan(...) + run(...)`, decode: 같은 패턴 — 합쳐서 최대 2회 |
| 메타데이터 형식 | `block_table` + `seq_lens` 를 그대로 커널에 전달 | **CSR 변환**: `kv_indptr`, `kv_indices`, `kv_last_page_len` (모두 int32) |
| Prefill/Decode 구별 | 절대 위치 mask 로 수학적 통합 → 분기 없음 | wrapper 클래스 자체가 분리 → **분기 강제** (vllm_multiseq 와 동일한 split-dispatch) |
| 한 forward 의 launch 수 | 1 | 1~2 (prefill 그룹 / decode 그룹 별도) |
| 외부 의존 | Triton 만 | `flashinfer-python` 휠 |

## 2. CSR 변환의 의미

FlashInfer 는 paged KV 를 vLLM 의 `block_table` 형식으로 받지 않는다. 대신
세 개의 1-D int32 텐서로 재포맷한 것을 요구한다.

예시 — 3개 시퀀스, page_size=16:

```
seq 0: 페이지 [3, 7, 11], 마지막 페이지 12개 토큰
seq 1: 페이지 [4, 8],     마지막 페이지 5개 토큰
seq 2: 페이지 [9],        마지막 페이지 16개 토큰

→ kv_indptr        = [0, 3, 5, 6]            # 누적 페이지 수
  kv_indices       = [3, 7, 11, 4, 8, 9]     # flat page ids
  kv_last_page_len = [12, 5, 16]             # 1..page_size
```

`MyFlashInferMetadataBuilder._build_paged_csr` 가 매 forward 직전에 이 변환을
수행한다. 실제 vLLM v1 의 `FlashInferMetadataBuilder` 가 하는 일과 동일.

> 학습용 단순화: 이 프로젝트는 변환을 Python 루프로 한다. 실제 vLLM 은 같은
> 변환을 GPU 친화적 cumsum + gather 로 처리해서 CPU↔GPU 동기화를 피한다.

## 3. 왜 unified 가 못 되는가

vllm_unified 의 핵심 트릭은 "절대 위치 mask 로 prefill/decode/chunked 를
수학적으로 통합" — 한 커널이 모두 처리. FlashInfer 에는 그 자유도가 없다.
**wrapper 가 prefill 용 / decode 용으로 클래스 자체가 갈라져 있고**, `plan()`
의 입력 자료구조도 다르다 (prefill 만 `qo_indptr` 가짐). 따라서 이 backend
는 vllm_multiseq 의 split-dispatch 단계에 멈춰 있다 — 그 위로 unified 로
올라갈 수 없는 게 라이브러리의 설계 결정.

이 한계가 실제 vLLM v1 의 `vllm/v1/attention/backends/flashinfer.py` 에서도
그대로 나타난다 — `_get_prefill_wrapper()` 와 `_get_decode_wrapper()` 가
분리되어 있다.

## 4. plan-then-run 패턴

FlashInfer wrapper 는 모두 2단계.

```
wrapper.plan(qo_indptr, kv_indptr, kv_indices, last_page_len, ...,
             num_qo_heads, num_kv_heads, head_dim, page_size, causal=True)
# CPU 로 보조 자료구조 구성 + JIT 커널 선택. 한 forward 당 1회.

output = wrapper.run(q, paged_kv_cache)
# 실제 GPU 커널 launch.
```

매 forward 마다 `plan()` 이 호출되는데, 여기서 발생하는 CPU 오버헤드 (특히
piecewise CUDA Graph 환경) 를 줄이려고 vLLM 은 `fast_decode_plan` 이라는
캐시 경로를 별도로 사용한다. 이 educational backend 는 그것 없이 매번
`plan()` 을 호출 — 학습 가시성을 위해.

## 5. Wrapper 재사용 (중요)

`_ensure_wrappers()` 는 첫 forward 에서 `BatchPrefillWithPagedKVCacheWrapper`
와 `BatchDecodeWithPagedKVCacheWrapper` 를 한 번만 만들고 이후 재사용한다.
이렇게 해야 FlashInfer 의 JIT 컴파일 결과 (in-memory + on-disk 2단계 캐시)
가 살아 있어 매번 재컴파일이 일어나지 않는다. workspace 버퍼도 두 wrapper
가 **공유** 한다 (FlashInfer 가 명시적으로 허용하는 사용 패턴).

## 6. KV cache layout — 같은 shape, 다른 해석

vllm_unified 와 같은 텐서를 받아온다:

```
kv_cache: (num_blocks, 2, block_size, Hkv, D)
```

vllm_unified 는 `kv_cache.unbind(1)` 로 K/V 를 분리한 뒤 자신의 Triton 커널에
`key_cache`, `value_cache` 두 개로 넘긴다. flashinfer 는 그 텐서를 통째로
`paged_kv_cache` 인자로 넘긴다. FlashInfer 가 내부적으로 `[..., 2, ...]` 축을
K/V 로 해석. layout 명 "NHD" 는 **N**(pages) **H**(heads) **D**(head_dim) 순서.

## 7. 알려진 한계 (이 프로젝트)

- `cudagraph_support = NEVER` — wrapper 의 `use_cuda_graph=True` 모드와
  workspace 사전 할당까지 챙기는 건 학습 범위 외
- `common_prefix_len > 0` (cascade) 미지원 — `MultiLevelCascadeAttentionWrapper`
  를 써야 하는데 분기 추가 필요
- Sliding window / alibi / soft cap / MLA 모두 assert 거부
- KV cache dtype: auto / fp16 / bf16. fp8 / nvfp4 미지원
- head_dim ∈ {64, 128, 256} (FlashInfer 제약)
- page_size ∈ {1, 16, 32, 64} (FlashInfer 제약 — vLLM `block_size` 와 일치해야)
- CSR 변환에서 Python 루프 사용 (B 가 매우 크면 GPU↔CPU 동기 비용 발생)

## 8. 의존성

```
vllm == 0.19.1
flashinfer-python >= 0.2.0   # plan/run API 사용. 0.6.x 호환 확인됨.
torch
```

> FlashInfer 0.2 미만 버전에서는 `plan()` 대신 `begin_forward()` 호출이 필요
> 할 수 있다 (alias 호환되는 버전도 존재).

## 9. 파일

- `pyproject.toml` — name `my-flashinfer-backend`, entry point
  `my_flashinfer_backend = "flashinfer_attention_backend:register"`
- `flashinfer_attention_backend.py` — 4단 구조 (Metadata / Builder / Impl /
  Backend) + register helper. Triton 커널 자리를 wrapper.run() 으로 대체.
- `NOTES.md` — 이 파일

## 10. 자매 프로젝트와의 비교 — 한 줄 요약

```
vllm_unified        직접 작성한 Triton  →  1 kernel,  block_table + seq_lens 직접 사용
flashinfer (이 폴더) 외부 라이브러리      →  2 wrappers, CSR 변환 후 plan → run
```

같은 인터페이스에 다른 본체. vLLM v1 의 `AttentionBackend` 추상이 이 두
양식을 모두 받아낸다는 것을 확인하는 게 이 프로젝트의 가치.

## 11. 학습 포인트 — vLLM 이 attention 라이브러리를 "갈아끼우는" 방법

이 프로젝트가 보여주는 vLLM v1 의 핵심 디자인 결정:

1. **Backend 추상은 라이브러리 친화적**. `AttentionBackend` 가 요구하는 것은
   shape, builder, impl 그뿐. 본체가 직접 작성한 커널이든 외부 wrapper 든
   같은 슬롯에 들어간다.

2. **Metadata 가 라이브러리별 자료구조 변환 지점**. 라이브러리가 다른 페이지
   인덱싱 형식을 요구하면, vLLM 의 공통 포맷 (`block_table` + `seq_lens`) 을
   각자 변환해서 들고 있는 게 builder 의 역할. 이게 모든 attention backend
   가 자기 builder 를 가지는 이유.

3. **KV write 는 backend 가 책임지고 처리**. `forward_includes_kv_cache_update
   = True` 와 `accept_output_buffer = True` 두 contract 만 지키면 vLLM 코어가
   forward 호출만 하고 빠진다.

vllm_unified 와 이 프로젝트의 backend 파일을 diff 해 보면 "라이브러리 swap
의 비용" 이 정확히 얼마인지 보인다 — Metadata 필드 추가 + Builder 의 CSR
변환 + Impl 의 plan/run 호출. 그 외에는 모두 그대로.
