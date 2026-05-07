# NOTES — flexattn

Qwen3-0.6B × **FlexAttention**-backed attention. vllm_attn 시리즈의 세 번째
사이드 트랙 — Triton 커널 자리를 PyTorch 내장 `flex_attention` + `BlockMask`
(torch 2.5+)로 갈아끼웠을 때 vLLM v1 의 `AttentionBackend` 인터페이스가
어떻게 적응하는지 보는 프로젝트.

> **자매 프로젝트**:
> - `../flashinfer/` — FlashInfer wrapper, split-dispatch, CSR 변환 필요
> - `../flashattn/` — FlashAttention-2, single varlen launch (unified)
> - `../ptr/vllm_unified/` — 직접 작성한 Triton 커널 1개, 3종 통합

## 1. 무엇인가

`torch.nn.attention.flex_attention.flex_attention` 은 PyTorch 2.5 부터 내장된
flexible attention 커널이다. `BlockMask` 객체가 희소 접근 패턴을 인코딩하고,
`create_block_mask(mask_mod, ...)` 가 Python 함수에서 `BlockMask` 를 구축한다.

이 프로젝트의 교육적 목표:
- **paged KV → logical index 브리지**를 `physical_to_logical` 역방향 테이블로
  구현하는 방법을 보여 준다
- `mask_mod` 클로저가 어떻게 물리적 KV 인덱스를 논리적 인덱스로 변환하는지
  추적 가능하게 노출한다
- 외부 라이브러리 없이 순수 PyTorch 로 완결되는 attention backend 를 제시한다

## 2. 같은 것 / 다른 것 (vs 자매 프로젝트)

| 항목 | vllm_unified (Triton) | flashinfer | **flexattn (이 프로젝트)** |
|---|---|---|---|
| 커널 소스 | 직접 작성 Triton | FlashInfer 외부 라이브러리 | PyTorch 내장 flex_attention |
| Prefill/Decode 분기 | 없음 — 수학적 통합 | 있음 — wrapper 클래스 분리 | **없음** — BlockMask 가 통합 |
| 한 forward 의 launch 수 | 1 | 1~2 | **1** |
| 메타데이터 변환 | block_table + seq_lens 직접 | CSR triple (kv_indptr 등) | **physical_to_logical 역표** |
| 외부 의존 | Triton | flashinfer-python | **없음** (torch >= 2.5 만) |
| KV split dim | unbind(1) — shape (B,2,bs,H,D) | unbind(1) | **unbind(0)** — shape (2,B,bs,H,D) |
| workspace / plan | 없음 | 128 MiB workspace + plan() | **없음** |
| BlockMask 구축 위치 | N/A | N/A | **builder.build()** (forward 아님) |
| torch.compile | 선택 가능 | 해당 없음 | **필수** (stock 과 동일) |

## 3. BlockMask 브리지 — physical→logical 역매핑 설명

### 왜 필요한가

flex_attention 의 `mask_mod(b, h, q_idx, kv_idx)` 는 **논리적** 인덱스를 받는다.
vLLM 의 paged KV cache 는 **물리적** 블록 번호로 토큰을 저장한다.
이 두 세계를 연결하는 것이 `physical_to_logical` 역표이다.

### 구조

```
block_table[req, logical_block] = physical_block   (vLLM 가 관리)
physical_to_logical[req, physical_block] = logical_block  (우리가 구축)
```

mask_mod 안에서:
```python
phys_block = kv_idx // block_size
offset     = kv_idx % block_size
log_block  = physical_to_logical[req, phys_block]   # -1 이면 미할당
logical_kv = log_block * block_size + offset
```

### 작은 예시 (block_size=16, 3 요청, num_gpu_blocks=8)

```
block_table (logical → physical):
  req 0: [phys 3, phys 7, phys 5]   (seq_len=44, 3블록)
  req 1: [phys 1, phys 4]           (seq_len=21, 2블록)
  req 2: [phys 6]                   (seq_len=16, 1블록)

physical_to_logical (physical → logical), -1 = 미할당:
  req 0: [-1, -1, -1,  0, -1, 2,  -1,  1]   (phys 0..7)
  req 1: [-1,  0, -1, -1,  1, -1, -1, -1]
  req 2: [-1, -1, -1, -1, -1, -1,  0, -1]
```

kv_idx=80 이 req 0 에서 어떤 논리 위치인가:
- phys_block = 80 // 16 = 5
- offset = 80 % 16 = 0
- log_block = physical_to_logical[0, 5] = 2
- logical_kv = 2 * 16 + 0 = 32

→ `mask_mod` 는 `logical_kv(32) <= logical_q` 를 평가해 causal mask 를 적용.
커널 자체는 paging 을 전혀 모른다 — 역표가 그 정보를 숨겨 준다.

**교육적 헤드라인**: "BlockMask 는 vLLM 의 물리 paged KV layout 과 순수-Python
mask_mod 사이의 브리지다 — mask_mod 는 역매핑을 통해 논리 토큰 인덱스를 받으므로
커널 자체는 paging 에 완전히 무관하다."

## 4. KV cache layout 비대칭 주의

이 프로젝트의 KV cache shape 은 `(2, num_blocks, block_size, Hkv, D)`.
dim-0 이 K/V 분리 축이므로 `kv_cache.unbind(0)` 으로 분리한다.

flashinfer 와 vllm_unified 는 `(num_blocks, 2, block_size, Hkv, D)` 이고
`unbind(1)` 을 사용한다. 같은 `triton_reshape_and_cache_flash` 를 호출하지만
두 레이아웃은 메모리에서 다르게 배치된다.

이 레이아웃은 stock `vllm.v1.attention.backends.flex_attention.FlexAttentionBackend`
와 동일하다 — 시작부터 호환성을 의도한 설계.

## 5. 왜 plan 없고, workspace 없고, torch.compile 없는가 (교육용)

**plan 없음**: flex_attention 에는 FlashInfer 의 2단계 plan/run 패턴이 없다.
`flex_attention(q, k, v, block_mask=...)` 한 번 호출로 끝. CPU 오버헤드가
없으므로 workspace 도 필요없다.

**workspace 없음**: flex_attention 은 내부 scratch buffer 를 자동 관리한다.
flashinfer 처럼 128 MiB 를 사전 할당할 필요가 없으므로 profile_run OOM 위험도
없다.

**torch.compile 필수**: `flex_attention` 을 eager 로 호출하면 `sdpa_dense` (math fallback) 로 dispatch되어, BlockMask sparsity 를 무시하고 전체 N×K logits 텐서를 할당하는 full matmul 을 실행한다 — 실제 LLM KV cache 크기에서 즉시 OOM. `torch.compile` 을 통해야만 BlockMask sparsity 를 실제로 존중하는 Triton sparse kernel 로 도달한다. 이것 자체가 하나의 교육 포인트다: BlockMask 의 sparsity 는 런타임 eager API 기능이 아니라 `torch.compile` 이 활성화해 주는 기능이다. stock vLLM 도 동일하게 `torch.compile(flex_attention, fullgraph=True)` 를 사용한다.

**BlockMask 를 builder 에서 구축**: 28개 Qwen3 레이어가 한 forward 에서 같은
메타데이터를 공유한다. forward 마다 build() 에서 1회 구축 → 28회 절약.

## 6. 알려진 한계 (이 프로젝트)

- `cudagraph_support = NEVER` — CUDAGraph 와 eager BlockMask 구축의 호환성
  검증 없음
- `common_prefix_len > 0` (cascade) 미지원
- Sliding window / alibi / soft cap / MLA / kv_sharing 모두 assert 거부
- KV cache dtype: auto / fp16 / bf16. fp8 / nvfp4 미지원 (`is_quantized_kv_cache`)
- `create_block_mask` 를 `torch.compile` 없이 호출 — 매 forward 마다 Python
  수준 sparsity 그리드 평가 (B×N/block_size 루프 오버헤드)
- block_size 가 kv_block_size 와 일치해야 함 (vLLM 기본 16 OK)
- 순수 causal decoder 전용 — encoder-only 또는 prefix LM 미지원

## 7. 의존성

```
vllm == 0.19.1
torch >= 2.5   # flex_attention + BlockMask 가 2.5부터 안정화
```

외부 라이브러리 불필요 — 이것이 flashinfer 대비 가장 큰 환경 단순화.

## 8. 파일

- `pyproject.toml` — name `my-flexattn-backend`, entry point
  `my_flexattn_backend = "flex_attn_attention_backend:register"`
- `flex_attn_attention_backend.py` — 4단 구조 (Metadata / Builder / Impl /
  Backend) + register helper. KV write → flat reshape → flex_attention 한 번.
- `NOTES.md` — 이 파일
- `qwen3_flexattn_attention.ipynb` — 커스텀 backend 교육용 노트북
- `qwen3_flexattn_reference.ipynb` — stock `FLEX_ATTENTION` 비교용 노트북

## 9. 자매 프로젝트 크로스링크

```
vllm_attn/
  ├─ ptr/vllm_unified/      Triton 1 커널, unified dispatch
  ├─ flashinfer/            FlashInfer wrapper, split dispatch, CSR 변환
  ├─ flashattn/             FlashAttention-2
  └─ flexattn/  (← 여기)   PyTorch flex_attention, BlockMask 브리지, no split
```

세 backend 의 `*_attention_backend.py` 를 나란히 diff 하면:
- **공통**: Backend/Impl/Metadata/Builder 4단, CUSTOM 슬롯, triton_reshape KV write
- **차이**: Metadata 변환 (CSR vs physical_to_logical), Impl.forward 의 kernel 호출
- **KV shape**: flashinfer/vllm_unified = `(B,2,bs,H,D)`, flexattn/flashattn = `(2,B,bs,H,D)`
