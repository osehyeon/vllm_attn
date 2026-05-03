# vllm_attn

vLLM 의 attention backend 아키텍처를 **6단계** 로 쪼개 재현한 학습용 저장소.
각 단계는 직전 단계에서 **"정확히 한 가지"** 만 바꾸는 파생 프로젝트이며,
마지막 단계가 vLLM v1 의 실제 `kernel_unified_attention_2d` 와 동일한 구조에 도달한다.

같은 6단계가 **두 가지 메모리 접근 표현**으로 각각 구현되어 있다 — 짝을 비교해서
보면 Triton 의 두 표현 양식의 차이가 명확해진다.

```
vllm_attn/
├── ptr/                          # raw pointer arithmetic 표현
│   ├── vllm_padded_decode/
│   ├── vllm_split/
│   ├── vllm_paged/
│   ├── vllm_multiseq/
│   ├── vllm_unified/
│   └── vllm_varlen/
├── block_ptr/                    # tl.make_block_ptr 표현 (같은 6단계)
│   ├── vllm_padded_decode/
│   ├── vllm_split/
│   ├── vllm_paged/
│   ├── vllm_multiseq/
│   ├── vllm_unified/
│   ├── vllm_varlen/
│   ├── README.md                 # block_ptr 변형 자체의 안내 + caveat
│   ├── BLOCK_PTR_MIGRATION.md    # ptr → block_ptr 변환에서 어려웠던 점
│   └── ptr_vs_block_ptr_examples.ipynb   # 두 표현의 1:1 교환 가능성 사례
└── colab_smoke_test.sh           # Colab 등 GPU 환경에서 12개 커널 일괄 검증
```

모든 구현은 **Triton 커널** 로, Qwen3-0.6B 위에서 vLLM 의 공식
`AttentionBackend` 스펙을 따라 plugin entry point 로 등록된다. 모든 단계는 실제
Qwen3-0.6B E2E 로 검증됨 (CUDA 환경 필요).

---

## 두 변형 한눈에

| 항목 | `ptr/` | `block_ptr/` |
|---|---|---|
| 메모리 접근 표현 | `tl.load(P + offs[:,None]*sa + offs_d[None,:]*sd, mask=...)` | `tl.make_block_ptr(...)` + `tl.load(bp, boundary_check=...)` |
| K transpose | `tl.trans(k)` | `order=(0, 1)` virtual transpose |
| Boundary 처리 | `mask=` 인자에 통합 | `boundary_check=` (load-time 한정) |
| Algorithmic mask (causal 등) | 같은 `mask=`에 통합 | `tl.where`로 분리 |
| Paged KV의 `block_table` 간접 인덱싱 | 한 tile에 여러 logical block 혼재 가능 | **`BLOCK_N == BLOCK_SIZE` 강제**, 매 iter fresh `make_block_ptr` |
| 커널 수학 | 동일 | 동일 |
| Wrapper 시그니처 | 동일 | 동일 |
| smoke test 출력 | 동일 (atol 기준 동등) | 동일 |
| 대응하는 vLLM upstream 코드 | vLLM v1 production 패턴 | upstream 패턴 아님 (arxiv:2511.11581 / IBM `vllm-triton-lib` idiom) |

자세한 차이·caveat·전환 노트는 **[`block_ptr/README.md`](./block_ptr/README.md)**
와 **[`block_ptr/BLOCK_PTR_MIGRATION.md`](./block_ptr/BLOCK_PTR_MIGRATION.md)**
참조. 두 표현의 1:1 교환 가능성 자체에 관심 있으면
**[`block_ptr/ptr_vs_block_ptr_examples.ipynb`](./block_ptr/ptr_vs_block_ptr_examples.ipynb)**.

---

## 전체 로드맵 (두 변형 공통)

```
vllm_padded_decode    prefill 커널 1개 + zero-pad decode                (O(s²), 독자적 교육용)
       ▼
vllm_split            prefill/decode 전용 커널 2개, Python gather       (O(s))
       ▼
vllm_paged            커널이 paged KV 직접 읽음                        (max_num_seqs=1, vLLM v0 decode)
       ▼
vllm_multiseq         multi-seq batch + backend split-dispatch         (vLLM v0 완성형)
       ▼
vllm_unified          커널 통합 + 절대 위치 causal mask                (수학은 v1, grid 는 seq-first)
       ▼
vllm_varlen           seq-aligned flat grid + find_seq_idx binary search (vLLM v1 실제 방식)
```

각 단계 사이의 변화는 **한 가지만**:

| 단계 전환 | 추가되는 것 |
|---|---|
| padded_decode → split | decode 전용 커널 |
| split → paged | 커널이 paged KV 직접 읽기 |
| paged → multiseq | multi-seq batch dispatch |
| multiseq → unified | 두 커널을 수학적으로 통합 (절대 위치 mask) |
| unified → varlen | grid 를 token-flat 으로 전환 + find_seq_idx |

---

## 각 단계 한 줄 요약

| 단계 | 핵심 |
|---|---|
| `vllm_padded_decode` | prefill 커널 하나로 decode 까지 처리 (q zero-pad 트릭) |
| `vllm_split` | decode 전용 커널 추가, backend 에서 Python gather |
| `vllm_paged` | 커널이 `block_table` 로 paged KV 직접 인덱싱 |
| `vllm_multiseq` | Backend 가 prefill/decode 그룹 분할 → 각 커널을 multi-seq batch 로 호출 |
| `vllm_unified` | 커널 1개가 prefill/decode/chunked 모두 처리 |
| `vllm_varlen` | grid 를 seq-aligned flat 으로 + `find_seq_idx` binary search |

각 sub-project 의 `NOTES.md` 에 상세 내용. `block_ptr/<단계>/NOTES.md` 끝에는
"Block-pointer conversion notes" 섹션이 추가되어 ptr 버전과의 차이를 정리한다.

---

## 사전 요구사항

- CUDA GPU (compute capability 8.0+)
- Python ≥ 3.10
- vLLM **0.19.1** (정확히 고정 — 내부 API drift 로 다른 버전에서 깨짐)

각 sub-project 가 자기 `.venv/` 를 가지며 독립 실행. 같은 venv 에 **둘 이상 설치
금지** (`py-modules` 이름 충돌). 특히 `ptr/<단계>` 와 `block_ptr/<단계>` 는
backend entry-point slot 이 같으므로 같은 venv 에 동시 설치 금지.

---

## 시작하기

### 한 sub-project 만 실행

```bash
cd vllm_attn/ptr/vllm_varlen           # 또는 block_ptr/vllm_varlen, 또는 다른 단계
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e .
uv pip install --python .venv/bin/python nbclient ipykernel jupyter
.venv/bin/jupyter notebook qwen3_triton_attention.ipynb
```

또는 `source .venv/bin/activate` 후 `jupyter notebook ...`.

### 12개 모두 일괄 검증 (Colab 권장)

`colab_smoke_test.sh` 가 모든 sub-project 의 `triton_attn.py` smoke test 를
순차 실행하고 PASS/FAIL 집계. Colab GPU 런타임 또는 다른 CUDA Linux 환경에서
사용:

```bash
# Colab 셀 (또는 로컬 CUDA Linux)
git clone <this-repo> && cd vllm_attn
bash colab_smoke_test.sh
```

상세 사용법은 스크립트 첫 부분 주석 참조.

### VS Code / Cursor 에서

sub-project 디렉토리를 열고 Kernel 선택에서 그 sub-project 의 `.venv/bin/python` 선택.
`qwen3_triton_attention.ipynb` 를 위에서 아래로 실행.

### 공통 cell 구조 (15셀, 두 변형 공통)

| 셀 | 내용 |
|---|---|
| 1–2 | 제목 + 설치 안내 |
| 3 | 환경 체크 (CUDA, vllm, triton 버전) |
| 4–5 | Qwen3-0.6B 스펙 표 |
| 6–7 | Triton 커널 설명 + smoke test (SDPA 대비) |
| 8 | Plugin entry point 설명 |
| 9 | entry point 등록 확인 |
| 10 | `LLM(...)` 로드 |
| 11 | `llm.generate(...)` 실행 |
| 12 | 실행 증거 (`fired` 로그) 해설 |
| 13 | CUSTOM 슬롯 점유 확인 |
| 14 | 다음 단계 |

---

## 공통 설계 원칙 (두 변형 모두에 해당)

1. **한 단계 한 변화** — 직전 단계와 `diff` 했을 때 변화가 의도한 한 가지로 국한
2. **Plugin entry point 자동 등록** — 수동 `register()` 호출 없이, `pyproject.toml` 의 `vllm.general_plugins` entry point 로 vLLM 이 시작될 때마다 모든 프로세스 (메인·엔진 코어·워커) 에서 자동 호출. `VLLM_ENABLE_V1_MULTIPROCESSING=0` 같은 환경변수 조작 불필요
3. **관찰 카운터 (`FIRE_COUNTER`)** — `Impl.forward` 가 실제 실행됐다는 증거를 엔진 코어 stderr 에 로그로 남김. subprocess 경계를 넘어 관찰 가능
4. **선언 매트릭스는 vLLM 기본 상속** — `supported_dtypes` / `get_supported_head_sizes` 등은 override 하지 않음. 실제 커널 한계는 `Impl.__init__` 의 assert 로만 방어
5. **vLLM 내부 유틸 재사용** — `triton_reshape_and_cache_flash` (KV write) 는 vLLM 것 그대로 사용

---

## vLLM 실제 코드와의 대응

| 단계 | 대응하는 vLLM 영역 |
|---|---|
| padded_decode | 독자적 (교육용 단순화) |
| split | — |
| paged | vLLM v0 의 `paged_attention_v1/v2_kernel` (`csrc/attention/attention_kernels.cu`) |
| multiseq | vLLM v0 의 prefill (xformers/FA varlen) + decode (paged) 스케줄러 경로 + split dispatch |
| unified | vLLM v1 의 `kernel_unified_attention_2d` 의 **수학 부분** (절대 위치 causal) |
| varlen | vLLM v1 의 `kernel_unified_attention_2d` 전체 (flat grid + `find_seq_idx` 포함). `vllm/v1/attention/ops/triton_unified_attention.py` |

**메모리 접근 표현 측면**: vLLM upstream 의 v0/v1 모두 `ptr/` 변형의 raw pointer
arithmetic 패턴을 쓴다. `block_ptr/` 변형은 학습용 비교 idiom으로,
upstream production 패턴이 아님 (`block_ptr/README.md` caveat 섹션 참조).

---

## 알려진 공통 한계

- vLLM **0.19.1 고정** — 다른 버전은 내부 API drift 로 깨질 수 있음
- `max_num_seqs > 1` 는 `multiseq` 부터, chunked prefill 실증은 `unified/varlen` 에서
- dtype fp16/bf16, head\_dim 은 2의 거듭제곱
- `enforce_eager=True` 고정 (cudagraph 미지원)
- sliding\_window / alibi / logits\_soft\_cap / kv\_sharing / MLA / sparse attention 미지원 (assert 로 거부)
- GQA 를 KV head grid 축으로 접는 최적화 (`BLOCK_M = BLOCK_Q × n_rep`) 는 varlen 단계까지 포함 안 됨. vLLM v1 실제 커널은 이걸 수행
- Split-k decode reduction 미적용 (긴 컨텍스트 decode 에서 vLLM 대비 낮은 utilization)

`block_ptr/` 변형 고유 한계:
- paged 커널의 `BLOCK_N` 이 `BLOCK_SIZE` (KV cache의 block_size) 에 고정됨 → KV
  iteration 횟수 증가 가능. `block_ptr/README.md` 참조.

---

## 학습 경로 추천

**처음 접하는 경우**: `ptr/vllm_padded_decode → ptr/vllm_split → ptr/vllm_paged`
순으로 보면서 "AttentionBackend 스펙을 따라 custom backend 를 쓰는 최소한" 을
이해. 그 다음 같은 단계의 `block_ptr/` 변형을 보면서 표현 차이 비교.

**vLLM 의 continuous batching 을 배우고 싶다면**: `ptr/vllm_multiseq` 부터
시작. prefill/decode batch 분류와 그 스케줄링 효과를 체감.

**v0 → v1 전환의 이유를 알고 싶다면**: `ptr/vllm_multiseq → ptr/vllm_unified → ptr/vllm_varlen`
을 연속으로. 절대 위치 mask 로 "구별이 사라지는" 순간 + grid 토폴로지가
varlen 화되는 순간.

**Triton 의 두 메모리 접근 표현이 궁금하다면**: 같은 단계의 `ptr/<단계>` 와
`block_ptr/<단계>` 의 `triton_attn.py` 를 나란히 diff. 또는
`block_ptr/ptr_vs_block_ptr_examples.ipynb` 에서 1:1 교환 가능성 사례 5개를
직접 실행.

**이미 vLLM 구조에 익숙하다면**: `ptr/vllm_unified` 와 `ptr/vllm_varlen` 의 NOTES
만 읽으면 로드맵 전체의 핵심 요약 확보.

---

## 각 sub-project 의 NOTES.md

```
ptr/vllm_padded_decode/NOTES.md
ptr/vllm_split/NOTES.md
ptr/vllm_paged/NOTES.md
ptr/vllm_multiseq/NOTES.md
ptr/vllm_unified/NOTES.md
ptr/vllm_varlen/NOTES.md

block_ptr/vllm_padded_decode/NOTES.md   (+ Block-pointer conversion notes 섹션)
block_ptr/vllm_split/NOTES.md
block_ptr/vllm_paged/NOTES.md
block_ptr/vllm_multiseq/NOTES.md
block_ptr/vllm_unified/NOTES.md
block_ptr/vllm_varlen/NOTES.md
```

코드보다 NOTES 먼저 읽으면 이해가 빠름.
