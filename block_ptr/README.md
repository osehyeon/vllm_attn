# vllm_attn — block_ptr 변형

`vllm_attn`의 6단계 학습용 저장소를 미러링한 변형. **유일한 차이는 Triton 커널이
raw pointer arithmetic 대신 `triton.language.make_block_ptr`(tile I/O 전용)을
쓴다는 것**이다. 그 외 단계 구분·인터페이스·smoke test 결과·plugin entry point
등록 방식은 모두 `vllm_attn`과 동일.

마지막 단계가 vLLM v1의 `kernel_unified_attention_2d`와 동일한 **수학·grid 구조**
에 도달하는 것은 같지만, **벡터화된 메모리 접근**의 표현 방식이 다르다.

---

## vllm_attn 와의 차이 (정직한 caveat)

이 저장소는 **block-pointer 기반 attention 커널의 표현 방법을 학습하기 위한 변형**
이며, **vLLM upstream의 실제 production 커널이 사용하는 패턴은 아니다**:

- vLLM v1의 `vllm/v1/attention/ops/triton_unified_attention.py`는
  `tl.make_block_ptr` 없이 raw pointer arithmetic(`Ptr + offs[:,None]*stride + ...`)을 쓴다.
- 본 저장소의 idiom은 arxiv:2511.11581 *"The Anatomy of a Triton Attention Kernel"*
  및 IBM `vllm-triton-lib`의 표현과 일치한다.
- 두 표현 모두 같은 PTX/SASS로 컴파일될 수 있으므로 **수학·정확성은 등가**이며,
  차이는 **가독성·유지보수·boundary 처리 추상화 수준**에 국한된다.

이 변형의 학습 가치:

1. paged KV의 간접 인덱싱(`block_table`)이 block_ptr 추상화에 어떤 제약을
   강제하는지 — `tl.advance`가 base 주소 변동을 따라갈 수 없으므로 매 iteration
   마다 fresh `tl.make_block_ptr`을 만들어야 함
2. K transpose를 `tl.trans` 대신 `make_block_ptr`의 `order=(0, 1)` virtual
   transpose로 표현하는 idiom
3. mask vs `boundary_check`의 책임 분리 — boundary는 load-time, causal/q_mask
   outer product는 여전히 `tl.where`

각 sub-project의 `NOTES.md` 끝에 "Block-pointer conversion notes" 섹션이
무엇이 바뀌고 무엇이 안 바뀌었는지 정확히 기록한다.

---

## 전체 로드맵 (vllm_attn과 동일)

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
vllm_varlen           seq-aligned flat grid + find_seq_idx binary search (vLLM v1 실제 grid 방식)
```

각 단계 사이의 변화는 한 가지만 (vllm_attn과 동일):

| 단계 전환 | 추가되는 것 |
|---|---|
| padded_decode → split | decode 전용 커널 |
| split → paged | 커널이 paged KV 직접 읽기 |
| paged → multiseq | multi-seq batch dispatch |
| multiseq → unified | 두 커널을 수학적으로 통합 (절대 위치 mask) |
| unified → varlen | grid 를 token-flat 으로 전환 + find_seq_idx |

---

## 각 프로젝트 한 줄 요약

| 프로젝트 | 핵심 | block_ptr 변환 시 추가 제약 |
|---|---|---|
| [vllm_padded_decode](./vllm_padded_decode/) | prefill 커널 하나로 decode 까지 처리 (q zero-pad 트릭) | 없음 (non-paged) |
| [vllm_split](./vllm_split/) | decode 전용 커널 추가, backend 에서 Python gather | 없음 (non-paged) |
| [vllm_paged](./vllm_paged/) | 커널이 `block_table` 로 paged KV 직접 인덱싱 | **`BLOCK_N == BLOCK_SIZE`** 강제 (loop 내 fresh `make_block_ptr`) |
| [vllm_multiseq](./vllm_multiseq/) | Backend 가 prefill/decode 그룹 분할 → 각 커널을 multi-seq batch 로 호출 | 동일 |
| [vllm_unified](./vllm_unified/) | 커널 1개가 prefill/decode/chunked 모두 처리 | 동일 + 커널에 `total_q_tokens` 인자 추가 |
| [vllm_varlen](./vllm_varlen/) | grid 를 seq-aligned flat 으로 + `find_seq_idx` binary search | 동일. `_find_seq_idx`는 raw scalar 그대로 (block_ptr은 tile I/O 전용) |

---

## 사전 요구사항

- CUDA GPU (compute capability 8.0+)
- Python ≥ 3.10
- vLLM **0.19.1** (정확히 고정 — 내부 API drift 로 다른 버전에서 깨짐)

각 프로젝트가 자기 `.venv/` 를 가지며 독립 실행. 같은 venv 에 **둘 이상 설치 금지** (`py-modules` 이름 충돌).

`vllm_attn`과 `vllm_attn/block_ptr`을 **동일 venv에 함께 설치하지 마라** —
backend 등록 entry point가 같은 슬롯(`triton_attention_backend:register`)을
가리키므로 충돌. 프로젝트별 별도 venv 사용은 안전.

---

## 시작하기

### 한 프로젝트만 실행

```bash
cd vllm_attn/block_ptr/vllm_varlen            # 또는 원하는 단계
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e .
uv pip install --python .venv/bin/python nbclient ipykernel jupyter
.venv/bin/jupyter notebook qwen3_triton_attention.ipynb
```

또는 `source .venv/bin/activate` 후 `jupyter notebook ...`.

### VS Code / Cursor 에서

프로젝트 디렉토리를 열고 Kernel 선택에서 그 프로젝트의 `.venv/bin/python` 선택.
`qwen3_triton_attention.ipynb` 를 위에서 아래로 실행.

### 공통 cell 구조 (15셀, vllm_attn과 동일)

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

## 공통 설계 원칙

1. **한 단계 한 변화** — 직전 프로젝트와 `diff` 했을 때 변화가 의도한 한 가지로 국한
2. **Plugin entry point 자동 등록** — 수동 `register()` 호출 없이, `pyproject.toml` 의 `vllm.general_plugins` entry point 로 vLLM 이 시작될 때마다 모든 프로세스 (메인·엔진 코어·워커) 에서 자동 호출
3. **관찰 카운터 (`FIRE_COUNTER`)** — 우리 `Impl.forward` 가 실제 실행됐다는 증거를 엔진 코어 stderr 에 로그로 남김
4. **선언 매트릭스는 vLLM 기본 상속** — `supported_dtypes` / `get_supported_head_sizes` 등은 override 하지 않음. 실제 커널 한계는 `Impl.__init__` 의 assert 로만 방어
5. **vLLM 내부 유틸 재사용** — `triton_reshape_and_cache_flash` (KV write) 는 vLLM 것 그대로 사용
6. **block_ptr는 tile I/O 전용** — `block_table`, `seq_lens`, `query_start_loc`, `_find_seq_idx`의 binary search 같은 스칼라/1-D int 로드는 raw pointer arithmetic 그대로

---

## vLLM 실제 코드와의 대응

| 우리 단계 | 대응하는 vLLM 영역 | 메모리 표현 |
|---|---|---|
| padded_decode | 독자적 (교육용 단순화) | block_ptr (변형) |
| split | — | block_ptr (변형) |
| paged | vLLM v0 의 `paged_attention_v1/v2_kernel` | block_ptr (변형); vLLM v0는 raw |
| multiseq | vLLM v0 의 prefill (xformers/FA varlen) + decode (paged) 스케줄러 경로 + split dispatch | block_ptr (변형) |
| unified | vLLM v1 의 `kernel_unified_attention_2d` 의 **수학 부분** | block_ptr (변형); vLLM v1은 raw |
| varlen | vLLM v1 의 `kernel_unified_attention_2d` 전체 | block_ptr (변형); vLLM v1은 raw |

마지막 두 단계의 수학·grid는 vLLM v1과 동일하지만, **메모리 접근 표현은
upstream vLLM의 raw pointer arithmetic과 다르다**. 자세한 caveat은 위 "vllm_attn
와의 차이" 섹션 참조.

---

## 알려진 공통 한계 (vllm_attn 와 동일)

- vLLM **0.19.1 고정**
- `max_num_seqs > 1` 는 `multiseq` 부터, chunked prefill 실증은 `unified/varlen` 에서
- dtype fp16/bf16, head\_dim 은 2의 거듭제곱
- `enforce_eager=True` 고정 (cudagraph 미지원)
- sliding\_window / alibi / logits\_soft\_cap / kv\_sharing / MLA / sparse attention 미지원 (assert 로 거부)
- GQA 를 KV head grid 축으로 접는 최적화 (`BLOCK_M = BLOCK_Q × n_rep`) 는 varlen 단계까지 포함 안 됨
- Split-k decode reduction 미적용

## block_ptr 변형 고유 한계

- **paged 커널의 `BLOCK_N` 이 `BLOCK_SIZE`(KV cache의 block_size)에 고정**
  됨. 원본 `vllm_attn`은 BLOCK_N을 dtype·head_dim heuristic으로 선택해 한
  iteration에 여러 logical block을 처리할 수 있었으나, 본 변형은 매 iteration이
  정확히 한 logical block에 대응. 정확성은 동일하지만 N tile이 작아져
  per-iteration 산술 강도가 낮아질 수 있음 (educational simplification cost).
- multiseq prefill / unified / varlen 커널은 **`total_q_tokens` 인자**가 추가됨
  (Q의 flat-packed parent shape 명시용). wrapper signature는 불변 — 호출부는
  `q.shape[0]`을 자동 전달.

---

## 학습 경로 추천

**`vllm_attn` 을 이미 본 사람**: 같은 단계의 NOTES.md 끝 "Block-pointer
conversion notes" 섹션만 비교해서 표현 차이만 빠르게 확인.

**처음 접하는 경우**: `vllm_attn` 쪽을 먼저 보고 raw pointer arithmetic 으로 한
바퀴 돌고 나서 본 저장소로 와서 같은 수학을 block_ptr로 어떻게 표현하는지 보면
순서가 자연스러움.

---

## 각 프로젝트의 NOTES.md

```
vllm_padded_decode/NOTES.md
vllm_split/NOTES.md
vllm_paged/NOTES.md
vllm_multiseq/NOTES.md
vllm_unified/NOTES.md
vllm_varlen/NOTES.md
```

각 NOTES.md는 vllm_attn 의 동명 NOTES와 본문이 동일하고, 끝에 "Block-pointer
conversion notes" 섹션이 추가되어 있음. 코드보다 NOTES 먼저 읽으면 이해가 빠름.
