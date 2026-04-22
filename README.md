# vllm_attn

vLLM 의 attention backend 아키텍처를 **6단계** 로 쪼개 재현한 학습용 저장소.
각 단계는 직전 단계에서 **"정확히 한 가지"** 만 바꾸는 파생 프로젝트이며,
마지막 단계가 vLLM v1 의 실제 `kernel_unified_attention_2d` 와 동일한 구조에 도달한다.

모든 구현은 **Triton 커널** 로, Qwen3-0.6B 위에서 vLLM 의 공식 `AttentionBackend`
스펙을 따라 plugin entry point 로 등록된다. 모든 단계는 실제 Qwen3-0.6B E2E 로 검증됨.

---

## 전체 로드맵

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

## 각 프로젝트 한 줄 요약

| 프로젝트 | 핵심 | 실측 증거 |
|---|---|---|
| [vllm_padded_decode](./vllm_padded_decode/) | prefill 커널 하나로 decode 까지 처리 (q zero-pad 트릭) | `fired prefill/decode`, Qwen3 출력 OK |
| [vllm_split](./vllm_split/) | decode 전용 커널 추가, backend 에서 Python gather | `fired (prefill)`, `fired (decode)` 분리 |
| [vllm_paged](./vllm_paged/) | 커널이 `block_table` 로 paged KV 직접 인덱싱 | `fired (paged)` + vLLM v0 decode 재현 |
| [vllm_multiseq](./vllm_multiseq/) | Backend 가 prefill/decode 그룹 분할 → 각 커널을 multi-seq batch 로 호출 | `num_seqs=4`, decode batching 실측 |
| [vllm_unified](./vllm_unified/) | 커널 1개가 prefill/decode/chunked 모두 처리 | `num_seqs=4 prefill-like=3 decode=1 chunked=1` (혼합 배치) |
| [vllm_varlen](./vllm_varlen/) | grid 를 seq-aligned flat 으로 + `find_seq_idx` binary search | unified 와 동일 출력, 내부 grid 는 varlen |

---

## 사전 요구사항

- CUDA GPU (compute capability 8.0+)
- Python ≥ 3.10
- vLLM **0.19.1** (정확히 고정 — 내부 API drift 로 다른 버전에서 깨짐)

각 프로젝트가 자기 `.venv/` 를 가지며 독립 실행. 같은 venv 에 **둘 이상 설치 금지** (`py-modules` 이름 충돌).

---

## 시작하기

### 한 프로젝트만 실행

```bash
cd vllm_varlen            # 또는 원하는 단계
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e .
uv pip install --python .venv/bin/python nbclient ipykernel jupyter
.venv/bin/jupyter notebook qwen3_triton_attention.ipynb
```

또는 `source .venv/bin/activate` 후 `jupyter notebook ...`.

### VS Code / Cursor 에서

프로젝트 디렉토리를 열고 Kernel 선택에서 그 프로젝트의 `.venv/bin/python` 선택.
`qwen3_triton_attention.ipynb` 를 위에서 아래로 실행.

### 공통 cell 구조 (15셀)

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

1. **한 단계 한 변화** — 직전 프로젝트와 `diff` 했을 때 변화가 의도한 한 가지로 국한되도록 유지
2. **Plugin entry point 자동 등록** — 수동 `register()` 호출 없이, `pyproject.toml` 의 `vllm.general_plugins` entry point 로 vLLM 이 시작될 때마다 모든 프로세스 (메인·엔진 코어·워커) 에서 자동 호출. `VLLM_ENABLE_V1_MULTIPROCESSING=0` 같은 환경변수 조작 불필요
3. **관찰 카운터 (`FIRE_COUNTER`)** — 우리 `Impl.forward` 가 실제 실행됐다는 증거를 엔진 코어 stderr 에 로그로 남김. subprocess 경계를 넘어 관찰 가능. multi-seq / chunked 단계에서는 `num_seqs`, `max_q_len`, `max_chunked_seqs` 등 진화하는 배치 구성까지 포착
4. **선언 매트릭스는 vLLM 기본 상속** — `supported_dtypes` / `get_supported_head_sizes` 등은 override 하지 않음. 실제 커널 한계는 `Impl.__init__` 의 assert 로만 방어
5. **vLLM 내부 유틸 재사용** — `triton_reshape_and_cache_flash` (KV write) 는 vLLM 것 그대로 사용. 우리가 손대는 곳은 어텐션 수학으로 범위 좁힘

---

## vLLM 실제 코드와의 대응

| 우리 단계 | 대응하는 vLLM 영역 |
|---|---|
| padded_decode | 독자적 (교육용 단순화) |
| split | — |
| paged | vLLM v0 의 `paged_attention_v1/v2_kernel` (`csrc/attention/attention_kernels.cu`) |
| multiseq | vLLM v0 의 prefill (xformers/FA varlen) + decode (paged) 스케줄러 경로 + split dispatch |
| unified | vLLM v1 의 `kernel_unified_attention_2d` 의 **수학 부분** (절대 위치 causal) |
| varlen | vLLM v1 의 `kernel_unified_attention_2d` 전체 (flat grid + `find_seq_idx` 포함). `vllm/v1/attention/ops/triton_unified_attention.py` |

---

## 알려진 공통 한계

- vLLM **0.19.1 고정** — 다른 버전은 내부 API drift 로 깨질 수 있음
- `max_num_seqs > 1` 는 `multiseq` 부터, chunked prefill 실증은 `unified/varlen` 에서
- dtype fp16/bf16, head\_dim 은 2의 거듭제곱
- `enforce_eager=True` 고정 (cudagraph 미지원)
- sliding\_window / alibi / logits\_soft\_cap / kv\_sharing / MLA / sparse attention 미지원 (assert 로 거부)
- GQA 를 KV head grid 축으로 접는 최적화 (`BLOCK_M = BLOCK_Q × n_rep`) 는 varlen 단계까지 포함 안 됨. vLLM v1 실제 커널은 이걸 수행
- Split-k decode reduction 미적용 (긴 컨텍스트 decode 에서 vLLM 대비 낮은 utilization)

---

## 학습 경로 추천

**처음 접하는 경우**: `padded_decode → split → paged` 순으로 보면서 "AttentionBackend 스펙을 따라 custom backend 를 쓰는 최소한" 을 이해.

**vLLM 의 continuous batching 을 배우고 싶다면**: `multiseq` 부터 시작. prefill/decode batch 분류와 그 스케줄링 효과를 체감.

**v0 → v1 전환의 이유를 알고 싶다면**: `multiseq → unified → varlen` 을 연속으로. 절대 위치 mask 로 "구별이 사라지는" 순간 + grid 토폴로지가 varlen 화되는 순간.

**이미 vLLM 구조에 익숙하다면**: `unified` 와 `varlen` 의 NOTES 만 읽으면 로드맵 전체의 핵심 요약 확보.

---

## 각 프로젝트의 NOTES.md

각 서브프로젝트에 상세 노트가 있음 — 이 프로젝트에서 바뀐 것, 핵심 설계 결정, 실측 결과, 알려진 함정, 다음 단계로의 연결이 담김. 코드보다 NOTES 먼저 읽으면 이해가 빠름.

```
vllm_padded_decode/NOTES.md
vllm_split/NOTES.md
vllm_paged/NOTES.md
vllm_multiseq/NOTES.md
vllm_unified/NOTES.md
vllm_varlen/NOTES.md
```
