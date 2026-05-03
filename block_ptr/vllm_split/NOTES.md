# NOTES — 구현 과정 요약

Qwen3-0.6B × 커스텀 Triton attention 을 vLLM 위에서 돌린 경험 정리.

## 1. 목표

Qwen3-0.6B (fp16, Q=16 / KV=8 / head\_dim=128) 의 attention 계산을
직접 작성한 Triton 커널로 교체한 상태로 vLLM 을 실행한다.

## 2. 경로 선택

vLLM v1 의 `AttentionBackend` 를 서브클래싱하고
`vllm.general_plugins` entry point 로 등록. 기각한 대안:

- **monkey-patch** (`self_attn.forward` 덮어쓰기) — paged KV cache 를 우회해야 해서 실질 교체 불가.
- **custom op** (`torch.compile`) — 초보자 범위 밖.
- **FlashInfer fork** — 빌드 체인 부담, 의도가 FlashInfer 개조가 돼 버림.

핵심 원칙: **"교체했다"고 주장하려면 런타임 로그로 실제 호출을 증명한다.**

## 3. 모듈 구조 (`triton_attention_backend.py`)

```
MyTritonMetadata         — dataclass. builder 가 만들고 Impl 이 읽는 per-forward 메타
MyTritonMetadataBuilder  — vLLM 이 매 forward 마다 호출해서 metadata 생성
MyTritonImpl             — AttentionImpl 서브클래스. forward() 에서 커널 실행
MyTritonBackend          — AttentionBackend 서브클래스. 위 셋을 묶는 staticmethod 4개
register()               — register_backend(AttentionBackendEnum.CUSTOM, "...") 헬퍼
FIRE_COUNTER             — 프로세스 로컬 호출 카운터 (entry/디버그용)
```

주요 플래그:
- `forward_includes_kv_cache_update = True` — KV write 를 forward 안에서 수행
- `accept_output_buffer = True` — vLLM 이 `output` 텐서를 미리 할당해 넘김, in-place 기록

## 4. 핵심 설계 결정

**(1) KV write 는 vLLM 내부 `triton_reshape_and_cache_flash` 재사용.**
slot\_mapping -1 패딩 처리 등 잔버그가 많은 부분이고 교육적 가치가 낮아
"검증된 것을 빌려오고" 우리가 손대는 곳은 어텐션 수학으로만 좁힘.

**(2) prefill / decode 를 분리된 전용 커널로 구현 (vLLM 네이밍 따름).**
`_fwd_kernel_prefill` — q_len == kv_len, causal mask (기존 커널).
`_fwd_kernel_decode` — q_len == 1, 전체 kv_len 을 BLOCK\_N 단위로 순회, causal mask 불필요
(마지막 쿼리는 모든 과거 key 를 볼 수 있음). Backend 는 `q_len == s_len` 분기로 둘 중 하나 호출.
Decode 비용: 이전 zero-pad 재활용 방식의 O(s\_len²) → 전용 커널로 **O(s\_len)**.
Block 크기는 vLLM 의 `get_block_size(dtype)` 휴리스틱 (fp32=32 / sm\_80+=128 / else=64)
+ head\_dim 기반 상한 (shared memory 제약).

**(3) 선언 매트릭스는 vLLM 기본 상속, 커널은 dtype/head\_dim generic.**
`supported_dtypes` / `supported_kv_cache_dtypes` / `get_supported_head_sizes` / `is_mla` / `supports_attn_type` 등은
`AttentionBackend` 의 기본값 그대로 상속 (override 하지 않음). 커널 자체는 `Out.dtype.element_ty` 로
출력 dtype 을 추론하고 `BLOCK_D: tl.constexpr` 로 head\_dim 을 받아 Triton JIT 가 자동 specialization.
fp16/bf16 × head\_dim ∈ {64, 128, 256} 조합을 소스 한 벌로 커버 (smoke test PASS).
여전히 `MyTritonImpl.__init__` 에서 DECODER / no sliding\_window·alibi·logits\_soft\_cap·kv\_sharing ·
cudagraph NEVER 는 assert 로 막는다 (실제 우리가 구현 안 한 범위).

## 5. 함정 모음

| # | 내용 | 해결 |
|---|---|---|
| 1 | `get_name()` 이 "MY\_TRITON" 처럼 enum 에 없는 문자열 반환 → `AttentionBackendEnum[name]` 에서 ValueError | `"CUSTOM"` 반환 (한 줄 수정) |
| 2 | `triton_attention()` 래퍼가 scale 을 하드코딩 → 다른 모델에서 조용한 수치 오류 | `scale=None` 파라미터 추가 + Impl 이 `self.scale` 전달 |
| 3 | fp32 누산 의도 모호 (`tl.dot(fp16, fp16)` 기본 출력은 fp16) | `out_dtype=tl.float32` 명시. smoke test max\_err 0.000488 |
| 4 | `AttentionImpl.__init__` 에 vLLM 이 새 kwarg 추가 가능 | `**kwargs` 로 흡수 + 노트북 cell 에서 `vllm==0.19.1` assert |
| 5 | 엔진 코어가 subprocess 로 spawn 되어 메인에서 호출한 `register()` 가 전달 안 되면 CUSTOM 이 조용히 fallback | `vllm.general_plugins` entry point 로 등록 → 모든 프로세스에서 자동 `register()`. TP>1 도 OK |

## 6. 검증 방법 & 실측

| 단계 | 방법 | 실측 |
|---|---|---|
| 커널 정확성 (prefill) | 랜덤 Q/K/V → SDPA 와 max\_abs\_err 비교 | **PASS** (fp16 × D ∈ {64,128,256}: 0.0005 / bf16: 0.004~0.008) |
| 커널 정확성 (decode) | q=1, kv\_len ∈ {32,128,1024} × SDPA 의 last-position 과 비교 | **PASS** (fp16: 0.0001~0.0005 / bf16: 0.0005~0.004) |
| Plugin 탐지 | `entry_points(group="vllm.general_plugins")` | **PASS** (`my_triton_backend -> triton_attention_backend:register`) |
| CUSTOM 슬롯 | `AttentionBackendEnum.CUSTOM.get_path()` | **PASS** (`triton_attention_backend.MyTritonBackend`) |
| 백엔드 실행 증거 | 엔진 코어 stderr 의 `MyTritonImpl.forward fired` 로그 | **PASS** (`(EngineCore pid=...) fired prefill tokens=5 / decode s_len=6`) |
| 출력 품질 | `"The capital of France is"` + max\_tokens=32 | **PASS** (`Paris. The capital of Italy is Rome. ...`) |

**핵심**: 실행 증거는 "출력이 그럴듯함"이 아니라 **우리 코드가 찍은 로그가 엔진 코어 pid 아래에 보임**.
`MyTritonImpl.forward` 안에서 vLLM 로거로 `WARNING` 한 줄을 찍어 subprocess 경계를 건너 보게 했다.
메인 프로세스의 `FIRE_COUNTER` 는 engine-core 와 별도 메모리라 0으로 남을 수 있으므로 단독 근거로 쓰지 말 것.

## 7. 파일

- `pyproject.toml` — `vllm==0.19.1` 고정 + `vllm.general_plugins` entry point
- `triton_attn.py` — Flash-style online softmax 커널 + `triton_attention(q, k, v, scale=None)` 래퍼
- `triton_attention_backend.py` — AttentionBackend 4개 클래스 + `register()` + fire 로그
- `qwen3_triton_attention.ipynb` — 15셀 튜토리얼

**사용법**: `pip install -e .` 한 번. 이후 `register()` 수동 호출·multiproc 플래그 불필요.

## 8. 알려진 한계

- decode 는 single-token 전용 커널 (split-k 미사용 단일 program). 장기 컨텍스트 throughput 은 vLLM 내장 백엔드 대비 낮음
- dtype fp16/bf16 지원, head\_dim 은 2의 거듭제곱 (Triton `tl.arange` 요구)
- `max_num_seqs=1`, `enforce_eager=True` 한정 (배치/cudagraph 미지원)
- sliding\_window · alibi · logits\_soft\_cap · kv\_sharing · MLA · sparse 미지원
- **vLLM 0.19.1 정확히 고정** — 내부 API drift 로 다른 버전에서 깨질 수 있음

---

## Block-pointer conversion notes

### 무엇이 바뀌었는가

**`_fwd_kernel_prefill`** — vllm_padded_decode 와 동일한 변환:

1. **Q load** (Snippet D): raw pointer → `tl.make_block_ptr(shape=(S,BLOCK_D), order=(1,0))` + `boundary_check=(0,)`. `mask=` 제거.
2. **K load** (Snippet A): `tl.make_block_ptr(shape=(BLOCK_D,S), strides=(stride_kd,stride_ks), order=(0,1))` — virtual transpose. `tl.trans()` 제거. `tl.advance(K_block_ptr, (0, BLOCK_N))`.
3. **V load** (Snippet A): `tl.make_block_ptr(shape=(S,BLOCK_D), order=(1,0))` + `tl.advance(V_block_ptr, (BLOCK_N, 0))`.
4. **O store** (Snippet F): raw pointer + mask → `tl.make_block_ptr` + `boundary_check=(0,)`.

**`_fwd_kernel_decode`** — 1-D Q 와 2-D K/V 변환:

1. **Q load** (Snippet E): `tl.load(Q + pid_bh*stride_qb + offs_d*stride_qd)` → `tl.make_block_ptr(shape=(BLOCK_D,), strides=(stride_qd,), order=(0,))`. boundary_check 불필요 (D 는 정확).
2. **K load**: K 는 transpose 안 함 — `(N, D)` 로 로드해서 `tl.sum(q[None,:]*k.to(tl.float32), axis=1)` reduction. `tl.make_block_ptr(shape=(S,BLOCK_D), order=(1,0))` + `tl.advance(K_block_ptr, (BLOCK_N, 0))`.
3. **V load**: 동일하게 `(N, D)` 2-D block_ptr + advance.
4. **O store**: 1-D block_ptr, boundary_check 불필요.

### 무엇이 안 바뀌었는가

- Online softmax 수학 (`m_i`, `l_i`, `alpha`, `p`, `acc`) — 완전 불변.
- GQA (`repeat_interleave` in wrapper) — 불변.
- Causal mask + `kv_mask` 의 `tl.where` — 불변.
- Wrapper 시그니처 `triton_attention_prefill`, `triton_attention_decode`, `triton_attention` alias — 불변.
- Smoke test 코드 — 불변.

### 정직한 caveat

- vLLM v1 본가는 raw pointer arithmetic 사용. 이 구현은 `arxiv:2511.11581` / IBM `vllm-triton-lib` 패턴.
- Decode 커널의 K/V 에 `tl.dot` 대신 `tl.sum(q[None,:]*k, axis=1)` reduction 을 사용하는 이유: decode 는 `[1, D]·[N, D]ᵀ` 를 계산하는데, 이를 `tl.dot([1,D], [D,N])` 으로 바꾸려면 K 를 transpose 해야 한다. Decode 에서는 tile 크기가 작아 element-wise reduction 이 충분하고, transpose 없이 `(N, D)` 그대로 로드하는 편이 block_ptr advance 와 자연스럽게 맞는다.
- `boundary_check=(0,)` 는 시퀀스 축 (dim 0) 의 out-of-range 를 처리한다. D 축은 항상 2의 거듭제곱이므로 boundary_check 불필요.
