# raw pointer → `tl.make_block_ptr` 변환에서 어려웠던 점

`vllm_attn`의 6개 단계를 모두 `tl.make_block_ptr` 기반으로 옮기면서 마주한
구체적 난점들과 해결 방법. 각 항목은 **(1) 무엇이 어려웠는지 → (2) 왜 어려웠는지
→ (3) 어떻게 해결했는지 → (4) before / after 코드** 순서로 정리.

> 결론부터: **paged KV의 간접 인덱싱과 mask vs boundary_check 책임 분리** 이 두
> 가지가 가장 비중이 컸다. 나머지는 표현 방식만 바꾸는 비교적 기계적 작업.

---

## 1. paged KV — `tl.advance`가 base 주소 변동을 따라갈 수 없음

### 무엇이 어려웠는가
non-paged 커널(`vllm_padded_decode`, `vllm_split` prefill)에서는 K/V가 단일
연속 텐서이므로, loop 밖에서 `make_block_ptr`을 한 번 만들고 매 iter `tl.advance(ptr, (0, BLOCK_N))`으로 자연스럽게 넘어갈 수 있다. **paged 커널은 이게 안 된다.**

### 왜 어려운가
paged KV cache는 `[num_blocks, block_size, num_kv_heads, head_dim]` 레이아웃이고,
시퀀스의 logical block n번째가 어느 physical block에 있는지는 `block_table[seq, n]`
간접 인덱싱으로만 알 수 있다. 즉 매 iteration마다 **base 주소가 비-uniform하게
점프한다.** `tl.advance`는 같은 부모 텐서 안에서 일정한 offset 만큼 이동하는
연산이므로, 이런 jump를 표현하지 못한다.

원본은 raw pointer arithmetic으로 한 N tile 안에 여러 logical block을 섞어
처리할 수 있었다 (`physical_block`을 벡터로 로드해서 gather):

```python
# 원본 vllm_attn/ptr/vllm_paged/triton_attn.py — 한 tile에 여러 logical block 섞임
for n in range(0, tl.cdiv(S, BLOCK_N)):
    offs_n = n * BLOCK_N + tl.arange(0, BLOCK_N)
    logical_block = offs_n // BLOCK_SIZE                  # [BLOCK_N] 벡터
    slot_in_block = offs_n % BLOCK_SIZE
    physical_block = tl.load(block_table + ... + logical_block, mask=kv_mask)
    cache_base = (
        physical_block[:, None] * stride_cache_block      # 행마다 다른 base!
        + slot_in_block[:, None] * stride_cache_slot
        + kv_head_idx * stride_cache_head
        + offs_d[None, :] * stride_cache_d
    )
    k = tl.load(K_cache + cache_base, mask=kv_mask[:, None], other=0.0)
    ...
```

`physical_block[:, None]` — 같은 N tile 안에서 행마다 다른 physical block을
가리키는 *gather* 패턴. block_ptr은 **단일 base + 정적 strides**를 요구하므로
이걸 직접 표현할 수 없다.

### 어떻게 해결했는가
**한 iteration이 정확히 하나의 logical block에 대응하도록 만들고, 매 iter마다
`make_block_ptr`을 새로 만든다.** 이 idiom은 IBM `vllm-triton-lib`과
arxiv:2511.11581 *"The Anatomy of a Triton Attention Kernel"* 에서 확인.

```python
# vllm_attn/block_ptr/vllm_paged/triton_attn.py — 한 iter = 한 logical block
for n in range(0, tl.cdiv(S, BLOCK_SIZE)):
    offs_n = n * BLOCK_SIZE + tl.arange(0, BLOCK_N)       # BLOCK_N == BLOCK_SIZE
    kv_mask = offs_n < S

    physical_block_idx = tl.load(block_table + batch_idx * stride_bt_seq + n)  # 스칼라!
    kv_block_base = (
        physical_block_idx * stride_cache_block
        + kv_head_idx * stride_cache_head
    )

    K_block_ptr = tl.make_block_ptr(
        base=K_cache + kv_block_base,                     # 단일 base, 정적 strides
        shape=(BLOCK_D, BLOCK_SIZE),
        strides=(stride_cache_d, stride_cache_slot),
        offsets=(0, 0),
        block_shape=(BLOCK_D, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V_cache + kv_block_base,
        shape=(BLOCK_SIZE, BLOCK_D),
        strides=(stride_cache_slot, stride_cache_d),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )
    k_t = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
    v   = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
    ...
```

핵심 차이:
- `physical_block_idx`가 **스칼라**가 됨 (한 iter에 한 block 만 다루므로)
- `make_block_ptr`이 loop *안에* 위치 (매 iter마다 base가 바뀌므로)
- `tl.advance` 사용 안 함

---

## 2. `BLOCK_N == BLOCK_SIZE` 강제 — 처리량 trade-off

### 무엇이 어려웠는가
원본은 `BLOCK_N`을 dtype·head_dim heuristic으로 자유 선택했다 (예: D=128, fp16,
Ampere 이상에서 BLOCK_N=64). 한 N tile이 여러 logical block에 걸칠 수 있다는
유연성. block_ptr 변환 후엔 이걸 포기해야 한다.

### 왜 어려운가
위 §1의 직접적 결과. 매 iter가 한 logical block에 대응하므로 `BLOCK_N`과
`BLOCK_SIZE`를 분리할 수 없다. KV cache의 `block_size`(보통 16)가 곧 N tile의
크기가 된다.

처리량 영향: D=128, fp16 Ampere에서
- 원본: BLOCK_N=64 → KV loop 횟수 `cdiv(S, 64)`
- 변환: BLOCK_N=16 → KV loop 횟수 `cdiv(S, 16)` (4배 증가)

per-iteration의 산술 강도(arithmetic intensity)도 같이 떨어진다.

### 어떻게 해결했는가
**정확성을 우선하고 처리량 비용을 honest하게 문서화한다.** 4개 paged 변형
모두에서 wrapper가 `BLOCK_N=BLOCK_SIZE`로 launch:

```python
# vllm_attn/block_ptr/vllm_paged/triton_attn.py wrapper
_fwd_kernel_prefill_paged[grid](
    ...,
    BLOCK_M=BLOCK,
    BLOCK_N=BLOCK_SIZE,                                   # 강제
    BLOCK_D=D, BLOCK_SIZE=BLOCK_SIZE,
)
```

correctness는 손상 없음 — 마지막 partial block은 `boundary_check=(1,)` +
`padding_option="zero"`가 처리하고, kv_mask가 algorithm 측면에서 추가 제거.

각 sub-project NOTES.md의 "BLOCK_N == BLOCK_SIZE 제약" 섹션에 educational
simplification cost임을 명시. 이 비용을 피하려면 §1의 gather 패턴으로 돌아가야
하는데, 그건 block_ptr 변환의 본질을 부정하는 것.

---

## 3. `mask=` 가 사라지고 `boundary_check=` 만 — 책임 분리 강제

### 무엇이 어려웠는가
원본은 `tl.load(ptr + ..., mask=combined_mask, other=0.0)` 한 줄에 (a) 텐서
경계 boundary, (b) causal mask, (c) q_len/kv_len outer-product mask를 모두
넣었다. block_ptr API는 `tl.load(block_ptr, ...)`에 `mask=` 인자를 받지 않는다.

### 왜 어려운가
block_ptr은 *load-time boundary*(텐서 끝을 넘는 lane 처리)만 책임진다.
"causal이라서 무시", "이 행은 다른 시퀀스라서 무시" 같은 *algorithmic mask*는
load 책임이 아니다. 둘을 분리해야 한다.

### 어떻게 해결했는가
**책임 분리:**
- 텐서 boundary → `boundary_check=(축,) + padding_option="zero"`로 처리 (load 시점)
- algorithmic mask → `tl.where(...)`로 그대로 유지 (softmax 직전)

```python
# 원본 (vllm_attn) — 한 줄에 모든 mask
qk = tl.where(causal_mask & kv_mask, qk, float("-inf"))
# K load는 mask로 boundary도 같이:
k = tl.load(K + ..., mask=kv_mask, other=0.0)

# 변환 (vllm_attn/block_ptr) — boundary는 load에, algorithmic은 tl.where에
k_t = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")  # boundary만
qk = tl.dot(q, k_t) * sm_scale
qk = tl.where(causal & kv_mask[None, :], qk, float("-inf"))             # 그대로
```

함정 — 처음엔 "boundary_check가 padding을 zero로 채우니 kv_mask가 필요 없을
것"이라고 착각하기 쉽다. 그러나 boundary_check는 텐서 *끝*만 처리하고, 시퀀스
중간의 mask 의미는 모른다. 특히 unified/varlen에서 flat-packed Q의 한 tile이
**다음 시퀀스의 영역으로 넘어가는** 경우 — 그 영역은 boundary 안이지만 다른
시퀀스 데이터이므로 `q_mask`가 여전히 필요. 이걸 놓치면 GPU에서 silently
잘못된 output.

---

## 4. K transpose — `tl.trans` 대신 `order=(0,1)` virtual transpose

### 무엇이 어려웠는가
원본은 K를 `[N, D]`로 로드한 뒤 `tl.trans(k)` → `[D, N]`을 만들어 `tl.dot(q, k_t)`
에 전달. block_ptr 패턴에서 자연스럽게 표현할 방법이 모호.

### 왜 어려운가
`tl.trans`를 쓰는 것 자체는 가능하지만 추가 op. block_ptr이 제공하는 `order`
인자로 더 깨끗하게 표현 가능 — *어떤 게 더 간결한지* 결정해야 했다.

### 어떻게 해결했는가
`make_block_ptr`의 `shape`/`strides`/`order`를 transposed view로 선언해서
**처음부터 `[D, N]` 모양으로 로드한다.** `tl.trans` 사라짐.

```python
# 원본 — K를 [N, D]로 로드 후 trans
k = tl.load(K + offs_n[None,:]*stride_ks + offs_d[:,None]*stride_kd, mask=...)
k_t = tl.trans(k)                                          # [D, N]
qk = tl.dot(q, k_t)

# 변환 — strides를 swap하고 order=(0,1)로 [D, N]으로 직접 로드
K_block_ptr = tl.make_block_ptr(
    base=K + pid_bh * stride_kb,
    shape=(BLOCK_D, S),                                    # transposed shape
    strides=(stride_kd, stride_ks),                        # swapped strides
    offsets=(0, 0),
    block_shape=(BLOCK_D, BLOCK_N),
    order=(0, 1),                                          # virtual transpose
)
k_t = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")  # [D, N]
qk = tl.dot(q, k_t)
```

이게 Triton의 공식 fused-attention 튜토리얼(2.1.x)에서 쓰는 표준 idiom. V는
`(N, D)` 그대로 둬서 transpose 안 함 — 비대칭이 약간 불편하지만 표현은 더 짧음.

> decode 커널은 K transpose가 아예 필요 없다 — `tl.dot` 대신
> `tl.sum(q[None,:] * k.to(tl.float32), axis=1)` 형태의 reduction을 쓰기
> 때문. K를 `(N, D)` 그대로 로드. 이 비대칭도 NOTES에 명시.

---

## 5. Flat-packed Q — base/shape/offsets 모호성 (`total_q_tokens` 추가)

### 무엇이 어려웠는가
multiseq/unified/varlen의 Q는 `[total_q_tokens, Hq, D]` 레이아웃 + `query_start_loc`로
시퀀스마다 다른 q_start. 시퀀스 i의 행 영역은 `Q[q_start_i : q_start_i+q_len_i, h, :]`.
이걸 block_ptr로 어떻게 잡을지 — base를 어디에 두고 shape를 무엇으로 둘지 — 가
모호했다.

### 왜 어려운가
두 가지 후보가 있었다:
- **(a)** base = `Q + q_head_idx * stride_h`, parent shape = `(total_q_tokens, D)`,
  offsets = `(q_start + pid_m*BLOCK_M, 0)` — 부모 텐서의 진짜 행 크기를 shape에
  넣음
- **(b)** base = `Q + q_start*stride_t + q_head_idx*stride_h`, parent shape = `(q_len, D)`,
  offsets = `(pid_m*BLOCK_M, 0)` — 시퀀스 로컬 view

(b)는 boundary_check가 자연스럽게 `q_mask` 일부를 대체할 수 있어서 유혹적.
하지만 `q_len`은 매 iter마다 다른 runtime 값이고, 본질적으로 시퀀스마다 다른
view를 만드는 게 boundary 의미를 모호하게 만든다. (a)가 정직.

(a)의 문제: `total_q_tokens`가 wrapper 호출 시점에는 알지만 **커널 인자로
전달이 안 되어 있었음.** 시그니처를 바꿔야 한다.

### 어떻게 해결했는가
**커널에 `total_q_tokens` runtime int 인자 1개 추가** (wrapper가 `q.shape[0]`
자동 전달). public Python wrapper signature는 한 글자도 안 바뀜 — backend는 무영향.

```python
# 변환 — Snippet C
Q_block_ptr = tl.make_block_ptr(
    base=Q + q_head_idx * stride_q_h,
    shape=(total_q_tokens, BLOCK_D),                       # 부모 텐서 진짜 크기
    strides=(stride_q_t, stride_q_d),
    offsets=(q_start + pid_m * BLOCK_M, 0),                # offsets는 runtime 가능
    block_shape=(BLOCK_M, BLOCK_D),
    order=(1, 0),
)
q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
# q_mask는 여전히 tl.where 단계에서 필요!
```

함정 — boundary_check가 `total_q_tokens` 끝의 padding은 처리하지만, **다음
시퀀스 영역으로 넘어가는 행**은 boundary 안이라 zero-pad 안 됨. 그래서 `q_mask =
offs_m_local < q_len`을 softmax mask에 포함시켜야 함 (§3과 같은 이슈의 다른
얼굴).

---

## 6. Scalar load는 의도적으로 raw pointer로 남기기

### 무엇이 어려웠는가
`block_table`, `seq_lens`, `query_start_loc`, varlen의 `_find_seq_idx` binary
search — 한 번에 int 하나 읽는 스칼라 로드들. "all kernels use block_ptr"
원칙에 따라 이것도 변환할까?

### 왜 어려운가
변환의 본질은 *tile* I/O 최적화 추상화다. 스칼라 1개 로드에 block_ptr을 씌우는
건 추상화 비용만 더하고 이득이 없다. 하지만 일관성을 깨는 결정에 정당화가 필요.

### 어떻게 해결했는가
**원칙을 명시적으로 좁힌다: "block_ptr은 tile I/O 전용. 스칼라/1-D int 로드는
raw pointer 유지."** 모든 NOTES.md에 이 원칙 기록.

특히 `_find_seq_idx`(varlen):
```python
# vllm_attn/block_ptr/vllm_varlen/triton_attn.py — 의도적으로 raw 유지
@triton.jit
def _find_seq_idx(cum_q_blocks_ptr, target_q_block, num_seqs):
    """Binary search — 스칼라 비교만 함. block_ptr 의미 없음."""
    lo, hi = 0, num_seqs - 1
    while lo < hi:
        mid = (lo + hi) // 2
        v = tl.load(cum_q_blocks_ptr + mid + 1)            # 스칼라 — raw
        if v <= target_q_block:
            lo = mid + 1
        else:
            hi = mid
    return lo
```

이걸 block_ptr로 바꾸면 코드는 더 길어지고 이득은 0. NOTES에 "scalar binary
search; block_ptr은 tile I/O 전용"으로 명시.

---

## 7. 1-D block_ptr — decode Q/O가 `[BLOCK_D]` 벡터인 경우

### 무엇이 어려웠는가
decode 커널의 Q는 한 토큰의 head vector — `[BLOCK_D]` 1-D. block_ptr이 1-D를
지원하는지 확인 필요했다.

### 어떻게 해결했는가
지원함. `block_shape`이 1-tuple이면 됨.

```python
Q_block_ptr = tl.make_block_ptr(
    base=Q + pid_bh * stride_qb,
    shape=(BLOCK_D,),
    strides=(stride_qd,),
    offsets=(0,),
    block_shape=(BLOCK_D,),
    order=(0,),
)
q = tl.load(Q_block_ptr).to(tl.float32)                    # 정확히 D개 lane — boundary_check 불필요
```

D는 head_dim (정확히 채워짐, partial 없음)이므로 `boundary_check` 자체가
필요 없다. 가장 깨끗한 케이스 — 어떤 함정도 없음.

---

## 8. (사후 발견) multiseq decode O 일관성 — 한 곳만 raw로 남겨졌던 문제

### 무엇이 어려웠는가
처음엔 "1-D scalar store에 block_ptr 이점이 없다"는 이유로 `vllm_multiseq`의
decode O store만 raw pointer로 남겼다. 그런데 같은 1-D store인 `vllm_split`,
`vllm_paged`의 decode O는 이미 block_ptr을 쓰고 있어 일관성 위반. reviewer가
잡아냄.

### 왜 어려운가 (의사 결정 측면)
"이득이 없으니 raw로 둔다"와 "표현 일관성을 위해 block_ptr로 한다"가 충돌.
저장소의 학습 목표(모든 tile I/O를 block_ptr로 표현)를 우선하면 후자가 답.

### 어떻게 해결했는가
일관성 우선 → 1-D block_ptr로 통일. 그리고 NOTES.md의 "이득 없음" 문구를
"이득보다 일관성을 위해 — split/paged decode O와 동일한 패턴"으로 수정.

```python
# 수정 후 — 다른 decode O와 동일한 패턴
O_block_ptr = tl.make_block_ptr(
    base=Out + batch_idx * stride_o_s + q_head_idx * stride_o_h,
    shape=(BLOCK_D,),
    strides=(stride_o_d,),
    offsets=(0,),
    block_shape=(BLOCK_D,),
    order=(0,),
)
tl.store(O_block_ptr, acc.to(io_dtype))
```

교훈: "이 케이스만 예외" 결정은 비슷한 다른 케이스를 모두 점검한 뒤에
정당화해야 한다. 한 변형 안에서 일관성이 깨지면 학습용 저장소의 가치가
떨어진다.

---

## 정리 — 변환 패턴 요약 (cheat sheet)

| 원본 패턴 | 변환 |
|---|---|
| `tl.load(P + offs[:,None]*sa + offs_d[None,:]*sd, mask=m, other=0.0)` | `make_block_ptr(...)` + `tl.load(bp, boundary_check=(0,), padding_option="zero")` + `tl.where`로 mask 분리 |
| `tl.trans(k)` (K transpose) | `make_block_ptr(shape=(D,N), strides=(sd,sn), order=(0,1))` |
| paged inner loop (`physical_block`을 벡터로 gather) | `BLOCK_N := BLOCK_SIZE` + 매 iter 새 `make_block_ptr` (`physical_block`은 스칼라) |
| flat-packed Q (`q_start + offs_m_local`) | `make_block_ptr(shape=(total_q_tokens, D), offsets=(q_start+pid_m*BLOCK_M, 0))` + 커널 인자 `total_q_tokens` 추가 |
| `tl.advance` | non-paged에서만. paged는 매 iter 새 `make_block_ptr` (advance 불가) |
| `block_table`, `seq_lens`, `_find_seq_idx` 등 스칼라 로드 | **변환 안 함 — raw pointer 유지** (block_ptr은 tile I/O 전용) |
| 1-D Q/O (decode) | 1-D `make_block_ptr` (`block_shape=(D,)`, `order=(0,)`) — boundary_check 불필요 |

## 최종 caveat (반복)

이 변환의 idiom은 **vLLM upstream의 production 패턴이 아니다.** vLLM v1의
`triton_unified_attention.py`는 raw pointer arithmetic 그대로 쓴다. 본 변환은
arxiv:2511.11581 / IBM `vllm-triton-lib` 표현과 일치 — 학습용 비교 저장소이지
"production이 이렇게 한다"의 reproduction이 아님.

GPU 환경에서 정확성 검증은 사용자 책임 (Darwin 환경에서는 정적 검증만
가능했음).
