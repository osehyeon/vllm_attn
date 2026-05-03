# NOTES — vllm_multiseq

Qwen3-0.6B × 커스텀 Triton attention. **Multi-seq batch dispatch** 단계.
vLLM v0 의 실제 스타일 — 커널은 prefill/decode 2개 분리, 각 커널이 multi-seq 를 한 launch 에 처리.

> **이전**: `../vllm_paged/` — 동일 구조지만 `max_num_seqs=1` 고정
> **다음**: `vllm_unified` (미래) — 한 커널로 prefill/decode 혼합 배치 + `find_seq_idx`

## 1. 이 프로젝트에서 추가된 것

**vLLM 이 한 forward 에 여러 시퀀스를 넣으면, 각 시퀀스가 prefill 인지 decode 인지에 따라
두 그룹으로 나눠 각각의 커널을 1회씩 호출** — 진짜 multi-seq batching.

### 커널 시그니처 변화 (paged → multiseq)

**Prefill** — 이제 여러 시퀀스의 q 토큰이 **flat-packed** 로 들어옴:
```
Q             : [total_q_tokens, Hq, D]
query_start_loc: [num_seqs + 1]         ← 각 seq 의 q 시작 위치
seq_lens       : [num_seqs]
block_table    : [num_seqs, max_blocks]

커널 내부:
  batch_idx 로부터 q_start, q_len, s_len 읽음
  causal: q_abs[:, None] >= offs_n[None, :]  where q_abs = s_len - q_len + local
```

**Decode** — 각 seq 당 1 토큰, batch 축으로 쌓음:
```
Q       : [num_decode_seqs, Hq, D]
grid    : (num_decode_seqs * Hq,)
커널 안에서 각 프로그램이 한 seq 의 1 q 토큰 × 전체 kv_len 처리
```

## 2. Backend 의 split-dispatch 로직

```python
q_lens = query_start_loc[1:] - query_start_loc[:-1]
is_prefill = q_lens == seq_lens
is_decode  = q_lens == 1

if is_prefill.any():
    prefill_idx = is_prefill.nonzero()
    # prefill seqs 의 q 토큰만 모아서 한 번의 kernel launch
    triton_attention_prefill_multiseq(q_gathered, ..., bt[prefill_idx], sl[prefill_idx], qsl)

if is_decode.any():
    decode_idx = is_decode.nonzero()
    # decode seqs 의 last q token 만 쌓아서 한 번의 kernel launch
    triton_attention_decode_multiseq(q_last_tokens, ..., bt[decode_idx], sl[decode_idx])
```

혼합 배치여도 **커널 호출은 각 그룹당 1회** — 이게 batching 의 핵심.

## 3. 제약

- **Chunked prefill 미지원** — 각 seq 는 pure prefill (q\_len == s\_len) 또는 pure decode (q\_len == 1). `q_len > 1 and q_len < s_len` 인 경우 backend 의 `assert` 로 거부.
  **참고**: prefill 커널은 이미 `q_abs = s_len - q_len + local` 절대 위치 mask 를 쓰므로 **커널은 chunked 를 할 수 있음**. Backend 의 assert 한 줄만 풀고 분류를 3-way (prefill / chunked / decode) 로 확장하면 v0 + `enable_chunked_prefill=True` 완전 재현. 이 프로젝트에서는 일부러 거부해서 "multi-seq 가 먼저, chunked 는 unified 에서" 단계를 유지.
- 나머지 제약은 `vllm_paged` 와 동일 (sliding_window/alibi/logits_soft_cap/kv_sharing/MLA/sparse 미지원)

## 4. 설계 결정

**(1) 커널 2개 유지, backend 가 분할 dispatch.**
vLLM v0 실제 방식. Unified 로 가는 대신 "batching 먼저, 통합은 나중" 전략.

**(2) Prefill 은 flat-packed Q + query_start_loc, Decode 는 `[num_seqs, Hq, D]` batch-stacked Q.**
두 경우의 shape 이 달라 커널 시그니처도 살짝 다르지만, 각자 맥락에 맞는 가장 간단한 형태.
Unified 에서는 이 두 케이스가 하나의 flat Q 로 합쳐짐.

**(3) Causal mask 를 절대 위치 기반으로 (`q_abs = s_len - q_len + local`).**
`paged` 에서는 `q_len == s_len` 이라 `offs_m >= offs_n` 상대 위치로 충분했음.
multi-seq 에서는 향후 chunked prefill 확장 여지도 있어 절대 위치가 더 일반적.

**(4) Backend 가 multi-seq 관찰 카운터 유지.**
`FIRE_COUNTER["max_prefill_seqs"]`, `max_decode_seqs` 등 — 새 최대값을 볼 때마다 로그 출력.
vLLM 스케줄러가 실제로 multi-seq batch 를 내리는지를 시각적으로 확인 가능.

## 5. 검증 (kernel 단독 smoke test)

**12/12 조합 PASS**:
- prefill × (fp16/bf16) × q_lens ∈ {(32,64,128), (17,31,97), (50,)} × D ∈ {64,128}
- decode  × (fp16/bf16) × s_lens ∈ {(32,128), (64,256,1024), (50,50,50,50)}

## 6. E2E 실측 (vLLM 에서 Qwen3-0.6B, max_num_seqs=4, 4 prompts 동시 제출)

```
MyTritonImpl.forward fired (prefill multiseq) num_seqs=2 tokens=13
MyTritonImpl.forward fired (decode  multiseq) num_seqs=2
MyTritonImpl.forward fired (decode  multiseq) num_seqs=4   ← 진짜 multi-seq batching
[0]  Paris. The capital of Italy is Rome. The capital of Spain is Madrid.
[1] ...? A. Mercury B. Venus C. Earth D. Mars
[2]  "Macbeth" in 1606. The play is based on
[3]  a group of developers who wanted to create a new way to write code that is
```

`num_seqs=4` 한 번의 decode 커널 호출이 4 시퀀스를 동시에 처리 — continuous batching 이 우리 backend 까지 도달.

## 7. 파일

- `pyproject.toml` — name `my-multiseq-backend`, entry point `my_multiseq_backend`
- `triton_attn.py` — multi-seq 커널 2개 + 래퍼 + `_pack_to_paged_multiseq` 헬퍼
- `triton_attention_backend.py` — 분할 dispatch 로직 + multi-seq 관찰 카운터
- `qwen3_triton_attention.ipynb` — 15셀, 4 prompts 동시 생성

## 8. 알려진 한계

- Chunked prefill 미지원 (q\_len > 1 and q\_len < s\_len 이면 assert)
- dtype fp16/bf16, head\_dim 2의 거듭제곱
- cudagraph 미지원 (`enforce_eager=True`)
- sliding\_window · alibi · logits\_soft\_cap · kv\_sharing · MLA · sparse 미지원
- **vLLM 0.19.1 정확히 고정**

## 9. 이 프로젝트의 위치

```
vllm_padded_decode    단일 prefill 커널 + zero-pad decode (O(s²))
       ▼
vllm_split            prefill + decode 전용 커널 (Python gather, O(s))
       ▼
vllm_paged            두 커널 유지, 커널이 paged KV 를 직접 읽음 (max_num_seqs=1)
       ▼
vllm_multiseq (← 여기) 두 커널이 각각 multi-seq 를 한 launch 에 처리. vLLM v0 스타일 batching
       ▼
vllm_unified (미래)    커널 1개로 prefill+decode 혼합 배치 통합. vLLM v1 스타일
```
