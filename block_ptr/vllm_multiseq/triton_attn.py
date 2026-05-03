"""Multi-seq paged attention kernels — block_ptr variant.

Same semantics as vllm_multiseq/triton_attn.py, but raw pointer arithmetic
for Q/K/V tile I/O is replaced with tl.make_block_ptr.

Two kernels remain (prefill + decode), same as vllm_paged, but now accept
**multiple sequences per launch**:
  - prefill kernel consumes a flat-packed Q  [total_q_tokens, Hq, D]
    with `query_start_loc` demarcating each seq's span. Causal mask uses
    absolute query position inside each sequence (`seq_len - q_len + local`).
  - decode kernel consumes Q [num_decode_seqs, Hq, D] — exactly one query
    token per sequence, indexed directly by batch_idx.

KV cache is still paged:
  key_cache/value_cache : [num_blocks, block_size, num_kv_heads, head_dim]
  block_table           : [num_seqs, max_blocks]
  seq_lens              : [num_seqs]

A mixed batch (prefill + decode) is split by the backend and dispatched to
the two kernels separately — "vLLM v0 style" multi-seq handling.
"""

import math
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Block-size heuristic (mirrors vllm/v1/attention/ops/triton_prefill_attention.py)
# ---------------------------------------------------------------------------

def _get_block_size(dtype: torch.dtype) -> int:
    if dtype == torch.float32:
        return 32
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return 128
    return 64


def _cap_block_for_head_dim(block: int, head_dim: int) -> int:
    if head_dim <= 64:
        return block
    if head_dim <= 128:
        return min(block, 64)
    if head_dim <= 256:
        return min(block, 32)
    return min(block, 16)


# ---------------------------------------------------------------------------
# Prefill kernel — multi-seq, flat-packed Q, paged KV, causal
# ---------------------------------------------------------------------------

@triton.jit
def _fwd_kernel_prefill_multiseq(
    Q,                            # [total_q_tokens, Hq, D] flat-packed
    Out,                          # same shape
    K_cache, V_cache,
    block_table,                  # [num_seqs, max_blocks]
    seq_lens,                     # [num_seqs]
    query_start_loc,              # [num_seqs + 1]
    # Q/Out strides
    stride_q_t, stride_q_h, stride_q_d,
    stride_o_t, stride_o_h, stride_o_d,
    # cache strides
    stride_cache_block, stride_cache_slot, stride_cache_head, stride_cache_d,
    stride_bt_seq,
    sm_scale,
    num_heads_q,
    num_queries_per_kv,
    total_q_tokens,               # new arg: q.shape[0], needed for make_block_ptr shape
    BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,        # BLOCK_N == BLOCK_SIZE enforced by wrapper
    BLOCK_D: tl.constexpr,
):
    # grid: (num_seqs * num_heads_q, max_q_blocks)
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    batch_idx = pid_bh // num_heads_q
    q_head_idx = pid_bh % num_heads_q
    kv_head_idx = q_head_idx // num_queries_per_kv

    S = tl.load(seq_lens + batch_idx)
    q_start = tl.load(query_start_loc + batch_idx)
    q_end = tl.load(query_start_loc + batch_idx + 1)
    q_len = q_end - q_start

    offs_m_local = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    q_mask = offs_m_local < q_len

    # Causal mask: absolute positions within the full sequence.
    q_abs = S - q_len + offs_m_local                      # [BLOCK_M]

    io_dtype = Out.dtype.element_ty

    # Snippet C: flat-packed Q via make_block_ptr
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_head_idx * stride_q_h,
        shape=(total_q_tokens, BLOCK_D),
        strides=(stride_q_t, stride_q_d),
        offsets=(q_start + pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )
    q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Snippet B: paged K/V — fresh make_block_ptr each iteration (BLOCK_N == BLOCK_SIZE)
    for n in range(0, tl.cdiv(S, BLOCK_SIZE)):
        n_start = n * BLOCK_SIZE
        offs_n = n_start + tl.arange(0, BLOCK_N)              # BLOCK_N == BLOCK_SIZE
        kv_mask = offs_n < S

        # Scalar load: one physical block index per iteration
        physical_block_idx = tl.load(block_table + batch_idx * stride_bt_seq + n)
        kv_block_base = physical_block_idx * stride_cache_block + kv_head_idx * stride_cache_head

        # K loaded as (BLOCK_D, BLOCK_N) — virtual transpose via order=(0,1)
        K_block_ptr = tl.make_block_ptr(
            base=K_cache + kv_block_base,
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

        qk = tl.dot(q, k_t) * sm_scale
        causal = q_abs[:, None] >= offs_n[None, :]
        qk = tl.where(q_mask[:, None] & kv_mask[None, :] & causal, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        acc = acc + tl.dot(p.to(io_dtype), v, out_dtype=tl.float32)
        m_i = m_new

    acc = acc / l_i[:, None]

    # Snippet F: output store via make_block_ptr
    O_block_ptr = tl.make_block_ptr(
        base=Out + q_head_idx * stride_o_h,
        shape=(total_q_tokens, BLOCK_D),
        strides=(stride_o_t, stride_o_d),
        offsets=(q_start + pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )
    tl.store(O_block_ptr, acc.to(io_dtype), boundary_check=(0,))


# ---------------------------------------------------------------------------
# Decode kernel — multi-seq, one q token per seq
# ---------------------------------------------------------------------------

@triton.jit
def _fwd_kernel_decode_multiseq(
    Q,                            # [num_decode_seqs, Hq, D]
    Out,                          # same
    K_cache, V_cache,
    block_table,                  # [num_decode_seqs, max_blocks]
    seq_lens,                     # [num_decode_seqs]
    stride_q_s, stride_q_h, stride_q_d,
    stride_o_s, stride_o_h, stride_o_d,
    stride_cache_block, stride_cache_slot, stride_cache_head, stride_cache_d,
    stride_bt_seq,
    sm_scale,
    num_heads_q,
    num_queries_per_kv,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,        # BLOCK_N == BLOCK_SIZE enforced by wrapper
    BLOCK_D: tl.constexpr,
):
    # grid: (num_decode_seqs * num_heads_q,)
    pid_bh = tl.program_id(0)
    batch_idx = pid_bh // num_heads_q
    q_head_idx = pid_bh % num_heads_q
    kv_head_idx = q_head_idx // num_queries_per_kv

    S = tl.load(seq_lens + batch_idx)

    io_dtype = Out.dtype.element_ty

    # Snippet E: 1-D Q via make_block_ptr
    Q_block_ptr = tl.make_block_ptr(
        base=Q + batch_idx * stride_q_s + q_head_idx * stride_q_h,
        shape=(BLOCK_D,),
        strides=(stride_q_d,),
        offsets=(0,),
        block_shape=(BLOCK_D,),
        order=(0,),
    )
    q = tl.load(Q_block_ptr).to(tl.float32)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    # Snippet B (decode variant): paged K/V — fresh make_block_ptr each iteration
    # K is loaded as [BLOCK_N, BLOCK_D] (no transpose needed for decode dot)
    for n in range(0, tl.cdiv(S, BLOCK_SIZE)):
        offs_n = n * BLOCK_N + tl.arange(0, BLOCK_N)
        kv_mask = offs_n < S

        physical_block_idx = tl.load(block_table + batch_idx * stride_bt_seq + n)
        kv_block_base = physical_block_idx * stride_cache_block + kv_head_idx * stride_cache_head

        K_block_ptr = tl.make_block_ptr(
            base=K_cache + kv_block_base,
            shape=(BLOCK_SIZE, BLOCK_D),
            strides=(stride_cache_slot, stride_cache_d),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V_cache + kv_block_base,
            shape=(BLOCK_SIZE, BLOCK_D),
            strides=(stride_cache_slot, stride_cache_d),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )
        k = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")

        qk = tl.sum(q[None, :] * k.to(tl.float32), axis=1) * sm_scale
        qk = tl.where(kv_mask, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)
        l_i = alpha * l_i + tl.sum(p, axis=0)
        acc = acc * alpha

        acc = acc + tl.sum(p[:, None] * v.to(tl.float32), axis=0)
        m_i = m_new

    acc = acc / l_i
    O_block_ptr = tl.make_block_ptr(
        base=Out + batch_idx * stride_o_s + q_head_idx * stride_o_h,
        shape=(BLOCK_D,),
        strides=(stride_o_d,),
        offsets=(0,),
        block_shape=(BLOCK_D,),
        order=(0,),
    )
    tl.store(O_block_ptr, acc.to(io_dtype))


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


def _common_check(q: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor) -> None:
    assert q.dtype in _SUPPORTED_DTYPES, f"dtype must be fp16/bf16, got {q.dtype}"
    assert key_cache.dtype == q.dtype == value_cache.dtype
    D = q.shape[-1]
    assert D > 0 and (D & (D - 1)) == 0, f"head_dim must be power of two, got {D}"
    assert key_cache.shape == value_cache.shape
    assert key_cache.ndim == 4


def triton_attention_prefill_multiseq(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Multi-seq prefill, batched in a single kernel launch."""
    assert q.ndim == 3, f"q must be [total_q_tokens, Hq, D], got {q.shape}"
    _common_check(q, key_cache, value_cache)
    total_q, Hq, D = q.shape
    Hkv = key_cache.shape[2]
    assert Hq % Hkv == 0
    num_seqs = seq_lens.shape[0]
    assert block_table.shape[0] == num_seqs
    assert query_start_loc.shape[0] == num_seqs + 1

    q = q.contiguous()
    o = torch.empty_like(q)

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    BLOCK_SIZE = key_cache.shape[1]
    BLOCK = _cap_block_for_head_dim(_get_block_size(q.dtype), D)
    max_q_len = int((query_start_loc[1:] - query_start_loc[:-1]).max().item())
    max_q_blocks = triton.cdiv(max_q_len, BLOCK)
    grid = (num_seqs * Hq, max_q_blocks)
    n_rep = Hq // Hkv

    _fwd_kernel_prefill_multiseq[grid](
        q, o,
        key_cache, value_cache,
        block_table, seq_lens, query_start_loc,
        q.stride(0), q.stride(1), q.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
        block_table.stride(0),
        scale,
        Hq, n_rep,
        total_q,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_M=BLOCK, BLOCK_N=BLOCK_SIZE, BLOCK_D=D,
    )
    return o


def triton_attention_decode_multiseq(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Multi-seq decode — one query token per seq, batched in a single launch."""
    assert q.ndim == 3, f"q must be [num_decode_seqs, Hq, D], got {q.shape}"
    _common_check(q, key_cache, value_cache)
    num_seqs, Hq, D = q.shape
    Hkv = key_cache.shape[2]
    assert Hq % Hkv == 0
    assert block_table.shape[0] == num_seqs
    assert seq_lens.shape == (num_seqs,)

    q = q.contiguous()
    o = torch.empty_like(q)

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    BLOCK_SIZE = key_cache.shape[1]
    grid = (num_seqs * Hq,)
    n_rep = Hq // Hkv

    _fwd_kernel_decode_multiseq[grid](
        q, o,
        key_cache, value_cache,
        block_table, seq_lens,
        q.stride(0), q.stride(1), q.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
        block_table.stride(0),
        scale,
        Hq, n_rep,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_N=BLOCK_SIZE, BLOCK_D=D,
    )
    return o


# ---------------------------------------------------------------------------
# Smoke-test helpers
# ---------------------------------------------------------------------------

def _pack_to_paged_multiseq(
    k_list: list[torch.Tensor],
    v_list: list[torch.Tensor],
    block_size: int,
):
    """Build a shared paged cache + per-seq block_table from a list of
    dense per-seq K/V tensors of possibly different lengths.
    Each k/v tensor has shape [Hkv, S_i, D]."""
    assert len(k_list) == len(v_list)
    num_seqs = len(k_list)
    Hkv, _, D = k_list[0].shape
    device, dtype = k_list[0].device, k_list[0].dtype

    blocks_per_seq = [triton.cdiv(k.shape[1], block_size) for k in k_list]
    total_blocks = sum(blocks_per_seq)
    max_blocks = max(blocks_per_seq)

    key_cache = torch.zeros(total_blocks, block_size, Hkv, D, dtype=dtype, device=device)
    value_cache = torch.zeros_like(key_cache)
    block_table = torch.zeros(num_seqs, max_blocks, dtype=torch.int32, device=device)

    cursor = 0
    for seq_idx, (k, v) in enumerate(zip(k_list, v_list)):
        S = k.shape[1]
        bps = blocks_per_seq[seq_idx]
        for bi in range(bps):
            phys = cursor + bi
            block_table[seq_idx, bi] = phys
            start = bi * block_size
            end = min(start + block_size, S)
            n = end - start
            key_cache[phys, :n] = k[:, start:end, :].transpose(0, 1)   # [Hkv,n,D] → [n,Hkv,D]
            value_cache[phys, :n] = v[:, start:end, :].transpose(0, 1)
        cursor += bps
    return key_cache, value_cache, block_table


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("GPU not available — skipping smoke test")
        raise SystemExit(0)

    import torch.nn.functional as F

    def _check_multi_prefill(dtype, D, q_lens, atol, label):
        Hq, Hkv = 16, 8
        q_per_seq, k_per_seq, v_per_seq, ref_per_seq = [], [], [], []
        for S in q_lens:
            q_s = torch.randn(Hq, S, D, dtype=dtype, device="cuda")
            k_s = torch.randn(Hkv, S, D, dtype=dtype, device="cuda")
            v_s = torch.randn(Hkv, S, D, dtype=dtype, device="cuda")
            q_per_seq.append(q_s)
            k_per_seq.append(k_s)
            v_per_seq.append(v_s)
            k_rep = k_s.repeat_interleave(Hq // Hkv, dim=0)
            v_rep = v_s.repeat_interleave(Hq // Hkv, dim=0)
            ref = F.scaled_dot_product_attention(
                q_s.unsqueeze(0), k_rep.unsqueeze(0), v_rep.unsqueeze(0), is_causal=True
            ).squeeze(0)
            ref_per_seq.append(ref)

        q_flat = torch.cat([q.transpose(0, 1) for q in q_per_seq], dim=0)  # [sum(S), Hq, D]
        cumulative = torch.cumsum(torch.tensor(q_lens, dtype=torch.int32), 0)
        query_start_loc = torch.cat(
            [torch.tensor([0], dtype=torch.int32), cumulative]
        ).to("cuda")
        seq_lens = torch.tensor(q_lens, dtype=torch.int32, device="cuda")

        key_cache, value_cache, block_table = _pack_to_paged_multiseq(
            k_per_seq, v_per_seq, block_size=16
        )
        ours = triton_attention_prefill_multiseq(
            q_flat, key_cache, value_cache, block_table, seq_lens, query_start_loc
        )

        max_err = 0.0
        for i, S in enumerate(q_lens):
            start = int(query_start_loc[i].item())
            ours_seq = ours[start:start + S].transpose(0, 1)  # [Hq, S, D]
            err = (ours_seq.float() - ref_per_seq[i].float()).abs().max().item()
            max_err = max(max_err, err)
        ok = max_err < atol
        print(f"  prefill  {label:32s} max_abs_err = {max_err:.6f}  →  {'PASS' if ok else 'FAIL'}")
        return ok

    def _check_multi_decode(dtype, s_lens, atol, label):
        Hq, Hkv, D = 16, 8, 128
        q_last_tokens, k_per_seq, v_per_seq, ref_lasts = [], [], [], []
        for S in s_lens:
            q_full = torch.randn(Hq, S, D, dtype=dtype, device="cuda")
            k_s = torch.randn(Hkv, S, D, dtype=dtype, device="cuda")
            v_s = torch.randn(Hkv, S, D, dtype=dtype, device="cuda")
            k_per_seq.append(k_s)
            v_per_seq.append(v_s)
            k_rep = k_s.repeat_interleave(Hq // Hkv, dim=0)
            v_rep = v_s.repeat_interleave(Hq // Hkv, dim=0)
            ref = F.scaled_dot_product_attention(
                q_full.unsqueeze(0), k_rep.unsqueeze(0), v_rep.unsqueeze(0), is_causal=True
            ).squeeze(0)
            ref_lasts.append(ref[:, -1, :])                      # [Hq, D]
            q_last_tokens.append(q_full[:, -1, :])               # [Hq, D]

        q_batched = torch.stack(q_last_tokens, dim=0)            # [num_seqs, Hq, D]
        seq_lens = torch.tensor(s_lens, dtype=torch.int32, device="cuda")
        key_cache, value_cache, block_table = _pack_to_paged_multiseq(
            k_per_seq, v_per_seq, block_size=16
        )

        ours = triton_attention_decode_multiseq(
            q_batched, key_cache, value_cache, block_table, seq_lens
        )
        max_err = 0.0
        for i in range(len(s_lens)):
            err = (ours[i].float() - ref_lasts[i].float()).abs().max().item()
            max_err = max(max_err, err)
        ok = max_err < atol
        print(f"  decode   {label:32s} max_abs_err = {max_err:.6f}  →  {'PASS' if ok else 'FAIL'}")
        return ok

    results = []
    print("prefill (multi-seq, varying q_lens) — paged read")
    for dtype, atol in [(torch.float16, 1e-2), (torch.bfloat16, 3e-2)]:
        for q_lens, D in [((32, 64, 128), 128), ((17, 31, 97), 128), ((50,), 64)]:
            results.append(_check_multi_prefill(
                dtype, D, q_lens, atol,
                f"{str(dtype).split('.')[-1]}/q_lens={q_lens}/D={D}",
            ))

    print("decode (multi-seq, varying kv_lens) — paged read")
    for dtype, atol in [(torch.float16, 1e-2), (torch.bfloat16, 3e-2)]:
        for s_lens in [(32, 128), (64, 256, 1024), (50, 50, 50, 50)]:
            results.append(_check_multi_decode(
                dtype, s_lens, atol,
                f"{str(dtype).split('.')[-1]}/s_lens={s_lens}",
            ))

    print("ALL PASS" if all(results) else "SOMETHING FAILED")
