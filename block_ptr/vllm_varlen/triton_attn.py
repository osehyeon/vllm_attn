"""Varlen attention — unified kernel with seq-aligned flat grid.
block_ptr variant.

Same semantics as vllm_varlen/triton_attn.py, but raw pointer arithmetic
for Q/K/V tile I/O is replaced with tl.make_block_ptr.

Same math as `vllm_unified` (one kernel for prefill + decode + chunked, via
absolute-position causal mask), but the **launch topology** now matches
vLLM v1's `kernel_unified_attention_2d`:

  Grid: (total_q_blocks_rounded, num_query_heads)
  Each program handles ONE sequence's BLOCK_Q tokens × ONE Q head.
  The sequence index is discovered via `find_seq_idx` (binary search over
  per-seq cumulative block counts). This "seq-aligned flat" layout keeps
  BLOCK_Q internally homogeneous (one seq per block) while exposing a
  token-flat grid dimension — no idle whole-sequence programs.

Q layout is still flat-packed [total_q_tokens, Hq, D], same as unified.
"""

import math
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Block-size heuristics
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
# Helpers
# ---------------------------------------------------------------------------

@triton.jit
def _find_seq_idx(cum_q_blocks_ptr, q_block_idx, num_seqs):
    """Binary search: largest seq such that cum_q_blocks[seq] <= q_block_idx.

    cum_q_blocks: [num_seqs + 1], cumulative count of BLOCK_Q blocks each
    sequence occupies in the flat grid (cum_q_blocks[0] == 0, cum_q_blocks[-1]
    == total_q_blocks_rounded).

    Intentionally uses raw scalar pointer arithmetic — this is a scalar binary
    search over a small array (num_seqs entries). block_ptr is for tile I/O;
    it adds no value here and would complicate the scalar reduction.
    """
    lo = 0
    hi = num_seqs
    while lo < hi:
        mid = (lo + hi) // 2
        start = tl.load(cum_q_blocks_ptr + mid)
        if start <= q_block_idx:
            lo = mid + 1
        else:
            hi = mid
    return lo - 1


# ---------------------------------------------------------------------------
# Varlen kernel — unified math, flat grid, binary search for seq_idx
# ---------------------------------------------------------------------------

@triton.jit
def _fwd_kernel_varlen(
    Q,                           # [total_q_tokens, Hq, D]
    Out,                         # same
    K_cache, V_cache,            # [num_blocks, block_size, Hkv, D]
    block_table,                 # [num_seqs, max_blocks]
    seq_lens,                    # [num_seqs]
    query_start_loc,             # [num_seqs + 1]
    cum_q_blocks,                # [num_seqs + 1]  — running count of BLOCK_Q blocks per seq
    stride_q_t, stride_q_h, stride_q_d,
    stride_o_t, stride_o_h, stride_o_d,
    stride_cache_block, stride_cache_slot, stride_cache_head, stride_cache_d,
    stride_bt_seq,
    sm_scale,
    num_seqs,
    num_queries_per_kv,
    total_q_tokens,              # new arg: q.shape[0], needed for make_block_ptr shape
    BLOCK_SIZE: tl.constexpr,
    BLOCK_Q: tl.constexpr,       # query tile = BLOCK_M conceptually
    BLOCK_N: tl.constexpr,       # BLOCK_N == BLOCK_SIZE enforced by wrapper
    BLOCK_D: tl.constexpr,
):
    # grid: (total_q_blocks_rounded, num_query_heads)
    q_block_global_idx = tl.program_id(0)
    q_head_idx = tl.program_id(1)
    kv_head_idx = q_head_idx // num_queries_per_kv

    # Which sequence does this block belong to?
    seq_idx = _find_seq_idx(cum_q_blocks, q_block_global_idx, num_seqs)

    # The block's offset WITHIN this sequence (local block index)
    seq_block_start = tl.load(cum_q_blocks + seq_idx)
    q_block_local_idx = q_block_global_idx - seq_block_start

    # Per-sequence info
    # Cast to int32 — make_block_ptr offsets only support 32-bit (Triton constraint).
    # query_start_loc may be int64 (PyTorch cumsum default).
    q_start = tl.load(query_start_loc + seq_idx).to(tl.int32)
    q_end = tl.load(query_start_loc + seq_idx + 1).to(tl.int32)
    q_len = q_end - q_start
    S = tl.load(seq_lens + seq_idx)

    # Local query offsets within the sequence's q span
    offs_m_local = q_block_local_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = offs_m_local < q_len

    # Absolute position in the full sequence (same formula as unified)
    q_abs = S - q_len + offs_m_local                      # [BLOCK_Q]

    io_dtype = Out.dtype.element_ty

    # Snippet C: flat-packed Q via make_block_ptr
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_head_idx * stride_q_h,
        shape=(total_q_tokens, BLOCK_D),
        strides=(stride_q_t, stride_q_d),
        offsets=(q_start + q_block_local_idx * BLOCK_Q, 0),
        block_shape=(BLOCK_Q, BLOCK_D),
        order=(1, 0),
    )
    q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")

    m_i = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q, BLOCK_D], dtype=tl.float32)

    # KV iteration — paged reads from this seq's block table (scalar seq_idx)
    # Snippet B: fresh make_block_ptr each iteration (BLOCK_N == BLOCK_SIZE)
    for n in range(0, tl.cdiv(S, BLOCK_SIZE)):
        n_start = n * BLOCK_SIZE
        offs_n = n_start + tl.arange(0, BLOCK_N)              # BLOCK_N == BLOCK_SIZE
        kv_mask = offs_n < S

        physical_block_idx = tl.load(block_table + seq_idx * stride_bt_seq + n)
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
        offsets=(q_start + q_block_local_idx * BLOCK_Q, 0),
        block_shape=(BLOCK_Q, BLOCK_D),
        order=(1, 0),
    )
    tl.store(O_block_ptr, acc.to(io_dtype), boundary_check=(0,))


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


def _common_check(q, key_cache, value_cache):
    assert q.dtype in _SUPPORTED_DTYPES, f"dtype must be fp16/bf16, got {q.dtype}"
    assert key_cache.dtype == q.dtype == value_cache.dtype
    D = q.shape[-1]
    assert D > 0 and (D & (D - 1)) == 0, f"head_dim must be power of two, got {D}"
    assert key_cache.shape == value_cache.shape
    assert key_cache.ndim == 4


def triton_attention_varlen(
    q: torch.Tensor,                # [total_q_tokens, Hq, D]
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,      # [num_seqs, max_blocks]
    seq_lens: torch.Tensor,         # [num_seqs]
    query_start_loc: torch.Tensor,  # [num_seqs + 1]
    scale: float | None = None,
) -> torch.Tensor:
    """Unified kernel with a **seq-aligned flat grid** launch topology.
    Same math as triton_attention_unified; the difference is that programs
    are dispatched per BLOCK_Q of tokens (discovered via binary search)
    rather than per sequence."""
    assert q.ndim == 3, f"q must be [total_q_tokens, Hq, D], got {q.shape}"
    _common_check(q, key_cache, value_cache)
    total_q, Hq, D = q.shape
    Hkv = key_cache.shape[2]
    assert Hq % Hkv == 0
    num_seqs = int(seq_lens.shape[0])
    assert block_table.shape[0] == num_seqs
    assert query_start_loc.shape[0] == num_seqs + 1

    q = q.contiguous()
    o = torch.empty_like(q)

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    BLOCK_SIZE = key_cache.shape[1]
    BLOCK = _cap_block_for_head_dim(_get_block_size(q.dtype), D)
    # BLOCK_Q: q token tile. Tie to BLOCK for simplicity.
    BLOCK_Q = BLOCK
    BLOCK_N = BLOCK_SIZE  # BLOCK_N == BLOCK_SIZE required for block_ptr paged KV

    # Per-seq rounded block counts; cumulative prefix for find_seq_idx.
    q_lens = query_start_loc[1:] - query_start_loc[:-1]           # [num_seqs]
    blocks_per_seq = (q_lens + BLOCK_Q - 1) // BLOCK_Q            # [num_seqs]
    cum_q_blocks = torch.zeros(num_seqs + 1, dtype=torch.int32, device=q.device)
    cum_q_blocks[1:] = torch.cumsum(blocks_per_seq.to(torch.int32), 0)
    total_q_blocks = int(cum_q_blocks[-1].item())

    n_rep = Hq // Hkv
    grid = (total_q_blocks, Hq)

    _fwd_kernel_varlen[grid](
        q, o,
        key_cache, value_cache,
        block_table, seq_lens, query_start_loc,
        cum_q_blocks,
        q.stride(0), q.stride(1), q.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
        block_table.stride(0),
        scale,
        num_seqs,
        n_rep,
        total_q,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_Q=BLOCK_Q, BLOCK_N=BLOCK_N, BLOCK_D=D,
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
            key_cache[phys, :n] = k[:, start:end, :].transpose(0, 1)
            value_cache[phys, :n] = v[:, start:end, :].transpose(0, 1)
        cursor += bps
    return key_cache, value_cache, block_table


# ---------------------------------------------------------------------------
# Smoke tests — every batch shape, now with varlen launch topology
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("GPU not available — skipping smoke test")
        raise SystemExit(0)

    import torch.nn.functional as F

    def _run(per_seq_specs, dtype, D, atol, label):
        Hq, Hkv = 16, 8
        full_q_per_seq, k_per_seq, v_per_seq, ref_per_seq = [], [], [], []
        for q_len, s_len in per_seq_specs:
            assert q_len <= s_len
            q_full = torch.randn(Hq, s_len, D, dtype=dtype, device="cuda")
            k_s = torch.randn(Hkv, s_len, D, dtype=dtype, device="cuda")
            v_s = torch.randn(Hkv, s_len, D, dtype=dtype, device="cuda")
            full_q_per_seq.append(q_full)
            k_per_seq.append(k_s)
            v_per_seq.append(v_s)
            k_rep = k_s.repeat_interleave(Hq // Hkv, dim=0)
            v_rep = v_s.repeat_interleave(Hq // Hkv, dim=0)
            ref_full = F.scaled_dot_product_attention(
                q_full.unsqueeze(0), k_rep.unsqueeze(0), v_rep.unsqueeze(0), is_causal=True
            ).squeeze(0)
            ref_per_seq.append(ref_full[:, s_len - q_len:, :])

        chunks = [full_q_per_seq[i][:, -per_seq_specs[i][0]:, :].transpose(0, 1)
                  for i in range(len(per_seq_specs))]
        q_flat = torch.cat(chunks, dim=0)

        q_lens = [s[0] for s in per_seq_specs]
        s_lens = [s[1] for s in per_seq_specs]
        cumulative = torch.cumsum(torch.tensor(q_lens, dtype=torch.int32), 0)
        query_start_loc = torch.cat(
            [torch.tensor([0], dtype=torch.int32), cumulative]
        ).to("cuda")
        seq_lens_t = torch.tensor(s_lens, dtype=torch.int32, device="cuda")

        key_cache, value_cache, block_table = _pack_to_paged_multiseq(
            k_per_seq, v_per_seq, block_size=16
        )
        ours = triton_attention_varlen(
            q_flat, key_cache, value_cache, block_table, seq_lens_t, query_start_loc
        )

        max_err = 0.0
        for i, q_len in enumerate(q_lens):
            start = int(query_start_loc[i].item())
            ours_seq = ours[start:start + q_len].transpose(0, 1)
            err = (ours_seq.float() - ref_per_seq[i].float()).abs().max().item()
            max_err = max(max_err, err)
        ok = max_err < atol
        print(f"  {label:55s} max_abs_err = {max_err:.6f}  →  {'PASS' if ok else 'FAIL'}")
        return ok

    results = []

    print("all prefill (q_len == s_len)")
    for dtype, atol in [(torch.float16, 1e-2), (torch.bfloat16, 3e-2)]:
        results.append(_run(
            [(32, 32), (64, 64), (128, 128)], dtype, 128, atol,
            f"{str(dtype).split('.')[-1]}/prefill q=s=(32,64,128)",
        ))

    print("all decode (q_len == 1)")
    for dtype, atol in [(torch.float16, 1e-2), (torch.bfloat16, 3e-2)]:
        results.append(_run(
            [(1, 32), (1, 128), (1, 1024)], dtype, 128, atol,
            f"{str(dtype).split('.')[-1]}/decode s=(32,128,1024)",
        ))

    print("mixed (prefill + decode)")
    for dtype, atol in [(torch.float16, 1e-2), (torch.bfloat16, 3e-2)]:
        results.append(_run(
            [(64, 64), (1, 200), (1, 50), (32, 32)], dtype, 128, atol,
            f"{str(dtype).split('.')[-1]}/mixed prefill+decode",
        ))

    print("chunked prefill")
    for dtype, atol in [(torch.float16, 1e-2), (torch.bfloat16, 3e-2)]:
        results.append(_run(
            [(16, 64), (32, 128)], dtype, 128, atol,
            f"{str(dtype).split('.')[-1]}/chunked q<s",
        ))

    print("ultimate mix: prefill + chunked + decode")
    for dtype, atol in [(torch.float16, 1e-2), (torch.bfloat16, 3e-2)]:
        results.append(_run(
            [(64, 64), (16, 128), (1, 200), (1, 50)], dtype, 128, atol,
            f"{str(dtype).split('.')[-1]}/prefill+chunked+decode",
        ))

    print("ALL PASS" if all(results) else "SOMETHING FAILED")
