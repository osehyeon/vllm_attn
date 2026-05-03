"""Unified attention kernel — single kernel for prefill + decode + chunked prefill.
block_ptr variant.

Same semantics as vllm_unified/triton_attn.py, but raw pointer arithmetic
for Q/K/V tile I/O is replaced with tl.make_block_ptr.

vLLM v1 style: one kernel handles everything via per-token absolute-position
causal mask (`q_abs = seq_len - q_len + local`). The kernel doesn't care
whether a sequence is in prefill, decode, or chunked-prefill state — all
reduce to the same math once q_abs is known.

KV layout (unchanged from the paged/multiseq projects):
  key_cache, value_cache : [num_blocks, block_size, num_kv_heads, head_dim]
  block_table            : [num_seqs, max_blocks]
  seq_lens               : [num_seqs]
  query_start_loc        : [num_seqs + 1]

Q is always flat-packed [total_q_tokens, num_q_heads, head_dim] with each
sequence's q tokens in a contiguous span.

Grid: (num_seqs * num_heads_q, max_q_blocks) — seq-first layout
(simpler than the flat grid + find_seq_idx that vLLM v1 actually uses; the
observation is what matters here, not the exact launch topology).
"""

import math
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Block-size heuristic
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
# Unified kernel — single code path for prefill / decode / chunked-prefill
# ---------------------------------------------------------------------------

@triton.jit
def _fwd_kernel_unified(
    Q,                            # [total_q_tokens, Hq, D]  flat-packed across seqs
    Out,                          # same shape
    K_cache, V_cache,             # [num_blocks, block_size, num_kv_heads, head_dim]
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

    # Local q offsets within this sequence's q-token span
    offs_m_local = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    q_mask = offs_m_local < q_len

    # Absolute query positions within the full sequence — works for prefill
    # (q_len == S → q_abs = [0, S)), decode (q_len == 1 → q_abs = [S-1]),
    # and chunked prefill (1 < q_len < S → q_abs = [S-q_len, S)).
    q_abs = S - q_len + offs_m_local                            # [BLOCK_M]

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

        # Causal with absolute positions
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
# Wrapper — single entry point covering all batch shapes
# ---------------------------------------------------------------------------

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


def _common_check(q, key_cache, value_cache):
    assert q.dtype in _SUPPORTED_DTYPES, f"dtype must be fp16/bf16, got {q.dtype}"
    assert key_cache.dtype == q.dtype == value_cache.dtype
    D = q.shape[-1]
    assert D > 0 and (D & (D - 1)) == 0, f"head_dim must be power of two, got {D}"
    assert key_cache.shape == value_cache.shape
    assert key_cache.ndim == 4


def triton_attention_unified(
    q: torch.Tensor,                # [total_q_tokens, Hq, D]
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,      # [num_seqs, max_blocks]
    seq_lens: torch.Tensor,         # [num_seqs]
    query_start_loc: torch.Tensor,  # [num_seqs + 1]
    scale: float | None = None,
) -> torch.Tensor:
    """Single kernel for every batch shape — pure prefill, pure decode,
    chunked-prefill, and any mixture thereof. The kernel uses each token's
    absolute position (`seq_len - q_len + local_idx`) so the distinction
    between the three doesn't exist at the math level."""
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
    max_q_blocks = triton.cdiv(max(max_q_len, 1), BLOCK)
    grid = (num_seqs * Hq, max_q_blocks)
    n_rep = Hq // Hkv

    _fwd_kernel_unified[grid](
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


# ---------------------------------------------------------------------------
# Smoke-test helpers — identical to multiseq (shared paged cache builder)
# ---------------------------------------------------------------------------

def _pack_to_paged_multiseq(
    k_list: list[torch.Tensor],    # each [Hkv, S_i, D]
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
# Smoke tests — exercise every batch shape the unified kernel should handle
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("GPU not available — skipping smoke test")
        raise SystemExit(0)

    import torch.nn.functional as F

    def _run(
        per_seq_specs,   # list of tuples: (q_len, s_len) for each seq
        dtype, D, atol, label,
    ):
        """Exercise a unified batch where each seq has its own (q_len, s_len).
        q_len == s_len → prefill, q_len == 1 → decode, 1 < q_len < s_len → chunked prefill.
        Reference: SDPA over full dense Q (length s_len), take last q_len positions."""
        Hq, Hkv = 16, 8
        full_q_per_seq, k_per_seq, v_per_seq = [], [], []
        ref_per_seq = []
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
            ).squeeze(0)                                        # [Hq, s_len, D]
            ref_per_seq.append(ref_full[:, s_len - q_len:, :])  # last q_len rows

        # Flat-pack only the last q_len tokens of each seq's Q (that's what
        # gets submitted in the unified batch)
        q_flat_chunks = [
            full_q_per_seq[i][:, -per_seq_specs[i][0]:, :].transpose(0, 1)
            for i in range(len(per_seq_specs))
        ]
        q_flat = torch.cat(q_flat_chunks, dim=0)  # [sum(q_len), Hq, D]

        q_lens = [spec[0] for spec in per_seq_specs]
        s_lens = [spec[1] for spec in per_seq_specs]
        cumulative = torch.cumsum(torch.tensor(q_lens, dtype=torch.int32), 0)
        query_start_loc = torch.cat(
            [torch.tensor([0], dtype=torch.int32), cumulative]
        ).to("cuda")
        seq_lens_t = torch.tensor(s_lens, dtype=torch.int32, device="cuda")

        key_cache, value_cache, block_table = _pack_to_paged_multiseq(
            k_per_seq, v_per_seq, block_size=16
        )
        ours = triton_attention_unified(
            q_flat, key_cache, value_cache, block_table, seq_lens_t, query_start_loc
        )

        max_err = 0.0
        for i, q_len in enumerate(q_lens):
            start = int(query_start_loc[i].item())
            ours_seq = ours[start:start + q_len].transpose(0, 1)   # [Hq, q_len, D]
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

    print("all decode (q_len == 1, varied s_len)")
    for dtype, atol in [(torch.float16, 1e-2), (torch.bfloat16, 3e-2)]:
        results.append(_run(
            [(1, 32), (1, 128), (1, 1024)], dtype, 128, atol,
            f"{str(dtype).split('.')[-1]}/decode s=(32,128,1024)",
        ))

    print("mixed (prefill + decode in same batch)")
    for dtype, atol in [(torch.float16, 1e-2), (torch.bfloat16, 3e-2)]:
        results.append(_run(
            [(64, 64), (1, 200), (1, 50), (32, 32)], dtype, 128, atol,
            f"{str(dtype).split('.')[-1]}/mixed prefill+decode",
        ))

    print("chunked prefill (q_len > 1, q_len < s_len)")
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
