"""Fused causal attention kernels — paged KV read inside the kernel.

Same two-kernel split as vllm_split, but now K/V are loaded via `block_table`
indirection directly inside the Triton kernel (vLLM v0 style). No more
Python-side `_gather_kv` shuffling.

Kernel naming follows vLLM:
    _fwd_kernel_prefill_paged : q_len == kv_len, causal mask
    _fwd_kernel_decode_paged  : q_len == 1, full kv_len, no causal mask

Cache layout (matches MyTritonBackend.get_kv_cache_shape):
    key_cache:   [num_blocks, block_size, num_kv_heads, head_dim]
    value_cache: [num_blocks, block_size, num_kv_heads, head_dim]
    block_table: [num_seqs, max_blocks]   (num_seqs == 1 here)

For key position `k` within a sequence:
    logical_block_idx   = k // BLOCK_SIZE
    offset_in_block     = k %  BLOCK_SIZE
    physical_block_idx  = block_table[seq_idx, logical_block_idx]
    key = key_cache[physical_block_idx, offset_in_block, kv_head, :]
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
# Prefill kernel — q_len == kv_len == S, causal, paged KV
# ---------------------------------------------------------------------------

@triton.jit
def _fwd_kernel_prefill_paged(
    Q, Out,
    K_cache, V_cache,
    block_table,
    seq_lens,
    # Q/Out strides: [B*Hq, S, D]
    stride_qb, stride_qs, stride_qd,
    stride_ob, stride_os, stride_od,
    # cache strides: [num_blocks, block_size, num_kv_heads, head_dim]
    stride_cache_block, stride_cache_slot, stride_cache_head, stride_cache_d,
    # block_table stride: [num_seqs, max_blocks]
    stride_bt_seq,
    sm_scale,
    num_heads_q,       # total Q heads per batch
    num_queries_per_kv,  # n_rep = Hq // Hkv
    BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # pid_bh: (batch*Hq) flat index; pid_m: query tile index
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    # Decompose into (batch_idx, q_head_idx) then kv_head via GQA
    batch_idx = pid_bh // num_heads_q
    q_head_idx = pid_bh % num_heads_q
    kv_head_idx = q_head_idx // num_queries_per_kv

    # This sequence's full context length (== q_len for prefill)
    S = tl.load(seq_lens + batch_idx)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    io_dtype = Out.dtype.element_ty

    # Load Q tile from the input tensor [B*Hq, S, D]
    q_ptr = Q + pid_bh * stride_qb
    q_mask = offs_m[:, None] < S
    q = tl.load(
        q_ptr + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
        mask=q_mask,
        other=0.0,
    )

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Iterate over KV tiles, loading via paged indirection
    for n in range(0, tl.cdiv(S, BLOCK_N)):
        offs_n = n * BLOCK_N + tl.arange(0, BLOCK_N)     # logical positions
        kv_mask = offs_n < S

        # Paged lookup: physical row = block_table[batch, k // BLOCK_SIZE] * BLOCK_SIZE + k % BLOCK_SIZE
        logical_block = offs_n // BLOCK_SIZE             # [BLOCK_N]
        slot_in_block = offs_n % BLOCK_SIZE              # [BLOCK_N]
        physical_block = tl.load(
            block_table + batch_idx * stride_bt_seq + logical_block,
            mask=kv_mask,
            other=0,
        )
        # Base offset into the cache for this (block, slot, kv_head) triple, per key position
        cache_base = (
            physical_block[:, None] * stride_cache_block
            + slot_in_block[:, None] * stride_cache_slot
            + kv_head_idx * stride_cache_head
            + offs_d[None, :] * stride_cache_d
        )  # [BLOCK_N, BLOCK_D]

        # K: [BLOCK_N, BLOCK_D] → transpose for dot
        k = tl.load(K_cache + cache_base, mask=kv_mask[:, None], other=0.0)
        k_t = tl.trans(k)                                 # [BLOCK_D, BLOCK_N]

        qk = tl.dot(q, k_t) * sm_scale
        causal = offs_m[:, None] >= offs_n[None, :]
        qk = tl.where(causal & kv_mask[None, :], qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        # V: [BLOCK_N, BLOCK_D]
        v = tl.load(V_cache + cache_base, mask=kv_mask[:, None], other=0.0)
        acc = acc + tl.dot(p.to(io_dtype), v, out_dtype=tl.float32)
        m_i = m_new

    acc = acc / l_i[:, None]
    o_ptr = Out + pid_bh * stride_ob
    tl.store(
        o_ptr + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od,
        acc.to(io_dtype),
        mask=q_mask,
    )


# ---------------------------------------------------------------------------
# Decode kernel — q_len == 1, full kv_len, no causal mask, paged KV
# ---------------------------------------------------------------------------

@triton.jit
def _fwd_kernel_decode_paged(
    Q, Out,
    K_cache, V_cache,
    block_table,
    seq_lens,
    stride_qb, stride_qd,      # Q: [B*Hq, D]
    stride_ob, stride_od,      # Out: [B*Hq, D]
    stride_cache_block, stride_cache_slot, stride_cache_head, stride_cache_d,
    stride_bt_seq,
    sm_scale,
    num_heads_q,
    num_queries_per_kv,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    batch_idx = pid_bh // num_heads_q
    q_head_idx = pid_bh % num_heads_q
    kv_head_idx = q_head_idx // num_queries_per_kv

    S = tl.load(seq_lens + batch_idx)
    offs_d = tl.arange(0, BLOCK_D)

    io_dtype = Out.dtype.element_ty

    q = tl.load(Q + pid_bh * stride_qb + offs_d * stride_qd).to(tl.float32)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for n in range(0, tl.cdiv(S, BLOCK_N)):
        offs_n = n * BLOCK_N + tl.arange(0, BLOCK_N)
        kv_mask = offs_n < S

        logical_block = offs_n // BLOCK_SIZE
        slot_in_block = offs_n % BLOCK_SIZE
        physical_block = tl.load(
            block_table + batch_idx * stride_bt_seq + logical_block,
            mask=kv_mask,
            other=0,
        )
        cache_base = (
            physical_block[:, None] * stride_cache_block
            + slot_in_block[:, None] * stride_cache_slot
            + kv_head_idx * stride_cache_head
            + offs_d[None, :] * stride_cache_d
        )

        k = tl.load(K_cache + cache_base, mask=kv_mask[:, None], other=0.0)
        qk = tl.sum(q[None, :] * k.to(tl.float32), axis=1) * sm_scale
        qk = tl.where(kv_mask, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)
        l_i = alpha * l_i + tl.sum(p, axis=0)
        acc = acc * alpha

        v = tl.load(V_cache + cache_base, mask=kv_mask[:, None], other=0.0)
        acc = acc + tl.sum(p[:, None] * v.to(tl.float32), axis=0)
        m_i = m_new

    acc = acc / l_i
    tl.store(
        Out + pid_bh * stride_ob + offs_d * stride_od,
        acc.to(io_dtype),
    )


# ---------------------------------------------------------------------------
# Wrappers — paged-aware
# ---------------------------------------------------------------------------

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


def _common_check(q: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor) -> None:
    assert q.dtype in _SUPPORTED_DTYPES, f"dtype must be fp16/bf16, got {q.dtype}"
    assert key_cache.dtype == q.dtype == value_cache.dtype, "q/cache dtypes must match"
    D = q.shape[-1]
    assert D > 0 and (D & (D - 1)) == 0, f"head_dim must be power of two, got {D}"
    assert key_cache.shape == value_cache.shape
    assert key_cache.ndim == 4, f"key_cache must be [num_blocks, block_size, Hkv, D], got {key_cache.shape}"


def triton_attention_prefill_paged(
    q: torch.Tensor,              # [B, Hq, S, D]
    key_cache: torch.Tensor,      # [num_blocks, block_size, Hkv, D]
    value_cache: torch.Tensor,    # same
    block_table: torch.Tensor,    # [B, max_blocks] int32
    seq_lens: torch.Tensor,       # [B] int32  (here B == 1)
    scale: float | None = None,
) -> torch.Tensor:
    """Prefill attention reading KV from paged cache.
    q_len == kv_len == seq_lens[i] (assumed for prefill; i.e. first forward)."""
    B, Hq, S, D = q.shape
    Hkv = key_cache.shape[2]
    _common_check(q, key_cache, value_cache)
    assert Hq % Hkv == 0
    assert seq_lens.shape == (B,)
    assert q.shape[-1] == key_cache.shape[-1]

    q = q.contiguous()
    q_3d = q.reshape(B * Hq, S, D)
    o_3d = torch.empty_like(q_3d)

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    BLOCK_SIZE = key_cache.shape[1]
    BLOCK = _cap_block_for_head_dim(_get_block_size(q.dtype), D)
    grid = (B * Hq, triton.cdiv(S, BLOCK))
    n_rep = Hq // Hkv

    _fwd_kernel_prefill_paged[grid](
        q_3d, o_3d,
        key_cache, value_cache,
        block_table, seq_lens,
        q_3d.stride(0), q_3d.stride(1), q_3d.stride(2),
        o_3d.stride(0), o_3d.stride(1), o_3d.stride(2),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
        block_table.stride(0),
        scale,
        Hq, n_rep,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_M=BLOCK, BLOCK_N=BLOCK, BLOCK_D=D,
    )
    return o_3d.reshape(B, Hq, S, D)


def triton_attention_decode_paged(
    q: torch.Tensor,              # [B, Hq, 1, D]
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Single-token decode attention reading KV from paged cache."""
    B, Hq, Q_len, D = q.shape
    Hkv = key_cache.shape[2]
    _common_check(q, key_cache, value_cache)
    assert Q_len == 1, f"decode expects q_len=1, got {Q_len}"
    assert Hq % Hkv == 0
    assert seq_lens.shape == (B,)

    q = q.contiguous()
    q_2d = q.reshape(B * Hq, D)
    o_2d = torch.empty_like(q_2d)

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    BLOCK_SIZE = key_cache.shape[1]
    BLOCK_N = _cap_block_for_head_dim(_get_block_size(q.dtype), D)
    grid = (B * Hq,)
    n_rep = Hq // Hkv

    _fwd_kernel_decode_paged[grid](
        q_2d, o_2d,
        key_cache, value_cache,
        block_table, seq_lens,
        q_2d.stride(0), q_2d.stride(1),
        o_2d.stride(0), o_2d.stride(1),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
        block_table.stride(0),
        scale,
        Hq, n_rep,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_N=BLOCK_N, BLOCK_D=D,
    )
    return o_2d.reshape(B, Hq, 1, D)


# ---------------------------------------------------------------------------
# Smoke test — build a paged cache from dense K/V, compare to SDPA
# ---------------------------------------------------------------------------

def _pack_to_paged(
    k_dense: torch.Tensor,   # [B, Hkv, S, D]
    v_dense: torch.Tensor,   # [B, Hkv, S, D]
    block_size: int,
):
    """Rearrange dense KV into paged cache + block_table.
    Each sequence gets a contiguous range of blocks."""
    B, Hkv, S, D = k_dense.shape
    blocks_per_seq = triton.cdiv(S, block_size)
    total_blocks = B * blocks_per_seq
    device, dtype = k_dense.device, k_dense.dtype

    key_cache = torch.zeros(total_blocks, block_size, Hkv, D, dtype=dtype, device=device)
    value_cache = torch.zeros_like(key_cache)
    block_table = torch.zeros(B, blocks_per_seq, dtype=torch.int32, device=device)

    for b in range(B):
        for bi in range(blocks_per_seq):
            phys = b * blocks_per_seq + bi
            block_table[b, bi] = phys
            start = bi * block_size
            end = min(start + block_size, S)
            n = end - start
            # [Hkv, n, D] → [n, Hkv, D]
            key_cache[phys, :n] = k_dense[b, :, start:end, :].transpose(0, 1)
            value_cache[phys, :n] = v_dense[b, :, start:end, :].transpose(0, 1)
    return key_cache, value_cache, block_table


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("GPU not available — skipping smoke test")
        raise SystemExit(0)

    import torch.nn.functional as F

    def _check_prefill(dtype, D, block_size, atol, label):
        B, Hq, Hkv, S = 1, 16, 8, 128
        q = torch.randn(B, Hq, S, D, dtype=dtype, device="cuda")
        k = torch.randn(B, Hkv, S, D, dtype=dtype, device="cuda")
        v = torch.randn(B, Hkv, S, D, dtype=dtype, device="cuda")
        # Dense reference
        k_rep = k.repeat_interleave(Hq // Hkv, dim=1)
        v_rep = v.repeat_interleave(Hq // Hkv, dim=1)
        ref = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)
        # Paged path
        key_cache, value_cache, block_table = _pack_to_paged(k, v, block_size)
        seq_lens = torch.full((B,), S, dtype=torch.int32, device="cuda")
        ours = triton_attention_prefill_paged(q, key_cache, value_cache, block_table, seq_lens)
        err = (ours.float() - ref.float()).abs().max().item()
        ok = err < atol
        print(f"  prefill {label:26s} max_abs_err = {err:.6f}  →  {'PASS' if ok else 'FAIL'}")
        return ok

    def _check_decode(dtype, kv_len, block_size, atol, label):
        B, Hq, Hkv, D = 1, 16, 8, 128
        q_full = torch.randn(B, Hq, kv_len, D, dtype=dtype, device="cuda")
        k = torch.randn(B, Hkv, kv_len, D, dtype=dtype, device="cuda")
        v = torch.randn(B, Hkv, kv_len, D, dtype=dtype, device="cuda")
        k_rep = k.repeat_interleave(Hq // Hkv, dim=1)
        v_rep = v.repeat_interleave(Hq // Hkv, dim=1)
        ref_full = F.scaled_dot_product_attention(q_full, k_rep, v_rep, is_causal=True)
        ref_last = ref_full[:, :, -1:, :]
        q_last = q_full[:, :, -1:, :].contiguous()
        key_cache, value_cache, block_table = _pack_to_paged(k, v, block_size)
        seq_lens = torch.full((B,), kv_len, dtype=torch.int32, device="cuda")
        ours = triton_attention_decode_paged(q_last, key_cache, value_cache, block_table, seq_lens)
        err = (ours.float() - ref_last.float()).abs().max().item()
        ok = err < atol
        print(f"  decode  {label:26s} max_abs_err = {err:.6f}  →  {'PASS' if ok else 'FAIL'}")
        return ok

    results = []
    print("prefill (q_len == kv_len, causal) — paged read")
    for dtype, atol in [(torch.float16, 1e-2), (torch.bfloat16, 3e-2)]:
        for D in (64, 128, 256):
            results.append(_check_prefill(dtype, D, 16, atol, f"{str(dtype).split('.')[-1]}/D={D}/BS=16"))

    print("decode (q_len=1, no causal) — paged read")
    for dtype, atol in [(torch.float16, 1e-2), (torch.bfloat16, 3e-2)]:
        for kv_len in (32, 128, 1024):
            results.append(_check_decode(dtype, kv_len, 16, atol, f"{str(dtype).split('.')[-1]}/S={kv_len}/BS=16"))

    # Different block sizes to check paged indexing
    print("varied block_size (fp16, S=128, D=128)")
    for bs in (8, 16, 32, 64):
        results.append(_check_prefill(torch.float16, 128, bs, 1e-2, f"prefill/BS={bs}"))
        results.append(_check_decode(torch.float16, 128, bs, 1e-2, f"decode/BS={bs}"))

    print("ALL PASS" if all(results) else "SOMETHING FAILED")
