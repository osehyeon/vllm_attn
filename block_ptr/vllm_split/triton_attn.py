"""Fused causal attention kernels — separate prefill and decode paths.

Naming follows vLLM conventions (see vllm/v1/attention/ops/triton_{prefill,decode}_attention.py):
    _fwd_kernel_prefill   : q_len == kv_len, causal mask
    _fwd_kernel_decode    : q_len == 1, full kv_len, no causal mask (last query sees all past)

Block sizes use the same dtype-based heuristic as vLLM's get_block_size().

block_ptr variant: Q/K/V/O loads and stores use tl.make_block_ptr.
Prefill K is transposed virtually via order=(0,1) eliminating tl.trans.
Decode uses 1-D Q block_ptr and 2-D K/V block_ptrs (no transpose needed).
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
    """Shrink the tile when head_dim is large so shared memory stays under
    the SM limit (roughly block × head_dim × dtype_size × ~4 buffers)."""
    if head_dim <= 64:
        return block
    if head_dim <= 128:
        return min(block, 64)
    if head_dim <= 256:
        return min(block, 32)
    return min(block, 16)


# ---------------------------------------------------------------------------
# Prefill kernel — q_len == kv_len == S, causal
# ---------------------------------------------------------------------------

@triton.jit
def _fwd_kernel_prefill(
    Q, K, V, Out,
    stride_qb, stride_qs, stride_qd,
    stride_kb, stride_ks, stride_kd,
    stride_vb, stride_vs, stride_vd,
    stride_ob, stride_os, stride_od,
    sm_scale,
    S,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    io_dtype = Out.dtype.element_ty

    # Load Q tile [BLOCK_M, BLOCK_D] via block_ptr (Snippet D)
    Q_block_ptr = tl.make_block_ptr(
        base=Q + pid_bh * stride_qb,
        shape=(S, BLOCK_D),
        strides=(stride_qs, stride_qd),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )
    q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # K transposed view (BLOCK_D, BLOCK_N) via order=(0,1) — Snippet A
    K_block_ptr = tl.make_block_ptr(
        base=K + pid_bh * stride_kb,
        shape=(BLOCK_D, S),
        strides=(stride_kd, stride_ks),
        offsets=(0, 0),
        block_shape=(BLOCK_D, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + pid_bh * stride_vb,
        shape=(S, BLOCK_D),
        strides=(stride_vs, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )

    for n in range(0, tl.cdiv(S, BLOCK_N)):
        offs_n = n * BLOCK_N + tl.arange(0, BLOCK_N)
        kv_mask = offs_n < S

        k_t = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")  # [BLOCK_D, BLOCK_N]
        qk = tl.dot(q, k_t) * sm_scale
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        qk = tl.where(causal_mask & kv_mask[None, :], qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")    # [BLOCK_N, BLOCK_D]
        acc = acc + tl.dot(p.to(io_dtype), v, out_dtype=tl.float32)
        m_i = m_new

        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    acc = acc / l_i[:, None]

    # Store output via block_ptr (Snippet F)
    O_block_ptr = tl.make_block_ptr(
        base=Out + pid_bh * stride_ob,
        shape=(S, BLOCK_D),
        strides=(stride_os, stride_od),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )
    tl.store(O_block_ptr, acc.to(io_dtype), boundary_check=(0,))


# ---------------------------------------------------------------------------
# Decode kernel — q_len == 1, full kv_len, no causal mask
# (the last query sees all past keys by construction; mask is trivially true)
# ---------------------------------------------------------------------------

@triton.jit
def _fwd_kernel_decode(
    Q, K, V, Out,
    stride_qb, stride_qd,
    stride_kb, stride_ks, stride_kd,
    stride_vb, stride_vs, stride_vd,
    stride_ob, stride_od,
    sm_scale,
    S,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)

    io_dtype = Out.dtype.element_ty

    # Load the single query token: [BLOCK_D] via 1-D block_ptr (Snippet E)
    Q_block_ptr = tl.make_block_ptr(
        base=Q + pid_bh * stride_qb,
        shape=(BLOCK_D,),
        strides=(stride_qd,),
        offsets=(0,),
        block_shape=(BLOCK_D,),
        order=(0,),
    )
    q = tl.load(Q_block_ptr).to(tl.float32)

    # Online softmax scalars (per program — no BLOCK_M dim)
    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    # K [S, BLOCK_D] — no transpose, use element-wise reduction
    K_block_ptr = tl.make_block_ptr(
        base=K + pid_bh * stride_kb,
        shape=(S, BLOCK_D),
        strides=(stride_ks, stride_kd),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + pid_bh * stride_vb,
        shape=(S, BLOCK_D),
        strides=(stride_vs, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )

    for n in range(0, tl.cdiv(S, BLOCK_N)):
        offs_n = n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S

        # K tile [BLOCK_N, BLOCK_D] — kept in input dtype, promoted at use-site
        k = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
        qk = tl.sum(q[None, :] * k.to(tl.float32), axis=1) * sm_scale
        qk = tl.where(mask_n, qk, float("-inf"))

        # Online softmax
        m_new = tl.maximum(m_i, tl.max(qk, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)                         # [BLOCK_N]
        l_i = alpha * l_i + tl.sum(p, axis=0)
        acc = acc * alpha

        # V tile [BLOCK_N, BLOCK_D]
        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
        acc = acc + tl.sum(p[:, None] * v.to(tl.float32), axis=0)  # [BLOCK_D]
        m_i = m_new

        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    acc = acc / l_i

    # 1-D output store — boundary_check not needed (D is exact)
    O_block_ptr = tl.make_block_ptr(
        base=Out + pid_bh * stride_ob,
        shape=(BLOCK_D,),
        strides=(stride_od,),
        offsets=(0,),
        block_shape=(BLOCK_D,),
        order=(0,),
    )
    tl.store(O_block_ptr, acc.to(io_dtype))


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


def _check_common(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    assert q.dtype in _SUPPORTED_DTYPES, f"dtype must be fp16/bf16, got {q.dtype}"
    assert k.dtype == q.dtype == v.dtype, "q/k/v dtypes must match"
    D = q.shape[-1]
    assert D > 0 and (D & (D - 1)) == 0, f"head_dim must be a power of two, got {D}"


def triton_attention_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """
    Full causal attention where q_len == kv_len.

    q: [B, Hq,  S, D]  fp16 or bf16
    k: [B, Hkv, S, D]  same dtype as q
    v: [B, Hkv, S, D]  same dtype as q
    returns o: [B, Hq, S, D]
    """
    B, Hq, S, D = q.shape
    Hkv = k.shape[1]
    _check_common(q, k, v)
    assert Hq % Hkv == 0, f"Hq ({Hq}) must be divisible by Hkv ({Hkv})"
    assert k.shape[2] == S and v.shape[2] == S, "prefill: q_len must equal kv_len"

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    n_rep = Hq // Hkv
    if n_rep > 1:
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    q_3d = q.reshape(B * Hq, S, D)
    k_3d = k.reshape(B * Hq, S, D)
    v_3d = v.reshape(B * Hq, S, D)
    o_3d = torch.empty_like(q_3d)

    BLOCK_D = D
    BLOCK = _cap_block_for_head_dim(_get_block_size(q.dtype), D)
    grid = (B * Hq, triton.cdiv(S, BLOCK))

    _fwd_kernel_prefill[grid](
        q_3d, k_3d, v_3d, o_3d,
        q_3d.stride(0), q_3d.stride(1), q_3d.stride(2),
        k_3d.stride(0), k_3d.stride(1), k_3d.stride(2),
        v_3d.stride(0), v_3d.stride(1), v_3d.stride(2),
        o_3d.stride(0), o_3d.stride(1), o_3d.stride(2),
        scale, S,
        BLOCK_M=BLOCK, BLOCK_N=BLOCK, BLOCK_D=BLOCK_D,
    )
    return o_3d.reshape(B, Hq, S, D)


def triton_attention_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """
    Single-token decode attention.

    q: [B, Hq,  1, D]  fp16 or bf16
    k: [B, Hkv, S, D]  past KV context (S >= 1, last position = new KV)
    v: [B, Hkv, S, D]
    returns o: [B, Hq, 1, D]
    """
    B, Hq, Q_len, D = q.shape
    Hkv = k.shape[1]
    S = k.shape[2]
    _check_common(q, k, v)
    assert Q_len == 1, f"decode expects q_len=1, got {Q_len}"
    assert Hq % Hkv == 0, f"Hq ({Hq}) must be divisible by Hkv ({Hkv})"

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    n_rep = Hq // Hkv
    if n_rep > 1:
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # Collapse (batch, head, q_len=1) → single dim for kernel
    q_2d = q.reshape(B * Hq, D)
    k_3d = k.reshape(B * Hq, S, D)
    v_3d = v.reshape(B * Hq, S, D)
    o_2d = torch.empty_like(q_2d)

    BLOCK_D = D
    BLOCK_N = _cap_block_for_head_dim(_get_block_size(q.dtype), D)
    grid = (B * Hq,)

    _fwd_kernel_decode[grid](
        q_2d, k_3d, v_3d, o_2d,
        q_2d.stride(0), q_2d.stride(1),
        k_3d.stride(0), k_3d.stride(1), k_3d.stride(2),
        v_3d.stride(0), v_3d.stride(1), v_3d.stride(2),
        o_2d.stride(0), o_2d.stride(1),
        scale, S,
        BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )
    return o_2d.reshape(B, Hq, 1, D)


# Backwards-compat alias — existing code (backend prefill branch, notebook
# smoke test) can keep calling triton_attention() unchanged.
triton_attention = triton_attention_prefill


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("GPU not available — skipping smoke test")
        raise SystemExit(0)

    import torch.nn.functional as F

    def _check_prefill(dtype, D, atol, label):
        B, Hq, Hkv, S = 1, 16, 8, 128
        q = torch.randn(B, Hq, S, D, dtype=dtype, device="cuda")
        k = torch.randn(B, Hkv, S, D, dtype=dtype, device="cuda")
        v = torch.randn(B, Hkv, S, D, dtype=dtype, device="cuda")
        k_rep = k.repeat_interleave(Hq // Hkv, dim=1)
        v_rep = v.repeat_interleave(Hq // Hkv, dim=1)
        ref = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)
        ours = triton_attention_prefill(q, k, v)
        err = (ours.float() - ref.float()).abs().max().item()
        ok = err < atol
        print(f"  prefill  {label:18s} max_abs_err = {err:.6f}  →  {'PASS' if ok else 'FAIL'}")
        return ok

    def _check_decode(dtype, kv_len, atol, label):
        B, Hq, Hkv, D = 1, 16, 8, 128
        # Full Q history for the SDPA reference
        q_full = torch.randn(B, Hq, kv_len, D, dtype=dtype, device="cuda")
        k = torch.randn(B, Hkv, kv_len, D, dtype=dtype, device="cuda")
        v = torch.randn(B, Hkv, kv_len, D, dtype=dtype, device="cuda")
        k_rep = k.repeat_interleave(Hq // Hkv, dim=1)
        v_rep = v.repeat_interleave(Hq // Hkv, dim=1)
        ref_full = F.scaled_dot_product_attention(q_full, k_rep, v_rep, is_causal=True)
        ref_last = ref_full[:, :, -1:, :]                     # [B, Hq, 1, D]
        # Our decode path receives only the last query token
        q_last = q_full[:, :, -1:, :].contiguous()            # [B, Hq, 1, D]
        ours = triton_attention_decode(q_last, k, v)
        err = (ours.float() - ref_last.float()).abs().max().item()
        ok = err < atol
        print(f"  decode   {label:18s} max_abs_err = {err:.6f}  →  {'PASS' if ok else 'FAIL'}")
        return ok

    results = []
    print("prefill (q_len == kv_len, causal)")
    for dtype, atol in [(torch.float16, 1e-2), (torch.bfloat16, 3e-2)]:
        for D in (64, 128, 256):
            results.append(
                _check_prefill(dtype, D, atol, f"{str(dtype).split('.')[-1]}/D={D}")
            )

    print("decode (q_len=1, no causal mask needed)")
    for dtype, atol in [(torch.float16, 1e-2), (torch.bfloat16, 3e-2)]:
        for kv_len in (32, 128, 1024):
            results.append(
                _check_decode(
                    dtype, kv_len, atol,
                    f"{str(dtype).split('.')[-1]}/S={kv_len}",
                )
            )

    print("ALL PASS" if all(results) else "SOMETHING FAILED")
