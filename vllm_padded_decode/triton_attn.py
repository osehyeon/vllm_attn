"""Fused causal attention — **single prefill kernel only**.

Decode is not a separate kernel here: the backend handles decode by
zero-padding q up to kv_len and calling this prefill kernel, then
extracting only the last query position. See NOTES.md for why.

Naming follows vLLM conventions
(cf. vllm/v1/attention/ops/triton_prefill_attention.py).
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
    offs_d = tl.arange(0, BLOCK_D)

    q_ptr = Q + pid_bh * stride_qb
    k_ptr = K + pid_bh * stride_kb
    v_ptr = V + pid_bh * stride_vb
    o_ptr = Out + pid_bh * stride_ob

    io_dtype = Out.dtype.element_ty

    q_mask = offs_m[:, None] < S
    q = tl.load(
        q_ptr + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
        mask=q_mask,
        other=0.0,
    )

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    for n in range(0, tl.cdiv(S, BLOCK_N)):
        offs_n = n * BLOCK_N + tl.arange(0, BLOCK_N)
        kv_mask = offs_n[None, :] < S

        k = tl.load(
            k_ptr + offs_n[None, :] * stride_ks + offs_d[:, None] * stride_kd,
            mask=kv_mask,
            other=0.0,
        )
        qk = tl.dot(q, k) * sm_scale
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        qk = tl.where(causal_mask & kv_mask, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v = tl.load(
            v_ptr + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd,
            mask=offs_n[:, None] < S,
            other=0.0,
        )
        acc = acc + tl.dot(p.to(io_dtype), v, out_dtype=tl.float32)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(
        o_ptr + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od,
        acc.to(io_dtype),
        mask=q_mask,
    )


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


def triton_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """
    Causal attention where q_len == kv_len.

    q: [B, Hq,  S, D]  fp16 or bf16
    k: [B, Hkv, S, D]  same dtype as q
    v: [B, Hkv, S, D]  same dtype as q
    returns o: [B, Hq, S, D]
    """
    B, Hq, S, D = q.shape
    Hkv = k.shape[1]
    assert q.dtype in _SUPPORTED_DTYPES, f"dtype must be fp16/bf16, got {q.dtype}"
    assert k.dtype == q.dtype == v.dtype, "q/k/v dtypes must match"
    assert D > 0 and (D & (D - 1)) == 0, f"head_dim must be a power of two, got {D}"
    assert Hq % Hkv == 0, f"Hq ({Hq}) must be divisible by Hkv ({Hkv})"
    assert k.shape[2] == S and v.shape[2] == S, "q_len must equal kv_len"

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


# ---------------------------------------------------------------------------
# Smoke test — prefill correctness across dtype × head_dim
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("GPU not available — skipping smoke test")
        raise SystemExit(0)

    import torch.nn.functional as F

    def _check(dtype, D, atol, label):
        B, Hq, Hkv, S = 1, 16, 8, 128
        q = torch.randn(B, Hq, S, D, dtype=dtype, device="cuda")
        k = torch.randn(B, Hkv, S, D, dtype=dtype, device="cuda")
        v = torch.randn(B, Hkv, S, D, dtype=dtype, device="cuda")
        k_rep = k.repeat_interleave(Hq // Hkv, dim=1)
        v_rep = v.repeat_interleave(Hq // Hkv, dim=1)
        ref = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)
        ours = triton_attention(q, k, v)
        err = (ours.float() - ref.float()).abs().max().item()
        ok = err < atol
        print(f"  {label:18s} max_abs_err = {err:.6f}  →  {'PASS' if ok else 'FAIL'}")
        return ok

    results = []
    print("prefill (q_len == kv_len, causal)")
    for dtype, atol in [(torch.float16, 1e-2), (torch.bfloat16, 3e-2)]:
        for D in (64, 128, 256):
            results.append(
                _check(dtype, D, atol, f"{str(dtype).split('.')[-1]}/D={D}")
            )

    print("ALL PASS" if all(results) else "SOMETHING FAILED")
