"""Fused causal attention kernels — separate prefill and **split-KV decode**.

Naming follows vLLM conventions, with FlashDecoding-style split-KV applied to
the decode path:

    _fwd_kernel_prefill          : q_len == kv_len, causal mask  (unchanged from vllm_split)
    _fwd_kernel_decode_simple    : q_len == 1, single-block-per-(b,h) decode  (KV_SPLITS == 1 fast path)
    _fwd_kernel_decode_partial   : q_len == 1, KV chunk per program            (phase 1 of split-KV)
    _fwd_kernel_decode_reduce    : log-sum-exp combine of KV partials          (phase 2 of split-KV)

The diff vs `vllm_split` is concentrated in the decode path. Prefill is byte-for-byte
identical because prefill already has plenty of grid parallelism (B × Hq × Q_blocks).

Split-KV (= FlashDecoding, Tri Dao et al. 2023.10) adds an extra grid axis to recover
parallelism in the small-batch / long-context decode regime, where the natural grid
B×Hq is smaller than num_SMs.
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
# Prefill kernel — q_len == kv_len == S, causal  (unchanged from vllm_split)
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
# Decode kernel (KV_SPLITS == 1 fast path) — same as vllm_split's decode
# Used when grid (B*Hq) already saturates the GPU and split-KV would be overhead.
# ---------------------------------------------------------------------------

@triton.jit
def _fwd_kernel_decode_simple(
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
    offs_d = tl.arange(0, BLOCK_D)

    io_dtype = Out.dtype.element_ty

    q = tl.load(Q + pid_bh * stride_qb + offs_d * stride_qd).to(tl.float32)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for n in range(0, tl.cdiv(S, BLOCK_N)):
        offs_n = n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S

        k = tl.load(
            K + pid_bh * stride_kb
            + offs_n[:, None] * stride_ks
            + offs_d[None, :] * stride_kd,
            mask=mask_n[:, None],
            other=0.0,
        )
        qk = tl.sum(q[None, :] * k.to(tl.float32), axis=1) * sm_scale
        qk = tl.where(mask_n, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)
        l_i = alpha * l_i + tl.sum(p, axis=0)
        acc = acc * alpha

        v = tl.load(
            V + pid_bh * stride_vb
            + offs_n[:, None] * stride_vs
            + offs_d[None, :] * stride_vd,
            mask=mask_n[:, None],
            other=0.0,
        )
        acc = acc + tl.sum(p[:, None] * v.to(tl.float32), axis=0)
        m_i = m_new

    acc = acc / l_i
    tl.store(
        Out + pid_bh * stride_ob + offs_d * stride_od,
        acc.to(io_dtype),
    )


# ---------------------------------------------------------------------------
# Split-KV decode — phase 1: partial m / l / acc per KV chunk
# Grid: (B*Hq, KV_SPLITS).  Each program owns a contiguous chunk of KV.
# ---------------------------------------------------------------------------

@triton.jit
def _fwd_kernel_decode_partial(
    Q, K, V,
    PartialAcc,                    # [B*Hq, KV_SPLITS, D]   fp32
    PartialM,                      # [B*Hq, KV_SPLITS]      fp32
    PartialL,                      # [B*Hq, KV_SPLITS]      fp32
    stride_qb, stride_qd,
    stride_kb, stride_ks, stride_kd,
    stride_vb, stride_vs, stride_vd,
    stride_pa_b, stride_pa_k, stride_pa_d,
    stride_pm_b, stride_pm_k,
    stride_pl_b, stride_pl_k,
    sm_scale,
    S,
    KV_SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_kv = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_D)

    # Determine this split's KV range. Use ceildiv chunks so the last split
    # may be shorter (or empty when KV_SPLITS > S, which is degenerate).
    chunk = tl.cdiv(S, KV_SPLITS)
    kv_start = pid_kv * chunk
    kv_end = tl.minimum(kv_start + chunk, S)

    q = tl.load(Q + pid_bh * stride_qb + offs_d * stride_qd).to(tl.float32)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    n_start = kv_start // BLOCK_N
    n_end = tl.cdiv(kv_end, BLOCK_N)

    # If kv_start >= S (degenerate), n_start >= n_end and the loop is empty.
    # Defaults (m=-inf, l=0, acc=0) are what the reduce kernel needs to ignore
    # this split (exp(-inf - finite) = 0).
    for n in range(n_start, n_end):
        offs_n = n * BLOCK_N + tl.arange(0, BLOCK_N)
        # Must be in this split's window AND in the valid sequence
        mask_n = (offs_n >= kv_start) & (offs_n < kv_end)

        k = tl.load(
            K + pid_bh * stride_kb
            + offs_n[:, None] * stride_ks
            + offs_d[None, :] * stride_kd,
            mask=mask_n[:, None],
            other=0.0,
        )
        qk = tl.sum(q[None, :] * k.to(tl.float32), axis=1) * sm_scale
        qk = tl.where(mask_n, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)
        l_i = alpha * l_i + tl.sum(p, axis=0)
        acc = acc * alpha

        v = tl.load(
            V + pid_bh * stride_vb
            + offs_n[:, None] * stride_vs
            + offs_d[None, :] * stride_vd,
            mask=mask_n[:, None],
            other=0.0,
        )
        acc = acc + tl.sum(p[:, None] * v.to(tl.float32), axis=0)
        m_i = m_new

    # Store partial results — DO NOT divide by l_i yet (reduce kernel does it).
    tl.store(
        PartialAcc + pid_bh * stride_pa_b + pid_kv * stride_pa_k
        + offs_d * stride_pa_d,
        acc,
    )
    tl.store(PartialM + pid_bh * stride_pm_b + pid_kv * stride_pm_k, m_i)
    tl.store(PartialL + pid_bh * stride_pl_b + pid_kv * stride_pl_k, l_i)


# ---------------------------------------------------------------------------
# Split-KV decode — phase 2: log-sum-exp combine of partials
# Grid: (B*Hq,).  Each program reads KV_SPLITS partials and emits one final O row.
#
# Math (numerically stable):
#   m_global       = max_k(partial_m[k])
#   scale[k]       = exp(partial_m[k] - m_global)
#   l_global       = sum_k(partial_l[k] * scale[k])
#   acc_global     = sum_k(partial_acc[k] * scale[k]) / l_global
# ---------------------------------------------------------------------------

@triton.jit
def _fwd_kernel_decode_reduce(
    PartialAcc,                    # [B*Hq, KV_SPLITS, D]
    PartialM,                      # [B*Hq, KV_SPLITS]
    PartialL,                      # [B*Hq, KV_SPLITS]
    Out,                           # [B*Hq, D]
    stride_pa_b, stride_pa_k, stride_pa_d,
    stride_pm_b, stride_pm_k,
    stride_pl_b, stride_pl_k,
    stride_ob, stride_od,
    KV_SPLITS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)

    offs_d = tl.arange(0, BLOCK_D)
    offs_k = tl.arange(0, KV_SPLITS)

    pm = tl.load(PartialM + pid_bh * stride_pm_b + offs_k * stride_pm_k)   # [KV_SPLITS]
    pl = tl.load(PartialL + pid_bh * stride_pl_b + offs_k * stride_pl_k)   # [KV_SPLITS]

    m_global = tl.max(pm, axis=0)
    # Splits with m == -inf (empty range) → scale = 0, contribute nothing
    scale = tl.exp(pm - m_global)                                          # [KV_SPLITS]

    l_global = tl.sum(pl * scale, axis=0)

    pa = tl.load(
        PartialAcc + pid_bh * stride_pa_b
        + offs_k[:, None] * stride_pa_k
        + offs_d[None, :] * stride_pa_d,
    )                                                                      # [KV_SPLITS, BLOCK_D]

    acc = tl.sum(pa * scale[:, None], axis=0) / l_global                   # [BLOCK_D]

    io_dtype = Out.dtype.element_ty
    tl.store(
        Out + pid_bh * stride_ob + offs_d * stride_od,
        acc.to(io_dtype),
    )


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
_MAX_KV_SPLITS = 16     # cap to keep partial buffers small + reduce kernel SMEM modest
_SPLIT_KV_THRESHOLD = 512   # below this S, split-KV overhead > benefit


def _check_common(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    assert q.dtype in _SUPPORTED_DTYPES, f"dtype must be fp16/bf16, got {q.dtype}"
    assert k.dtype == q.dtype == v.dtype, "q/k/v dtypes must match"
    D = q.shape[-1]
    assert D > 0 and (D & (D - 1)) == 0, f"head_dim must be a power of two, got {D}"


def _pow2_floor(n: int) -> int:
    """Largest power of 2 ≤ n (≥ 1). Required because the reduce kernel uses
    `tl.arange(0, KV_SPLITS)` which Triton compiles only for power-of-2 lengths.
    """
    if n < 1:
        return 1
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def _choose_kv_splits(grid_bh: int, S: int) -> int:
    """Heuristic: if grid (B*Hq) already saturates the GPU, no split.
    Otherwise split enough to reach ~2× SM count, rounded down to a power of 2
    (constrained by `tl.arange` requirement in the reduce kernel).
    """
    if S < _SPLIT_KV_THRESHOLD:
        return 1
    if not torch.cuda.is_available():
        return 1
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    target = num_sms * 2
    if grid_bh >= target:
        return 1
    needed = (target + grid_bh - 1) // grid_bh
    return _pow2_floor(min(_MAX_KV_SPLITS, needed))


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
    kv_splits: int | None = None,
) -> torch.Tensor:
    """
    Single-token decode attention with FlashDecoding-style split-KV.

    q: [B, Hq,  1, D]  fp16 or bf16
    k: [B, Hkv, S, D]  past KV context (S >= 1, last position = new KV)
    v: [B, Hkv, S, D]
    kv_splits: optional override for the split-KV depth; None → heuristic.
    returns o: [B, Hq, 1, D]
    """
    B, Hq, Q_len, D = q.shape
    Hkv = k.shape[1]
    S = k.shape[2]
    _check_common(q, k, v)
    assert Q_len == 1, f"decode expects q_len=1, got {Q_len}"
    assert Hq % Hkv == 0, f"Hq ({Hq}) must be divisible by Hkv ({Hkv})"
    assert S >= 1, f"decode expects kv_len >= 1, got {S}"

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    n_rep = Hq // Hkv
    if n_rep > 1:
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    q_2d = q.reshape(B * Hq, D)
    k_3d = k.reshape(B * Hq, S, D)
    v_3d = v.reshape(B * Hq, S, D)
    o_2d = torch.empty_like(q_2d)

    BLOCK_D = D
    BLOCK_N = _cap_block_for_head_dim(_get_block_size(q.dtype), D)

    grid_bh = B * Hq
    if kv_splits is None:
        kv_splits = _choose_kv_splits(grid_bh, S)
    # Clamp to [1, min(MAX, S)] AND power of 2 (reduce kernel uses tl.arange).
    kv_splits = _pow2_floor(min(max(1, kv_splits), _MAX_KV_SPLITS, S))

    if kv_splits == 1:
        # Fast path: identical to vllm_split's decode kernel.
        grid = (grid_bh,)
        _fwd_kernel_decode_simple[grid](
            q_2d, k_3d, v_3d, o_2d,
            q_2d.stride(0), q_2d.stride(1),
            k_3d.stride(0), k_3d.stride(1), k_3d.stride(2),
            v_3d.stride(0), v_3d.stride(1), v_3d.stride(2),
            o_2d.stride(0), o_2d.stride(1),
            scale, S,
            BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        )
        return o_2d.reshape(B, Hq, 1, D)

    # Split-KV path: phase-1 partial + phase-2 reduce
    partial_acc = torch.empty(
        (grid_bh, kv_splits, D), dtype=torch.float32, device=q.device,
    )
    partial_m = torch.empty(
        (grid_bh, kv_splits), dtype=torch.float32, device=q.device,
    )
    partial_l = torch.empty(
        (grid_bh, kv_splits), dtype=torch.float32, device=q.device,
    )

    grid_partial = (grid_bh, kv_splits)
    _fwd_kernel_decode_partial[grid_partial](
        q_2d, k_3d, v_3d,
        partial_acc, partial_m, partial_l,
        q_2d.stride(0), q_2d.stride(1),
        k_3d.stride(0), k_3d.stride(1), k_3d.stride(2),
        v_3d.stride(0), v_3d.stride(1), v_3d.stride(2),
        partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2),
        partial_m.stride(0), partial_m.stride(1),
        partial_l.stride(0), partial_l.stride(1),
        scale, S,
        KV_SPLITS=kv_splits, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )

    grid_reduce = (grid_bh,)
    _fwd_kernel_decode_reduce[grid_reduce](
        partial_acc, partial_m, partial_l, o_2d,
        partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2),
        partial_m.stride(0), partial_m.stride(1),
        partial_l.stride(0), partial_l.stride(1),
        o_2d.stride(0), o_2d.stride(1),
        KV_SPLITS=kv_splits, BLOCK_D=BLOCK_D,
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
        print(f"  prefill  {label:22s} max_abs_err = {err:.6f}  →  {'PASS' if ok else 'FAIL'}")
        return ok

    def _check_decode(dtype, kv_len, kv_splits, atol, label):
        B, Hq, Hkv, D = 1, 16, 8, 128
        q_full = torch.randn(B, Hq, kv_len, D, dtype=dtype, device="cuda")
        k = torch.randn(B, Hkv, kv_len, D, dtype=dtype, device="cuda")
        v = torch.randn(B, Hkv, kv_len, D, dtype=dtype, device="cuda")
        k_rep = k.repeat_interleave(Hq // Hkv, dim=1)
        v_rep = v.repeat_interleave(Hq // Hkv, dim=1)
        ref_full = F.scaled_dot_product_attention(q_full, k_rep, v_rep, is_causal=True)
        ref_last = ref_full[:, :, -1:, :]
        q_last = q_full[:, :, -1:, :].contiguous()
        ours = triton_attention_decode(q_last, k, v, kv_splits=kv_splits)
        err = (ours.float() - ref_last.float()).abs().max().item()
        ok = err < atol
        split_label = "auto" if kv_splits is None else f"ns={kv_splits}"
        print(
            f"  decode   {label:14s} {split_label:>8s}  "
            f"max_abs_err = {err:.6f}  →  {'PASS' if ok else 'FAIL'}"
        )
        return ok

    results = []
    print("prefill (q_len == kv_len, causal)")
    for dtype, atol in [(torch.float16, 1e-2), (torch.bfloat16, 3e-2)]:
        for D in (64, 128, 256):
            results.append(
                _check_prefill(dtype, D, atol, f"{str(dtype).split('.')[-1]}/D={D}")
            )

    print("decode (q_len=1, split-KV)")
    # Sweep kv_len AND kv_splits to exercise both phase-1+2 and the simple fast path
    for dtype, atol in [(torch.float16, 1e-2), (torch.bfloat16, 3e-2)]:
        for kv_len in (32, 128, 1024, 4096):
            for ns in (None, 1, 2, 4, 8):
                if ns is not None and ns > kv_len:
                    continue
                results.append(
                    _check_decode(
                        dtype, kv_len, ns, atol,
                        f"{str(dtype).split('.')[-1]}/S={kv_len}",
                    )
                )

    print("ALL PASS" if all(results) else "SOMETHING FAILED")
