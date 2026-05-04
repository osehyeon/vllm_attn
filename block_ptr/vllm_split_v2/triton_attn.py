"""Fused causal attention kernels — separate prefill and **split-KV decode**.
block_ptr variant of vllm_split_v2.

Naming follows vLLM conventions, with FlashDecoding-style split-KV applied to
the decode path. All Q/K/V/O loads and stores use `tl.make_block_ptr`:

    _fwd_kernel_prefill          : q_len == kv_len, causal mask  (= block_ptr/vllm_split prefill, unchanged)
    _fwd_kernel_decode_simple    : q_len == 1, KV_SPLITS == 1 fast path  (= block_ptr/vllm_split decode)
    _fwd_kernel_decode_partial   : q_len == 1, KV chunk per program  (split-KV phase 1)
    _fwd_kernel_decode_reduce    : log-sum-exp combine of KV partials  (split-KV phase 2)

Block-pointer specifics for split-KV:
    - partial kernel starts K_block_ptr / V_block_ptr at `n_start * BLOCK_N`
      (BLOCK_N-aligned floor of `kv_start`) so the loop advances naturally
    - chunk window mask `(offs_n >= kv_start) & (offs_n < kv_end)` is applied
      via `tl.where` since block_ptr's `boundary_check` only checks against
      the global S, not against the chunk boundary
    - reduce kernel uses 2-D block_ptr `[KV_SPLITS, BLOCK_D]` for partial_acc
      and 1-D block_ptr `[KV_SPLITS]` for partial_m, partial_l
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
# Prefill kernel — q_len == kv_len == S, causal  (unchanged from block_ptr/vllm_split)
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

    # K transposed view (BLOCK_D, BLOCK_N) via order=(0,1) — virtual transpose
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

        k_t = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
        qk = tl.dot(q, k_t) * sm_scale
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        qk = tl.where(causal_mask & kv_mask[None, :], qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
        acc = acc + tl.dot(p.to(io_dtype), v, out_dtype=tl.float32)
        m_i = m_new

        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    acc = acc / l_i[:, None]

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
# Decode kernel (KV_SPLITS == 1 fast path) — same as block_ptr/vllm_split decode
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

    io_dtype = Out.dtype.element_ty

    Q_block_ptr = tl.make_block_ptr(
        base=Q + pid_bh * stride_qb,
        shape=(BLOCK_D,),
        strides=(stride_qd,),
        offsets=(0,),
        block_shape=(BLOCK_D,),
        order=(0,),
    )
    q = tl.load(Q_block_ptr).to(tl.float32)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

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

        k = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
        qk = tl.sum(q[None, :] * k.to(tl.float32), axis=1) * sm_scale
        qk = tl.where(mask_n, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)
        l_i = alpha * l_i + tl.sum(p, axis=0)
        acc = acc * alpha

        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
        acc = acc + tl.sum(p[:, None] * v.to(tl.float32), axis=0)
        m_i = m_new

        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    acc = acc / l_i

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
# Split-KV decode — phase 1 (block_ptr variant)
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

    chunk = tl.cdiv(S, KV_SPLITS)
    kv_start = pid_kv * chunk
    kv_end = tl.minimum(kv_start + chunk, S)

    Q_block_ptr = tl.make_block_ptr(
        base=Q + pid_bh * stride_qb,
        shape=(BLOCK_D,),
        strides=(stride_qd,),
        offsets=(0,),
        block_shape=(BLOCK_D,),
        order=(0,),
    )
    q = tl.load(Q_block_ptr).to(tl.float32)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    # BLOCK_N-aligned iteration window covering [kv_start, kv_end).
    # The chunk boundary may not align with BLOCK_N; we mask with tl.where below.
    n_start = kv_start // BLOCK_N
    n_end = tl.cdiv(kv_end, BLOCK_N)

    K_block_ptr = tl.make_block_ptr(
        base=K + pid_bh * stride_kb,
        shape=(S, BLOCK_D),
        strides=(stride_ks, stride_kd),
        offsets=(n_start * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + pid_bh * stride_vb,
        shape=(S, BLOCK_D),
        strides=(stride_vs, stride_vd),
        offsets=(n_start * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )

    for n in range(n_start, n_end):
        offs_n = n * BLOCK_N + tl.arange(0, BLOCK_N)
        # Combined chunk + sequence mask (boundary_check only handles offs_n < S).
        mask_n = (offs_n >= kv_start) & (offs_n < kv_end)

        k = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
        qk = tl.sum(q[None, :] * k.to(tl.float32), axis=1) * sm_scale
        qk = tl.where(mask_n, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)
        l_i = alpha * l_i + tl.sum(p, axis=0)
        acc = acc * alpha

        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
        acc = acc + tl.sum(p[:, None] * v.to(tl.float32), axis=0)
        m_i = m_new

        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    # Store partial_acc via 1-D block_ptr (D dim of the [pid_bh, pid_kv, :] slice)
    PartialAcc_block_ptr = tl.make_block_ptr(
        base=PartialAcc + pid_bh * stride_pa_b + pid_kv * stride_pa_k,
        shape=(BLOCK_D,),
        strides=(stride_pa_d,),
        offsets=(0,),
        block_shape=(BLOCK_D,),
        order=(0,),
    )
    tl.store(PartialAcc_block_ptr, acc)

    # Scalars m_i and l_i — block_ptr is overkill, use raw pointer arithmetic.
    tl.store(PartialM + pid_bh * stride_pm_b + pid_kv * stride_pm_k, m_i)
    tl.store(PartialL + pid_bh * stride_pl_b + pid_kv * stride_pl_k, l_i)


# ---------------------------------------------------------------------------
# Split-KV decode — phase 2 (block_ptr variant)
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
    KV_SPLITS: tl.constexpr,       # power of 2 (tl.arange compatibility)
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)

    # Load partial_m, partial_l via 1-D block_ptr (KV_SPLITS dim)
    PartialM_block_ptr = tl.make_block_ptr(
        base=PartialM + pid_bh * stride_pm_b,
        shape=(KV_SPLITS,),
        strides=(stride_pm_k,),
        offsets=(0,),
        block_shape=(KV_SPLITS,),
        order=(0,),
    )
    PartialL_block_ptr = tl.make_block_ptr(
        base=PartialL + pid_bh * stride_pl_b,
        shape=(KV_SPLITS,),
        strides=(stride_pl_k,),
        offsets=(0,),
        block_shape=(KV_SPLITS,),
        order=(0,),
    )
    pm = tl.load(PartialM_block_ptr)
    pl = tl.load(PartialL_block_ptr)

    m_global = tl.max(pm, axis=0)
    scale = tl.exp(pm - m_global)                     # [KV_SPLITS]
    l_global = tl.sum(pl * scale, axis=0)

    # Load partial_acc via 2-D block_ptr [KV_SPLITS, BLOCK_D]
    PartialAcc_block_ptr = tl.make_block_ptr(
        base=PartialAcc + pid_bh * stride_pa_b,
        shape=(KV_SPLITS, BLOCK_D),
        strides=(stride_pa_k, stride_pa_d),
        offsets=(0, 0),
        block_shape=(KV_SPLITS, BLOCK_D),
        order=(1, 0),
    )
    pa = tl.load(PartialAcc_block_ptr)

    acc = tl.sum(pa * scale[:, None], axis=0) / l_global

    O_block_ptr = tl.make_block_ptr(
        base=Out + pid_bh * stride_ob,
        shape=(BLOCK_D,),
        strides=(stride_od,),
        offsets=(0,),
        block_shape=(BLOCK_D,),
        order=(0,),
    )
    tl.store(O_block_ptr, acc.to(Out.dtype.element_ty))


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
_MAX_KV_SPLITS = 16
_SPLIT_KV_THRESHOLD = 512


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
    Single-token decode attention with FlashDecoding-style split-KV (block_ptr).

    q: [B, Hq,  1, D]  fp16 or bf16
    k: [B, Hkv, S, D]  past KV context (S >= 1, last position = new KV)
    v: [B, Hkv, S, D]
    kv_splits: optional override for the split-KV depth; None → heuristic.
               Effective value is rounded down to a power of 2 (reduce kernel
               requires `tl.arange(0, KV_SPLITS)` to be power-of-2).
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
    kv_splits = _pow2_floor(min(max(1, kv_splits), _MAX_KV_SPLITS, S))

    if kv_splits == 1:
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

    print("decode (q_len=1, split-KV, block_ptr)")
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
