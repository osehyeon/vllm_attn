"""vLLM v1 AttentionBackend — FlexAttention-backed attention (educational).

Sibling of ../flashinfer/ and ../flashattn/. Same Backend/Impl/Metadata/Builder
4-tier shape, same `AttentionBackendEnum.CUSTOM` slot, same KV-write path.
The kernel body swaps to `torch.nn.attention.flex_attention.flex_attention` +
`BlockMask` — a PyTorch builtin since 2.5, no external library dependency.

Three things fall out of that swap and they are the whole point of this project:

1. **BlockMask is the bridge**. flex_attention does not accept vLLM's paged
   `block_table` directly — the mask_mod function receives logical token
   indices, so we must build a `physical_to_logical` inverse table that maps
   (request_id, physical_block_id) → logical_block_index.  The BlockMask is
   built once per forward in `MetadataBuilder.build()` (not in `Impl.forward`)
   so all 28 Qwen3 layers share one precomputed mask.

2. **Single dispatch — no split**. flex_attention with a BlockMask handles
   prefill, decode, and chunked prefill in a single launch.  No split between
   prefill and decode wrappers (unlike flashinfer which must split).

3. **No workspace, no plan**. flex_attention has no persistent GPU scratch
   buffer.  The `profile_run` OOM lesson from flashinfer does not apply here;
   the only precomputation is the BlockMask construction in `build()`.

Register via:
    from flex_attn_attention_backend import register
    register()
Then:
    LLM(..., attention_backend=AttentionBackendEnum.CUSTOM)
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)

# torch.compile is REQUIRED, not optional.
# Eager flex_attention dispatches to sdpa_dense (math fallback) which does a
# full N×K matmul over the entire flat KV cache — OOM on any real LLM.
# torch.compile reaches the Triton sparse kernel that actually respects
# BlockMask sparsity. This matches stock vLLM's flex_attention.py exactly.
# FORCE_USE_FLEX_ATTENTION=True bypasses the flex_decoding path and uses the
# main flex_attention (prefill) kernel — required on some GPU architectures
# (e.g. SM 12.0 / Blackwell) where the flex_decoding autotuner finds no
# valid Triton choices for short decode sequences.
def _run_flex_attention(q, k, v, score_mod, block_mask, scale, enable_gqa):
    return flex_attention(
        q, k, v,
        score_mod=score_mod,
        block_mask=block_mask,
        scale=scale,
        enable_gqa=enable_gqa,
        # FORCE_USE_FLEX_ATTENTION=True routes to the main prefill kernel and
        # away from flex_decoding which fails autotuning on SM 12.0.
        # BLOCK_M/N must divide the BlockMask BLOCK_SIZE (16) — so 16.
        kernel_options={"FORCE_USE_FLEX_ATTENTION": True, "BLOCK_M": 16, "BLOCK_N": 16},
    )

_flex_attention_compiled = torch.compile(_run_flex_attention, fullgraph=True)

from vllm.logger import init_logger

_logger = init_logger(__name__)

from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import AttentionSpec

FIRE_COUNTER: dict[str, int] = {
    "fires": 0,
    "max_num_seqs": 0,
    "max_q_len": 0,
    "max_chunked_seqs": 0,
}
_LOG_FIRE = os.environ.get("MY_FLEXATTN_BACKEND_LOG", "1") != "0"


# ---------------------------------------------------------------------------
# 1. Metadata
# ---------------------------------------------------------------------------

@dataclass
class MyFlexAttnMetadata(AttentionMetadata):
    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor       # (B+1,) int32
    seq_lens: torch.Tensor              # (B,)   int32
    block_table: torch.Tensor           # (B, max_blocks) int32
    slot_mapping: torch.Tensor          # (num_actual_tokens,) int64
    # FlexAttention-specific
    doc_ids: torch.Tensor               # (num_actual_tokens,) int32
    decode_offset: torch.Tensor         # (B,) int32 — seq_lens - q_lens
    physical_to_logical: torch.Tensor   # (B, num_gpu_blocks) int64
    block_mask: Any                     # BlockMask, built by builder.build()
    total_cache_tokens: int             # num_gpu_blocks * block_size
    block_size: int
    num_gpu_blocks: int


# ---------------------------------------------------------------------------
# Helper: physical_to_logical inverse table
# ---------------------------------------------------------------------------

def _build_physical_to_logical(
    block_table: torch.Tensor,   # (B, max_blocks) int32
    seq_lens: torch.Tensor,      # (B,) int32
    block_size: int,
    num_gpu_blocks: int,
) -> torch.Tensor:
    """Build the inverse mapping: (B, num_gpu_blocks) int64.

    physical_to_logical[req, pb] = logical block index for request req whose
    physical block id is pb, or -1 if that physical block is not in req.

    Sentinel = -1 so that mask_mod can cheaply check `>= 0` for validity.
    The scatter_ uses 'amax' reduce — if the same physical block appears at
    multiple logical indices (shouldn't happen for standard paged attn), the
    highest logical index wins.  We also mask out garbage values beyond the
    valid block count so random block_table padding doesn't corrupt the table.
    """
    device = block_table.device
    B, max_blocks = block_table.shape

    # Sentinel -1: physical blocks not belonging to this request get -1.
    p2l = torch.full((B, num_gpu_blocks), -1, dtype=torch.long, device=device)

    # Number of valid blocks per sequence (ceil-div).
    num_blocks_per_seq = (seq_lens.to(torch.long) + block_size - 1) // block_size  # (B,)

    # Mask out garbage values in block_table beyond valid block count.
    col_idx = torch.arange(max_blocks, device=device).unsqueeze(0)  # (1, max_blocks)
    valid_mask = col_idx < num_blocks_per_seq.unsqueeze(1)           # (B, max_blocks)

    # logical_indices[b, j] = j (the logical block index for column j).
    logical_indices = col_idx.expand(B, max_blocks)  # (B, max_blocks)

    # Zero out invalid entries so scatter_ target is always in [0, num_gpu_blocks).
    safe_phys = torch.where(valid_mask, block_table.to(torch.long), 0)
    safe_logi = torch.where(valid_mask, logical_indices, 0)

    p2l.scatter_reduce_(-1, safe_phys, safe_logi, reduce="amax")

    # Block 0 is always the vLLM null block — reset it to -1 for safety.
    p2l[:, 0] = -1
    return p2l


# ---------------------------------------------------------------------------
# Helper: mask_mod closure (paged causal attention)
# ---------------------------------------------------------------------------

def _make_paged_causal_mask_mod(
    doc_ids: torch.Tensor,           # (num_actual_tokens,) int32
    decode_offset: torch.Tensor,     # (B,) int32
    query_start_loc: torch.Tensor,   # (B+1,) int32
    physical_to_logical: torch.Tensor,  # (B, num_gpu_blocks) int64
    seq_lens: torch.Tensor,          # (B,) int32
    block_size: int,
) -> Any:
    """Return the mask_mod closure for paged causal attention.

    The closure receives (b, h, q_idx, kv_idx) where:
      - q_idx  is a flat index into the packed query batch (0..num_actual_tokens-1)
      - kv_idx is a flat index into the concatenated physical KV cache
                (0..num_gpu_blocks*block_size-1)

    Physical→logical remapping is the bridge between vLLM's paged layout and
    flex_attention's logical-index world.  See NOTES.md section 3 for a worked
    example.

    Attribution: this structure mirrors stock vLLM's
    `FlexAttentionMetadata.get_paged_mask_mod()` (Apache-2.0), simplified
    for plain causal decoder attention only.
    """
    # Capture block_size as a Python int for tracing stability.
    _block_size = int(block_size)

    def mask_mod(
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        # Map flat query index → request id.
        req = doc_ids[q_idx]

        # Logical query position within the full sequence.
        local_q = q_idx - query_start_loc[req]
        logical_q = local_q + decode_offset[req]

        # Map flat KV index → physical block + offset → logical KV position.
        phys_block = kv_idx // _block_size
        offset_in_block = kv_idx % _block_size
        log_block = physical_to_logical[req, phys_block]   # int64, -1 if invalid
        logical_kv = log_block * _block_size + offset_in_block

        # Validity: block must be allocated AND token within sequence.
        is_allocated = log_block >= 0
        within_seq = logical_kv < seq_lens[req].to(torch.long)
        is_valid = is_allocated & within_seq

        # Causal + paged validity.
        return torch.where(is_valid, logical_kv <= logical_q, False)

    return mask_mod


# ---------------------------------------------------------------------------
# 2. MetadataBuilder
# ---------------------------------------------------------------------------

class MyFlexAttnMetadataBuilder(AttentionMetadataBuilder[MyFlexAttnMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: "AttentionSpec",
        layer_names: list[str],
        vllm_config: "VllmConfig",
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.device = device
        self.block_size = kv_cache_spec.block_size
        self.cache_config = vllm_config.cache_config

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MyFlexAttnMetadata:
        if common_prefix_len > 0:
            raise NotImplementedError(
                "cascade attention (common_prefix_len > 0) not supported in this educational backend"
            )

        query_start_loc = common_attn_metadata.query_start_loc   # (B+1,)
        seq_lens = common_attn_metadata.seq_lens                 # (B,)
        block_table = common_attn_metadata.block_table_tensor    # (B, max_blocks)
        slot_mapping = common_attn_metadata.slot_mapping
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len
        device = block_table.device

        B = int(seq_lens.numel())
        block_size = self.block_size

        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None, (
            "MyFlexAttnMetadataBuilder requires num_gpu_blocks to be set in cache_config"
        )
        total_cache_tokens = num_gpu_blocks * block_size

        # doc_ids[i] = request index for flat query token i.
        q_lens = query_start_loc[1:] - query_start_loc[:-1]  # (B,)
        doc_ids = torch.repeat_interleave(
            torch.arange(B, device=device, dtype=torch.int32), q_lens
        )

        # decode_offset[i] = number of already-computed tokens for request i.
        decode_offset = (seq_lens - q_lens).to(torch.int32)

        # physical_to_logical inverse table — the heart of the paged bridge.
        physical_to_logical = _build_physical_to_logical(
            block_table, seq_lens, block_size, num_gpu_blocks
        )

        # Build BlockMask once here; shared across all 28 attention layers.
        # Educational note: create_block_mask is called without torch.compile
        # so each forward triggers a Python-level evaluation of mask_mod on
        # a small grid to discover sparsity — fine for eager debugging.
        mask_mod = _make_paged_causal_mask_mod(
            doc_ids, decode_offset, query_start_loc,
            physical_to_logical, seq_lens, block_size,
        )
        block_mask = create_block_mask(
            mask_mod,
            B=None,  # single flat batch dimension
            H=None,  # all heads share the same mask
            Q_LEN=num_actual_tokens,
            KV_LEN=total_cache_tokens,
            device=device,
            BLOCK_SIZE=block_size,
        )

        return MyFlexAttnMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            block_table=block_table,
            slot_mapping=slot_mapping,
            doc_ids=doc_ids,
            decode_offset=decode_offset,
            physical_to_logical=physical_to_logical,
            block_mask=block_mask,
            total_cache_tokens=total_cache_tokens,
            block_size=block_size,
            num_gpu_blocks=num_gpu_blocks,
        )


# ---------------------------------------------------------------------------
# 3. Impl
# ---------------------------------------------------------------------------

class MyFlexAttnImpl(AttentionImpl[MyFlexAttnMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ) -> None:
        assert alibi_slopes is None, "alibi_slopes not supported (educational limitation)"
        assert sliding_window is None, "sliding_window not supported (educational limitation)"
        assert logits_soft_cap is None, "logits_soft_cap not supported (educational limitation)"
        assert attn_type == AttentionType.DECODER, (
            f"only DECODER attention supported, got {attn_type!r}"
        )
        assert kv_sharing_target_layer_name is None, "kv_sharing not supported"
        if kv_cache_dtype not in ("auto", "float16", "bfloat16"):
            raise AssertionError(
                f"fp8/nvfp4 kv_cache not supported (is_quantized_kv_cache), got {kv_cache_dtype!r}"
            )
        if kwargs:
            warnings.warn(
                f"MyFlexAttnImpl: ignoring unexpected kwargs {list(kwargs)}",
                stacklevel=2,
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.kv_cache_dtype = kv_cache_dtype

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,        # [num_tokens, Hq, D]
        key: torch.Tensor,          # [num_tokens, Hkv, D]
        value: torch.Tensor,        # [num_tokens, Hkv, D]
        kv_cache: torch.Tensor,     # [2, num_blocks, block_size, Hkv, D]
        attn_metadata: MyFlexAttnMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Profiling run / empty cache — short-circuit identical to flashinfer.
        if attn_metadata is None or kv_cache.numel() == 0:
            assert output is not None
            return output.fill_(0)

        assert output is not None
        N = attn_metadata.num_actual_tokens
        Hkv, Hq, D = self.num_kv_heads, self.num_heads, self.head_size

        # 1) Write incoming K/V into the paged cache.
        # KV cache layout is (2, num_blocks, block_size, Hkv, D).
        # unbind(0) splits dim-0 → key_cache (num_blocks, block_size, Hkv, D).
        # Note asymmetry vs flashinfer/vllm_unified which use unbind(1).
        key_cache, value_cache = kv_cache.unbind(0)
        from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
            triton_reshape_and_cache_flash,
        )
        triton_reshape_and_cache_flash(
            key[:N], value[:N],
            key_cache, value_cache,
            attn_metadata.slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale, layer._v_scale,
        )

        # 2) Reshape from paged (num_blocks, block_size, Hkv, D) to flat
        #    4-D (1, Hkv, total_tokens, D) that flex_attention expects.
        key_flat = key_cache.view(-1, Hkv, D)[None].permute(0, 2, 1, 3)   # (1, Hkv, total, D)
        val_flat = value_cache.view(-1, Hkv, D)[None].permute(0, 2, 1, 3) # (1, Hkv, total, D)
        q4d      = query[None, :N].permute(0, 2, 1, 3)                     # (1, Hq,  N,     D)

        # 3) Single flex_attention launch — BlockMask encodes paging + causality.
        # Must use the compiled version: eager dispatches to sdpa_dense (math
        # fallback) which allocates the full N×K logits tensor → OOM.
        # FORCE_USE_FLEX_ATTENTION bypasses flex_decoding (which fails on some
        # architectures, e.g. SM 12.0 / Blackwell, due to missing autotuned
        # kernel choices) and uses the main Triton sparse path.
        out = _flex_attention_compiled(
            q4d,
            key_flat,
            val_flat,
            None,  # score_mod
            attn_metadata.block_mask,
            self.scale,
            (Hkv != Hq),  # enable_gqa
        )  # (1, Hq, N, D)

        output[:N].copy_(out.permute(0, 2, 1, 3).squeeze(0))

        # 4) Observability.
        FIRE_COUNTER["fires"] += 1
        q_lens = attn_metadata.query_start_loc[1:] - attn_metadata.query_start_loc[:-1]
        num_seqs = int(q_lens.numel())
        max_q = int(attn_metadata.max_query_len)
        n_chunked = int((q_lens > 1).sum().item())  # seqs with q_len > 1 (chunked/prefill)

        new_seqs = num_seqs > FIRE_COUNTER["max_num_seqs"]
        new_q = max_q > FIRE_COUNTER["max_q_len"]
        new_chunked = n_chunked > FIRE_COUNTER["max_chunked_seqs"]
        if new_seqs:
            FIRE_COUNTER["max_num_seqs"] = num_seqs
        if new_q:
            FIRE_COUNTER["max_q_len"] = max_q
        if new_chunked:
            FIRE_COUNTER["max_chunked_seqs"] = n_chunked

        if _LOG_FIRE and (FIRE_COUNTER["fires"] == 1 or new_seqs or new_q or new_chunked):
            _logger.warning(
                "MyFlexAttnImpl.forward fired num_seqs=%d max_q_len=%d tokens=%d chunked=%d",
                num_seqs, max_q, N, n_chunked,
            )

        return output


# ---------------------------------------------------------------------------
# 4. Backend
# ---------------------------------------------------------------------------

class MyFlexAttnBackend(AttentionBackend):
    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type[MyFlexAttnImpl]:
        return MyFlexAttnImpl

    @staticmethod
    def get_builder_cls() -> type[MyFlexAttnMetadataBuilder]:
        return MyFlexAttnMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # Stock FlexAttn KV layout: (2, num_blocks, block_size, Hkv, D).
        # K/V split on dim-0 (unbind(0)) — different from flashinfer/vllm_unified
        # which use (num_blocks, 2, block_size, Hkv, D) and unbind(1).
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        # flex_attention imposes no hard head_dim restriction (unlike FlashInfer).
        return [64, 96, 128, 256]


# ---------------------------------------------------------------------------
# 5. Registration helper
# ---------------------------------------------------------------------------

def register() -> None:
    """Register MyFlexAttnBackend into the CUSTOM slot.

    Call this before constructing LLM(), then pass:
        LLM(..., attention_backend=AttentionBackendEnum.CUSTOM)
    """
    register_backend(
        AttentionBackendEnum.CUSTOM,
        "flex_attn_attention_backend.MyFlexAttnBackend",
    )


# ---------------------------------------------------------------------------
# Quick import / syntax smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import ast
    import pathlib

    src = pathlib.Path(__file__).read_text()
    ast.parse(src)
    print("syntax OK")

    assert "fires" in FIRE_COUNTER
    assert "max_chunked_seqs" in FIRE_COUNTER
    assert MyFlexAttnBackend.get_name() == "CUSTOM"
    assert MyFlexAttnBackend.get_kv_cache_shape(4, 16, 8, 128) == (2, 4, 16, 8, 128)
    assert MyFlexAttnBackend.get_supported_head_sizes() == [64, 96, 128, 256]
    print("import smoke test PASSED")
