"""vLLM v1 AttentionBackend — FlashAttention-backed attention (educational).

Sibling project of ../flashinfer/ and ../ptr/vllm_unified/. Same Backend /
Impl / Metadata / MetadataBuilder 4-tier shape, same `AttentionBackendEnum.CUSTOM`
slot, same KV-write path.

Three teaching points distinguish this backend from its siblings:

1. **Zero-cost builder**. `flash_attn_varlen_func` natively accepts vLLM's
   `block_table` and `cu_seqlens` directly — no CSR conversion, no plan(),
   no workspace buffer. The builder's `build()` is a ~15-line pass-through
   that copies `CommonAttentionMetadata` fields into `MyFlashAttnMetadata`
   unchanged. Compare with flashinfer (needs CSR conversion + plan() +
   128 MiB workspace) and vllm_unified (hand-written Triton kernel launch).

2. **Single launch, no split-dispatch**. A single `flash_attn_varlen_func`
   call handles prefill, decode, and chunked prefill uniformly. No prefill /
   decode classification in Impl.forward. This matches vllm_unified's
   structural unification (one kernel), but via an external library instead
   of a hand-written Triton kernel.

3. **KV layout asymmetry**. This backend uses `get_kv_cache_shape()` returning
   `(2, num_blocks, block_size, Hkv, D)` — dim-0 K/V split — so
   `kv_cache.unbind(0)` yields `(key_cache, value_cache)`. The flashinfer
   sibling uses `(num_blocks, 2, block_size, Hkv, D)` — dim-1 split —
   requiring `unbind(1)`. This is the point: vLLM's `AttentionBackend`
   interface lets each backend choose its own layout, and siblings can differ.

Register via:
    from flash_attn_attention_backend import register
    register()
Then:
    LLM(..., attention_backend=AttentionBackendEnum.CUSTOM)
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch

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

# flash_attn_varlen_func is imported lazily inside Impl to keep this module
# loadable on machines that have vLLM but not a GPU.

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import AttentionSpec

# Counters for observability. Single-launch: no prefill/decode split, so we
# track n_chunked (like vllm_unified) instead of prefill_calls/decode_calls.
FIRE_COUNTER: dict[str, int] = {
    "fires": 0,
    "max_num_seqs": 0,
    "max_q_len": 0,
    "max_chunked_seqs": 0,
}
_LOG_FIRE = os.environ.get("MY_FLASHATTN_BACKEND_LOG", "1") != "0"


# ---------------------------------------------------------------------------
# 1. Metadata
# ---------------------------------------------------------------------------

@dataclass
class MyFlashAttnMetadata(AttentionMetadata):
    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor   # (B+1,) int32  == cu_seqlens_q
    seq_lens: torch.Tensor          # (B,)   int32  == seqused_k
    block_table: torch.Tensor       # (B, max_blocks) int32
    slot_mapping: torch.Tensor      # (num_actual_tokens,) int64


# ---------------------------------------------------------------------------
# 2. MetadataBuilder
# ---------------------------------------------------------------------------

class MyFlashAttnMetadataBuilder(AttentionMetadataBuilder[MyFlashAttnMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: "AttentionSpec",
        layer_names: list[str],
        vllm_config: "VllmConfig",
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # No workspace, no wrappers — nothing to initialize. This is the
        # zero-cost builder: flash_attn_varlen_func needs no pre-allocated
        # scratch buffer or plan step, so there is nothing to do here.

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MyFlashAttnMetadata:
        if common_prefix_len > 0:
            raise NotImplementedError(
                "cascade attention (common_prefix_len > 0) not supported in this educational backend"
            )

        # Pure pass-through: vLLM's CommonAttentionMetadata fields ARE the
        # flash_attn_varlen_func arguments — no conversion needed.
        # Compare with flashinfer's builder which does a full CSR conversion.
        return MyFlashAttnMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
        )


# ---------------------------------------------------------------------------
# 3. Impl
# ---------------------------------------------------------------------------

class MyFlashAttnImpl(AttentionImpl[MyFlashAttnMetadata]):

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
        assert head_size % 8 == 0 and head_size <= 256, (
            f"FA2 requires head_dim % 8 == 0 and head_dim <= 256, got {head_size}"
        )
        assert alibi_slopes is None, "alibi_slopes not supported"
        assert sliding_window is None, "sliding_window not supported"
        assert logits_soft_cap is None, "logits_soft_cap not supported"
        assert attn_type == AttentionType.DECODER, f"attn_type={attn_type!r} not supported"
        assert kv_sharing_target_layer_name is None, "kv_sharing not supported"
        if kv_cache_dtype not in ("auto", "float16", "bfloat16"):
            raise AssertionError(f"kv_cache_dtype={kv_cache_dtype!r} not supported (no fp8 on sm_120)")
        if kwargs:
            warnings.warn(
                f"MyFlashAttnImpl: ignoring unexpected kwargs {list(kwargs)}",
                stacklevel=2,
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,        # [num_tokens, Hq, D]
        key: torch.Tensor,          # [num_tokens, Hkv, D]
        value: torch.Tensor,        # [num_tokens, Hkv, D]
        kv_cache: torch.Tensor,     # [2, num_blocks, block_size, Hkv, D]
        attn_metadata: MyFlashAttnMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Profiling run / empty cache — same short-circuit as all siblings.
        # Safe here: no workspace was allocated, so nothing is mis-accounted.
        if attn_metadata is None or kv_cache.numel() == 0:
            assert output is not None
            return output.fill_(0)

        assert output is not None, "accept_output_buffer=True so output must be pre-allocated"
        assert query.dtype in (torch.float16, torch.bfloat16), (
            f"only fp16/bf16 supported, got {query.dtype}"
        )

        N = attn_metadata.num_actual_tokens

        # 1) KV write — store current tokens into paged cache.
        #    Layout: (2, num_blocks, block_size, Hkv, D), unbind(0) splits K/V.
        #    TEACHING POINT: flashinfer uses (num_blocks, 2, ...) and unbind(1).
        #    Different layout, same triton_reshape_and_cache_flash call otherwise.
        key_cache, value_cache = kv_cache.unbind(0)   # each [num_blocks, block_size, Hkv, D]
        from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
            triton_reshape_and_cache_flash,
        )
        triton_reshape_and_cache_flash(
            key[:N], value[:N],
            key_cache, value_cache,
            attn_metadata.slot_mapping,
            "auto",
            layer._k_scale, layer._v_scale,
        )

        # 2) Single varlen call — prefill + decode + chunked uniformly.
        #    No split-dispatch. query_start_loc IS cu_seqlens_q.
        #    seq_lens IS seqused_k. block_table IS block_table. Pass-through.
        from vllm.vllm_flash_attn import flash_attn_varlen_func
        flash_attn_varlen_func(
            q=query[:N],
            k=key_cache,
            v=value_cache,
            out=output[:N],
            cu_seqlens_q=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            seqused_k=attn_metadata.seq_lens,
            max_seqlen_k=attn_metadata.max_seq_len,
            block_table=attn_metadata.block_table,
            softmax_scale=self.scale,
            causal=True,
            fa_version=2,
        )

        # 3) Observability — same structure as vllm_unified (no prefill/decode
        #    breakdown since there's no split-dispatch).
        num_seqs = int(attn_metadata.seq_lens.shape[0])
        q_lens = attn_metadata.query_start_loc[1:] - attn_metadata.query_start_loc[:-1]
        max_q = int(q_lens.max().item()) if q_lens.numel() > 0 else 0
        n_chunked = int(((q_lens > 1) & (q_lens < attn_metadata.seq_lens)).sum().item())

        FIRE_COUNTER["fires"] += 1
        new_max_seqs = num_seqs > FIRE_COUNTER["max_num_seqs"]
        new_max_q = max_q > FIRE_COUNTER["max_q_len"]
        new_chunked = n_chunked > FIRE_COUNTER["max_chunked_seqs"]
        if new_max_seqs:
            FIRE_COUNTER["max_num_seqs"] = num_seqs
        if new_max_q:
            FIRE_COUNTER["max_q_len"] = max_q
        if new_chunked:
            FIRE_COUNTER["max_chunked_seqs"] = n_chunked
        if _LOG_FIRE and (
            FIRE_COUNTER["fires"] == 1 or new_max_seqs or new_max_q or new_chunked
        ):
            _logger.warning(
                "MyFlashAttnImpl.forward fired num_seqs=%d max_q_len=%d "
                "chunked=%d tokens=%d",
                num_seqs, max_q, n_chunked, N,
            )

        return output


# ---------------------------------------------------------------------------
# 4. Backend
# ---------------------------------------------------------------------------

class MyFlashAttnBackend(AttentionBackend):
    # Same two contract additions as siblings:
    #  - Impl writes directly into the provided output buffer
    #  - Impl handles the KV-cache write internally
    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type[MyFlashAttnImpl]:
        return MyFlashAttnImpl

    @staticmethod
    def get_builder_cls() -> type[MyFlashAttnMetadataBuilder]:
        return MyFlashAttnMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # FA2 paged layout: dim-0 splits K vs V.
        # unbind(0) → key_cache [num_blocks, block_size, Hkv, D]
        #              value_cache [num_blocks, block_size, Hkv, D]
        # NOTE: flashinfer sibling uses (num_blocks, 2, ...) — dim-1 split.
        # This is the KV layout asymmetry teaching point.
        if block_size % 16 != 0:
            raise ValueError("block_size must be a multiple of 16 (FA2 requirement)")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        # FA2: head_dim % 8 == 0 and head_dim <= 256. Return common values.
        return [64, 96, 128, 192, 256]


# ---------------------------------------------------------------------------
# 5. Registration helper
# ---------------------------------------------------------------------------

def register() -> None:
    """Register MyFlashAttnBackend into the CUSTOM slot.

    Call this before constructing LLM(), then pass:
        LLM(..., attention_backend=AttentionBackendEnum.CUSTOM)
    """
    register_backend(
        AttentionBackendEnum.CUSTOM,
        "flash_attn_attention_backend.MyFlashAttnBackend",
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
    assert "max_num_seqs" in FIRE_COUNTER
    assert "max_q_len" in FIRE_COUNTER
    assert "max_chunked_seqs" in FIRE_COUNTER
    assert MyFlashAttnBackend.get_name() == "CUSTOM"
    assert MyFlashAttnBackend.get_kv_cache_shape(4, 16, 8, 128) == (2, 4, 16, 8, 128)
    assert MyFlashAttnBackend.get_supported_head_sizes() == [64, 96, 128, 192, 256]
    print("import smoke test PASSED")
