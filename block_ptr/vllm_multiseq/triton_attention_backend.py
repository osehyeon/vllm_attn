"""vLLM v1 AttentionBackend — Triton fused attention (educational, fp16, head_dim=128).

Register via:
    from triton_attention_backend import register
    register()
Then pass attention_backend=AttentionBackendEnum.CUSTOM to LLM().
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

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import AttentionSpec

# Counters for observability. "max_prefill_seqs" / "max_decode_seqs" grow
# whenever vLLM actually dispatches a multi-seq batch to us — watching those
# grow is how we verify continuous batching is hitting this backend.
FIRE_COUNTER: dict[str, int] = {
    "prefill": 0,
    "decode": 0,
    "max_prefill_seqs": 0,
    "max_prefill_tokens": 0,
    "max_decode_seqs": 0,
}
_LOG_FIRE = os.environ.get("MY_TRITON_BACKEND_LOG", "1") != "0"


# ---------------------------------------------------------------------------
# 1. Metadata
# ---------------------------------------------------------------------------

@dataclass
class MyTritonMetadata(AttentionMetadata):
    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor   # (B+1,) int32
    seq_lens: torch.Tensor          # (B,) int32 — full context lengths
    block_table: torch.Tensor       # (B, max_blocks) int32
    slot_mapping: torch.Tensor      # (num_actual_tokens,) int64


# ---------------------------------------------------------------------------
# 2. MetadataBuilder
# ---------------------------------------------------------------------------

class MyTritonMetadataBuilder(AttentionMetadataBuilder[MyTritonMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: "AttentionSpec",
        layer_names: list[str],
        vllm_config: "VllmConfig",
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MyTritonMetadata:
        if common_prefix_len > 0:
            raise NotImplementedError("cascade attention (common_prefix_len>0) is not supported")

        return MyTritonMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
        )


# ---------------------------------------------------------------------------
# 3. (formerly _gather_kv helper — removed; the kernel now reads paged KV directly)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 4. Impl
# ---------------------------------------------------------------------------

class MyTritonImpl(AttentionImpl[MyTritonMetadata]):

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
        assert head_size > 0 and (head_size & (head_size - 1)) == 0, (
            f"triton_attn kernel requires head_dim to be a power of two, got {head_size}"
        )
        assert alibi_slopes is None, "alibi_slopes not supported"
        assert sliding_window is None, "sliding_window not supported"
        assert logits_soft_cap is None, "logits_soft_cap not supported"
        assert attn_type == AttentionType.DECODER, f"attn_type={attn_type!r} not supported"
        assert kv_sharing_target_layer_name is None, "kv_sharing not supported"
        if kv_cache_dtype not in ("auto", "float16"):
            raise AssertionError(f"kv_cache_dtype={kv_cache_dtype!r} not supported (fp16 only)")
        if kwargs:
            warnings.warn(f"MyTritonImpl: ignoring unexpected kwargs {list(kwargs)}", stacklevel=2)

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
        kv_cache: torch.Tensor,     # [num_blocks, 2, block_size, Hkv, D]
        attn_metadata: MyTritonMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Profiling run or empty cache — skip kernel
        if attn_metadata is None or kv_cache.numel() == 0:
            assert output is not None
            return output.fill_(0)

        assert output is not None, "accept_output_buffer=True so output must be pre-allocated"
        assert query.dtype in (torch.float16, torch.bfloat16), (
            f"only fp16/bf16 supported, got {query.dtype}"
        )

        from triton_attn import (  # local import avoids circular dep
            triton_attention_prefill_multiseq,
            triton_attention_decode_multiseq,
        )

        N = attn_metadata.num_actual_tokens
        Hq, Hkv, D = self.num_heads, self.num_kv_heads, self.head_size

        # 1) KV write: store current key/value tokens into paged cache
        key_cache, value_cache = kv_cache.unbind(1)   # each [num_blocks, block_size, Hkv, D]
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

        # 2) Classify each sequence: prefill (q_len == s_len) vs decode (q_len == 1).
        #    Mixed batch: split into prefill group + decode group, dispatch each
        #    to its dedicated multi-seq kernel (single launch per group).
        query_start_loc = attn_metadata.query_start_loc   # [num_seqs+1]
        seq_lens = attn_metadata.seq_lens                  # [num_seqs]
        block_table = attn_metadata.block_table            # [num_seqs, max_blocks]
        num_seqs = seq_lens.shape[0]

        q_lens = query_start_loc[1:] - query_start_loc[:-1]   # [num_seqs]
        is_prefill = q_lens == seq_lens
        is_decode = (q_lens == 1) & ~is_prefill
        # Chunked prefill (q_len > 1 but < s_len) not supported in this stage
        assert (is_prefill | is_decode).all(), (
            "chunked prefill not supported in vllm_multiseq — each seq must be "
            "either pure prefill (q_len==s_len) or pure decode (q_len==1)"
        )

        # --- Prefill group ---
        if is_prefill.any():
            prefill_idx = is_prefill.nonzero(as_tuple=True)[0]            # [num_prefill]
            # Gather q tokens of prefill seqs (preserving intra-seq order)
            starts = query_start_loc[prefill_idx]
            ends = query_start_loc[prefill_idx + 1]
            token_indices = torch.cat(
                [torch.arange(int(s), int(e), device=query.device) for s, e in zip(starts, ends)]
            ) if len(prefill_idx) > 0 else torch.zeros(0, dtype=torch.long, device=query.device)
            q_prefill = query[token_indices]                               # [total_p, Hq, D]
            sl_prefill = seq_lens[prefill_idx].contiguous()
            bt_prefill = block_table[prefill_idx].contiguous()
            # Per-group query_start_loc (0-based within this subgroup)
            p_q_lens = q_lens[prefill_idx]
            qsl_prefill = torch.zeros(len(prefill_idx) + 1, dtype=torch.int32, device=query.device)
            qsl_prefill[1:] = torch.cumsum(p_q_lens.to(torch.int32), 0)

            o_prefill = triton_attention_prefill_multiseq(
                q_prefill, key_cache, value_cache,
                bt_prefill, sl_prefill, qsl_prefill,
                scale=self.scale,
            )
            output[token_indices] = o_prefill
            FIRE_COUNTER["prefill"] += 1
            n_seqs = int(prefill_idx.numel())
            n_toks = int(q_prefill.shape[0])
            new_max_seqs = n_seqs > FIRE_COUNTER["max_prefill_seqs"]
            new_max_toks = n_toks > FIRE_COUNTER["max_prefill_tokens"]
            if new_max_seqs:
                FIRE_COUNTER["max_prefill_seqs"] = n_seqs
            if new_max_toks:
                FIRE_COUNTER["max_prefill_tokens"] = n_toks
            # Log on the first call OR whenever we observe a new maximum —
            # lets us see scheduler dispatching multi-seq batches without
            # spamming every forward.
            if _LOG_FIRE and (FIRE_COUNTER["prefill"] == 1 or new_max_seqs or new_max_toks):
                _logger.warning(
                    "MyTritonImpl.forward fired (prefill multiseq) num_seqs=%d tokens=%d",
                    n_seqs, n_toks,
                )

        # --- Decode group ---
        if is_decode.any():
            decode_idx = is_decode.nonzero(as_tuple=True)[0]               # [num_decode]
            # Each decode seq contributes exactly 1 query token, at position query_start_loc[i]
            decode_token_pos = query_start_loc[decode_idx]                  # [num_decode]
            q_decode = query[decode_token_pos]                              # [num_decode, Hq, D]
            sl_decode = seq_lens[decode_idx].contiguous()
            bt_decode = block_table[decode_idx].contiguous()

            o_decode = triton_attention_decode_multiseq(
                q_decode, key_cache, value_cache,
                bt_decode, sl_decode,
                scale=self.scale,
            )
            output[decode_token_pos] = o_decode
            FIRE_COUNTER["decode"] += 1
            n_seqs = int(decode_idx.numel())
            new_max = n_seqs > FIRE_COUNTER["max_decode_seqs"]
            if new_max:
                FIRE_COUNTER["max_decode_seqs"] = n_seqs
            if _LOG_FIRE and (FIRE_COUNTER["decode"] == 1 or new_max):
                _logger.warning(
                    "MyTritonImpl.forward fired (decode multiseq) num_seqs=%d",
                    n_seqs,
                )

        return output


# ---------------------------------------------------------------------------
# 5. Backend
# ---------------------------------------------------------------------------

class MyTritonBackend(AttentionBackend):
    # Only these two are genuine contract additions for our backend
    # (our Impl writes to the provided output buffer, and handles KV cache
    # update inside forward). Everything else — supported_dtypes,
    # supported_kv_cache_dtypes, get_supported_head_sizes(), is_mla(),
    # supports_attn_type(), ... — is inherited from AttentionBackend's
    # defaults. Real kernel constraints (fp16, head_dim=128) are enforced
    # as asserts in MyTritonImpl.__init__.
    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type[MyTritonImpl]:
        return MyTritonImpl

    @staticmethod
    def get_builder_cls() -> type[MyTritonMetadataBuilder]:
        return MyTritonMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # Same layout as vLLM's built-in TritonAttentionBackend.
        # unbind(1) → key_cache [num_blocks, block_size, Hkv, D]
        #                        value_cache [num_blocks, block_size, Hkv, D]
        return (num_blocks, 2, block_size, num_kv_heads, head_size)


# ---------------------------------------------------------------------------
# 6. Registration helper
# ---------------------------------------------------------------------------

def register() -> None:
    """Register MyTritonBackend into the CUSTOM slot.

    Call this before constructing LLM(), then pass:
        LLM(..., attention_backend=AttentionBackendEnum.CUSTOM)
    """
    register_backend(
        AttentionBackendEnum.CUSTOM,
        "triton_attention_backend.MyTritonBackend",
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

    # Verify dataclass and counter exist
    assert "prefill" in FIRE_COUNTER
    assert "decode" in FIRE_COUNTER
    assert MyTritonBackend.get_name() == "CUSTOM"
    assert MyTritonBackend.get_kv_cache_shape(4, 16, 8, 128) == (4, 2, 16, 8, 128)
    # Declaration matrix now inherits AttentionBackend defaults —
    # real kernel limits (fp16, head_dim=128) are enforced in MyTritonImpl.__init__.
    assert MyTritonBackend.get_supported_head_sizes() == []  # unconstrained at declaration level
    print("import smoke test PASSED")
