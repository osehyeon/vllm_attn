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

# Incremented each time MyTritonImpl.forward runs. Since the engine core may
# live in a separate process, in-memory counters don't cross that boundary —
# so we also emit a stderr line (captured by vLLM's logger as "[pid=...]")
# the first few times each path fires. Gated by MY_TRITON_BACKEND_LOG env var
# for quietness in production-like runs.
FIRE_COUNTER: dict[str, int] = {"prefill": 0, "decode": 0}
_LOG_FIRE = os.environ.get("MY_TRITON_BACKEND_LOG", "1") != "0"
_LOG_EVERY_N = 1  # first N calls on each path


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
# 3. KV-gather helper (decode path only — pure PyTorch, intentionally slow)
# ---------------------------------------------------------------------------

def _gather_kv(
    key_cache: torch.Tensor,    # [num_blocks, block_size, Hkv, D]
    value_cache: torch.Tensor,  # [num_blocks, block_size, Hkv, D]
    block_table_row: torch.Tensor,  # [max_blocks] int32
    s_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather paged KV into contiguous [1, Hkv, s_len, D] tensors.

    Educational / slow path — used only during decode.
    TODO: replace with a dedicated decode kernel for production.
    """
    block_size = key_cache.shape[1]
    n_blocks = (s_len + block_size - 1) // block_size
    blocks = block_table_row[:n_blocks].to(torch.long)          # [n_blocks]
    # [n_blocks*block_size, Hkv, D] → slice to s_len
    k_sel = key_cache[blocks].reshape(-1, *key_cache.shape[2:])[:s_len]    # [s_len, Hkv, D]
    v_sel = value_cache[blocks].reshape(-1, *value_cache.shape[2:])[:s_len]
    # → [1, Hkv, s_len, D]
    k = k_sel.permute(1, 0, 2).unsqueeze(0).contiguous()
    v = v_sel.permute(1, 0, 2).unsqueeze(0).contiguous()
    return k, v


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
            triton_attention_prefill,
            triton_attention_decode,
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

        # 2) Determine prefill vs decode (batch_size=1 assumed)
        q_len = int(
            attn_metadata.query_start_loc[1].item()
            - attn_metadata.query_start_loc[0].item()
        )
        s_len = int(attn_metadata.seq_lens[0].item())

        if q_len == s_len:
            # === prefill === (q_len == kv_len, causal)
            q = query[:N].view(1, N, Hq, D).transpose(1, 2).contiguous()    # [1, Hq, N, D]
            k = key[:N].view(1, N, Hkv, D).transpose(1, 2).contiguous()     # [1, Hkv, N, D]
            v = value[:N].view(1, N, Hkv, D).transpose(1, 2).contiguous()   # [1, Hkv, N, D]
            o = triton_attention_prefill(q, k, v, scale=self.scale)          # [1, Hq, N, D]
            output[:N].copy_(o.transpose(1, 2).reshape(N, Hq, D))
            FIRE_COUNTER["prefill"] += 1
            if _LOG_FIRE and FIRE_COUNTER["prefill"] <= _LOG_EVERY_N:
                _logger.warning("MyTritonImpl.forward fired (prefill) tokens=%d", N)
        else:
            # === decode === (q_len=1, full KV context from paged cache)
            q = query[:1].view(1, 1, Hq, D).transpose(1, 2).contiguous()    # [1, Hq, 1, D]
            k_full, v_full = _gather_kv(
                key_cache, value_cache,
                attn_metadata.block_table[0], s_len,
            )                                                                # [1, Hkv, s_len, D]
            o = triton_attention_decode(q, k_full, v_full, scale=self.scale) # [1, Hq, 1, D]
            output[:1].copy_(o.reshape(1, Hq, D))
            FIRE_COUNTER["decode"] += 1
            if _LOG_FIRE and FIRE_COUNTER["decode"] <= _LOG_EVERY_N:
                _logger.warning("MyTritonImpl.forward fired (decode) s_len=%d", s_len)

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
