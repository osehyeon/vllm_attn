"""vLLM v1 AttentionBackend — FlashInfer-backed attention (educational).

Sibling project of ../ptr/vllm_unified/. Same Backend/Impl/Metadata/Builder
4-tier shape, same `AttentionBackendEnum.CUSTOM` slot, same KV-write path.
The only thing that changes is the kernel body: instead of a hand-written
Triton kernel, this backend dispatches into FlashInfer's prefill / decode
wrappers.

Three consequences fall out of that swap and they are the whole point of
this project:

1. **CSR conversion**. FlashInfer does not accept vLLM's `(block_table,
   seq_lens)` paged format directly — it wants three int32 1-D tensors
   (`kv_indptr`, `kv_indices`, `kv_last_page_len`). The MetadataBuilder
   does the conversion every forward.

2. **Split dispatch**. FlashInfer ships separate wrapper classes for
   prefill (q_len > 1) and decode (q_len == 1) and the `plan()` inputs
   differ between them. So this backend stays at the vllm_multiseq
   split-dispatch stage — it cannot collapse to one launch the way
   vllm_unified does, because the library itself splits.

3. **Workspace lives on the builder**. FlashInfer needs a ~128 MiB scratch
   buffer for split-K reductions. We allocate it lazily inside the
   MetadataBuilder (mirroring stock `vllm.v1.attention.backends.flashinfer`),
   not the Impl, so vLLM's `profile_run` triggers the allocation through
   `build() → plan()` and the memory profiler sees it. If the workspace
   is allocated inside `Impl.forward` instead, the dummy run short-circuits
   on `kv_cache.numel() == 0` and the 128 MiB never enters the profile peak
   — KV cache then claims the entire GPU and the first real forward OOMs.

Register via:
    from flashinfer_attention_backend import register
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

# FlashInfer is imported lazily inside Impl to keep this module loadable on
# machines that have vLLM but not the FlashInfer wheel.

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import AttentionSpec

FIRE_COUNTER: dict[str, int] = {
    "fires": 0,
    "prefill_calls": 0,
    "decode_calls": 0,
    "max_num_seqs": 0,
    "max_q_len": 0,
}
_LOG_FIRE = os.environ.get("MY_FLASHINFER_BACKEND_LOG", "1") != "0"


# ---------------------------------------------------------------------------
# 1. Metadata
# ---------------------------------------------------------------------------

@dataclass
class MyFlashInferMetadata(AttentionMetadata):
    num_actual_tokens: int
    slot_mapping: torch.Tensor                # (num_actual_tokens,) int64

    # --- Prefill subgroup (q_len > 1) ----------------------------------
    prefill_count: int
    prefill_token_slice: torch.Tensor         # (sum_p_q_len,) int64 — gather/scatter
    # The wrapper is `plan()`-ed by the builder on this batch's CSR triple
    # before this metadata reaches `Impl.forward`. forward() just calls
    # `.run()`. None when prefill_count == 0.
    prefill_wrapper: Any | None

    # --- Decode subgroup (q_len == 1) ----------------------------------
    decode_count: int
    decode_token_slice: torch.Tensor          # (D,) int64
    decode_wrapper: Any | None                # see prefill_wrapper note

    # --- Observability -------------------------------------------------
    prefill_max_q_len: int                    # for the "fired" log line


# ---------------------------------------------------------------------------
# 2. MetadataBuilder
# ---------------------------------------------------------------------------

def _build_paged_csr(
    block_table: torch.Tensor,   # (B, max_blocks)
    seq_lens: torch.Tensor,      # (B,)
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert vLLM's (block_table, seq_lens) into FlashInfer's CSR triple.

    Returns (kv_indptr, kv_indices, kv_last_page_len), all int32, all on
    block_table's device. FlashInfer requires int32 here — int64 silently
    misindexes.

    Educational implementation: a Python list-comprehension. Production
    vLLM does this on the GPU via cumsum + gather.
    """
    device = block_table.device
    B = int(seq_lens.numel())
    if B == 0:
        empty = torch.empty(0, dtype=torch.int32, device=device)
        zero_indptr = torch.zeros(1, dtype=torch.int32, device=device)
        return zero_indptr, empty, empty

    pages_per_seq = (seq_lens + page_size - 1) // page_size       # ceil-div, (B,)
    last_page_len = ((seq_lens - 1) % page_size) + 1              # (B,) in [1, page_size]

    kv_indptr = torch.zeros(B + 1, dtype=torch.int32, device=device)
    torch.cumsum(pages_per_seq.to(torch.int32), dim=0, out=kv_indptr[1:])

    rows = [block_table[i, :int(pages_per_seq[i])] for i in range(B)]
    kv_indices = torch.cat(rows).to(torch.int32)
    return kv_indptr, kv_indices, last_page_len.to(torch.int32)


class MyFlashInferMetadataBuilder(AttentionMetadataBuilder[MyFlashInferMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    # Mirrors stock `vllm.v1.attention.backends.flashinfer`. The default
    # there is 256 MiB and is exposed via `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE`;
    # we use a smaller fixed value because this educational backend skips the
    # cascade / TRTLLM paths that need the larger budget.
    _WORKSPACE_BYTES = 128 * 1024 * 1024

    def __init__(
        self,
        kv_cache_spec: "AttentionSpec",
        layer_names: list[str],
        vllm_config: "VllmConfig",
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.device = device
        self.page_size = kv_cache_spec.block_size

        # Same sources stock vLLM's FlashInferMetadataBuilder uses. The
        # builder is per-backend (not per-layer), so caching once is fine.
        model_config = vllm_config.model_config
        self.num_qo_heads = model_config.get_num_attention_heads(
            vllm_config.parallel_config,
        )
        self.num_kv_heads = kv_cache_spec.num_kv_heads
        self.head_dim = kv_cache_spec.head_size
        self.q_data_type = model_config.dtype

        # Lazily allocated; first `build()` triggers them. During vLLM's
        # `profile_run` this is what makes the 128 MiB visible to the
        # memory profiler so KV-cache reservation accounts for it.
        self._workspace_buffer: torch.Tensor | None = None
        self._prefill_wrapper: Any | None = None
        self._decode_wrapper: Any | None = None

    def _get_workspace_buffer(self) -> torch.Tensor:
        if self._workspace_buffer is None:
            self._workspace_buffer = torch.empty(
                self._WORKSPACE_BYTES, dtype=torch.uint8, device=self.device,
            )
        return self._workspace_buffer

    def _get_prefill_wrapper(self) -> Any:
        if self._prefill_wrapper is None:
            import flashinfer
            self._prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self._get_workspace_buffer(), kv_layout="NHD",
            )
        return self._prefill_wrapper

    def _get_decode_wrapper(self) -> Any:
        if self._decode_wrapper is None:
            import flashinfer
            self._decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                self._get_workspace_buffer(), kv_layout="NHD",
            )
        return self._decode_wrapper

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MyFlashInferMetadata:
        if common_prefix_len > 0:
            raise NotImplementedError(
                "cascade attention (common_prefix_len > 0) not supported in this educational backend"
            )

        query_start_loc = common_attn_metadata.query_start_loc   # (B+1,)
        seq_lens = common_attn_metadata.seq_lens                 # (B,)
        block_table = common_attn_metadata.block_table_tensor    # (B, max_blocks)
        device = block_table.device

        q_lens = query_start_loc[1:] - query_start_loc[:-1]
        prefill_idx = torch.nonzero(q_lens > 1, as_tuple=False).squeeze(-1)
        decode_idx = torch.nonzero(q_lens == 1, as_tuple=False).squeeze(-1)

        # ---- Prefill subgroup ---------------------------------------
        prefill_wrapper: Any | None = None
        prefill_max_q_len = 1
        if prefill_idx.numel() > 0:
            p_seq_lens = seq_lens[prefill_idx]
            p_block_table = block_table[prefill_idx]
            p_kv_indptr, p_kv_indices, p_last_page_len = _build_paged_csr(
                p_block_table, p_seq_lens, self.page_size,
            )
            # qo_indptr: cumulative q lengths inside the prefill subgroup
            p_q_lens = q_lens[prefill_idx].to(torch.int32)
            p_qo_indptr = torch.zeros(p_q_lens.numel() + 1, dtype=torch.int32, device=device)
            torch.cumsum(p_q_lens, dim=0, out=p_qo_indptr[1:])
            # token slice into the full flat-packed Q
            ranges = [
                torch.arange(int(query_start_loc[i]), int(query_start_loc[i + 1]),
                             device=device, dtype=torch.long)
                for i in prefill_idx.tolist()
            ]
            p_token_slice = (torch.cat(ranges) if ranges
                             else torch.empty(0, dtype=torch.long, device=device))
            prefill_max_q_len = int(p_q_lens.max().item())

            # plan() here, not in forward(): allocates workspace + selects the
            # JIT kernel module on CPU. During profile_run this is what makes
            # the 128 MiB enter the profiler's peak.
            prefill_wrapper = self._get_prefill_wrapper()
            prefill_wrapper.plan(
                qo_indptr=p_qo_indptr,
                paged_kv_indptr=p_kv_indptr,
                paged_kv_indices=p_kv_indices,
                paged_kv_last_page_len=p_last_page_len,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_dim,
                page_size=self.page_size,
                causal=True,
                q_data_type=self.q_data_type,
            )
            p_token_slice_out = p_token_slice
        else:
            p_token_slice_out = torch.empty(0, dtype=torch.long, device=device)

        # ---- Decode subgroup ----------------------------------------
        decode_wrapper: Any | None = None
        if decode_idx.numel() > 0:
            d_seq_lens = seq_lens[decode_idx]
            d_block_table = block_table[decode_idx]
            d_kv_indptr, d_kv_indices, d_last_page_len = _build_paged_csr(
                d_block_table, d_seq_lens, self.page_size,
            )
            # decode contributes exactly 1 token per request, at qsl[i]
            d_token_slice = query_start_loc[decode_idx].to(torch.long)

            decode_wrapper = self._get_decode_wrapper()
            decode_wrapper.plan(
                indptr=d_kv_indptr,
                indices=d_kv_indices,
                last_page_len=d_last_page_len,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                page_size=self.page_size,
                q_data_type=self.q_data_type,
            )
        else:
            d_token_slice = torch.empty(0, dtype=torch.long, device=device)

        return MyFlashInferMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            slot_mapping=common_attn_metadata.slot_mapping,
            prefill_count=int(prefill_idx.numel()),
            prefill_token_slice=p_token_slice_out,
            prefill_wrapper=prefill_wrapper,
            decode_count=int(decode_idx.numel()),
            decode_token_slice=d_token_slice,
            decode_wrapper=decode_wrapper,
            prefill_max_q_len=prefill_max_q_len,
        )


# ---------------------------------------------------------------------------
# 3. Impl
# ---------------------------------------------------------------------------

class MyFlashInferImpl(AttentionImpl[MyFlashInferMetadata]):
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
        assert head_size in (64, 128, 256), (
            f"FlashInfer requires head_dim ∈ {{64,128,256}}, got {head_size}"
        )
        assert alibi_slopes is None, "alibi_slopes not supported"
        assert sliding_window is None, "sliding_window not supported"
        assert logits_soft_cap is None, "logits_soft_cap not supported"
        assert attn_type == AttentionType.DECODER, f"attn_type={attn_type!r} not supported"
        assert kv_sharing_target_layer_name is None, "kv_sharing not supported"
        if kv_cache_dtype not in ("auto", "float16", "bfloat16"):
            raise AssertionError(f"kv_cache_dtype={kv_cache_dtype!r} not supported")
        if kwargs:
            warnings.warn(
                f"MyFlashInferImpl: ignoring unexpected kwargs {list(kwargs)}",
                stacklevel=2,
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads

        # No workspace / wrappers held here — they live on the builder so
        # vLLM's `profile_run` (which runs `build()` but short-circuits in
        # `forward()` on empty kv_cache) can still see the workspace allocation.

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,        # [num_tokens, Hq, D]
        key: torch.Tensor,          # [num_tokens, Hkv, D]
        value: torch.Tensor,        # [num_tokens, Hkv, D]
        kv_cache: torch.Tensor,     # [num_blocks, 2, block_size, Hkv, D]
        attn_metadata: MyFlashInferMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Profiling run / empty cache — same short-circuit as the Triton siblings.
        # Safe now: workspace allocation already happened in `build()`.
        if attn_metadata is None or kv_cache.numel() == 0:
            assert output is not None
            return output.fill_(0)

        assert output is not None, "accept_output_buffer=True so output must be pre-allocated"
        assert query.dtype in (torch.float16, torch.bfloat16), (
            f"only fp16/bf16 supported, got {query.dtype}"
        )

        N = attn_metadata.num_actual_tokens

        # 1) KV write — identical to vllm_unified. The KV cache layout we
        #    asked vLLM to allocate is the same as what FlashInfer wants
        #    (NHD), so this path needs no change.
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

        # FlashInfer reads the [num_blocks, 2, page_size, Hkv, D] tensor as
        # one paged_kv argument; no need to split into key_cache/value_cache.
        paged_kv = kv_cache

        # 2) Prefill subgroup — wrapper was already plan()-ed by the builder.
        if attn_metadata.prefill_count > 0:
            q_pre = query.index_select(0, attn_metadata.prefill_token_slice)  # [sum_p_q, Hq, D]
            o_pre = attn_metadata.prefill_wrapper.run(q_pre, paged_kv)
            output.index_copy_(0, attn_metadata.prefill_token_slice, o_pre)
            FIRE_COUNTER["prefill_calls"] += 1

        # 3) Decode subgroup
        if attn_metadata.decode_count > 0:
            q_dec = query.index_select(0, attn_metadata.decode_token_slice)   # [D_count, Hq, D]
            o_dec = attn_metadata.decode_wrapper.run(q_dec, paged_kv)
            output.index_copy_(0, attn_metadata.decode_token_slice, o_dec)
            FIRE_COUNTER["decode_calls"] += 1

        # 4) Observability
        FIRE_COUNTER["fires"] += 1
        num_seqs = attn_metadata.prefill_count + attn_metadata.decode_count
        max_q = attn_metadata.prefill_max_q_len
        new_seqs = num_seqs > FIRE_COUNTER["max_num_seqs"]
        new_q = max_q > FIRE_COUNTER["max_q_len"]
        if new_seqs:
            FIRE_COUNTER["max_num_seqs"] = num_seqs
        if new_q:
            FIRE_COUNTER["max_q_len"] = max_q
        if _LOG_FIRE and (FIRE_COUNTER["fires"] == 1 or new_seqs or new_q):
            _logger.warning(
                "MyFlashInferImpl.forward fired num_seqs=%d (prefill=%d decode=%d) "
                "max_q_len=%d tokens=%d",
                num_seqs, attn_metadata.prefill_count, attn_metadata.decode_count, max_q, N,
            )

        return output


# ---------------------------------------------------------------------------
# 4. Backend
# ---------------------------------------------------------------------------

class MyFlashInferBackend(AttentionBackend):
    # Same two contract additions as vllm_unified:
    #  - Impl writes directly into the provided output buffer
    #  - Impl handles the KV-cache write internally
    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type[MyFlashInferImpl]:
        return MyFlashInferImpl

    @staticmethod
    def get_builder_cls() -> type[MyFlashInferMetadataBuilder]:
        return MyFlashInferMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # FlashInfer NHD layout. Same shape as vllm_unified, so the KV-write
        # path (triton_reshape_and_cache_flash, taking unbind(1)) is shared
        # without modification. FlashInfer's run() consumes the whole tensor
        # as a single paged_kv argument.
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128, 256]


# ---------------------------------------------------------------------------
# 5. Registration helper
# ---------------------------------------------------------------------------

def register() -> None:
    """Register MyFlashInferBackend into the CUSTOM slot.

    Call this before constructing LLM(), then pass:
        LLM(..., attention_backend=AttentionBackendEnum.CUSTOM)
    """
    register_backend(
        AttentionBackendEnum.CUSTOM,
        "flashinfer_attention_backend.MyFlashInferBackend",
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
    assert "prefill_calls" in FIRE_COUNTER
    assert "decode_calls" in FIRE_COUNTER
    assert MyFlashInferBackend.get_name() == "CUSTOM"
    assert MyFlashInferBackend.get_kv_cache_shape(4, 16, 8, 128) == (4, 2, 16, 8, 128)
    assert MyFlashInferBackend.get_supported_head_sizes() == [64, 128, 256]
    print("import smoke test PASSED")
