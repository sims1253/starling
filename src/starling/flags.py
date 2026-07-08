"""Runtime feature flags for optional megakernel optimisations.

The megakernel pipeline has several optimisations that trade numerical
exactness for speed.  Some are byte-exact (safe to always enable); others break
byte-exactness and must be opt-in.  This module provides a single source of
truth for which optimisations are active, with a process-global default and a
context manager for scoped overrides.

Flags
-----
* ``multistep_graph`` (default **True**) -- use :class:`MultiStepLLMMega`
  (K-step CUDA-graph capture) instead of :class:`FusedLLMMega` (single-step).
  **Byte-exact**: greedy = greedy, the only change is *when* the argmax runs
  and *when* the host syncs.  Safe to leave on.
* ``batched_encoder`` (default **False**) -- enable a batched-encoder fast path
  in :class:`BatchedPipeline` (encode all B streams in one forward instead of
  per-stream).  **Breaks byte-exactness**: the conformer's BatchNorm
  (running_var ~4e-10) amplifies batch-size-dependent reduction differences
  ~316x per block; measured ~5.2 max-abs diff in the encoder hidden.  Only
  enable with ``tolerance_mode=True``.
* ``tolerance_mode`` (default **False**) -- master switch allowing
  ~5e-3 mean-abs numerical differences from byte-exactness-breaking
  optimisations (e.g. ``batched_encoder``).  When False the pipeline must be
  end-to-end byte-exact with the golden reference.

Usage
-----
::

    from starling.flags import OptFlags, flags, get_default_flags

    # Use the process default (multistep on, byte-exact).
    pipe = MegaPipeline(model, proc)

    # Scope a temporary override.
    with flags(tolerance_mode=True, batched_encoder=True):
        pipe_batched = BatchedPipeline(model, proc, max_batch_size=8)
        ...  # tolerance-matched, faster encoder

    # Pass explicit flags at construction.
    pipe = MegaPipeline(model, proc, flags=OptFlags(multistep_graph=False))
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class OptFlags:
    """Runtime feature flags for optional optimisations.

    Defaults preserve byte-exactness (the safe baseline).  Flags that break
    byte-exactness (``batched_encoder``) require ``tolerance_mode=True``.
    """

    multistep_graph: bool = True
    """Use :class:`MultiStepLLMMega` (K-step graph) instead of
    :class:`FusedLLMMega` (single-step).  **Byte-exact** -- safe."""

    batched_encoder: bool = False
    """Enable the batched-encoder fast path in :class:`BatchedPipeline`.
    **Breaks byte-exactness** (BatchNorm running_var amplifies batch-dependent
    diffs ~316x/block).  Requires ``tolerance_mode=True``."""

    tolerance_mode: bool = False
    """Master switch: allow ~5e-3 mean-abs numerical differences.  When False
    the pipeline is end-to-end byte-exact with the golden reference."""

    fused_qkv: bool = True
    """Fuse the per-layer q/k/v projections into one cuBLAS GEMM and the
    gate/up projections into one GEMM (cuts 5 launches -> 2 per layer).
    **Byte-exact**: concatenating weights/biases is associative over the
    matmul+add, so the per-element result is identical.  Safe to leave on."""

    sdpa_attention: bool = False
    """Replace the manual ``matmul -> +mask -> softmax -> matmul`` attention
    with :func:`torch.nn.functional.scaled_dot_product_attention` using the
    **math** backend and ``enable_gqa=True``.  Removes the GQA
    ``_repeat_kv`` materialisation (a full K/V copy per layer per token) and
    collapses 4 launches into 1.  Nearly byte-exact (same fp32 softmax over
    the same scores) but cuBLAS picks a different Q@K^T algorithm under the
    fused QKV layout, which can drift ~1 ULP and occasionally flip a near-tie
    argmax over a long decode.  Leave off for strict byte-exactness."""

    flash_attention: bool = False
    """Use SDPA's flash/efficient backend instead of the math backend.
    **Breaks byte-exactness**: fused flash attention does its softmax in
    fp32 register-tiled blocks rather than materialising the full score
    matrix, so per-element diffs are ~1e-3 (typically no argmax flips over a
    full decode).  Requires ``tolerance_mode=True``."""

    fp8_attention: bool = False
    """Cast Q/K/V to fp8e4m3 for the attention matmuls (Q@K^T and attn@V),
    with fp32 accumulation.  On Blackwell (sm_120) this is the largest
    remaining decode speedup for long audio.  **Breaks byte-exactness**
    (~1e-2 score diffs); requires ``tolerance_mode=True`` and re-validation
    against the WER bench."""

    fp8_weights: bool = False
    """Cast the decoder-layer projection weights (q/k/v/o, gate/up/down) to
    fp8e4m3 with dynamic per-token activation scaling, via ``torch._scaled_mm``
    (Blackwell fp8 tensor cores).  Halves the weight bandwidth that dominates
    decode (~57% of the captured step is these GEMVs per the profiler).  The
    **lm_head stays bf16** (its vocab-wide argmax has fp8-fragile near-ties).
    **Breaks byte-exactness** (fp8 weight rounding); requires
    ``tolerance_mode=True`` and forces ``fused_qkv`` (it reads the
    pre-concatenated qkv/gate-up weights).

    .. note::
       The current ``torch._scaled_mm`` path is **correct but slower at
       batch=1** decode (the cutlass GEMM + per-token activation quant overhead
       outweighs the bandwidth win; measured 0.63x).  It pays off at larger M
       (prefill / batched decode) and is the scaffolding for a future fused
       fp8 dequant-GEMV Triton kernel (analogous to the fp4 one in
       ``llm_kernels.py``) that would deliver the decode speedup.  See
       ``starling.granite.fp8`` for the full measured breakdown."""

    # ------------------------------------------------------------------
    # Ablatable optimisations (wiki-driven).  Each defaults to off so the
    # baseline remains byte-exact; the ablation harness flips them on/off to
    # measure individual benefit.  See benchmarks/bench_ablate.py.
    # ------------------------------------------------------------------

    rope_alloc_free: bool = True
    """RoPE ``rotate_half`` via precomputed index+sign buffers instead of
    ``torch.cat`` (2 fewer allocations/layer/step).  **Byte-exact** -- same
    arithmetic over the same elements.  On by default; flip off to ablate."""

    lm_head_scale_fold: bool = True
    """Fold ``LLM_LOGITS_SCALING`` into a pre-scaled lm_head weight (one fewer
    elementwise divide/step, folded into the cuBLAS GEMM epilogue).
    **Byte-exact** (fp32 rescale of weights then re-cast).  On by default."""

    gemm_epilogue_fusion: bool = False
    """Fold RMSNorm + residual + SiLU-gate into the adjacent cuBLASLt GEMM
    epilogues (CODA-style).  Removes ~4-5% of decode-step elementwise launches.
    **Experimental** -- under construction; off by default."""

    chunk_prefill_overlap: bool = True
    """Long-audio: run chunk N+1's encoder prefill on a second CUDA stream
    while chunk N's LLM decode runs on the default stream (prefill is
    compute-bound, decode is bandwidth-bound -- they don't compete).  Pure
    scheduling; **byte-exact** (identical per-chunk work, just overlapped)."""

    nvfp4_weights: bool = False
    """Load the LLM weights at NVFP4 (4-bit microscaling) with dequant fused
    into the GEMV.  Halves the weight bandwidth that dominates decode (51% of
    step time per the profile).  **Requires quantized weights + a QAD fine-tune
    to preserve WER** -- not yet implemented; the loader gates on this flag.
    **Breaks byte-exactness**; requires ``tolerance_mode=True``."""

    nvfp4_lm_head_only: bool = False
    """Quantize only the final LM-head projection to NVFP4, keeping decoder
    layers in bf16. Experimental WER-gated variant of ``nvfp4_weights`` for
    testing selective BF16 retention. **Breaks byte-exactness**; requires
    ``tolerance_mode=True``."""

    kv_cache_compression: bool = False
    """Compress the encoder KV cache via spectral calibration (only the 3-4%
    of head dims carrying signal are kept; see kv-cache-spectral-compression).
    Targets the encoder, not the LLM.  **Experimental** -- requires a
    calibration pass to confirm ASR encoder KV is as low-rank as LLM KV."""

    slim_draft_head: bool = False
    """Speculative decoding: replace the draft LM-head with a low-rank
    factorisation (SlimSpec, arXiv:2605.10453).  4-5x faster draft head, no
    acceptance ceiling.  Composes with the existing CTC-BPE draft path."""

    def __post_init__(self) -> None:
        """Validate flag combinations at construction time."""
        if self.batched_encoder and not self.tolerance_mode:
            raise ValueError(
                "batched_encoder=True requires tolerance_mode=True (it breaks "
                "byte-exactness). Set tolerance_mode=True or batched_encoder=False."
            )
        if self.flash_attention and not self.tolerance_mode:
            raise ValueError(
                "flash_attention=True requires tolerance_mode=True (flash softmax "
                "is not bit-exact with the reference). Set tolerance_mode=True."
            )
        if self.fp8_attention and not self.tolerance_mode:
            raise ValueError(
                "fp8_attention=True requires tolerance_mode=True (fp8 attention "
                "matmuls are not bit-exact). Set tolerance_mode=True."
            )
        if self.fp8_attention and not self.flash_attention:
            # fp8 attention reuses the flash backend's SDPA path; force it on.
            self.flash_attention = True
        if self.fp8_weights and not self.fused_qkv:
            # fp8 packing reads from the pre-concatenated qkv/gate-up weights
            # built by _fuse_layer_weights; force fused_qkv on.
            self.fused_qkv = True
        if self.fp8_weights and not self.tolerance_mode:
            raise ValueError(
                "fp8_weights=True requires tolerance_mode=True (fp8 weights are "
                "not bit-exact). Set tolerance_mode=True or fp8_weights=False."
            )
        if self.nvfp4_weights and not self.fused_qkv:
            # NVFP4 packing reads from the pre-concatenated QKV/gate-up weights
            # built by _fuse_layer_weights; force fused_qkv on (mirrors the
            # fp8_attention -> flash_attention dependency).
            self.fused_qkv = True
        if self.nvfp4_lm_head_only and not self.fused_qkv:
            self.fused_qkv = True
        if self.nvfp4_weights and not self.tolerance_mode:
            raise ValueError(
                "nvfp4_weights=True requires tolerance_mode=True (4-bit weights "
                "are not bit-exact). Set tolerance_mode=True."
            )
        if self.nvfp4_lm_head_only and not self.tolerance_mode:
            raise ValueError(
                "nvfp4_lm_head_only=True requires tolerance_mode=True (4-bit "
                "weights are not bit-exact). Set tolerance_mode=True."
            )


# ---------------------------------------------------------------------------
# process-global default flags
# ---------------------------------------------------------------------------
_DEFAULT_FLAGS = OptFlags()


def get_default_flags() -> OptFlags:
    """Return the process-global default :class:`OptFlags` instance."""
    return _DEFAULT_FLAGS


def set_default_flags(fl: OptFlags) -> None:
    """Replace the process-global default flags."""
    global _DEFAULT_FLAGS
    _DEFAULT_FLAGS = fl


@contextmanager
def flags(**overrides):
    """Temporarily override the global default flags within a ``with`` scope.

    Only the given keyword overrides change; all others inherit the current
    global default.  The original default is restored on exit (even on error).

    Example::

        with flags(tolerance_mode=True):
            ...  # byte-exactness-breaking opts allowed here
        # back to byte-exact default here
    """
    global _DEFAULT_FLAGS
    saved = _DEFAULT_FLAGS
    new = OptFlags(
        multistep_graph=overrides.get("multistep_graph", saved.multistep_graph),
        batched_encoder=overrides.get("batched_encoder", saved.batched_encoder),
        tolerance_mode=overrides.get("tolerance_mode", saved.tolerance_mode),
        fused_qkv=overrides.get("fused_qkv", saved.fused_qkv),
        sdpa_attention=overrides.get("sdpa_attention", saved.sdpa_attention),
        flash_attention=overrides.get("flash_attention", saved.flash_attention),
        fp8_attention=overrides.get("fp8_attention", saved.fp8_attention),
        fp8_weights=overrides.get("fp8_weights", saved.fp8_weights),
        rope_alloc_free=overrides.get("rope_alloc_free", saved.rope_alloc_free),
        lm_head_scale_fold=overrides.get("lm_head_scale_fold", saved.lm_head_scale_fold),
        gemm_epilogue_fusion=overrides.get("gemm_epilogue_fusion", saved.gemm_epilogue_fusion),
        chunk_prefill_overlap=overrides.get("chunk_prefill_overlap", saved.chunk_prefill_overlap),
        nvfp4_weights=overrides.get("nvfp4_weights", saved.nvfp4_weights),
        nvfp4_lm_head_only=overrides.get("nvfp4_lm_head_only", saved.nvfp4_lm_head_only),
        kv_cache_compression=overrides.get("kv_cache_compression", saved.kv_cache_compression),
        slim_draft_head=overrides.get("slim_draft_head", saved.slim_draft_head),
    )
    _DEFAULT_FLAGS = new
    try:
        yield new
    finally:
        _DEFAULT_FLAGS = saved
