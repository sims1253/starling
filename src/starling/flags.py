"""Runtime options for the Python megakernel pipelines.

``tolerance_mode`` permits experimental paths that change arithmetic. It does
not enforce an error bound or guarantee transcript accuracy. Compare quality
on representative audio before enabling these paths in production. Default
flags are the baseline configuration, not a promise of identical output across
hardware, libraries, or reference implementations.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, replace


@dataclass
class OptFlags:
    """Pipeline options, with experimental arithmetic disabled by default."""

    multistep_graph: bool = True
    """Capture K decode steps per CUDA graph instead of one."""

    batched_encoder: bool = False
    """Encode streams together; batch-dependent reductions can change output."""

    tolerance_mode: bool = False
    """Allow the approximate paths listed in ``__post_init__``."""

    fused_qkv: bool = True
    """Combine Q/K/V and gate/up projections into two GEMMs per layer."""

    sdpa_attention: bool = False
    """Use SDPA's math backend; reduction order can change rounding."""

    flash_attention: bool = False
    """Use SDPA's flash/efficient backend."""

    fp8_attention: bool = False
    """Cast attention Q/K/V to fp8; implies ``flash_attention``."""

    rope_alloc_free: bool = True
    """Apply RoPE using precomputed index and sign buffers."""

    lm_head_scale_fold: bool = True
    """Fold logits scaling into the LM-head weights."""

    chunk_prefill_overlap: bool = True
    """Overlap the next audio chunk's encoder prefill with LLM decode."""

    fp8_weights: bool = False
    """Use fp8 decoder weights with Triton dequant-GEMV; keep LM-head bf16."""

    gemm_epilogue_fusion: bool = False
    """Fold RMSNorm scaling into QKV and gate/up GEMVs; changes bf16 rounding."""

    nvfp4_weights: bool = False
    """Quantize decoder projections to NVFP4 with fused dequant-GEMV."""

    nvfp4_lm_head_only: bool = False
    """Quantize only the LM-head projection to NVFP4."""

    def __post_init__(self) -> None:
        for name in (
            "batched_encoder", "sdpa_attention", "flash_attention", "fp8_attention",
            "fp8_weights", "gemm_epilogue_fusion", "nvfp4_weights", "nvfp4_lm_head_only",
        ):
            if getattr(self, name) and not self.tolerance_mode:
                raise ValueError(f"{name}=True requires tolerance_mode=True")
        if self.fp8_attention:
            self.flash_attention = True
        if (self.fp8_weights or self.nvfp4_weights or self.nvfp4_lm_head_only
                or self.gemm_epilogue_fusion):
            self.fused_qkv = True


# ---------------------------------------------------------------------------
# process-global default flags
# ---------------------------------------------------------------------------
_DEFAULT_FLAGS = OptFlags()
# Reentrant: ``flags()`` must hold the lock across the *entire* yielded context
# (not just the snapshot/restore swap) so overlapping scopes on the same thread
# (e.g. a nested ``flags()`` inside another ``flags()``) cannot restore each
# other's snapshots.  A plain Lock would deadlock on such re-entry; RLock does not.
_FLAGS_LOCK = threading.RLock()


def get_default_flags() -> OptFlags:
    """Return the process-global default :class:`OptFlags` instance."""
    return _DEFAULT_FLAGS


def set_default_flags(fl: OptFlags) -> None:
    """Replace the process-global default flags."""
    global _DEFAULT_FLAGS
    with _FLAGS_LOCK:
        _DEFAULT_FLAGS = fl


@contextmanager
def flags(**overrides):
    """Temporarily override the global default flags within a ``with`` scope.

    Only the given keyword overrides change; all others inherit the current
    global default.  The original default is restored on exit (even on error).

    The RLock is held for the *whole* yielded body, serialising overlapping
    scopes: a concurrent ``flags()`` block on another thread waits until this
    scope exits before it can snapshot+override, so neither scope can observe
    or restore a snapshot belonging to the other.  On the same thread the
    reentrant lock permits clean nesting (inner exit restores the outer's
    override, outer exit restores the original).

    Example::

        with flags(tolerance_mode=True):
            ...  # approximate paths allowed here
        # previous defaults restored here
    """
    global _DEFAULT_FLAGS
    with _FLAGS_LOCK:
        saved = _DEFAULT_FLAGS
        new = replace(saved, **overrides)
        _DEFAULT_FLAGS = new
        try:
            yield new
        finally:
            _DEFAULT_FLAGS = saved
