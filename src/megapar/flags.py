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
* ``quantized_weights`` (default **False**) -- enable weight-only INT8
  quantisation of the Granite LLM decoder weights (per-output-row channelwise
  scales) with a fused Triton dequant-GEMM, via :class:`megapar.quant.QuantLLMMega`
  / :class:`BatchedQuantLLMMega`.  **Breaks byte-exactness** (INT8 weight
  rounding).  Empirically this is currently **slower** than the bf16 cuBLAS path
  on the RTX 5090 (the dequant overhead + Triton's lower per-shape bandwidth
  efficiency eat the 2x weight-traffic reduction), so it ships only for
  completeness / future re-evaluation.  Requires ``tolerance_mode=True``.
* ``tolerance_mode`` (default **False**) -- master switch allowing
  ~5e-3 mean-abs numerical differences from byte-exactness-breaking
  optimisations (e.g. ``batched_encoder``).  When False the pipeline must be
  end-to-end byte-exact with the golden reference.

Usage
-----
::

    from megapar.flags import OptFlags, flags, get_default_flags

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

    quantized_weights: bool = False
    """Enable weight-only INT8 quantisation of the LLM decoder weights
    (:mod:`megapar.quant`).  **Breaks byte-exactness** (INT8 weight rounding).
    Currently slower than bf16 on the RTX 5090 -- shipped for completeness.
    Requires ``tolerance_mode=True``."""

    tolerance_mode: bool = False
    """Master switch: allow ~5e-3 mean-abs numerical differences.  When False
    the pipeline is end-to-end byte-exact with the golden reference."""

    def __post_init__(self) -> None:
        """Validate flag combinations at construction time."""
        if self.batched_encoder and not self.tolerance_mode:
            raise ValueError(
                "batched_encoder=True requires tolerance_mode=True (it breaks "
                "byte-exactness). Set tolerance_mode=True or batched_encoder=False."
            )
        if self.quantized_weights and not self.tolerance_mode:
            raise ValueError(
                "quantized_weights=True requires tolerance_mode=True (INT8 weight "
                "rounding breaks byte-exactness). Set tolerance_mode=True or "
                "quantized_weights=False."
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
        quantized_weights=overrides.get("quantized_weights", saved.quantized_weights),
        tolerance_mode=overrides.get("tolerance_mode", saved.tolerance_mode),
    )
    _DEFAULT_FLAGS = new
    try:
        yield new
    finally:
        _DEFAULT_FLAGS = saved
