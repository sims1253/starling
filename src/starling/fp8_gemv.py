"""Shared FP8 (e4m3) weight-only quantization for M=1 decoder GEMVs.

Decode is memory-bandwidth bound on the LLM weights: the per-layer projection
GEMVs (qkv/o/gateup/down) are ~57% of the captured decode step, each a pure
weight read at batch=1.  Casting those weights to fp8e4m3 halves the weight
traffic -- this module provides a **fused dequant-GEMV** that realizes that
bandwidth win.

This module is now a thin compatibility shim over :mod:`starling._kernels`,
which dispatches to the Triton fused GEMV (Linux) or a stock-PyTorch
dequant-then-:func:`~torch.nn.functional.linear` correctness path (Windows).
The public API is unchanged so existing callers and tests that import directly
from ``starling.fp8_gemv`` keep working::

    from starling.fp8_gemv import fp8_linear, quantize_weight_e4m3, FP8_DTYPE, FP8_MAX

Performance note
----------------
On Linux (triton backend) this is the purpose-built fused GEMV described in the
original docstring: streams the fp8 weight, dequantizes each element to fp32 in
registers, accumulates the dot product in fp32.  On Windows (torch backend) it
dequantizes the full weight to bf16 and calls cuBLAS -- correct but it does NOT
realize the bandwidth win, so on Windows prefer leaving ``fp8_weights`` off
(the default) unless this correctness path is acceptable.  The fp8 path is
opt-in via :attr:`starling.flags.OptFlags.fp8_weights`.
"""

from __future__ import annotations

from ._kernels import FP8_DTYPE, FP8_MAX, fp8_linear, quantize_weight_e4m3

__all__ = ["FP8_DTYPE", "FP8_MAX", "fp8_linear", "quantize_weight_e4m3"]
