"""Granite compatibility exports for the shared fused FP8 GEMV."""

from ..fp8_gemv import FP8_DTYPE, FP8_MAX, fp8_linear, quantize_weight_e4m3

__all__ = ["FP8_DTYPE", "FP8_MAX", "fp8_linear", "quantize_weight_e4m3"]
