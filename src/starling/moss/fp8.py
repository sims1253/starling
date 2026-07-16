"""MOSS compatibility exports for the shared fused FP8 dequant-GEMV.

MOSS decode projections always have one activation row.  The shared Triton
kernel streams row-major FP8 weights and keeps the bf16 activation unquantized,
avoiding both the activation-scaling launches and the general GEMM overhead of
``torch._scaled_mm``.  The lm_head remains bf16.

On an RTX 5090, ``benchmarks.moss.bench_fp8_gemv`` measures the four projection
shapes at 806.3 us/layer with ``_scaled_mm`` versus 266.3 us/layer here (3.03x).
The medium fixture's integrated decode improved from 306 to 405 token/s; a
35-second streaming fixture ran at 23.9x realtime with 0.00% WER.
"""

from ..fp8_gemv import FP8_DTYPE, FP8_MAX, fp8_linear, quantize_weight_e4m3

__all__ = ["FP8_DTYPE", "FP8_MAX", "fp8_linear", "quantize_weight_e4m3"]
