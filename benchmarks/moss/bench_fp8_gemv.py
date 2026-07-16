"""A/B benchmark for MOSS FP8 decode GEMVs.

Compares the current dynamic-activation ``torch._scaled_mm`` path with the
weight-only fused Triton GEMV used by Granite.  The four shapes are the fused
QKV, attention output, fused gate/up, and MLP down projections in one MOSS
decoder layer.  All calls use M=1, matching autoregressive decode.

Run: ``uv run python -m benchmarks.moss.bench_fp8_gemv``
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from starling.fp8_gemv import (  # noqa: E402
    FP8_DTYPE,
    FP8_MAX,
    fp8_linear as triton_linear,
    quantize_weight_e4m3 as quantize_triton,
)
from starling.parakeet.gpu_lock import with_gpu_lock  # noqa: E402


SHAPES = {
    "qkv": (4096, 2048),
    "o": (2048, 2048),
    "gate_up": (12288, 2048),
    "down": (2048, 6144),
}


def quantize_scaled_mm(weight):
    """The replaced MOSS layout, retained as the benchmark baseline."""
    amax = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = amax / FP8_MAX
    weight_fp8 = (weight / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE).contiguous()
    return weight_fp8.t(), scale.reshape(1, -1).float()


def scaled_mm_linear(x, weight_fp8_kn, weight_scale):
    """The replaced dynamic-activation ``torch._scaled_mm`` call."""
    amax = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    x_scale = amax / FP8_MAX
    x_fp8 = (x / x_scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return torch._scaled_mm(
        x_fp8,
        weight_fp8_kn,
        scale_a=x_scale.float(),
        scale_b=weight_scale,
        out_dtype=torch.bfloat16,
    )


def _time_us(fn, *, warmup: int = 20, reps: int = 100) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(7):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / reps)
    return statistics.median(samples)


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    with with_gpu_lock(
        session="moss",
        model="synthetic-moss-fp8-gemv",
        eta_min=2,
        note="MOSS scaled_mm vs Triton FP8 GEMV A/B",
    ):
        torch.manual_seed(0)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"{'projection':<10} {'shape':>13} {'scaled_mm us':>13} "
              f"{'triton us':>10} {'speedup':>9} {'max |delta|':>12}")
        total_scaled = 0.0
        total_triton = 0.0
        for name, (out_features, in_features) in SHAPES.items():
            weight = torch.randn(
                out_features, in_features, device="cuda", dtype=torch.bfloat16
            ) * 0.02
            x = torch.randn(1, in_features, device="cuda", dtype=torch.bfloat16)
            scaled_mm_weight, scaled_mm_scale = quantize_scaled_mm(weight)
            triton_weight, triton_scale = quantize_triton(weight)
            scaled_mm_call = lambda: scaled_mm_linear(
                x, scaled_mm_weight, scaled_mm_scale
            )
            triton_call = lambda: triton_linear(x, triton_weight, triton_scale)
            scaled_mm_us = _time_us(scaled_mm_call)
            triton_us = _time_us(triton_call)
            scaled_mm_output = scaled_mm_call()
            triton_output = triton_call()
            delta = (
                scaled_mm_output.float() - triton_output.float()
            ).abs().max().item()
            total_scaled += scaled_mm_us
            total_triton += triton_us
            print(f"{name:<10} {str((out_features, in_features)):>13} "
                  f"{scaled_mm_us:>13.1f} {triton_us:>10.1f} "
                  f"{scaled_mm_us / triton_us:>8.2f}x "
                  f"{delta:>12.5f}")
            del weight, x, scaled_mm_weight, scaled_mm_scale
            del triton_weight, triton_scale
            torch.cuda.empty_cache()
        print(f"{'layer total':<24} {total_scaled:>13.1f} {total_triton:>10.1f} "
              f"{total_scaled / total_triton:>8.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
