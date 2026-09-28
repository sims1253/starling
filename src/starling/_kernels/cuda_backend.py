"""CUDA C++ fused kernels compiled from ``cuda/backend.cu``.

The dispatcher initializes this extension before returning the CUDA backend.
Direct imports compile on the first operation. PyTorch caches the build in
``STARLING_CUDA_BUILD_DIR`` or ``~/.cache/starling/cuda_ext``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch

from .base import FP8_DTYPE, FP8_MAX

_CUDA_SRC = Path(__file__).resolve().parent / "cuda" / "backend.cu"


@lru_cache(maxsize=1)
def _module():
    """JIT-compile the CUDA extension (cached across the process)."""
    from torch.utils.cpp_extension import load

    cache_dir = os.environ.get(
        "STARLING_CUDA_BUILD_DIR",
        str(Path.home() / ".cache" / "starling" / "cuda_ext"),
    )
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    ext = load(
        name="starling_cuda_kernels",
        sources=[str(_CUDA_SRC)],
        build_directory=cache_dir,
        verbose=False,
        with_cuda=True,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )
    return ext


def _ext():
    """Return the compiled extension (compiles on first call)."""
    return _module()


def _require_nonempty(*tensors: torch.Tensor) -> None:
    """Reject empty launches before C++ divides by a dimension or launches grid 0."""
    if any(t.numel() == 0 for t in tensors):
        raise ValueError("CUDA fused kernels require non-empty tensors")


def _require_cuda(*tensors: torch.Tensor) -> None:
    if any(not t.is_cuda for t in tensors):
        raise ValueError("CUDA fused kernels require CUDA tensors")


def _require_row_dimension(tensor: torch.Tensor) -> None:
    if tensor.ndim == 0:
        raise ValueError("CUDA fused kernels require a tensor with a final dimension")


# ---------------------------------------------------------------------------
# Public fused ops (same signatures as triton_backend / torch_backend)
# ---------------------------------------------------------------------------

def fused_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over the last dim, fp32 internally, bf16 in/out (CUDA fused)."""
    _require_nonempty(x, weight)
    _require_row_dimension(x)
    N = weight.numel()
    if x.shape[-1] != N or weight.ndim != 1:
        raise ValueError("CUDA fused_rmsnorm requires x.shape[-1] == weight.numel() and 1D weight")
    if not weight.is_contiguous():
        raise ValueError("CUDA fused_rmsnorm requires contiguous weight")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("CUDA fused_rmsnorm requires bf16 x and weight")
    _require_cuda(x, weight)
    M = x.numel() // N
    x2 = x.reshape(M, N)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    y = _ext().fused_rmsnorm(x2, weight, float(eps))
    return y.view_as(x)


def fused_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SiLU(gate) * up fused into one kernel, fp32 internally (CUDA fused)."""
    _require_nonempty(gate, up)
    _require_row_dimension(gate)
    if gate.shape != up.shape:
        raise ValueError("CUDA fused_silu_mul requires gate and up to have the same shape")
    if gate.dtype != torch.bfloat16 or up.dtype != torch.bfloat16:
        raise ValueError("CUDA fused_silu_mul requires bf16 gate and up")
    _require_cuda(gate, up)
    N = gate.shape[-1]
    M = gate.numel() // N
    g2 = gate.reshape(M, N)
    u2 = up.reshape(M, N)
    if not g2.is_contiguous():
        g2 = g2.contiguous()
    if not u2.is_contiguous():
        u2 = u2.contiguous()
    out = _ext().fused_silu_mul(g2, u2)
    return out.view_as(gate)


def residual_add(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """x + alpha*y fused (CUDA). alpha=1.0 fast path is plain x+y."""
    _require_nonempty(x, y)
    _require_row_dimension(x)
    if x.shape != y.shape:
        raise ValueError("CUDA residual_add requires x and y to have the same shape")
    if x.dtype != torch.bfloat16 or y.dtype != torch.bfloat16:
        raise ValueError("CUDA residual_add requires bf16 x and y")
    _require_cuda(x, y)
    N = x.shape[-1]
    M = x.numel() // N
    x2 = x.reshape(M, N)
    y2 = y.reshape(M, N)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    if not y2.is_contiguous():
        y2 = y2.contiguous()
    z = _ext().residual_add(x2, y2, float(alpha))
    return z.view_as(x)


def quantize_weight_e4m3(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel symmetric absmax fp8 quantization (shared recipe)."""
    amax = weight.abs().amax(dim=1).clamp(min=1e-8)
    scale = amax / FP8_MAX
    w_fp8 = (weight / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE).contiguous()
    return w_fp8, scale.float()


def fp8_linear(x: torch.Tensor, w_fp8: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    """x @ W^T with an fp8 weight via the fused dequant-GEMV (CUDA)."""
    _require_nonempty(x, w_fp8, w_scale)
    _require_row_dimension(x)
    if (w_fp8.ndim != 2 or x.numel() != w_fp8.shape[1] or x.shape[-1] != w_fp8.shape[1]
            or w_scale.numel() != w_fp8.shape[0]):
        raise ValueError("CUDA fp8_linear requires one x row of length K and one scale per output row")
    if not w_fp8.is_contiguous() or not w_scale.is_contiguous():
        raise ValueError("CUDA fp8_linear requires contiguous weights and scales")
    if x.dtype != torch.bfloat16 or w_fp8.dtype != FP8_DTYPE or w_scale.dtype != torch.float32:
        raise ValueError("CUDA fp8_linear requires bf16 x, fp8 weight and fp32 scale")
    _require_cuda(x, w_fp8, w_scale)
    return _ext().fp8_linear(x, w_fp8, w_scale)


def fused_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embedding to Q and K in one kernel launch (CUDA).

    Mirrors the triton/torch launchers: flatten q/k to (B*heads, hd), take the
    single seq=1 position from cos/sin, and require fp32 cos/sin for the kernel.
    """
    _require_nonempty(q, k, cos, sin)
    _require_row_dimension(cos)
    _require_row_dimension(sin)
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("CUDA fused_rope requires four-dimensional q and k")
    B, n_q, _, hd = q.shape
    if hd not in (64, 128) or k.shape[-1] != hd or cos.shape[-1] != hd or sin.shape[-1] != hd:
        raise ValueError("CUDA fused_rope requires head_dim=64 or 128 for q, k, cos and sin")
    if q.shape[0] != k.shape[0] or q.shape[2] != 1 or k.shape[2] != 1:
        raise ValueError("CUDA fused_rope requires matching batch sizes and one decode position")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise ValueError("CUDA fused_rope requires bf16 q and k")
    _require_cuda(q, k, cos, sin)
    n_kv = k.shape[1]
    q_flat = q.reshape(B * n_q, hd).contiguous()
    k_flat = k.reshape(B * n_kv, hd).contiguous()
    cos_flat = cos.reshape(-1, hd)[0:1].reshape(hd).to(torch.float32).contiguous()
    sin_flat = sin.reshape(-1, hd)[0:1].reshape(hd).to(torch.float32).contiguous()
    qo, ko = _ext().fused_rope(q_flat, k_flat, cos_flat, sin_flat)
    return qo.view_as(q), ko.view_as(k)


def compute_rstd(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Scalar rstd = rsqrt(mean(x^2)+eps) as a (1,) fp32 tensor (CUDA)."""
    _require_nonempty(x)
    _require_row_dimension(x)
    if x.numel() != x.shape[-1]:
        raise ValueError("CUDA compute_rstd requires one row")
    if x.dtype != torch.bfloat16:
        raise ValueError("CUDA compute_rstd requires bf16 x")
    _require_cuda(x)
    return _ext().compute_rstd(x, float(eps))


def fused_gemv_normscale(
    x: torch.Tensor, w_scaled: torch.Tensor, rstd: torch.Tensor
) -> torch.Tensor:
    """GEMV (M=1) of x @ w_scaled^T with rstd folded into the epilogue (CUDA)."""
    _require_nonempty(x, w_scaled, rstd)
    _require_row_dimension(x)
    if (w_scaled.ndim != 2 or x.numel() != w_scaled.shape[1]
            or x.shape[-1] != w_scaled.shape[1] or rstd.numel() != 1):
        raise ValueError("CUDA fused_gemv_normscale requires one x row of length K and scalar rstd")
    if not w_scaled.is_contiguous():
        raise ValueError("CUDA fused_gemv_normscale requires contiguous weights")
    if x.dtype != torch.bfloat16 or w_scaled.dtype != torch.bfloat16 or rstd.dtype != torch.float32:
        raise ValueError("CUDA fused_gemv_normscale requires bf16 x/weight and fp32 rstd")
    _require_cuda(x, w_scaled, rstd)
    return _ext().fused_gemv_normscale(x, w_scaled, rstd)


def fp4_gemv_fused(
    x: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """Fused NVFP4 dequant-GEMV (M=1): streams nibble-packed codes + fp8 scales (CUDA)."""
    _require_nonempty(x, codes, scales)
    _require_row_dimension(x)
    if (codes.ndim != 2 or codes.shape[1] % 8 or x.numel() != codes.shape[1] * 2
            or x.shape[-1] != codes.shape[1] * 2):
        raise ValueError("CUDA fp4_gemv_fused requires K divisible by 16 and one x row of length K")
    if scales.shape != (codes.shape[0], codes.shape[1] // 8):
        raise ValueError("CUDA fp4_gemv_fused requires one scale per 16 values")
    if not codes.is_contiguous() or not scales.is_contiguous():
        raise ValueError("CUDA fp4_gemv_fused requires contiguous codes and scales")
    if x.dtype != torch.bfloat16 or codes.dtype != torch.uint8 or scales.dtype != FP8_DTYPE:
        raise ValueError("CUDA fp4_gemv_fused requires bf16 x, uint8 codes and fp8 scales")
    _require_cuda(x, codes, scales)
    return _ext().fp4_gemv_fused(x, codes, scales)


# Autotune control: CUDA kernels use fixed launch configs (no autotune sweep),
# so report False for API compatibility.
AUTOTUNE = False


def set_autotune(enabled: bool) -> None:
    """No-op: the CUDA backend uses fixed launch configs (no autotuning)."""


def autotune_enabled() -> bool:
    """Always False for the CUDA backend (fixed launch configs)."""
    return False


__all__ = [
    "fused_rmsnorm",
    "fused_silu_mul",
    "residual_add",
    "fused_rope",
    "quantize_weight_e4m3",
    "fp8_linear",
    "compute_rstd",
    "fused_gemv_normscale",
    "fp4_gemv_fused",
    "set_autotune",
    "autotune_enabled",
    "AUTOTUNE",
    "FP8_DTYPE",
    "FP8_MAX",
]
