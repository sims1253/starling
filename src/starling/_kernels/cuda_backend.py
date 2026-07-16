"""CUDA C++ kernel backend for cross-platform fused decode kernels.

This is the **third** backend, selected automatically where triton is
unavailable (Windows) but a CUDA toolkit + compiler is present.  It gives
Windows the same fused-kernel performance as Linux's Triton path: the kernels
are compiled from ``cuda/backend.cu`` via
:func:`torch.utils.cpp_extension.load_inline` on first use (JIT, cached under
``~/.cache/starling``), then reused.

Why a CUDA backend at all?
--------------------------
On the torch (stock-PyTorch) backend the three elementwise kernels are correct
and byte-exact, but unfused -- each issues 3-4 separate kernels.  Across a
28-layer Moss decode step that costs ~0.85 ms/step (see
``benchmarks/bench_kernels.py``), and the fp8 dequant-GEMV is 6-16x slower than
the Triton fused version.  The CUDA backend closes both gaps with single-launch
fused kernels, recovering full Linux/Triton performance on Windows.

Compilation
-----------
The first ``import``/call triggers a one-time nvcc compile (10-60 s depending on
the machine); the shared object is cached, so subsequent runs are instant.  If
compilation fails (no CUDA toolkit / no compiler), :func:`get_backend` falls
back to the torch backend automatically (see ``__init__.py``).  On Linux the
triton backend remains the default (faster autotuning, no first-run compile).

Backend selection
-----------------
* Linux + triton        -> triton (default, max perf)
* Windows + CUDA toolkit -> cuda (full perf; this module)
* Windows, no toolkit    -> torch (correctness; slower fp8)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch

from .base import FP8_DTYPE, FP8_MAX  # noqa: F401  (re-exported public constants)

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


# ---------------------------------------------------------------------------
# Public fused ops (same signatures as triton_backend / torch_backend)
# ---------------------------------------------------------------------------

def fused_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over the last dim, fp32 internally, bf16 in/out (CUDA fused)."""
    N = weight.numel()
    M = x.numel() // N
    x2 = x.reshape(M, N)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    y = _ext().fused_rmsnorm(x2, weight, float(eps))
    return y.view_as(x)


def fused_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SiLU(gate) * up fused into one kernel, fp32 internally (CUDA fused)."""
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
    return _ext().fp8_linear(x, w_fp8, w_scale)


def fused_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embedding to Q and K in one kernel launch (CUDA).

    Mirrors the triton/torch launchers: flatten q/k to (B*heads, hd), take the
    single seq=1 position from cos/sin, and require fp32 cos/sin for the kernel.
    """
    B, n_q, _, hd = q.shape
    n_kv = k.shape[1]
    q_flat = q.reshape(B * n_q, hd).contiguous()
    k_flat = k.reshape(B * n_kv, hd).contiguous()
    cos_flat = cos.reshape(-1, hd)[0:1].reshape(hd).to(torch.float32).contiguous()
    sin_flat = sin.reshape(-1, hd)[0:1].reshape(hd).to(torch.float32).contiguous()
    qo, ko = _ext().fused_rope(q_flat, k_flat, cos_flat, sin_flat)
    return qo.view_as(q), ko.view_as(k)


def compute_rstd(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Scalar rstd = rsqrt(mean(x^2)+eps) as a (1,) fp32 tensor (CUDA)."""
    return _ext().compute_rstd(x, float(eps))


def fused_gemv_normscale(
    x: torch.Tensor, w_scaled: torch.Tensor, rstd: torch.Tensor
) -> torch.Tensor:
    """GEMV (M=1) of x @ w_scaled^T with rstd folded into the epilogue (CUDA)."""
    return _ext().fused_gemv_normscale(x, w_scaled, rstd)


def fp4_gemv_fused(
    x: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """Fused NVFP4 dequant-GEMV (M=1): streams nibble-packed codes + fp8 scales (CUDA)."""
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
