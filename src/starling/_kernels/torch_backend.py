"""Cross-platform PyTorch fallback backend for Starling's fused decode kernels.

This is the **correctness-first, stock-PyTorch** implementation of the public
kernel interface defined in :mod:`starling._kernels.base`.  It is selected
automatically on platforms where the ``triton`` package is unavailable -- most
notably **Windows**, for which Triton publishes no official wheels -- letting
the same consumer code (``from . import llm_kernels as _k``) run unchanged.

Design goals
------------
1. **No triton dependency.**  This module must import successfully on a machine
   without ``triton`` installed.  It uses only stock ``torch`` ops.
2. **Match the eager PyTorch reference exactly.**  The byte-exactness goldens are
   captured from the *eager* path (what ``transformers`` does), and the hand-
   written Triton kernels in :mod:`starling.granite.llm_kernels` were carefully
   written to replicate that eager rounding.  Matching the recipes below is
   therefore matching both.  The load-bearing detail throughout is the
   **intermediate bf16 truncation order**: we normalize/scale in fp32, truncate
   to bf16 *before* the next op, exactly as the model's eager code does.
3. **Same public interface** as the triton backend, so the dispatch in
   :mod:`starling._kernels.__init__` can delegate transparently.

What this backend is NOT
------------------------
It is not a performance backend.  The FP8/FP4 GEMV paths here are
**dequant-then-matmul** "correctness paths": they materialize the full bf16
weight, so they do NOT realize the bandwidth win of the triton fused kernels.
On Windows prefer leaving ``OptFlags.fp8_weights`` off (bf16 GEMV via
``F.linear``) unless the slower correctness path is acceptable.  The
elementwise ops (RMSNorm/SiLU/residual/RoPE) are already near-free at decode
shapes, so fusing them buys little; correctness is what matters there.

Autotune control (``AUTOTUNE``, ``set_autotune``, ``autotune_enabled``) is
exposed for API compatibility only -- there is no autotuning concept in stock
PyTorch, so ``set_autotune`` is a no-op and ``autotune_enabled`` always returns
``False``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Re-export the canonical FP8 (e4m3) constants from the shared base so that
# every backend module and every consumer reads the same source of truth.
from .base import FP8_DTYPE, FP8_MAX  # noqa: F401  (re-exported public names)

__all__ = [
    # constants (re-exported)
    "FP8_DTYPE",
    "FP8_MAX",
    # elementwise decode kernels
    "fused_rmsnorm",
    "fused_silu_mul",
    "residual_add",
    "fused_rope",
    # granite GEMM-epilogue fusion (experimental)
    "compute_rstd",
    "fused_gemv_normscale",
    # FP8 weight-only decode GEMV (opt-in)
    "quantize_weight_e4m3",
    "fp8_linear",
    # FP4 fused GEMV (granite, experimental) -- correctness path
    "fp4_gemv_fused",
    # autotune control (no-ops in this backend)
    "AUTOTUNE",
    "set_autotune",
    "autotune_enabled",
]


# =========================================================================== #
# Autotune control -- NO-OPS in the torch backend.
#
# There is no autotuning concept in stock PyTorch, so these exist purely for
# API compatibility with the triton backend (which sweeps num_warps/num_stages).
# =========================================================================== #
AUTOTUNE: bool = False


def set_autotune(enabled: bool) -> None:
    """No-op.  Stock PyTorch has no autotuning concept; kept for API compat."""
    # Deliberately does nothing -- there is no AUTOTUNE flag to flip here.  The
    # triton backend's set_autotune toggles its kernel autotuner; this backend
    # has nothing equivalent, so we just accept and discard the argument.
    return None


def autotune_enabled() -> bool:
    """Always returns ``False`` -- the torch backend never autotunes."""
    return False


# =========================================================================== #
# Elementwise decode kernels
# =========================================================================== #
def fused_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over the last dim, fp32 internally, bf16 out.

    Matches GraniteRMSNorm / Qwen3RMSNorm exactly: compute the variance in fp32,
    normalize, **truncate the normalized hidden to bf16**, THEN multiply by the
    bf16 weight.  Doing the weight product in fp32 and truncating once gives a
    different (wrong-vs-golden) result.

    ``x`` is ``(*, N)`` with ``N == weight.numel()``; arbitrary leading dims are
    handled via last-dim broadcasting (equivalent to the triton launcher's
    reshape-to-(M,N)-then-view_as behavior, since the reduction is per-row).
    """
    x_f32 = x.float()
    var = x_f32.pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    x_normed = (x_f32 * rstd).to(x.dtype)  # truncate to bf16 (matches model RMSNorm)
    return x_normed * weight               # bf16 * bf16


def fused_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU ``silu(gate) * up``, fp32 silu, bf16 out.

    Matches ATen's intermediate truncation: compute SiLU in fp32, **truncate to
    bf16 BEFORE multiplying by ``up``**.  Computing the whole op in fp32 and
    truncating once gives different rounding than the eager reference.

    ``gate`` and ``up`` are ``(*, N)`` with the same shape.
    """
    silu_g = F.silu(gate.float()).to(gate.dtype)  # fp32 silu, truncate to bf16
    return silu_g * up                            # bf16 * bf16


def residual_add(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """``x + alpha * y`` residual connection, bf16 in/out.

    Matches the model's ``residual + delta * multiplier``: when ``alpha == 1.0``
    (the Qwen3 / MOSS case) take the fast path of a plain ``x + y``.  Otherwise
    compute the scaled delta in fp32, **truncate to bf16, THEN add the residual**
    (also bf16).  This matches the triton ``_residual_scale_kernel`` truncation
    order; computing ``x + alpha*y`` wholly in fp32 and truncating once differs.
    """
    if alpha == 1.0:
        return x + y
    scaled = (y.float() * alpha).to(y.dtype)  # fp32 scale, truncate to bf16
    return x + scaled


def fused_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embedding to Q and K.

    Args:
        q: ``(B, n_q_heads, 1, head_dim)`` bf16.
        k: ``(B, n_kv_heads, 1, head_dim)`` bf16.
        cos, sin: ``(B, 1, 1, head_dim)`` or broadcastable to ``(1, head_dim)``.

    Returns:
        ``(q_rotated, k_rotated)`` with the same shapes/dtype as the inputs.

    Matches the triton ``_rope_kernel``: ``rotate_half(x) = cat(-x[d/2:], x[:d/2])``,
    and each product (``x*cos``, ``rotate_half(x)*sin``) is **truncated to bf16
    BEFORE the two are added**.  As in the triton launcher, the single decode
    position is taken from ``cos``/``sin`` via ``cos.reshape(-1, hd)[0:1]`` and
    applied identically to every head.
    """
    B, n_q, _, hd = q.shape
    n_kv = k.shape[1]
    q_dtype = q.dtype

    # Flatten to (B * heads, hd) per tensor, matching the triton launcher.
    q_flat = q.reshape(B * n_q, hd)
    k_flat = k.reshape(B * n_kv, hd)

    # Take the single decode position (seq=1) from cos/sin, exactly as the
    # triton launcher does: reshape to (-1, hd), take row [0:1] -> (1, hd),
    # broadcast across all (B*heads) rows.
    cos_flat = cos.reshape(-1, hd)[0:1].to(q_dtype)  # (1, hd)
    sin_flat = sin.reshape(-1, hd)[0:1].to(q_dtype)  # (1, hd)

    def _rotate_half(t: torch.Tensor) -> torch.Tensor:
        half = t.shape[-1] // 2
        # rotate_half(x) = cat(-x[d/2:], x[:d/2]) along the last dim.
        return torch.cat((-t[..., half:], t[..., :half]), dim=-1)

    def _apply(t: torch.Tensor, c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        t_rot = _rotate_half(t)
        # Truncate each product to bf16 BEFORE adding (matches _rope_kernel).
        prod1 = (t * c).to(q_dtype)
        prod2 = (t_rot * s).to(q_dtype)
        return prod1 + prod2

    q_out = _apply(q_flat, cos_flat, sin_flat)
    k_out = _apply(k_flat, cos_flat, sin_flat)
    return q_out.view_as(q), k_out.view_as(k)


# =========================================================================== #
# Granite GEMM-epilogue fusion (experimental)
#
# These two ops let the RMSNorm scale commute into the next M=1 GEMV:
#   rmsnorm(x) @ W^T  ==  (x @ (gamma .* W)^T) * rstd
# ``compute_rstd`` produces the scalar rstd; ``fused_gemv_normscale`` is the
# GEMV with rstd folded into the epilogue.  See llm_kernels.py for the (non-byte-
# exact) numerical note -- these match the triton math, materializing the matmul.
# =========================================================================== #
def compute_rstd(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Scalar ``rstd = rsqrt(mean(x^2) + eps)`` as a ``(1,)`` fp32 tensor.

    ``x`` is ``(N,)`` or ``(1, N)`` (a single decode row).  Matches the triton
    ``_rstd_kernel``: flatten, fp32 sum-of-squares / N, rsqrt(+ eps).
    """
    x1 = x.reshape(-1)
    var = x1.float().pow(2).mean()
    return torch.rsqrt(var + eps).reshape(1)  # (1,) fp32


def fused_gemv_normscale(
    x: torch.Tensor, w_scaled: torch.Tensor, rstd: torch.Tensor
) -> torch.Tensor:
    """M=1 GEMV of ``x @ w_scaled^T`` with the RMSNorm rstd folded into the epilogue.

    Args:
        x: ``(1, K)`` or ``(K,)`` activation.
        w_scaled: ``(OUT, K)`` weight already prescaled by gamma.
        rstd: ``(1,)`` fp32 scalar from :func:`compute_rstd`.

    Returns:
        ``(OUT,)`` tensor matching ``w_scaled.dtype``.

    This is the correctness path: it materializes the matmul normally via
    ``F.linear`` and folds rstd into the fp32 epilogue, matching the triton
    ``_gemv_normscale_kernel`` math (``out = rstd * (x @ w_scaled^T)``).
    """
    x1 = x.reshape(-1)
    # x1.unsqueeze(0) is (1, K); F.linear computes x1 @ w_scaled^T -> (1, OUT).
    out = F.linear(x1.unsqueeze(0), w_scaled).squeeze(0)  # (OUT,)
    return (out.float() * rstd).to(out.dtype)             # fold rstd


# =========================================================================== #
# FP8 weight-only decode GEMV (opt-in via OptFlags.fp8_weights)
# =========================================================================== #
def quantize_weight_e4m3(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize an ``(N, K)`` linear weight to fp8e4m3, per-output-channel.

    Per-output-channel symmetric absmax scaling.  Matches
    :func:`starling.fp8_gemv.quantize_weight_e4m3` EXACTLY (same clamp floor,
    same scale formula, same fp8 cast) so the dequant path is bit-consistent.

    Returns:
        ``(w_fp8, scale)`` where ``w_fp8`` is a row-major ``(N, K)`` fp8e4m3
        tensor and ``scale`` is the ``(N,)`` fp32 per-output-channel dequant
        scale.
    """
    amax = weight.abs().amax(dim=1).clamp(min=1e-8)  # (N,)
    scale = amax / FP8_MAX                           # (N,)
    w_fp8 = (weight / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE).contiguous()
    return w_fp8, scale.float()


def fp8_linear(
    x: torch.Tensor, w_fp8: torch.Tensor, w_scale: torch.Tensor
) -> torch.Tensor:
    """``x @ W^T`` with an fp8 weight, M=1 GEMV.

    CORRECTNESS PATH -- dequant-then-matmul.  Materializes the full bf16 weight
    so it does NOT realize the bandwidth win of the triton fused kernel.  Used on
    Windows / when triton is absent.  The fp8 path is opt-in
    (``OptFlags.fp8_weights``); on Windows prefer leaving it off (bf16 GEMV via
    ``F.linear``) unless this correctness path is acceptable.

    Args:
        x: ``(1, K)`` or ``(K,)`` bf16 activation.
        w_fp8: ``(N, K)`` row-major fp8 weight from :func:`quantize_weight_e4m3`.
        w_scale: ``(N,)`` or ``(1, N)`` fp32 per-output-channel scale.

    Returns:
        ``(1, N)`` bf16 result, matching ``F.linear(x, W)`` to fp8 precision.
    """
    K = w_fp8.shape[1]
    N = w_fp8.shape[0]
    x1 = x.reshape(-1)[:K].contiguous()
    scale1d = w_scale.reshape(-1)
    # Dequant fp8 -> bf16: w_bf16 = w_fp8.to(bf16) * scale[:, None].
    w_bf16 = w_fp8.to(torch.bfloat16) * scale1d[:, None].to(torch.bfloat16)
    out = F.linear(x1.unsqueeze(0), w_bf16).squeeze(0)  # (N,) = x @ w_bf16^T
    return out.view(1, N).to(torch.bfloat16)


# =========================================================================== #
# FP4 fused GEMV (granite, experimental) -- CORRECTNESS PATH
#
# Nibble-packed NVFP4 storage (see starling.granite.fp4.quantize_fp4_packed):
#   codes  : (OUT, K // 2) uint8 -- even k -> low nibble, odd k -> high nibble
#   scales : (OUT, K // 16) float8_e4m3fn -- one fp8 scale per 16-elem block
# Reconstruction:  w ~= scale_fp8 * e2m1_level(code) / 6.0
#   where e2m1_level uses bit 3 as sign and bits[2:0] indexing the 8 positive
#   magnitude levels (0, 0.5, 1, 1.5, 2, 3, 4, 6).
# =========================================================================== #
_FP4_LEVELS: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_FP4_BLOCK_SIZE: int = 16


def fp4_gemv_fused(
    x: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """M=1 GEMV with a nibble-packed NVFP4 weight.

    CORRECTNESS PATH -- dequant-then-matmul.  Unpacks the nibble codes, dequants
    to bf16 via ``w ~= scale_fp8 * e2m1_level(code) / 6.0``, then runs
    ``F.linear``.  This materializes the full bf16 weight so it does NOT realize
    the bandwidth win of the triton fused kernel.  Used on Windows / when triton
    is absent; the recipe matches :mod:`starling.granite.fp4` and the triton
    ``_fp4_gemv_kernel`` exactly (same levels, same sign bit, same /6.0).

    Args:
        x: ``(1, K)`` or ``(K,)`` bf16 activation.
        codes: ``(OUT, K // 2)`` uint8 nibble-packed e2m1 codes (even ``k`` ->
            low nibble, odd ``k`` -> high nibble).
        scales: ``(OUT, K // 16)`` ``float8_e4m3fn`` block scales.

    Returns:
        ``(OUT,)`` bf16 result.
    """
    K_half = codes.shape[1]
    K = K_half * 2
    OUT = codes.shape[0]

    # --- unpack nibbles -> (OUT, K) e2m1 codes ---
    raw = codes.to(torch.int32)            # (OUT, K//2)
    lo = raw & 0xF                          # even k  (low nibble)
    hi = (raw >> 4) & 0xF                   # odd  k  (high nibble)
    # Interleave [lo[0], hi[0], lo[1], hi[1], ...] -> (OUT, K).
    codes_flat = torch.stack([lo, hi], dim=-1).reshape(OUT, K).to(torch.int32)

    # --- e2m1 dequant: bit 3 = sign, bits[2:0] = magnitude level index ---
    levels_t = x.new_tensor(_FP4_LEVELS, dtype=torch.float32)  # (8,) on x's device
    idx = (codes_flat & 0x7).long()                            # (OUT, K)
    sign = ((codes_flat >> 3) & 0x1).bool()                    # (OUT, K)
    mag = levels_t[idx]                                        # (OUT, K) fp32
    level = torch.where(sign, -mag, mag)                       # (OUT, K) fp32 in [-6, 6]

    # --- per-block fp8 scale: each scale covers 16 consecutive elements ---
    scale_f32 = scales.to(torch.float32)                       # (OUT, K//16)
    # Expand block scales to per-element: repeat each scale 16x along dim 1.
    scale_exp = scale_f32.repeat_interleave(_FP4_BLOCK_SIZE, dim=1)  # (OUT, K)

    w = scale_exp * level * (1.0 / 6.0)                        # (OUT, K) fp32
    w_bf16 = w.to(torch.bfloat16)

    # --- M=1 GEMV: out = x @ w_bf16^T ---
    x1 = x.reshape(-1)[:K].contiguous()
    out = F.linear(x1.unsqueeze(0), w_bf16).squeeze(0)         # (OUT,)
    return out
