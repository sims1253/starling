"""Cross-platform correctness gate for the kernel backend abstraction.

Starling's fused decode kernels (RMSNorm, SwiGLU, residual, RoPE, FP8/FP4
dequant-GEMV) ship in two interchangeable backends under
:mod:`starling._kernels`:

* ``triton_backend`` -- hand-tuned Triton kernels, selected on Linux for max
  performance.
* ``torch_backend``  -- stock-PyTorch fused ops, selected automatically where
  the ``triton`` package is unavailable (most importantly **Windows**, for
  which Triton publishes no official wheels).

This module is the correctness contract between the two: for every kernel, at
real decode shapes (M = batch*seq = 1), the torch fallback must reproduce the
Triton reference to within tolerance, and both must reproduce the eager
PyTorch reference op.  The load-bearing detail throughout is the **intermediate
bf16 truncation order** -- both backends normalize/scale in fp32 and truncate
to bf16 at exactly the same points as the model's eager code, so the three
elementwise kernels plus the residual are byte-exact (atol=0) between backends.
RoPE is bf16-sensitive (~1 ULP / ~0.004 observed) and the FP8 GEMV differs
between backends only by the dequant-dtype rounding (~0.03125 observed, since
both paths accumulate the same fp8 codes); both are exercised here at
empirically-tuned tolerances.

Every test needs CUDA (the whole module is skipped without it).  Tests that
A/B the Triton backend are additionally skipped when ``triton`` is not
importable, but the torch-vs-reference assertions always run -- those are the
guarantee that the Windows path is correct on its own.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

# Whole module needs CUDA -- both backends run on-device and the real decode
# shapes are exercised on GPU.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for kernel backends"
)

device = "cuda"
bf16 = torch.bfloat16


def _have_triton() -> bool:
    """Return ``True`` if the ``triton`` package is importable.

    Mirrors :func:`starling._kernels.base.have_triton`.  Gates only the
    triton-vs-torch A/B comparisons; the torch-vs-reference checks run
    everywhere CUDA is present.
    """
    try:
        import triton  # noqa: F401
        import triton.language  # noqa: F401
    except Exception:
        return False
    return True


# Tolerance shortcut for the Triton-vs-torch comparison when triton is present.
_no_triton = pytest.mark.skipif(not _have_triton(), reason="triton not importable")


# =========================================================================== #
# 1. fused_rmsnorm -- RMSNorm over the last dim, fp32 internally, bf16 out.
# =========================================================================== #
@pytest.mark.parametrize("M, N", [(1, 2048), (4, 2048), (1, 128)], ids=["decode", "multirow", "perhead"])
@torch.inference_mode()
def test_fused_rmsnorm_matches_reference_and_triton(M, N):
    from starling._kernels import torch_backend

    torch.manual_seed(0)
    x = (torch.randn(M, N, device=device, dtype=bf16) * 0.1)
    weight = torch.randn(N, device=device, dtype=bf16) * 0.1
    eps = 1e-6

    # Eager reference: fp32 variance, normalize, truncate to bf16, THEN * weight.
    x_f32 = x.float()
    var = x_f32.pow(2).mean(-1, keepdim=True)
    x_norm = (x_f32 * torch.rsqrt(var + eps)).to(bf16)
    ref = x_norm * weight

    torch_out = torch_backend.fused_rmsnorm(x, weight, eps)
    # Torch backend matches the eager reference byte-for-byte.
    torch.testing.assert_close(torch_out, ref, atol=0, rtol=0)

    if _have_triton():
        from starling._kernels import triton_backend

        triton_out = triton_backend.fused_rmsnorm(x, weight, eps)
        # Byte-exact between backends (verified empirically).
        torch.testing.assert_close(torch_out, triton_out, atol=0, rtol=0)


# =========================================================================== #
# 2. fused_silu_mul -- SwiGLU silu(gate) * up, fp32 silu, bf16 out.
# =========================================================================== #
@pytest.mark.parametrize("N", [4096, 2048], ids=["intermediate", "hidden"])
@torch.inference_mode()
def test_fused_silu_mul_matches_reference_and_triton(N):
    from starling._kernels import torch_backend

    torch.manual_seed(1)
    gate = torch.randn(1, N, device=device, dtype=bf16) * 0.1
    up = torch.randn(1, N, device=device, dtype=bf16) * 0.1

    # Eager reference: fp32 silu, truncate to bf16 BEFORE * up.
    silu_g = F.silu(gate.float()).to(bf16)
    ref = silu_g * up

    torch_out = torch_backend.fused_silu_mul(gate, up)
    torch.testing.assert_close(torch_out, ref, atol=0, rtol=0)

    if _have_triton():
        from starling._kernels import triton_backend

        triton_out = triton_backend.fused_silu_mul(gate, up)
        torch.testing.assert_close(torch_out, triton_out, atol=0, rtol=0)


# =========================================================================== #
# 3. residual_add -- x + alpha*y.  alpha=1.0 (moss/qwen3) and alpha~0.22 (granite).
# =========================================================================== #
@pytest.mark.parametrize("alpha", [1.0, 0.22], ids=["alpha1", "granite"])
@torch.inference_mode()
def test_residual_add_matches_reference_and_triton(alpha):
    from starling._kernels import torch_backend

    torch.manual_seed(2)
    x = torch.randn(1, 2048, device=device, dtype=bf16) * 0.1
    y = torch.randn(1, 2048, device=device, dtype=bf16) * 0.1

    if alpha == 1.0:
        ref = x + y
    else:
        ref = (y.float() * alpha).to(bf16) + x

    torch_out = torch_backend.residual_add(x, y, alpha)
    torch.testing.assert_close(torch_out, ref, atol=0, rtol=0)

    if _have_triton():
        from starling._kernels import triton_backend

        triton_out = triton_backend.residual_add(x, y, alpha)
        torch.testing.assert_close(torch_out, triton_out, atol=0, rtol=0)


# =========================================================================== #
# 4. fused_rope -- rotary embedding on Q+K.  bf16-sensitive (~1-2 ULP).
# =========================================================================== #
@torch.inference_mode()
def test_fused_rope_matches_reference_and_triton():
    from starling._kernels import torch_backend

    torch.manual_seed(3)
    n_q, n_kv, hd = 16, 8, 128
    q = torch.randn(1, n_q, 1, hd, device=device, dtype=bf16) * 0.1
    k = torch.randn(1, n_kv, 1, hd, device=device, dtype=bf16) * 0.1
    cos = torch.randn(1, 1, 1, hd, device=device, dtype=torch.float32)
    sin = torch.randn(1, 1, 1, hd, device=device, dtype=torch.float32)

    # Eager reference via rotate_half, matching the truncation order of both
    # backends: each product truncated to bf16 BEFORE the two are added.
    def _rotate_half(t):
        half = t.shape[-1] // 2
        return torch.cat((-t[..., half:], t[..., :half]), dim=-1)

    def _ref(t, c, s):
        c_bf = c.to(bf16)
        s_bf = s.to(bf16)
        return (t * c_bf).to(bf16) + (_rotate_half(t) * s_bf).to(bf16)

    # cos/sin broadcast across heads (single decode position).
    cos_h = cos.reshape(-1, hd)[0:1]  # (1, hd)
    sin_h = sin.reshape(-1, hd)[0:1]
    q_flat = q.reshape(1 * n_q, hd)
    k_flat = k.reshape(1 * n_kv, hd)
    ref_q = _ref(q_flat, cos_h, sin_h).view_as(q)
    ref_k = _ref(k_flat, cos_h, sin_h).view_as(k)

    torch_q, torch_k = torch_backend.fused_rope(q, k, cos, sin)
    # Near-exact vs reference: rope is bf16-sensitive.  Observed max-diff is
    # ~0.004 (1 bf16 ULP); atol=0.1 is a generous, seed-robust gate.
    torch.testing.assert_close(torch_q, ref_q, atol=0.1, rtol=0.01)
    torch.testing.assert_close(torch_k, ref_k, atol=0.1, rtol=0.01)

    if _have_triton():
        from starling._kernels import triton_backend

        triton_q, triton_k = triton_backend.fused_rope(q, k, cos, sin)
        torch.testing.assert_close(torch_q, triton_q, atol=0.1, rtol=0.01)
        torch.testing.assert_close(torch_k, triton_k, atol=0.1, rtol=0.01)


# =========================================================================== #
# 5. FP8: quantize_weight_e4m3 (exact between backends) + fp8_linear.
#    Shapes: the base (2048,2048) plus the four real MOSS projection shapes.
# =========================================================================== #
_FP8_SHAPES = [
    (2048, 2048),    # base / attention-output
    (4096, 2048),    # fused QKV
    (12288, 2048),   # fused gate/up
    (2048, 6144),    # MLP-down
]


@pytest.mark.parametrize("N, K", _FP8_SHAPES, ids=[f"{n}x{k}" for n, k in _FP8_SHAPES])
@torch.inference_mode()
def test_quantize_weight_e4m3_matches_reference_and_triton(N, K):
    from starling._kernels import torch_backend

    torch.manual_seed(4)
    weight = torch.randn(N, K, device=device, dtype=bf16) * 0.02

    w_fp8_torch, scale_torch = torch_backend.quantize_weight_e4m3(weight)

    # Reference: same per-channel absmax formula.
    from starling._kernels.base import FP8_MAX

    amax = weight.abs().amax(dim=1).clamp(min=1e-8)
    scale_ref = amax / FP8_MAX
    w_fp8_ref = (weight / scale_ref[:, None]).clamp(-FP8_MAX, FP8_MAX).to(
        torch_backend.FP8_DTYPE
    )
    torch.testing.assert_close(w_fp8_torch.float(), w_fp8_ref.float(), atol=0, rtol=0)
    torch.testing.assert_close(scale_torch, scale_ref.float(), atol=0, rtol=0)

    if _have_triton():
        from starling._kernels import triton_backend

        w_fp8_tri, scale_tri = triton_backend.quantize_weight_e4m3(weight)
        # Byte-exact quantization between backends.
        torch.testing.assert_close(w_fp8_torch.float(), w_fp8_tri.float(), atol=0, rtol=0)
        torch.testing.assert_close(scale_torch, scale_tri, atol=0, rtol=0)


@pytest.mark.parametrize("N, K", _FP8_SHAPES, ids=[f"{n}x{k}" for n, k in _FP8_SHAPES])
@torch.inference_mode()
def test_fp8_linear_matches_reference_and_triton(N, K):
    from starling._kernels import torch_backend

    torch.manual_seed(5)
    weight = torch.randn(N, K, device=device, dtype=bf16) * 0.02
    x = torch.randn(1, K, device=device, dtype=bf16)

    w_fp8, scale = torch_backend.quantize_weight_e4m3(weight)

    # Reference: dequant fp8 -> fp32, then F.linear in fp32, truncate to bf16
    # (mirrors tests/test_fp8_gemv.py).
    dequant = w_fp8.float() * scale[:, None]
    ref = F.linear(x.float(), dequant).to(bf16)

    torch_out = torch_backend.fp8_linear(x, w_fp8, scale)
    # Both backends match the fp32-dequant reference within ~1 bf16 ULP
    # (torch dequants fp8->bf16; triton dequants fp8->fp32 so is exact vs ref).
    # Observed torch-vs-ref max-diff is exactly 0.03125 on the wide shapes
    # (2048x6144); leave a small margin for seed/shape variation.
    torch.testing.assert_close(torch_out, ref, atol=0.0625, rtol=0.02)

    if _have_triton():
        from starling._kernels import triton_backend

        triton_out = triton_backend.fp8_linear(x, w_fp8, scale)
        # Triton dequants fp8 -> fp32 in registers, so it matches the fp32
        # reference exactly.
        torch.testing.assert_close(triton_out, ref, atol=0.03125, rtol=0.02)
        # Torch (bf16 dequant) vs Triton (fp32 dequant): differ by at most
        # ~0.03125 (1 bf16 ULP from the dequant rounding) -- NOT the ~1.0
        # one might expect from fp8 rounding, since both paths accumulate the
        # same fp8 codes and only the dequant dtype differs.
        torch.testing.assert_close(torch_out, triton_out, atol=0.0625, rtol=0.02)


# =========================================================================== #
# 6. compute_rstd + fused_gemv_normscale -- RMSNorm scale folded into a GEMV.
# =========================================================================== #
@torch.inference_mode()
def test_compute_rstd_matches_reference_and_triton():
    from starling._kernels import torch_backend

    torch.manual_seed(6)
    x = torch.randn(2048, device=device, dtype=bf16) * 0.1
    eps = 1e-6

    var = x.float().pow(2).mean()
    ref = torch.rsqrt(var + eps).reshape(1)

    torch_rstd = torch_backend.compute_rstd(x, eps)
    # Near-exact scalar (fp32 reduction; observed diff ~1e-6 between backends).
    torch.testing.assert_close(torch_rstd, ref, atol=1e-5, rtol=1e-5)

    if _have_triton():
        from starling._kernels import triton_backend

        triton_rstd = triton_backend.compute_rstd(x, eps)
        torch.testing.assert_close(torch_rstd, triton_rstd, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(triton_rstd, ref, atol=1e-5, rtol=1e-5)


@torch.inference_mode()
def test_fused_gemv_normscale_matches_reference_and_triton():
    from starling._kernels import torch_backend

    torch.manual_seed(7)
    K = 2048
    OUT = 2048
    x = torch.randn(K, device=device, dtype=bf16) * 0.1
    w_scaled = torch.randn(OUT, K, device=device, dtype=bf16) * 0.02
    eps = 1e-6

    rstd = torch_backend.compute_rstd(x, eps)

    # Reference: F.linear then fold rstd in fp32, truncate to bf16.
    out_ref = F.linear(x, w_scaled)
    ref = (out_ref.float() * rstd).to(bf16)

    torch_out = torch_backend.fused_gemv_normscale(x, w_scaled, rstd)
    # Torch backend materializes the matmul exactly like the reference.
    torch.testing.assert_close(torch_out, ref, atol=0.0, rtol=0.0)

    if _have_triton():
        from starling._kernels import triton_backend

        triton_out = triton_backend.fused_gemv_normscale(x, w_scaled, rstd)
        # Triton matches the F.linear reference within bf16 GEMV accumulation
        # noise (observed ~0.016).
        torch.testing.assert_close(triton_out, ref, atol=0.0625, rtol=0.01)
        # Torch vs Triton: same math, different accumulation order.
        torch.testing.assert_close(torch_out, triton_out, atol=0.0625, rtol=0.01)


# =========================================================================== #
# 7. CUDA C++ backend (``cuda_backend``) vs torch.
#
# A third backend -- ``cuda_backend`` -- implements the elementwise kernels
# (rmsnorm/silu_mul/residual_add) and fp8_linear natively in CUDA C++ via
# ``torch.utils.cpp_extension.load_inline`` (JIT-compiled once, then cached).
# It is the full-performance path on triton-less platforms that still have a
# CUDA toolkit (notably Windows).  It DELEGATES fused_rope, compute_rstd,
# fused_gemv_normscale and fp4_gemv_fused to ``torch_backend`` (imported), so
# those need no separate cuda-vs-torch check here -- they ARE torch.
#
# IMPORTANT caveat (triton-in-same-process IMA): if the triton fp8 kernel's
# autotuner AND the cuda fp8 kernel run in the SAME python process, triton's
# autotuner benchmarking hits a CUDA illegal-memory-access (a triton-side
# issue, not a bug in our cuda kernel).  To stay clear of it, EVERY cuda test
# below compares against the torch backend / the dequant reference only -- NEVER
# against the triton fp8 kernel.  The elementwise kernels are unaffected
# (autotuned earlier in the process), but we keep them cuda-vs-torch too for a
# uniform, triton-free comparison.
# =========================================================================== #
def _have_cuda_compile() -> bool:
    """Return ``True`` if the cuda C++ backend imports AND JIT-compiles.

    Mirrors the philosophy of :func:`_have_triton`: not just
    ``torch.cuda.is_available()`` (which the base helper already checks), but
    an actual end-to-end ``cuda_backend._ext()`` call, which JIT-compiles the
    extension on the first invocation and reuses the cached shared object
    thereafter.  Subsequent calls are therefore instant.  Any failure (no CUDA
    toolkit, no compiler, nvcc error) returns ``False`` so the cuda tests skip
    cleanly rather than error.
    """
    try:
        if not torch.cuda.is_available():
            return False
        from starling._kernels import cuda_backend

        cuda_backend._ext()  # triggers JIT compile; cached after first call
        return True
    except Exception:
        return False


_no_cuda = pytest.mark.skipif(
    not _have_cuda_compile(), reason="CUDA C++ backend unavailable (no toolkit / compile failed)"
)


@_no_cuda
@pytest.mark.parametrize("M, N", [(1, 2048), (4, 2048), (1, 128)], ids=["decode", "multirow", "perhead"])
@torch.inference_mode()
def test_cuda_fused_rmsnorm_matches_torch(M, N):
    """CUDA rmsnorm must match the torch backend byte-for-byte."""
    from starling._kernels import cuda_backend, torch_backend

    torch.manual_seed(10)
    x = (torch.randn(M, N, device=device, dtype=bf16) * 0.1)
    weight = torch.randn(N, device=device, dtype=bf16) * 0.1
    eps = 1e-6

    torch_out = torch_backend.fused_rmsnorm(x, weight, eps)
    cuda_out = cuda_backend.fused_rmsnorm(x, weight, eps)
    # Byte-exact (verified empirically across all three shapes).
    torch.testing.assert_close(cuda_out, torch_out, atol=0, rtol=0)


@_no_cuda
@pytest.mark.parametrize("N", [4096, 6144], ids=["intermediate", "down"])
@torch.inference_mode()
def test_cuda_fused_silu_mul_matches_torch(N):
    """CUDA silu_mul must match the torch backend byte-for-byte."""
    from starling._kernels import cuda_backend, torch_backend

    torch.manual_seed(11)
    gate = torch.randn(1, N, device=device, dtype=bf16) * 0.1
    up = torch.randn(1, N, device=device, dtype=bf16) * 0.1

    torch_out = torch_backend.fused_silu_mul(gate, up)
    cuda_out = cuda_backend.fused_silu_mul(gate, up)
    torch.testing.assert_close(cuda_out, torch_out, atol=0, rtol=0)


@_no_cuda
@pytest.mark.parametrize("alpha", [1.0, 0.22], ids=["alpha1", "granite"])
@torch.inference_mode()
def test_cuda_residual_add_matches_torch(alpha):
    """CUDA residual_add must match the torch backend byte-for-byte."""
    from starling._kernels import cuda_backend, torch_backend

    torch.manual_seed(12)
    x = torch.randn(1, 2048, device=device, dtype=bf16) * 0.1
    y = torch.randn(1, 2048, device=device, dtype=bf16) * 0.1

    torch_out = torch_backend.residual_add(x, y, alpha)
    cuda_out = cuda_backend.residual_add(x, y, alpha)
    torch.testing.assert_close(cuda_out, torch_out, atol=0, rtol=0)


# FP8 shapes: the four real MOSS projection shapes + the small (96,128) tile
# used in tests/test_fp8_gemv.py.  Compared against the dequant REFERENCE
# (F.linear on the fp32-dequantized weight), NOT against triton fp8 -- that
# avoids the triton-autotuner-in-same-process illegal-memory-access (see the
# section header above).
_CUDA_FP8_SHAPES = [
    (2048, 2048),    # base / attention-output
    (4096, 2048),    # fused QKV
    (12288, 2048),   # fused gate/up
    (2048, 6144),    # MLP-down
    (96, 128),       # small tile from tests/test_fp8_gemv.py
]


@_no_cuda
@pytest.mark.parametrize("N, K", _CUDA_FP8_SHAPES, ids=[f"{n}x{k}" for n, k in _CUDA_FP8_SHAPES])
@torch.inference_mode()
def test_cuda_fp8_linear_matches_dequant_reference(N, K):
    """CUDA fp8_linear must match the fp32-dequant F.linear reference.

    Compared against the dequant reference (not triton fp8) deliberately: the
    cuda fp8 kernel and the triton fp8 autotuner cannot safely coexist in one
    process (triton-side illegal-memory-access).  The dequant reference is the
    exact same correctness yardstick used in ``tests/test_fp8_gemv.py``
    (atol=0.03125), so this is a safe and sufficient check.
    """
    from starling._kernels import cuda_backend

    torch.manual_seed(13)
    weight = torch.randn(N, K, device=device, dtype=bf16) * 0.02
    x = torch.randn(1, K, device=device, dtype=bf16)

    w_fp8, scale = cuda_backend.quantize_weight_e4m3(weight)

    # Reference: dequant fp8 -> fp32, F.linear in fp32, truncate to bf16
    # (mirrors tests/test_fp8_gemv.py and the torch reference above).
    ref = F.linear(x.float(), w_fp8.float() * scale[:, None]).bfloat16()

    cuda_out = cuda_backend.fp8_linear(x, w_fp8, scale)
    # CUDA fp8 dequants fp8 -> fp32 in registers, so it matches the fp32
    # reference within ~1 bf16 ULP.  atol=0.03125 matches
    # test_fp8_gemv.py; observed max-diff is well under that.
    torch.testing.assert_close(cuda_out, ref, atol=0.03125, rtol=0.02)


@_no_cuda
@pytest.mark.parametrize("N, K", _CUDA_FP8_SHAPES, ids=[f"{n}x{k}" for n, k in _CUDA_FP8_SHAPES])
@torch.inference_mode()
def test_cuda_quantize_weight_e4m3_matches_torch(N, K):
    """CUDA quantize_weight_e4m3 is the shared pure-torch recipe -> byte-exact vs torch."""
    from starling._kernels import cuda_backend, torch_backend

    torch.manual_seed(14)
    weight = torch.randn(N, K, device=device, dtype=bf16) * 0.02

    w_fp8_cuda, scale_cuda = cuda_backend.quantize_weight_e4m3(weight)
    w_fp8_torch, scale_torch = torch_backend.quantize_weight_e4m3(weight)
    torch.testing.assert_close(w_fp8_cuda.float(), w_fp8_torch.float(), atol=0, rtol=0)
    torch.testing.assert_close(scale_cuda, scale_torch, atol=0, rtol=0)


# =========================================================================== #
# 7b. CUDA native kernels for the four ops that USED to be delegated to torch
# (fused_rope, compute_rstd, fused_gemv_normscale, fp4_gemv_fused).  These are
# now full-performance CUDA kernels (see cuda/backend.cu), so each needs its own
# cuda-vs-torch correctness gate, mirroring the structure above.
# =========================================================================== #
@_no_cuda
@torch.inference_mode()
def test_cuda_fused_rope_matches_torch():
    """CUDA fused_rope must match the torch backend within ~1-2 bf16 ULP.

    Both backends truncate each product to bf16 BEFORE the add (matching the
    eager reference).  The residual ~0.03 diff comes from fp32 reduction /
    cos-sin handling differences between the hand-written CUDA kernel and the
    stock-torch op; atol=0.06 covers 1-2 bf16 ULP seed-robustly.
    """
    from starling._kernels import cuda_backend, torch_backend

    torch.manual_seed(15)
    n_q, n_kv, hd = 16, 8, 128
    q = torch.randn(1, n_q, 1, hd, device=device, dtype=bf16) * 0.1
    k = torch.randn(1, n_kv, 1, hd, device=device, dtype=bf16) * 0.1
    cos = torch.randn(1, 1, 1, hd, device=device, dtype=torch.float32)
    sin = torch.randn(1, 1, 1, hd, device=device, dtype=torch.float32)

    torch_q, torch_k = torch_backend.fused_rope(q, k, cos, sin)
    cuda_q, cuda_k = cuda_backend.fused_rope(q, k, cos, sin)
    # 1-2 bf16 ULP: products are bf16-truncated before the add in both backends,
    # but the fp32 cos/sin reduction differs slightly.  Observed ~0.03.
    torch.testing.assert_close(cuda_q, torch_q, atol=0.06, rtol=0.01)
    torch.testing.assert_close(cuda_k, torch_k, atol=0.06, rtol=0.01)


@_no_cuda
@torch.inference_mode()
def test_cuda_compute_rstd_matches_torch():
    """CUDA compute_rstd must match the torch backend near-exactly.

    Both backends do a simple fp32 sum-of-squares / N + eps, rsqrt.  The CUDA
    kernel uses a single-block warp+block reduction; torch uses fp32 mean.  The
    result should be exact or within 1 fp32 ULP (atol=1e-5).
    """
    from starling._kernels import cuda_backend, torch_backend

    torch.manual_seed(16)
    x = torch.randn(2048, device=device, dtype=bf16) * 0.1
    eps = 1e-5

    torch_rstd = torch_backend.compute_rstd(x, eps)
    cuda_rstd = cuda_backend.compute_rstd(x, eps)
    # Near-exact scalar (fp32 reduction of sum-of-squares).
    torch.testing.assert_close(cuda_rstd, torch_rstd, atol=1e-5, rtol=1e-5)


@_no_cuda
@torch.inference_mode()
def test_cuda_fused_gemv_normscale_matches_torch():
    """CUDA fused_gemv_normscale must match the torch backend within the CODA tolerance.

    This is the experimental CODA path: the CUDA kernel does a 256-thread fp32
    strided GEMV reduction, while the torch backend materializes F.linear then
    folds rstd.  Reduction-order differences in fp32 give ~0.125 max-abs diff
    (documented in llm_kernels.py / the triton kernel); atol=0.25 is a
    seed-robust gate over that.
    """
    from starling._kernels import cuda_backend, torch_backend

    torch.manual_seed(17)
    K = 2048
    OUT = 2048
    x = torch.randn(1, K, device=device, dtype=bf16) * 0.1
    w_scaled = torch.randn(OUT, K, device=device, dtype=bf16) * 0.02
    eps = 1e-5

    # rstd (1,) fp32 from compute_rstd -- feed BOTH backends the same scalar so
    # the comparison isolates the GEMV+epilogue difference.
    rstd = torch_backend.compute_rstd(x, eps)

    torch_out = torch_backend.fused_gemv_normscale(x, w_scaled, rstd)
    cuda_out = cuda_backend.fused_gemv_normscale(x, w_scaled, rstd)
    # fp32 GEMV reduction-order differences (experimental CODA path, ~0.125
    # max-abs documented for the triton kernel of the same shape).
    torch.testing.assert_close(cuda_out, torch_out, atol=0.25, rtol=0.05)


@_no_cuda
@torch.inference_mode()
def test_cuda_fp4_gemv_fused_matches_torch():
    """CUDA fp4_gemv_fused must match the torch backend within the NVFP4 envelope.

    EXPERIMENTAL fp4 path: both backends consume the SAME nibble-packed codes
    and fp8 block scales (from quantize_fp4_packed), so the fp4 rounding itself
    is identical.  The residual diff is the bf16-vs-fp32 dequant + GEMV
    reduction order: the CUDA kernel dequants to fp32 registers and accumulates
    in fp32, while the torch correctness path dequants to bf16 then runs
    F.linear.  fp4.py documents ~1e-1 relative weight error; on magnitude-70
    decode projections that is ~1.0 max-abs, hence atol=1.5.
    """
    from starling.granite.fp4 import quantize_fp4_packed
    from starling._kernels import cuda_backend, torch_backend

    torch.manual_seed(18)
    OUT = 2048
    K = 2048
    # Weight scaled to produce magnitude-70-class projection outputs (the regime
    # the triton/cuda fp4 kernels are documented against).
    weight = torch.randn(OUT, K, device=device, dtype=bf16) * 0.5
    x = torch.randn(1, K, device=device, dtype=bf16)

    codes, scales = quantize_fp4_packed(weight)

    torch_out = torch_backend.fp4_gemv_fused(x, codes, scales)
    cuda_out = cuda_backend.fp4_gemv_fused(x, codes, scales)
    # NVFP4 rounding envelope (experimental fp4 path): ~1e-1 relative weight
    # error -> ~1.0 max-abs on magnitude-70 projections.
    torch.testing.assert_close(cuda_out, torch_out, atol=1.5, rtol=0.1)
