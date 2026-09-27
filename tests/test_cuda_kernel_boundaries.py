"""Deterministic boundary corpus for the eight CUDA C++ kernels.

These tests exercise actual launches, including odd tails and strided activation
views. They are finite checks, not an all-input equivalence proof. Run them under
``scripts/sanitizer_gate.py`` for device memory and synchronization checks.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_DEVICE = "cuda"
_BF16 = torch.bfloat16


def _rand(*shape: int) -> torch.Tensor:
    return torch.randn(shape, device=_DEVICE, dtype=_BF16) * 0.1


@pytest.mark.parametrize("n", [1, 31, 32, 33, 127, 128, 129, 1023, 1024, 1025])
@torch.inference_mode()
def test_rmsnorm_boundary(n: int) -> None:
    from starling._kernels import cuda_backend, torch_backend

    torch.manual_seed(n)
    x = _rand(2, n * 2)[:, ::2]  # non-contiguous view, logical width n
    weight = _rand(n)
    actual = cuda_backend.fused_rmsnorm(x, weight, 1e-6)
    expected = torch_backend.fused_rmsnorm(x, weight, 1e-6)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, atol=0.015625, rtol=0.01)


@pytest.mark.parametrize("n", [1, 31, 32, 33, 255, 256, 257, 1025])
@torch.inference_mode()
def test_silu_and_residual_boundary(n: int) -> None:
    from starling._kernels import cuda_backend, torch_backend

    torch.manual_seed(n + 100)
    gate = _rand(2, n * 2)[:, ::2]
    up = _rand(2, n * 2)[:, ::2]
    actual = cuda_backend.fused_silu_mul(gate, up)
    expected = torch_backend.fused_silu_mul(gate, up)
    torch.testing.assert_close(actual, expected, atol=0.0005, rtol=0.01)
    for alpha in (0.0, 1.0, 0.22):
        actual = cuda_backend.residual_add(gate, up, alpha)
        expected = torch_backend.residual_add(gate, up, alpha)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("batch,n_q,n_kv,hd", [
    (1, 1, 1, 128), (1, 3, 2, 128), (1, 17, 9, 128), (2, 3, 2, 128),
    (1, 14, 2, 64), (2, 3, 2, 64),
])
@torch.inference_mode()
def test_rope_head_boundary(batch: int, n_q: int, n_kv: int, hd: int) -> None:
    from starling._kernels import cuda_backend, torch_backend

    torch.manual_seed(n_q)
    q = _rand(batch, n_q, 1, hd)
    k = _rand(batch, n_kv, 1, hd)
    cos = torch.randn((1, 1, 1, hd), device=_DEVICE)
    sin = torch.randn((1, 1, 1, hd), device=_DEVICE)
    actual = cuda_backend.fused_rope(q, k, cos, sin)
    expected = torch_backend.fused_rope(q, k, cos, sin)
    for output, reference in zip(actual, expected):
        torch.testing.assert_close(output, reference, atol=0.06, rtol=0.01)


@pytest.mark.parametrize("n", [1, 31, 32, 33, 1023, 1024, 1025])
@torch.inference_mode()
def test_rstd_boundary(n: int) -> None:
    from starling._kernels import cuda_backend, torch_backend

    torch.manual_seed(n + 200)
    x = _rand(n * 2)[::2]
    actual = cuda_backend.compute_rstd(x, 1e-6)
    expected = torch_backend.compute_rstd(x, 1e-6)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("k,out", [(1, 1), (255, 3), (256, 3), (257, 3), (1025, 2)])
@torch.inference_mode()
def test_fp8_gemv_boundary(k: int, out: int) -> None:
    from starling._kernels import cuda_backend

    torch.manual_seed(k + 300)
    x = _rand(k * 2)[::2]
    weight = _rand(out, k)
    codes, scales = cuda_backend.quantize_weight_e4m3(weight)
    actual = cuda_backend.fp8_linear(x, codes, scales)
    expected = torch.nn.functional.linear(x.float(), codes.float() * scales[:, None]).to(_BF16).unsqueeze(0)
    torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0.02)


@pytest.mark.parametrize("k,out", [(1, 1), (255, 3), (256, 3), (257, 3), (1025, 2)])
@torch.inference_mode()
def test_normscale_gemv_boundary(k: int, out: int) -> None:
    from starling._kernels import cuda_backend, torch_backend

    torch.manual_seed(k + 400)
    x = _rand(k * 2)[::2]
    weight = _rand(out, k)
    rstd = torch_backend.compute_rstd(x, 1e-6)
    actual = cuda_backend.fused_gemv_normscale(x, weight, rstd)
    expected = torch_backend.fused_gemv_normscale(x, weight, rstd)
    torch.testing.assert_close(actual, expected, atol=0.25, rtol=0.05)


@pytest.mark.parametrize("k,out", [(16, 1), (32, 3), (256, 3), (272, 3), (1040, 2)])
@torch.inference_mode()
def test_fp4_gemv_boundary(k: int, out: int) -> None:
    from starling._kernels import cuda_backend, torch_backend

    torch.manual_seed(k + 500)
    x = _rand(k * 2)[::2]
    codes = torch.randint(0, 256, (out, k // 2), dtype=torch.uint8, device=_DEVICE)
    scales = torch.ones((out, k // 16), device=_DEVICE).to(torch.float8_e4m3fn)
    actual = cuda_backend.fp4_gemv_fused(x, codes, scales)
    expected = torch_backend.fp4_gemv_fused(x, codes, scales)
    torch.testing.assert_close(actual, expected, atol=0.25, rtol=0.05)


@torch.inference_mode()
def test_empty_dimensions_rejected_before_launch() -> None:
    from starling._kernels import cuda_backend

    empty = torch.empty((1, 0), device=_DEVICE, dtype=_BF16)
    one = torch.ones((1, 1), device=_DEVICE, dtype=_BF16)
    cases = (
        lambda: cuda_backend.fused_rmsnorm(empty, empty.reshape(-1), 1e-6),
        lambda: cuda_backend.fused_silu_mul(empty, empty),
        lambda: cuda_backend.residual_add(empty, empty),
        lambda: cuda_backend.compute_rstd(empty, 1e-6),
        lambda: cuda_backend.fp8_linear(one, torch.empty((1, 0), device=_DEVICE, dtype=torch.float8_e4m3fn), one.float()),
        lambda: cuda_backend.fused_gemv_normscale(one, empty, one.float()),
        lambda: cuda_backend.fp4_gemv_fused(one, torch.empty((1, 0), device=_DEVICE, dtype=torch.uint8), empty.to(torch.float8_e4m3fn)),
        lambda: cuda_backend.fused_rope(empty.reshape(1, 1, 1, 0), empty.reshape(1, 1, 1, 0), empty, empty),
    )
    for call in cases:
        with pytest.raises(ValueError, match="non-empty"):
            call()


@torch.inference_mode()
def test_rope_rejects_unsupported_head_dim() -> None:
    from starling._kernels import cuda_backend

    q = _rand(1, 1, 1, 32)
    with pytest.raises(ValueError, match="head_dim=64 or 128"):
        cuda_backend.fused_rope(q, q, q, q)


@torch.inference_mode()
def test_short_buffers_and_mismatched_shapes_rejected_before_launch() -> None:
    from starling._kernels import cuda_backend

    one = torch.ones((1, 1), device=_DEVICE, dtype=_BF16)
    two = torch.ones((1, 2), device=_DEVICE, dtype=_BF16)
    fp8 = torch.ones((1, 2), device=_DEVICE).to(torch.float8_e4m3fn)
    codes = torch.ones((1, 8), device=_DEVICE, dtype=torch.uint8)
    scale = torch.ones((1, 1), device=_DEVICE).to(torch.float8_e4m3fn)
    cases = (
        lambda: cuda_backend.fused_rmsnorm(one, two.reshape(-1), 1e-6),
        lambda: cuda_backend.fused_silu_mul(one, two),
        lambda: cuda_backend.residual_add(one, two),
        lambda: cuda_backend.compute_rstd(two.expand(2, 2), 1e-6),
        lambda: cuda_backend.fp8_linear(one, fp8, one.float()),
        lambda: cuda_backend.fused_gemv_normscale(one, two, one.float()),
        lambda: cuda_backend.fp4_gemv_fused(one, codes, scale),
        lambda: cuda_backend.fused_rope(_rand(1, 1, 2, 128), _rand(1, 1, 1, 128), _rand(1, 1, 1, 128), _rand(1, 1, 1, 128)),
    )
    for call in cases:
        with pytest.raises(ValueError):
            call()
