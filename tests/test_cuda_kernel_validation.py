"""Host-side CUDA wrapper guards; runnable with CPU-only PyTorch in CI."""

from __future__ import annotations

import pytest
import torch


def test_empty_and_short_inputs_fail_before_compiling_cuda() -> None:
    from starling._kernels import cuda_backend

    one = torch.ones((1, 1), dtype=torch.bfloat16)
    two = torch.ones((1, 2), dtype=torch.bfloat16)
    empty = torch.empty((1, 0), dtype=torch.bfloat16)
    fp8 = torch.ones((1, 2)).to(torch.float8_e4m3fn)
    codes = torch.ones((1, 8), dtype=torch.uint8)
    scale = torch.ones((1, 1)).to(torch.float8_e4m3fn)

    invalid_calls = (
        lambda: cuda_backend.fused_rmsnorm(empty, empty.reshape(-1), 1e-6),
        lambda: cuda_backend.fused_rmsnorm(one, two.reshape(-1), 1e-6),
        lambda: cuda_backend.fused_silu_mul(one, two),
        lambda: cuda_backend.residual_add(one, two),
        lambda: cuda_backend.compute_rstd(empty, 1e-6),
        lambda: cuda_backend.compute_rstd(two.expand(2, 2), 1e-6),
        lambda: cuda_backend.fp8_linear(one, fp8, one.float()),
        lambda: cuda_backend.fused_gemv_normscale(one, two, one.float()),
        lambda: cuda_backend.fp4_gemv_fused(one, codes, scale),
        lambda: cuda_backend.fused_rope(one.reshape(1, 1, 1, 1), one.reshape(1, 1, 1, 1), one, one),
    )
    for call in invalid_calls:
        with pytest.raises(ValueError):
            call()
