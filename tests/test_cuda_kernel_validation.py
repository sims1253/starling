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


def test_gemv_extra_rows_and_raw_pointer_dtype_mismatches_fail() -> None:
    from starling._kernels import cuda_backend

    bf16 = torch.ones((1, 4), dtype=torch.bfloat16)
    half = bf16.to(torch.float16)
    fp8 = torch.ones((2, 4)).to(torch.float8_e4m3fn)
    scale = torch.ones(2, dtype=torch.float32)
    codes = torch.ones((2, 8), dtype=torch.uint8)
    fp4_scale = torch.ones((2, 1)).to(torch.float8_e4m3fn)
    fp4_x = torch.ones((1, 16), dtype=torch.bfloat16)
    rope_bf16 = torch.ones((1, 1, 1, 64), dtype=torch.bfloat16)
    rope_half = rope_bf16.to(torch.float16)

    invalid_calls = (
        lambda: cuda_backend.fused_rmsnorm(half, bf16.reshape(-1), 1e-6),
        lambda: cuda_backend.fused_silu_mul(half, bf16),
        lambda: cuda_backend.residual_add(half, bf16),
        lambda: cuda_backend.compute_rstd(half, 1e-6),
        lambda: cuda_backend.fp8_linear(half, fp8, scale),
        lambda: cuda_backend.fp8_linear(bf16, fp8.to(torch.bfloat16), scale),
        lambda: cuda_backend.fp8_linear(bf16, fp8, scale.to(torch.float16)),
        lambda: cuda_backend.fused_gemv_normscale(half, bf16.repeat(2, 1), scale[:1]),
        lambda: cuda_backend.fp4_gemv_fused(fp4_x, codes.to(torch.int32), fp4_scale),
        lambda: cuda_backend.fp4_gemv_fused(fp4_x, codes, fp4_scale.to(torch.bfloat16)),
        lambda: cuda_backend.fused_rope(rope_half, rope_bf16, rope_bf16, rope_bf16),
        lambda: cuda_backend.fp8_linear(bf16.repeat(2, 1), fp8, scale),
        lambda: cuda_backend.fused_gemv_normscale(bf16.repeat(2, 1), bf16.repeat(2, 1), scale[:1]),
        lambda: cuda_backend.fp4_gemv_fused(fp4_x.repeat(2, 1), codes, fp4_scale),
    )
    for call in invalid_calls:
        with pytest.raises(ValueError):
            call()
