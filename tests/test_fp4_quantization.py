"""CPU regressions for FP4 block scaling in both storage formats."""

import pytest
import torch

from starling.granite.fp4 import dequantize_fp4, quantize_fp4, quantize_fp4_packed


def quantize(weight, packed):
    if not packed:
        return quantize_fp4(weight)
    codes, scales = quantize_fp4_packed(weight)
    # Decode the documented nibble layout into the reference format.
    unpacked = torch.stack((codes & 15, codes >> 4), dim=-1).reshape(-1, 16)
    return unpacked, scales.float().reshape(-1), (weight.shape, 0)


@pytest.mark.parametrize("packed", [False, True], ids=["reference", "packed"])
@pytest.mark.parametrize("background", [1.0, -1.0], ids=["mixed-sign", "all-negative"])
def test_negative_outliers_and_sign_reversal(packed, background):
    weight = torch.full((2, 16), background)
    weight[:, 0] = -6
    weight[1] *= 2
    codes, scales, meta = quantize(weight, packed)
    reversed_codes, reversed_scales, reversed_meta = quantize(-weight, packed)

    torch.testing.assert_close(scales, torch.tensor([6.0, 12.0]), rtol=0, atol=0)
    torch.testing.assert_close(reversed_scales, scales, rtol=0, atol=0)
    assert torch.equal(reversed_codes, codes ^ 8)
    # These weights are representable exactly; neither sign may clip an outlier.
    torch.testing.assert_close(dequantize_fp4(codes, scales, meta),
                               weight.bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(dequantize_fp4(reversed_codes, reversed_scales, reversed_meta),
                               -weight.bfloat16(), rtol=0, atol=0)
