"""Focused correctness and CUDA-graph tests for the shared FP8 GEMV."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@torch.inference_mode()
def test_fp8_gemv_matches_dequantized_weight_reference():
    from starling.fp8_gemv import fp8_linear, quantize_weight_e4m3

    torch.manual_seed(0)
    weight = torch.randn(96, 128, device="cuda", dtype=torch.bfloat16) * 0.02
    x = torch.randn(1, 128, device="cuda", dtype=torch.bfloat16)
    weight_fp8, scale = quantize_weight_e4m3(weight)

    actual = fp8_linear(x, weight_fp8, scale)
    dequantized = weight_fp8.float() * scale[:, None]
    expected = torch.nn.functional.linear(x.float(), dequantized).bfloat16()

    assert weight_fp8.shape == weight.shape
    assert weight_fp8.is_contiguous()
    torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0.02)


@torch.inference_mode()
def test_fp8_gemv_survives_many_varying_graph_replays():
    """Multiple captured shapes must not alias workspace or corrupt outputs."""
    from starling.fp8_gemv import fp8_linear, quantize_weight_e4m3

    torch.manual_seed(1)
    # The real fused QKV, attention-output, fused gate/up, and MLP-down
    # projection shapes from one MOSS decoder layer.
    shapes = [(4096, 2048), (2048, 2048), (12288, 2048), (2048, 6144)]
    bundles = []
    for out_features, in_features in shapes:
        weight = torch.randn(
            out_features, in_features, device="cuda", dtype=torch.bfloat16
        ) * 0.02
        inputs = [
            torch.randn(1, in_features, device="cuda", dtype=torch.bfloat16)
            for _ in range(2)
        ]
        x = inputs[0].clone()
        weight_fp8, scale = quantize_weight_e4m3(weight)
        # Trigger Triton autotuning before capture.
        expected = []
        for input_value in inputs:
            x.copy_(input_value)
            expected.append(fp8_linear(x, weight_fp8, scale).clone())
        x.copy_(inputs[0])
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = fp8_linear(x, weight_fp8, scale)
        # Captured graphs do not retain Python tensor owners. Keep every static
        # input alive exactly as the decoder object does in production.
        bundles.append((graph, output, expected, x, weight_fp8, scale, inputs))

    for replay in range(400):
        graph, output, expected, x, *_owners, inputs = bundles[replay % len(bundles)]
        variant = (replay // len(bundles)) % len(inputs)
        x.copy_(inputs[variant])
        graph.replay()
        if replay % 40 == 0:
            torch.cuda.synchronize()
            torch.testing.assert_close(output, expected[variant], atol=0, rtol=0)

    torch.cuda.synchronize()
    for graph, output, expected, x, *_owners, inputs in bundles:
        x.copy_(inputs[-1])
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected[-1], atol=0, rtol=0)
