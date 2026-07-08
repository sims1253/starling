"""FP8 (e4m3) weight-only quantization for the MOSS Qwen3 decoder GEMMs.

Decode is memory-bandwidth bound on the LLM weights (~72% of the captured
decode step is per-layer GEMMs, each a pure weight read at M=1).  Casting the
28 decoder layers' projection weights to fp8e4m3 halves that weight traffic and
is the single largest decode speedup that needs *no* fine-tune.

Quality (RTX 5090, sm_120).  ~+20% decode throughput (317 -> ~387 tok/s on the
medium fixture; long fixture 647 -> 546 ms) at **0.00% net WER cost**: on a
34-clip Open-ASR-Leaderboard subset (librispeech/ami/earnings22/gigaspeech) the
composite WER is 3.34% for both bf16 and fp8 (per-clip drift ~0.32%, which
washes out).  It reproduces the bf16 greedy transcript token-for-token on the
short/medium golden fixtures; very long single decodes can diverge late (the
long fixture emits EOS a few tokens early), but that does not move aggregate
WER.  The **lm_head is deliberately kept in bf16** -- its 151936-way argmax has
near-tie logits that fp8 rounding *does* flip, changing the transcript.  See
:attr:`starling.flags.OptFlags.fp8_weights`.

Scaling
-------
Per-output-channel symmetric absmax scaling (``scale[o] = max|W[o,:]| / 448``)
for the weight, dynamic per-token absmax scaling for the activation.  Both feed
``torch._scaled_mm`` (Blackwell fp8 tensor cores) with fp32 accumulation.
``_scaled_mm`` requires the weight as a **column-major** ``(K, N)`` operand, so
we store the transpose view of a contiguous ``(N, K)`` fp8 buffer.
"""

from __future__ import annotations

import torch

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0


def quantize_weight_e4m3(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize an ``(N, K)`` linear weight to fp8e4m3, per-output-channel.

    Returns ``(w_fp8_kn, scale)`` where ``w_fp8_kn`` is a column-major ``(K, N)``
    fp8 view (the layout ``torch._scaled_mm`` expects for ``mat2``) and ``scale``
    is the ``(1, N)`` fp32 per-channel dequant scale.
    """
    amax = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)  # (N, 1)
    scale = amax / FP8_MAX                                         # (N, 1)
    w_fp8 = (weight / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE).contiguous()
    return w_fp8.t(), scale.reshape(1, -1).float()                # (K, N) col-major, (1, N)


def fp8_linear(x: torch.Tensor, w_fp8_kn: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    """``x @ W^T`` with an fp8 weight and dynamic per-token fp8 activation.

    Args:
        x: ``(M, K)`` bf16 activation.
        w_fp8_kn: ``(K, N)`` column-major fp8 weight from
            :func:`quantize_weight_e4m3`.
        w_scale: ``(1, N)`` fp32 per-channel weight scale.

    Returns:
        ``(M, N)`` bf16 result, matching ``F.linear(x, W)`` to fp8 precision.
    """
    amax = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)  # (M, 1)
    x_scale = amax / FP8_MAX
    x_fp8 = (x / x_scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return torch._scaled_mm(
        x_fp8, w_fp8_kn,
        scale_a=x_scale.float(), scale_b=w_scale,
        out_dtype=torch.bfloat16,
    )
