"""Eager audio encoder for HojoAI/Hojo-ASR-V1 (tower + Conformer bottleneck).

This is the novel component of the Hojo pipeline. The forward path is

    mel (T, 128) -> Qwen3-Omni audio tower (32 layers) ->
    WeNet Conformer bottleneck (2 layers) -> ln_speech ->
    speech_embeddings (1, N, 2560)

WHY EAGER (NOT CUDA-GRAPH-CAPTURED)
-----------------------------------
Unlike ARK (whose Whisper+MLP encoder is graph-friendly) -- and exactly like
higgs -- Hojo's encoder is **not** byte-exactly CUDA-graph-capturable:

1. The Qwen3-Omni audio tower builds a block-diagonal 4D attention mask from
   ``cu_seqlens``. Constructing ``cu_seqlens`` calls ``mask.sum().item()`` per
   chunk in a Python loop (host sync), and the mask itself is a
   ``torch.full(...).min()`` tensor whose block-zeroing runs in a Python
   ``for i in range(1, len(cu_seqlens))`` loop with data-dependent slice bounds.
2. The tower packs the input via ``spectrogram[valid_mask]`` (dynamic boolean
   indexing) and splits it with ``.split(chunk_lengths.tolist(), ...)`` whose
   split sizes are computed from the input length at runtime.
3. The Conformer conv module does ``x.masked_fill_(~mask_pad, 0.0)`` on a
   dynamic-shape boolean padding mask.

Each of these emits host-dependent / dynamic-shape kernels that abort CUDA
graph capture or change the captured graph's behavior per input length. higgs
proved (and we confirm for Hojo) that eager-encoder + graphed-decode is a valid,
performant design: the encoder runs **once per clip** and is a small fraction
of the end-to-end cost relative to the beam-4 decode loop.

So this module runs the model's own eager ``encode_speech`` unchanged, which is
byte-exact with the reference (and the golden oracle). The "mega" naming is kept
for API symmetry with the other packages' ``FusedEncoder``; a future graphable
rewrite (e.g. precomputing the block mask on the host and threading a static
4D mask through the tower) would slot in here without changing the pipeline.

Public API
----------
``FusedEncoder(model)``
``FusedEncoder(mel, mel_len) -> (speech_embeddings, speech_attn)``
"""

from __future__ import annotations

from typing import Any

import torch

from .config import AUTOMIX_DTYPE


class FusedEncoder:
    """Eager audio encoder wrapper for Hojo-ASR-V1 (tower + Conformer).

    Drives the reference ``HOJO_ASR.encode_speech`` under fp16 autocast (the
    reference runs the whole ``infer`` under ``autocast_context()``; matching it
    keeps the bf16 decoder / f32 encoder casts byte-exact with the golden).

    Args:
        model: The loaded ``HOJO_ASR`` (provides ``encode_speech`` and
            ``autocast_context``).
        device: Target device (unused beyond bookkeeping; the model is already
            on ``device`` from :func:`starling.hojo.loader.load_model`).
    """

    def __init__(self, model: Any, device: str = "cuda") -> None:
        self.model = model
        self.device = device

    @torch.inference_mode()
    def __call__(
        self,
        spectrogram: torch.Tensor,
        spectrogram_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode mel -> speech embeddings + attention mask.

        Args:
            spectrogram: ``(B, T, 128)`` float32 mel tensor (transposed layout).
            spectrogram_lengths: ``(B,)`` int64 per-utterance mel lengths.

        Returns:
            ``(speech_embeddings, speech_attn)`` where ``speech_embeddings`` is
            ``(B, N, 2560)`` float32 and ``speech_attn`` is ``(B, N)`` float32
            (the bottleneck attention mask the decoder consumes as its
            ``attention_mask``).
        """
        # The reference wraps encode_speech in fp16 autocast. spectrogram is the
        # (B, T, 128) transposed Whisper mel; encode_speech handles the rest.
        autocast_dtype = getattr(torch, AUTOMIX_DTYPE, torch.float16)
        with self.model.autocast_context(autocast_dtype):
            speech_embeddings, speech_attn = self.model.encode_speech(
                spectrogram, spectrogram_lengths
            )
        return speech_embeddings, speech_attn

    def forward_eager(
        self,
        spectrogram: torch.Tensor,
        spectrogram_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Alias for :meth:`__call__` (API symmetry with ark/higgs)."""
        return self(spectrogram, spectrogram_lengths)
