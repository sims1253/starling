"""CUDA-graph-captured audio encoder for ARK-ASR-3B.

The audio path is a Whisper encoder (32 layers, d_model 1280) followed by a
fixed MLP adapter that merges every ``merge_factor`` frames into one LLM token
of hidden size 2048. The encoder runs once per utterance and is not on the
critical decode loop, but it still launches hundreds of small kernels per call.

This module captures the adapter's full forward (Whisper + layer norm + MLP)
into a per-shape ``torch.cuda.CUDAGraph`` so the whole encoder collapses to a
single ``graph.replay()`` per utterance. The captured graph runs the model's own
ops unchanged, so the output is byte-exact with the eager reference.

Public API
----------
``FusedEncoder(audio_encoder)``
``FusedEncoder(audios) -> audio_features``  (B, N, 2048) bf16
"""

from __future__ import annotations

from typing import Any

import torch

from .config import ENCODER_NUM_MEL_BINS


class FusedEncoder:
    """CUDA-graph capture of ``audio_encoder(audios)``.

    One graph is captured per ``(batch, mel_T)`` shape (shape-keyed cache). On
    the first call for a new shape the encoder is warmed up three times on a
    side stream (so cuBLAS/cuDNN lazy init settles) and then captured; later
    calls with the same shape copy the input into the static input buffer and
    replay the graph.

    Args:
        audio_encoder: The ``AudioMLPAdapter`` (``model.audio_encoder``).
        device: Target device.
        dtype: Encoder dtype (bf16 is the checkpoint dtype).
    """

    def __init__(
        self,
        audio_encoder: Any,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.audio_encoder = audio_encoder
        self.device = device
        self.dtype = dtype
        # Shape-keyed capture cache: (B, mel_T) -> CUDAGraph.
        self._graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        # Static input/output buffers per shape (kept alive alongside the graph).
        self._static_in: dict[tuple[int, int], torch.Tensor] = {}
        self._static_out: dict[tuple[int, int], torch.Tensor] = {}

    # ------------------------------------------------------------------ #
    # graph capture (lazy, per shape)
    # ------------------------------------------------------------------ #
    def _get_graph(self, B: int, mel_T: int) -> torch.cuda.CUDAGraph:
        """Capture (if needed) and return the graph for shape ``(B, mel_T)``."""
        key = (B, mel_T)
        if key in self._graphs:
            return self._graphs[key]

        static_audios = torch.zeros(
            B, ENCODER_NUM_MEL_BINS, mel_T, dtype=self.dtype, device=self.device
        )
        self._static_in[key] = static_audios

        # Warmup on a side stream (3 iters) so lazy initialisations settle BEFORE
        # capture. All warmup ops happen on the side stream's memory pool.
        device = torch.device(self.device)
        side = torch.cuda.Stream(device=device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            for _ in range(3):
                _ = self.audio_encoder(static_audios)
        torch.cuda.current_stream(device).wait_stream(side)

        # Capture.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = self.audio_encoder(static_audios)
        self._static_out[key] = static_out

        self._graphs[key] = graph
        return graph

    # ------------------------------------------------------------------ #
    # public forward
    # ------------------------------------------------------------------ #
    def __call__(self, audios: torch.Tensor) -> torch.Tensor:
        """Run the fused encoder.

        Args:
            audios: ``(B, 128, mel_T)`` mel tensor (cast to bf16 if needed).

        Returns:
            ``(B, N, 2048)`` bf16 audio features. The returned tensor is a
            contiguous clone so the caller owns it (the static output buffer is
            reused on the next call).
        """
        if audios.dtype != self.dtype:
            audios = audios.to(self.dtype)
        if audios.device.type != self.device:
            audios = audios.to(self.device)
        B, _, mel_T = audios.shape
        key = (int(B), int(mel_T))
        graph = self._get_graph(*key)
        self._static_in[key].copy_(audios)
        graph.replay()
        return self._static_out[key].clone().contiguous()

    def forward_eager(self, audios: torch.Tensor) -> torch.Tensor:
        """Direct ``audio_encoder(audios)`` call for A/B testing against the graph."""
        if audios.dtype != self.dtype:
            audios = audios.to(self.dtype)
        return self.audio_encoder(audios)
