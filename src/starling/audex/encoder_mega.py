"""CUDA-graph-captured audio encoder for Audex-2B (Qwen2AudioEncoder).

The audio path is a standard ``Qwen2AudioEncoder`` (Whisper-large-v3 shaped:
conv frontend → 32-layer transformer → avg-pooler that halves 1500→750 frames).
It runs once per utterance and is not on the critical decode loop, but it still
launches hundreds of small kernels per call.

This module captures the encoder's full forward into a per-shape
``torch.cuda.CUDAGraph`` so the whole path collapses to a single
``graph.replay()``. The captured graph runs the model's own ops unchanged, so
the output is byte-exact with the eager reference. Pattern ported from
``starling.ark.encoder_mega`` (same Qwen2AudioEncoder underneath).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch

from .config import AUDIO_NUM_MEL_BINS


class FusedEncoder:
    """CUDA-graph capture of ``audio_encoder(input_features)``.

    One graph per ``(B, mel_T)`` shape. For 30 s ASR clips the mel is always
    ``(1, 128, 3000)``, so typically one graph is captured.

    Args:
        audio_encoder: A loaded ``Qwen2AudioEncoder`` on cuda (eval, bf16).
        device: Target device.
        dtype: Encoder dtype (bf16).
        max_graphs: Max distinct shapes kept resident (LRU eviction beyond).
    """

    def __init__(
        self,
        audio_encoder: Any,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        *,
        max_graphs: int = 32,
    ) -> None:
        self.audio_encoder = audio_encoder
        self.device = device
        self.dtype = dtype
        self.max_graphs = max(1, int(max_graphs))
        self._graphs: OrderedDict[tuple[int, int], torch.cuda.CUDAGraph] = OrderedDict()
        self._static_in: dict[tuple[int, int], torch.Tensor] = {}
        self._static_out: dict[tuple[int, int], torch.Tensor] = {}

    def _free_graph(self, key: tuple[int, int]) -> None:
        graph = self._graphs.pop(key, None)
        self._static_in.pop(key, None)
        self._static_out.pop(key, None)
        if graph is not None:
            try:
                graph.reset()
            except Exception:
                pass
            del graph
            torch.cuda.empty_cache()

    def _evict_if_needed(self) -> None:
        while len(self._graphs) >= self.max_graphs:
            key, _ = self._graphs.popitem(last=False)
            self._free_graph(key)

    def _get_graph(self, B: int, mel_T: int) -> torch.cuda.CUDAGraph:
        key = (B, mel_T)
        graph = self._graphs.get(key)
        if graph is not None:
            self._graphs.move_to_end(key)
            return graph

        self._evict_if_needed()

        static_in = torch.zeros(
            B, AUDIO_NUM_MEL_BINS, mel_T, dtype=self.dtype, device=self.device
        )
        self._static_in[key] = static_in

        device_obj = torch.device(self.device)
        side = torch.cuda.Stream(device=device_obj)
        side.wait_stream(torch.cuda.current_stream(device_obj))
        with torch.cuda.stream(side):
            for _ in range(3):
                _ = self.audio_encoder(input_features=static_in, return_dict=True)
        torch.cuda.current_stream(device_obj).wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = self.audio_encoder(input_features=static_in, return_dict=True)
        self._static_out[key] = static_out.last_hidden_state

        self._graphs[key] = graph
        return graph

    def __call__(self, input_features: torch.Tensor) -> torch.Tensor:
        """Run the encoder; return ``last_hidden_state`` ``(B, 750, 1280)``.

        The returned tensor is a contiguous clone so the caller owns it.
        """
        if input_features.dtype != self.dtype:
            input_features = input_features.to(self.dtype)
        if input_features.device.type != self.device:
            input_features = input_features.to(self.device)
        B, _, mel_T = input_features.shape
        key = (int(B), int(mel_T))
        graph = self._get_graph(*key)
        self._static_in[key].copy_(input_features)
        graph.replay()
        return self._static_out[key].clone().contiguous()

    def forward_eager(self, input_features: torch.Tensor) -> torch.Tensor:
        """Direct encoder call for A/B testing."""
        if input_features.dtype != self.dtype:
            input_features = input_features.to(self.dtype)
        out = self.audio_encoder(
            input_features=input_features, return_dict=True
        )
        return out.last_hidden_state


__all__ = ["FusedEncoder"]
