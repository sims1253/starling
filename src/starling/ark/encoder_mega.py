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

from collections import OrderedDict
from typing import Any

import torch

from .config import ENCODER_NUM_MEL_BINS

# Hard cap on the number of distinct ``(B, mel_T)`` graphs retained. The ARK
# encoder is a Whisper encoder with *global* bidirectional self-attention, so
# padding the mel to coarser buckets would change every output row and break
# byte-exactness. Instead each distinct mel length captures its own graph, and
# the cache is bounded by an LRU eviction policy: when a new shape arrives and
# the cache is full, the least-recently-used graph is dropped and its private
# CUDA-graph memory pool is released (``del graph; torch.cuda.empty_cache()``).
# This keeps memory bounded across benchmarks that feed clips of many different
# lengths (voxpopuli, ami, earnings22, gigaspeech, ...) without leaking the
# ~6-12GB of pool memory each captured graph holds.
MAX_ENCODER_GRAPHS: int = 512


class FusedEncoder:
    """CUDA-graph capture of ``audio_encoder(audios)``.

    One graph is captured per ``(batch, mel_T)`` shape (shape-keyed cache). On
    the first call for a new shape the encoder is warmed up three times on a
    side stream (so cuBLAS/cuDNN lazy init settles) and then captured; later
    calls with the same shape copy the input into the static input buffer and
    replay the graph.

    The cache is bounded to ``MAX_ENCODER_GRAPHS`` entries via LRU eviction so
    that processing clips of many different lengths cannot accumulate CUDA-graph
    memory without bound (the original cause of the OOM/reboot). Evicted graphs
    release their private memory pool explicitly.

    Args:
        audio_encoder: The ``AudioMLPAdapter`` (``model.audio_encoder``).
        device: Target device.
        dtype: Encoder dtype (bf16 is the checkpoint dtype).
        max_graphs: Maximum number of distinct shapes whose graphs are kept
            resident (LRU eviction beyond this). Defaults to
            ``MAX_ENCODER_GRAPHS``.
    """

    def __init__(
        self,
        audio_encoder: Any,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        *,
        max_graphs: int = MAX_ENCODER_GRAPHS,
        graph_pool=None,
    ) -> None:
        self.audio_encoder = audio_encoder
        self.device = device
        self.dtype = dtype
        self.max_graphs = max(1, int(max_graphs))
        # Shared CUDA graph pool (from the pipeline). All captures share it so
        # evicting one graph (``del``) frees only its blocks; the pool + other
        # graphs stay valid. Without it eviction corrupts the context.
        self.graph_pool = graph_pool
        # Shape-keyed capture cache: (B, mel_T) -> CUDAGraph. ``OrderedDict`` so
        # the least-recently-used entry can be evicted when a new shape arrives
        # (move_to_end on every access). Each entry also pins the static input /
        # output buffers that the captured graph reads/writes.
        self._graphs: OrderedDict[tuple[int, int], torch.cuda.CUDAGraph] = OrderedDict()
        self._static_in: dict[tuple[int, int], torch.Tensor] = {}
        self._static_out: dict[tuple[int, int], torch.Tensor] = {}

    # ------------------------------------------------------------------ #
    # graph capture (lazy, per shape) + LRU eviction
    # ------------------------------------------------------------------ #
    def _free_graph(self, key: tuple[int, int]) -> None:
        """Release a captured graph + its static buffers.

        Each graph has its OWN private pool (capture used pool=None), so
        ``graph.reset()`` deterministically frees that pool's blocks without
        affecting any other graph. This is the safe eviction path; a shared
        pool + ``del`` left freed-but-unrecycled blocks -> cudaErrorIllegalAddress.
        """
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
        """Evict the LRU entry while the cache exceeds ``max_graphs``."""
        while len(self._graphs) >= self.max_graphs:
            # popitem(last=False) -> least-recently-used (head of the OrderedDict).
            key, _ = self._graphs.popitem(last=False)
            self._free_graph(key)

    def _get_graph(self, B: int, mel_T: int) -> torch.cuda.CUDAGraph:
        """Capture (if needed) and return the graph for shape ``(B, mel_T)``.

        Marks the entry most-recently-used on hit. On miss, evicts the LRU entry
        first (if the cache is full) and captures a fresh graph.
        """
        key = (B, mel_T)
        graph = self._graphs.get(key)
        if graph is not None:
            self._graphs.move_to_end(key)  # mark most-recently-used
            return graph

        # New shape: make room (evict the LRU graph + free its pool) so the
        # resident-graph count never exceeds ``max_graphs``.
        self._evict_if_needed()

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

        # Capture with a PRIVATE pool per graph (pool=None -> PyTorch allocates
        # a fresh private pool). On eviction we ``graph.reset()`` to
        # deterministically free that pool's blocks. A SHARED pool was tried but
        # ``del``-based eviction left freed-but-unrecycled blocks that corrupted
        # the next capture -> cudaErrorIllegalAddress. Private + reset is safe.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = self.audio_encoder(static_audios)
        # Clone the output OUT of the graph's private pool so reading it after
        # eviction (reset) doesn't dangle. __call__ also clones on read, but
        # this belt-and-suspenders guards the stored reference.
        self._static_out[key] = static_out

        self._graphs[key] = graph  # appended at the MRU tail
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
