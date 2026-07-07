"""CUDA-graph-captured Conformer encoder for parakeet-unified-en-0.6b.

Mirrors the sibling :class:`starling.parakeet.encoder_graph.GraphedEncoder`
mechanics but wraps the hand-built :class:`~starling.parakeet_unified.modeling.ConformerEncoder`
(the unified model is NeMo-free; there is no HF ``model.get_audio_features``
to call -- we capture ``encoder(features, lengths)`` directly).

Why graph the encoder
---------------------
The 24-layer Conformer is launch-overhead bound at small-medium batch (hundreds
of tiny per-layer kernels, each ~us of host launch latency). A CUDA graph
collapses those into one replay. On the TDT sibling this measured ~1.36x at
B8 medium with byte-exact output (max_diff 0.0); the same applies here.

Static-buffer strategy
----------------------
The graph reads two static buffers (allocated once per shape, tagged with
``torch._dynamo.mark_static_address``) and writes its outputs at fixed addresses:

* ``static_inp``    (B, F=128, T)  -- the mel features
* ``static_lengths`` (B,)           -- per-element mel-frame counts

On each call the new data is copied in and the graph is replayed. The captured
``encoded`` (B, T_enc, 1024) and ``enc_lengths`` (B,) live at fixed addresses;
``__call__`` returns clones so callers cannot mutate the captured state.

Shape caching
-------------
One graph per ``(B, T)`` shape is cached, so the one-off capture cost is
amortised across same-shape calls (the production-realistic shape: capture
once, encode many).
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

try:
    from torch._dynamo import mark_static_address as _mark_static
except Exception:  # pragma: no cover - older torch
    def _mark_static(t):  # type: ignore[misc]
        return t


def _mark_many(tensors) -> None:
    for t in tensors:
        try:
            _mark_static(t)
        except Exception:
            pass


class GraphedEncoder:
    """Capture ``encoder(features, lengths)`` into one CUDA graph; encode many.

    The graph is shape-specific (``B``, ``T`` fixed at capture time); one
    :class:`GraphedEncoder` caches one captured graph per ``(B, T)`` shape so
    the capture cost is amortised across same-shape calls.

    Args:
        encoder: a loaded :class:`~starling.parakeet_unified.modeling.ConformerEncoder`
            on cuda (eval mode).
        warmup_iters: side-stream warmup iterations before graph capture
            (stabilises cudnn/cublas autotune for the conv subsampling + the
            24 conformer layers).
    """

    def __init__(
        self,
        encoder,
        *,
        warmup_iters: int = 3,
        max_cached_shapes: int = 512,
        graph_pool=None,
    ) -> None:
        self.encoder = encoder
        self.warmup_iters = int(warmup_iters)
        self.max_cached_shapes = int(max_cached_shapes)
        self.graph_pool = graph_pool  # shared torch.cuda.graph_pool_handle() or None
        # (B, T) -> bundle: static_inp, static_lengths, static_out, static_lens, graph
        self._graphs: Dict[Tuple[int, int], dict] = {}

    # ------------------------------------------------------------------ #
    # shape-keyed capture (amortise capture across same-shape calls)
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def _capture(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> dict:
        """Allocate static buffers for this ``(B, T)`` shape and capture."""
        B = int(features.shape[0])
        T_mel = int(features.shape[-1])

        static_inp = torch.empty_like(features)
        static_lengths = torch.empty_like(lengths)
        _mark_many([static_inp, static_lengths])
        static_inp.copy_(features)
        static_lengths.copy_(lengths)

        # warmup on a side stream (stabilises cudnn/cublas autotune before
        # capture; mirrors the GraphedDecoder capture pattern in decode_mega).
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(self.warmup_iters):
                self.encoder(static_inp, static_lengths)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        # Capture with a PRIVATE pool per graph (mirrors the TDT sibling): on
        # eviction ``graph.reset()`` deterministically frees that pool's blocks
        # without corrupting other captured graphs (shared-pool + ``del`` left
        # freed-but-unrecycled blocks -> cudaErrorIllegalAddress on this build).
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out, static_lens = self.encoder(static_inp, static_lengths)

        bundle = {
            "B": B,
            "T_mel": T_mel,
            "static_inp": static_inp,
            "static_lengths": static_lengths,
            "static_out": static_out,
            "static_lens": static_lens,
            "graph": graph,
        }
        # Private-pool LRU eviction: reset() frees this graph's pool without
        # affecting others; re-capture reproduces the identical graph.
        if len(self._graphs) >= self.max_cached_shapes:
            import gc

            old = self._graphs.pop(next(iter(self._graphs)))
            try:
                old["graph"].reset()
            except Exception:
                pass
            del old
            gc.collect()
        self._graphs[(B, T_mel)] = bundle
        return bundle

    # ------------------------------------------------------------------ #
    # encode
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def __call__(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the captured encoder for this batch; return fresh clones.

        Args:
            features: ``(B, F=128, T)`` mel features on cuda.
            lengths: ``(B,)`` per-element mel-frame counts on cuda.

        Returns:
            ``(encoded, enc_lengths)`` clones of the captured outputs (so the
            caller cannot mutate the captured state). Byte-exact with the eager
            ``encoder(features, lengths)`` path (max_diff 0.0).
        """
        B = int(features.shape[0])
        T_mel = int(features.shape[-1])
        key = (B, T_mel)
        bundle = self._graphs.get(key)
        if bundle is None:
            bundle = self._capture(features, lengths)

        bundle["static_inp"].copy_(features)
        bundle["static_lengths"].copy_(lengths)
        bundle["graph"].replay()

        return (
            bundle["static_out"].clone(),
            bundle["static_lens"].clone(),
        )


__all__ = ["GraphedEncoder"]
