"""CUDA-graph-captured Parakeet encoder for CohereLabs/cohere-transcribe-03-2026.

Same I/O and byte-exact output as the stock
``model.model.encoder(input_features, attention_mask)`` call, but captured into
a single ``torch.cuda.CUDAGraph`` and served by ``graph.replay()``.

The 48-layer FastConformer encoder is launch-overhead bound (hundreds of tiny
per-layer kernels with sequential dependencies); a CUDA graph collapses those
launches into one replay. One graph per ``(B, T_mel)`` shape is cached so the
one-off capture cost is amortised across same-shape calls.

This is the same pattern as ``starling.parakeet.encoder_graph.GraphedEncoder``,
adapted to capture the cohere model's own ``model.model.encoder`` forward
(which returns a ``BaseModelOutput`` with ``last_hidden_state`` instead of the
parakeet TDT's ``pooler_output``).
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
    """Capture the Parakeet encoder forward into one CUDA graph; encode many.

    The graph is shape-specific (``B``, ``T_mel`` fixed at capture time); one
    :class:`GraphedEncoder` caches one captured graph per ``(B, T_mel)`` shape.

    Args:
        encoder: ``model.model.encoder`` of a loaded ``CohereAsrForConditionalGeneration``
            on cuda (eval mode, bf16).
        warmup_iters: side-stream warmup iterations before graph capture
            (stabilises cudnn/cublas autotune for the conv subsampling + the
            48 conformer layers).
    """

    def __init__(self, encoder, *, warmup_iters: int = 3) -> None:
        self.encoder = encoder
        self.warmup_iters = int(warmup_iters)
        # (B, T_mel) -> bundle: static_inp, static_mask, static_out, graph
        self._graphs: Dict[Tuple[int, int], dict] = {}

    # ------------------------------------------------------------------ #
    # shape-keyed capture (amortise capture across same-shape calls)
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def _capture(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict:
        """Allocate static buffers for this ``(B, T_mel)`` shape and capture."""
        B, T_mel = int(input_features.shape[0]), int(input_features.shape[1])

        static_inp = torch.empty_like(input_features)
        static_mask = torch.empty_like(attention_mask)
        _mark_many([static_inp, static_mask])
        static_inp.copy_(input_features)
        static_mask.copy_(attention_mask)

        # warmup on a side stream (stabilises cudnn/cublas autotune before
        # capture; mirrors starling.parakeet.encoder_graph).
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(self.warmup_iters):
                self.encoder(
                    input_features=static_inp, attention_mask=static_mask
                )
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = self.encoder(
                input_features=static_inp, attention_mask=static_mask
            )

        bundle = {
            "B": B,
            "T_mel": T_mel,
            "static_inp": static_inp,
            "static_mask": static_mask,
            "static_out": static_out,
            "graph": graph,
        }
        self._graphs[(B, T_mel)] = bundle
        return bundle

    # ------------------------------------------------------------------ #
    # encode
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def __call__(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the captured encoder; return a *fresh* ``last_hidden_state`` clone.

        Args:
            input_features: ``(B, T_mel, 128)`` bf16 mel features on cuda.
            attention_mask: ``(B, T_mel)`` attention mask on cuda.

        Returns:
            ``(B, T_enc, 1280)`` encoder hidden states (a clone of the captured
            output, so callers cannot mutate the captured state that the next
            replay would overwrite). Byte-exact with the eager encoder path.
        """
        B, T_mel = int(input_features.shape[0]), int(input_features.shape[1])
        key = (B, T_mel)
        bundle = self._graphs.get(key)
        if bundle is None:
            bundle = self._capture(input_features, attention_mask)

        bundle["static_inp"].copy_(input_features)
        bundle["static_mask"].copy_(attention_mask)
        bundle["graph"].replay()

        return bundle["static_out"].last_hidden_state.clone()
