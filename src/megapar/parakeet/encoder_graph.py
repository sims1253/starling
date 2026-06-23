"""CUDA-graph-captured Conformer encoder for nvidia/parakeet-tdt-0.6b-v3.

Same I/O and byte-exact output as the stock
``model.get_audio_features(input_features, attention_mask)`` call, but captured
into a single ``torch.cuda.CUDAGraph`` and served by ``graph.replay()``.

Why graph the encoder
---------------------
The stock 24-layer Conformer encoder is ~43 ms at batch=8 medium but only ~10%
GPU-busy: it is launch-overhead bound (hundreds of tiny per-layer kernels with
sequential dependencies, each paying ~us of host launch latency). A CUDA graph
collapses those launches into one replay, measured at ~32 ms (1.36x, byte-exact,
max_diff 0.0). See ``outputs/parakeet/pipeline_bench_graphed.json`` for the
integrated RTF impact.

Static-buffer strategy
----------------------
The graph reads two static buffers (allocated once per shape, tagged with
``torch._dynamo.mark_static_address``) and writes its outputs at fixed addresses:

* ``static_inp``  (B, T_mel, 128) -- the bf16 mel features
* ``static_mask`` (B, T_mel)       -- the bool attention mask

On each call, the new input data is copied into the static buffers and the graph
is replayed. The captured encoder output (a ``ParakeetEncoderModelOutput``) is
stored by reference; ``__call__`` returns a *fresh* dataclass whose
``pooler_output`` and ``attention_mask`` are clones of the static output tensors,
so callers cannot mutate the captured state (a subsequent replay would otherwise
overwrite it). The decode step consumes ``pooler_output.contiguous()`` and
``attention_mask.sum(-1)`` -- neither of which mutates -- so the clone is
defensive but never required by the integrated pipeline.

Shape caching
-------------
One graph per ``(B, T_mel)`` shape is cached in a dict, so the one-off capture
cost is amortised across all calls of the same shape (the production-realistic
shape: capture once, encode many). A first call for a new shape pays capture;
every later same-shape call is a dict lookup + buffer copy + replay.
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
    """Capture ``model.get_audio_features`` into one CUDA graph; encode many inputs.

    The graph is shape-specific (``B``, ``T_mel`` fixed at capture time); one
    :class:`GraphedEncoder` caches one captured graph per ``(B, T_mel)`` shape so
    the capture cost is amortised across same-shape calls.

    Args:
        model: a loaded ``ParakeetForTDT`` on cuda (eval mode, bf16).
        warmup_iters: side-stream warmup iterations before graph capture
            (stabilises cudnn/cublas autotune for the conv subsampling + the
            24 conformer layers).
    """

    def __init__(self, model, *, warmup_iters: int = 3) -> None:
        self.model = model
        self.warmup_iters = int(warmup_iters)
        # (B, T_mel) -> bundle(dict): static_inp, static_mask, static_out, graph
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

        # static input buffers (fixed GPU addresses for the graph's lifetime)
        static_inp = torch.empty_like(input_features)
        static_mask = torch.empty_like(attention_mask)
        _mark_many([static_inp, static_mask])
        static_inp.copy_(input_features)
        static_mask.copy_(attention_mask)

        # warmup on a side stream (stabilises cudnn/cublas autotune before
        # capture; mirrors the GraphedDecoder capture pattern in decode_mega).
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(self.warmup_iters):
                self.model.get_audio_features(
                    input_features=static_inp, attention_mask=static_mask
                )
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        # capture the encoder forward into a CUDAGraph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = self.model.get_audio_features(
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
    ):
        """Run the captured encoder for this batch; return a fresh output.

        Args:
            input_features: ``(B, T_mel, 128)`` bf16 mel features on cuda.
            attention_mask: ``(B, T_mel)`` bool attention mask on cuda.

        Returns:
            A new ``ParakeetEncoderModelOutput``-typed object whose
            ``pooler_output`` (B, T_enc, 640) and ``attention_mask`` (B, T_enc)
            are clones of the static captured output, so the caller cannot
            mutate the captured state. ``last_hidden_state`` etc. are left at
            their dataclass defaults (the integrated pipeline only reads
            ``pooler_output`` and ``attention_mask``).

        Byte-exact with the eager ``model.get_audio_features`` path (max_diff
        0.0); the graph only removes host launch overhead.
        """
        B, T_mel = int(input_features.shape[0]), int(input_features.shape[1])
        key = (B, T_mel)
        bundle = self._graphs.get(key)
        if bundle is None:
            bundle = self._capture(input_features, attention_mask)

        # copy the new input data into the static buffers and replay
        bundle["static_inp"].copy_(input_features)
        bundle["static_mask"].copy_(attention_mask)
        bundle["graph"].replay()

        out = bundle["static_out"]
        out_cls = type(out)
        # fresh dataclass; clone pooler_output + attention_mask so callers can't
        # mutate the captured state that the next replay would overwrite.
        return out_cls(
            pooler_output=out.pooler_output.clone(),
            attention_mask=out.attention_mask.clone(),
        )
