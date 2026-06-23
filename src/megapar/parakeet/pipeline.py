"""Integrated GPU megakernel pipeline for nvidia/parakeet-tdt-0.6b-v3.

Wires the three byte-exact-verified components into one end-to-end audio->text
path that never leaves the GPU except for the final text decode:

    audio_list (list[np.ndarray])
        -> GpuMelExtractor          (GPU mel, 15.8x faster at B8) -> (B,T,128) bf16
        -> GraphedEncoder / model.get_audio_features (24-layer Conformer encoder)
                                    -> pooler (B,T_enc,640)
        -> GraphedDecoder.decode    (CUDA-graph TDT decode, 6.65x) -> list[str]

The :class:`GraphedDecoder` is shape-specific: capture allocates static buffers
keyed on ``(B, T_enc)`` and builds one ``torch.cuda.CUDAGraph``. The pipeline
caches one captured decoder per shape so the one-off capture cost is amortised
across all calls of the same shape (the production-realistic shape: capture once,
decode many). A first call for a new shape pays capture; every later same-shape
call is a dict lookup + graph replays.

Public API
----------
:class:`MegaParakeetPipeline`
    ``MegaParakeetPipeline(model_id=...).transcribe(audio_list) -> list[str]``
    ``MegaParakeetPipeline(...).transcribe_with_timing(audio_list)
        -> (list[str], {"mel_ms","encoder_ms","decode_ms","total_ms"})``
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from .decode_mega import GraphedDecoder
from .encoder_graph import CompiledEncoder, GraphedEncoder
from .mel_gpu import GpuMelExtractor


# Valid encoder backends. ``encoder_mode`` takes precedence over the legacy
# boolean ``use_graphed_encoder`` flag (kept for backward compatibility).
ENCODER_MODES = ("eager", "graphed", "compiled")


class MegaParakeetPipeline:
    """End-to-end GPU ASR: GPU mel -> Conformer encoder -> graphed TDT decode.

    All three stages run on-device; the only host touch in the hot path is the
    per-step device->host token sync inside the graphed decode loop (intrinsic to
    the TDT loop, see :mod:`decode_mega`) and the final
    ``processor.batch_decode`` of the emitted token ids.

    Args:
        model_id: HuggingFace model id (default ``nvidia/parakeet-tdt-0.6b-v3``).
        device: target device (default ``"cuda"``).
        dtype: encoder/decoder dtype (default ``torch.bfloat16``). The GPU mel
            extractor runs in float32 internally; its output is cast to ``dtype``
            for the encoder (matching the baseline numerics and the oracle path).
        use_graphed_encoder: legacy bool flag. If ``encoder_mode`` is None, True
            selects the CUDA-graphed encoder (default), False selects the eager
            encoder. Ignored when ``encoder_mode`` is given.
        encoder_mode: encoder backend, one of ``ENCODER_MODES``:
            * ``"graphed"`` (default): :class:`GraphedEncoder` -- a CUDA-graph
              capture of ``model.get_audio_features`` that removes per-layer
              launch overhead (~1.36x faster at B8 medium, **byte-exact**).
            * ``"eager"``: the stock ``model.get_audio_features`` path (kept for
              byte-exactness A/B testing).
            * ``"compiled"``: :class:`CompiledEncoder` -- torch.compile
              (``reduce-overhead``) + BatchNorm1d fold. Fuses the elementwise /
              memop glue for extra speed but is **NOT guaranteed byte-exact**
              with eager/graphed; the correctness gate is a text-level
              transcript match vs the oracle. The encoder folds the conv-module
              BatchNorm1d into the depthwise conv (a fresh model is loaded for
              this mode, so the graphed/eager stock path is preserved for A/B).
    """

    def __init__(
        self,
        model_id: str = "nvidia/parakeet-tdt-0.6b-v3",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        use_graphed_encoder: bool = True,
        encoder_mode: str | None = None,
    ) -> None:
        # Local import: constructing the pipeline pays the HF import cost; keep
        # it out of module import time so `import pipeline` is cheap.
        from transformers import AutoModelForTDT, AutoProcessor

        self.model_id = model_id
        self.device = torch.device(device)
        self.dtype = dtype

        # Resolve the encoder backend. `encoder_mode` (explicit) wins over the
        # legacy `use_graphed_encoder` bool.
        if encoder_mode is None:
            encoder_mode = "graphed" if use_graphed_encoder else "eager"
        if encoder_mode not in ENCODER_MODES:
            raise ValueError(
                f"encoder_mode={encoder_mode!r} not in {ENCODER_MODES}"
            )
        self.encoder_mode = encoder_mode
        self.use_graphed_encoder = encoder_mode == "graphed"

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForTDT.from_pretrained(
            model_id, dtype=dtype, device_map=str(self.device)
        )
        self.model.eval()

        # (1) GPU mel extractor (float32 internally; cast to dtype in transcribe)
        self.mel = GpuMelExtractor(self.processor, device=str(self.device))

        # (2) encoder backend: "graphed" -> GraphedEncoder (CUDA-graph capture,
        # byte-exact), "compiled" -> CompiledEncoder (torch.compile + BN fold,
        # not guaranteed byte-exact), "eager" -> None (stock path).
        if encoder_mode == "graphed":
            self._graphed_encoder = GraphedEncoder(self.model)
        elif encoder_mode == "compiled":
            self._graphed_encoder = CompiledEncoder(self.model)
        else:
            self._graphed_encoder = None

        # (3) graphed decoder template; capture is shape-specific and is cached
        # per (B, T_enc) so the one-off capture cost is amortised across calls.
        # Pre-build (don't capture) so __init__ has no shape-dependent GPU work.
        self._decoders: Dict[Tuple[int, int], GraphedDecoder] = {}

        # pad token id (needed by the decoder's output buffer at capture time)
        self.pad_id = self.processor.tokenizer.pad_token_id

    # ------------------------------------------------------------------ #
    # shape-keyed graphed-decoder cache (amortise capture across calls)
    # ------------------------------------------------------------------ #
    def _get_decoder(
        self, pooler: torch.Tensor, valid_lengths: torch.Tensor
    ) -> GraphedDecoder:
        """Return a captured :class:`GraphedDecoder` for this ``(B, T_enc)``.

        On the first call for a shape, captures the graph on the current
        representative encoder output (capture is shape-only; the resulting
        graph is reused by ``decode`` for any same-shape input). Subsequent
        same-shape calls are a dict lookup, so capture is amortised.
        """
        B, T_enc, _ = pooler.shape
        key = (int(B), int(T_enc))
        dec = self._decoders.get(key)
        if dec is None:
            dec = GraphedDecoder(self.model).capture(
                pooler, valid_lengths, self.pad_id
            )
            self._decoders[key] = dec
        return dec

    # ------------------------------------------------------------------ #
    # encoder dispatch (graphed / compiled / eager)
    # ------------------------------------------------------------------ #
    def _run_encoder(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the 24-layer Conformer encoder + projector; return pooler + lengths.

        Dispatches on ``self.encoder_mode``: the graphed path replays a
        cached CUDA graph (byte-exact with eager; one capture per
        ``(B, T_mel)`` shape, amortised); the compiled path runs
        torch.compile + BN fold (NOT guaranteed byte-exact; one compile
        warmup per shape, amortised); the eager path is the stock
        ``model.get_audio_features``. All return the projector pooler output
        ``(B, T_enc, 640)`` as a contiguous tensor and the per-element valid
        encoder-frame lengths from ``attention_mask.sum(-1)``.
        """
        if self._graphed_encoder is not None:
            enc = self._graphed_encoder(input_features, attention_mask)
        else:
            enc = self.model.get_audio_features(
                input_features=input_features, attention_mask=attention_mask
            )
        pooler = enc.pooler_output.contiguous()
        valid_lengths = enc.attention_mask.to(torch.long).sum(-1).contiguous()
        return pooler, valid_lengths

    # ------------------------------------------------------------------ #
    # end-to-end transcription
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def transcribe(self, audio_list: List[np.ndarray]) -> List[str]:
        """Audio -> text end-to-end on GPU.

        Args:
            audio_list: list of 1D float32 mono arrays at 16 kHz (varying
                lengths); padded to the longest within the batch by the mel
                extractor.

        Returns:
            list of ``B`` decoded text strings (``skip_special_tokens=True``),
            byte-exact with the stock ``model.generate`` greedy path.
        """
        # (1) GPU mel; cast to bf16 for the encoder (matches the oracle path:
        # the baseline feeds bf16 features to the bf16 encoder).
        input_features, attention_mask = self.mel(audio_list)
        input_features = input_features.to(self.dtype)

        # (2) 24-layer Conformer encoder -> projector pooler output (graphed or
        # eager; both byte-exact). Graph capture is shape-keyed and amortised.
        pooler, valid_lengths = self._run_encoder(input_features, attention_mask)

        # (3) CUDA-graph TDT decode (shape-cached; capture amortised).
        decoder = self._get_decoder(pooler, valid_lengths)
        return decoder.decode(pooler, valid_lengths, self.processor)

    # ------------------------------------------------------------------ #
    # transcribe + per-stage timing (cuda events; for the benchmark)
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def transcribe_with_timing(
        self, audio_list: List[np.ndarray]
    ) -> Tuple[List[str], dict]:
        """Like :meth:`transcribe` but also return per-stage ms via cuda events.

        Returns ``(texts, timing)`` where ``timing`` has keys ``mel_ms``,
        ``encoder_ms``, ``decode_ms``, ``total_ms`` (all floats, ms). Each stage
        is bracketed by its own cuda-event pair + synchronize, so the stages do
        not overlap; ``total_ms`` is their sum. ``decode_ms`` includes
        ``processor.batch_decode`` (it is part of the integrated path). The
        graph capture (first call for a new shape) happens in ``_get_decoder``
        (decode) and inside ``GraphedEncoder`` (encoder) and is NOT counted in
        ``encoder_ms``/``decode_ms`` -- it is amortised across calls.
        """
        device = self.device

        def _timed(fn):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn()
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end), out

        # (1) mel
        mel_ms, (input_features, attention_mask) = _timed(
            lambda: self.mel(audio_list)
        )
        input_features = input_features.to(self.dtype)

        # (2) encoder (graphed or eager; graph capture is shape-keyed + amortised,
        # so after warmup _run_encoder is a dict lookup + graph replay)
        encoder_ms, (pooler, valid_lengths) = _timed(
            lambda: self._run_encoder(input_features, attention_mask)
        )

        # (3) decode (capture amortised -- _get_decoder is a dict hit after warmup)
        decoder = self._get_decoder(pooler, valid_lengths)
        decode_ms, texts = _timed(
            lambda: decoder.decode(pooler, valid_lengths, self.processor)
        )

        timing = {
            "mel_ms": mel_ms,
            "encoder_ms": encoder_ms,
            "decode_ms": decode_ms,
            "total_ms": mel_ms + encoder_ms + decode_ms,
        }
        return texts, timing
