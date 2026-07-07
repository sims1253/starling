"""Integrated GPU megakernel pipeline for nvidia/parakeet-unified-en-0.6b.

Wires the three byte-exact-verified components into one end-to-end audio->text
path that never leaves the GPU except for the final sentencepiece decode:

    audio_list (list[np.ndarray])
        -> GpuMelExtractor                 (GPU mel) -> (B, 128, T)
        -> GraphedEncoder / eager encoder  (24-layer Conformer) -> (B, T_enc, 1024)
        -> GraphedDecoder                  (CUDA-graph greedy RNN-T) -> list[str]

The :class:`GraphedDecoder` is shape-specific: capture allocates static buffers
keyed on ``(B, T_enc)`` and builds one ``torch.cuda.CUDAGraph``. The pipeline
caches one captured decoder per shape so the one-off capture cost is amortised
across all calls of the same shape (the production-realistic shape: capture
once, decode many).

Public API
----------
:class:`MegaParakeetUnifiedPipeline`
    ``MegaParakeetUnifiedPipeline(...).transcribe(audio_list) -> list[str]``
    ``MegaParakeetUnifiedPipeline(...).transcribe_with_timing(audio_list)
        -> (list[str], {"mel_ms","encoder_ms","decode_ms","total_ms"})``
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from . import config as C
from .decode_mega import GraphedDecoder
from .encoder_graph import GraphedEncoder
from .loader import load_state_dict
from .mel_gpu import GpuMelExtractor
from .modeling import ConformerEncoder, RNNTDecoder, RNNTJoint
from .tokenizer import ParakeetUnifiedTokenizer

ENCODER_MODES = ("eager", "graphed")


def _load_modules(device, dtype):
    """Load the mel + encoder + decoder + joint + tokenizer from the .nemo."""
    sd = load_state_dict(device=device, dtype=dtype)
    mel = GpuMelExtractor(sd, device=device)
    enc = ConformerEncoder().to(device).to(dtype).eval()
    enc.load_state_dict_prefixed(sd)
    dec = RNNTDecoder().to(device).to(dtype).eval()
    dec.load_state_dict(
        {k[len("decoder."):]: v for k, v in sd.items() if k.startswith("decoder.")},
        strict=True,
    )
    joint = RNNTJoint().to(device).to(dtype).eval()
    joint.load_state_dict(
        {k[len("joint."):]: v for k, v in sd.items() if k.startswith("joint.")},
        strict=True,
    )
    # flatten the LSTM weights so cudnn does not need to recompact on every call
    # (silences the RNN-flatten warning + lets the captured decode graph replay
    # without an internal weight-copy).
    dec.prediction.dec_rnn.lstm.flatten_parameters()
    tok = ParakeetUnifiedTokenizer()
    return mel, enc, dec, joint, tok


class MegaParakeetUnifiedPipeline:
    """End-to-end GPU ASR for parakeet-unified-en-0.6b (NeMo-free port).

    audio -> GPU mel -> 24-layer Conformer encoder -> CUDA-graph greedy RNN-T
    decode -> text. All stages run on-device; the only host touch in the hot
    path is the per-replay device->host token sync inside the graphed decode
    loop (intrinsic to the RNN-T loop) and the final sentencepiece detokenize.

    Args:
        model_id: HuggingFace model id (default
            ``nvidia/parakeet-unified-en-0.6b``). Only used to resolve the
            .nemo checkpoint via the HF cache.
        device: target device (default ``"cuda"``).
        dtype: encoder/decoder dtype. ``torch.float32`` (default) reproduces the
            fp32 golden reference byte-for-byte; ``torch.bfloat16`` runs faster
            and is byte-identical to the bf16 eager path (the fp32-vs-bf16
            difference is the standard sub-ULP Conformer rounding that flips an
            occasional argmax).
        encoder_mode: ``"graphed"`` (default, CUDA-graph-captured encoder,
            byte-exact with eager max_diff 0.0) or ``"eager"`` (the stock
            hand-built ``ConformerEncoder.forward``).
        steps_per_replay: number of RNN-T emission steps captured into ONE
            graph replay. ``None`` (default) auto-selects by encoded length on
            RTX 5090: K=16 for short clips, K=64 for longer clips. The host syncs once per K steps
            instead of once per step. ``1`` reproduces one-step-per-replay.
        max_cached_shapes: cap on the per-shape encoder + decoder graph caches.
    """

    def __init__(
        self,
        model_id: str = C.MODEL_ID,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        encoder_mode: str = "graphed",
        steps_per_replay: int | None = None,
        max_cached_shapes: int = 512,
    ) -> None:
        if encoder_mode not in ENCODER_MODES:
            raise ValueError(
                f"encoder_mode={encoder_mode!r} not in {ENCODER_MODES}"
            )
        self.model_id = model_id
        self.device = torch.device(device)
        self.dtype = dtype
        self.encoder_mode = encoder_mode
        self.steps_per_replay = (
            None if steps_per_replay is None else max(1, int(steps_per_replay))
        )
        self.max_cached_shapes = int(max_cached_shapes)

        self.mel, self.encoder, self.decoder, self.joint, self.tokenizer = (
            _load_modules(self.device, dtype)
        )

        # ONE shared CUDA graph pool for the encoder + decoder captures.
        self._graph_pool = torch.cuda.graph_pool_handle()
        self._graphed_encoder = (
            GraphedEncoder(
                self.encoder,
                max_cached_shapes=self.max_cached_shapes,
                graph_pool=self._graph_pool,
            )
            if encoder_mode == "graphed"
            else None
        )
        # decoder cache (per (B, T_enc) shape); built lazily on first decode
        self._decoders: Dict[Tuple[int, int], GraphedDecoder] = {}

    # ------------------------------------------------------------------ #
    # shape-keyed graphed-decoder cache (amortise capture across calls)
    # ------------------------------------------------------------------ #
    def _get_decoder(
        self, encoded: torch.Tensor, valid_lengths: torch.Tensor
    ) -> GraphedDecoder:
        """Return a captured :class:`GraphedDecoder` for this ``(B, T_enc)``."""
        B, T_enc, _ = encoded.shape
        key = (int(B), int(T_enc))
        dec = self._decoders.get(key)
        if dec is None:
            if len(self._decoders) >= self.max_cached_shapes:
                import gc

                old_dec = self._decoders.pop(next(iter(self._decoders)))
                try:
                    if getattr(old_dec, "graph", None) is not None:
                        old_dec.graph.reset()
                except Exception:
                    pass
                del old_dec
                gc.collect()
            dec = GraphedDecoder(
                self.decoder, self.joint,
                blank_id=C.BLANK_ID, vocab_size=C.VOCAB_SIZE,
                max_symbols=C.MAX_SYMBOLS_PER_STEP,
                pred_hidden=C.PRED_HIDDEN, n_layers=C.PRED_RNN_LAYERS,
                steps_per_replay=self._steps_for_shape(T_enc),
                graph_pool=self._graph_pool,
            ).capture(encoded, valid_lengths)
            self._decoders[key] = dec
        return dec

    def _steps_for_shape(self, t_enc: int) -> int:
        """Replay chunk size for this encoded length.

        RTX 5090 sweeps on 2026-07-06: short fixture (T=93) prefers K=16;
        medium fixture (T=279) prefers K=64. Explicit constructor values still
        override this auto policy.
        """
        if self.steps_per_replay is not None:
            return self.steps_per_replay
        return 16 if int(t_enc) <= 128 else 64

    # ------------------------------------------------------------------ #
    # encoder dispatch (graphed / eager)
    # ------------------------------------------------------------------ #
    def _run_encoder(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._graphed_encoder is not None:
            return self._graphed_encoder(features, lengths)
        with torch.inference_mode():
            return self.encoder(features, lengths)

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
            list of ``B`` decoded text strings, byte-exact with the eager
            greedy RNN-T oracle (:func:`decode_eager.greedy_decode`) at the
            pipeline's dtype.
        """
        features, lengths = self.mel(audio_list)
        features = features.to(self.dtype)
        encoded, valid_lengths = self._run_encoder(features, lengths)
        decoder = self._get_decoder(encoded, valid_lengths)
        ids_per_batch = decoder.decode(encoded, valid_lengths)
        return [self.tokenizer.ids_to_text(ids) for ids in ids_per_batch]

    # ------------------------------------------------------------------ #
    # transcribe + per-stage timing (cuda events; for the benchmark)
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def transcribe_with_timing(
        self, audio_list: List[np.ndarray]
    ) -> Tuple[List[str], dict]:
        """Like :meth:`transcribe` but also return per-stage ms via cuda events.

        Returns ``(texts, timing)`` where ``timing`` has keys ``mel_ms``,
        ``encoder_ms``, ``decode_ms``, ``total_ms`` (floats, ms). Each stage is
        bracketed by its own cuda-event pair + synchronize. Graph capture (first
        call for a new shape) is amortised and NOT counted.
        """

        def _timed(fn):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn()
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end), out

        mel_ms, (features, lengths) = _timed(lambda: self.mel(audio_list))
        features = features.to(self.dtype)

        encoder_ms, (encoded, valid_lengths) = _timed(
            lambda: self._run_encoder(features, lengths)
        )

        decoder = self._get_decoder(encoded, valid_lengths)
        decode_ms, ids_per_batch = _timed(
            lambda: decoder.decode(encoded, valid_lengths)
        )
        texts = [self.tokenizer.ids_to_text(ids) for ids in ids_per_batch]

        timing = {
            "mel_ms": mel_ms,
            "encoder_ms": encoder_ms,
            "decode_ms": decode_ms,
            "total_ms": mel_ms + encoder_ms + decode_ms,
        }
        return texts, timing

    @torch.inference_mode()
    def prewarm(self, durations_s: List[float] | None = None) -> None:
        """Pre-capture CUDA graphs for common audio durations.

        Args:
            durations_s: list of durations (seconds) to pre-capture. Default
                covers common live utterance lengths: [5, 10, 30].
        """
        if durations_s is None:
            durations_s = [5.0, 10.0, 30.0]
        sr = C.SAMPLE_RATE
        for dur in durations_s:
            n = int(dur * sr)
            dummy = np.zeros(n, dtype=np.float32)
            self.transcribe([dummy])
        torch.cuda.synchronize()


__all__ = ["MegaParakeetUnifiedPipeline", "ENCODER_MODES"]
