"""End-to-end graphed ASR pipeline for CohereLabs/cohere-transcribe-03-2026.

    mel (B, T, 128) -> GraphedEncoder (cudagraph) -> encoder hidden (B, S, 1280)
                   -> GraphedDecoder (K-step graphed seq2seq decode) -> token ids
                   -> processor.batch_decode -> transcript text

Both the encoder graph and the decode graph are byte-exact vs the stock eager
reference (``starling.cohere.reference``), so the end-to-end transcript
reproduces the golden reference exactly.

Public API
----------
``CohereMegaPipeline(model, processor)``
``CohereMegaPipeline.from_pretrained()``
``CohereMegaPipeline.transcribe(wav, language='en', ...) -> (text, ids)``
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from .config import MAX_AUDIO_CLIP_S, SAMPLE_RATE
from .decode_mega import GraphedDecoder
from .encoder_graph import GraphedEncoder


class CohereMegaPipeline:
    """End-to-end graphed ASR pipeline: graphed encoder + K-step graphed decode."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        max_cache_len: int = 1024,
        steps_per_replay: int = 16,
        warmup_iters: int = 4,
        encoder_warmup_iters: int = 3,
        use_graphed_encoder: bool = True,
    ) -> None:
        self.model = model
        self.processor = processor
        self.dtype = getattr(model, "dtype", torch.bfloat16)

        self.encoder = GraphedEncoder(
            model.model.encoder, warmup_iters=encoder_warmup_iters
        ) if use_graphed_encoder else None

        self.decoder = GraphedDecoder(
            model,
            max_cache_len=max_cache_len,
            steps_per_replay=steps_per_replay,
            warmup_iters=warmup_iters,
        )
        # captured graph is (B, prompt_len, S)-specific; re-capture on any change
        self._captured_key: Optional[tuple] = None

    @classmethod
    def from_pretrained(
        cls,
        *,
        max_cache_len: int = 1024,
        steps_per_replay: int = 8,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        **kwargs: Any,
    ) -> "CohereMegaPipeline":
        from .loader import load_model_and_processor

        model, processor = load_model_and_processor(dtype=dtype, device=device)
        return cls(
            model, processor,
            max_cache_len=max_cache_len, steps_per_replay=steps_per_replay,
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    def _encode(self, input_features: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.encoder is not None:
            return self.encoder(input_features, attention_mask)
        with torch.inference_mode():
            return self.model.model.encoder(
                input_features=input_features, attention_mask=attention_mask
            ).last_hidden_state

    def _ensure_capture(self, dec_in: torch.Tensor, enc_h: torch.Tensor, enc_mask: torch.Tensor) -> None:
        B, T = dec_in.shape
        S = enc_h.shape[1]
        key = (B, T, S)
        if self._captured_key != key:
            self.decoder.capture(dec_in, enc_h, enc_mask)
            self._captured_key = key

    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def transcribe(
        self,
        wav,
        *,
        language: str = "en",
        sampling_rate: int = SAMPLE_RATE,
        max_new_tokens: int = 300,
    ) -> tuple[list[str], torch.Tensor]:
        """End-to-end ASR: audio waveform(s) -> transcript text.

        Args:
            wav: a 1D/2D numpy array / torch tensor / list of arrays. Audio longer
                than ``max_audio_clip_s`` is auto-chunked by the processor (batch
                dim > 1); each chunk is decoded independently.
            language: source language code (one of the model's 14 supported langs).

        Returns ``(texts, ids)`` where ``texts`` is a list of ``B`` transcript
        strings and ``ids`` is a ``(B, n_gen)`` CPU long tensor of generated ids.
        """
        import numpy as np

        if isinstance(wav, torch.Tensor):
            wav = wav.numpy()
        inp = self.processor(
            wav, sampling_rate=sampling_rate, language=language, return_tensors="pt"
        )
        feat = inp["input_features"].to(self.dtype).cuda()
        amask = inp["attention_mask"].cuda()
        dec_in = inp["decoder_input_ids"].cuda()

        enc_h = self._encode(feat, amask)
        B, S = enc_h.shape[:2]
        neg = torch.finfo(self.dtype).min
        enc_mask = torch.zeros(B, 1, 1, S, device=enc_h.device, dtype=enc_h.dtype)

        self._ensure_capture(dec_in, enc_h, enc_mask)
        ids = self.decoder.decode(dec_in, enc_h, enc_mask, max_new_tokens=max_new_tokens)

        # decode text: full sequence = prompt + generated, per row
        full = torch.cat([dec_in.cpu().expand(ids.shape[0], -1), ids.cpu()], dim=1)
        texts = self.processor.batch_decode(full, skip_special_tokens=True)
        return texts, ids
