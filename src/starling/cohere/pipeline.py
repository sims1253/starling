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

from collections import OrderedDict
from typing import Any

import torch

from .config import MAX_AUDIO_CLIP_S, SAMPLE_RATE
from .decode_mega import GraphedDecoder
from .encoder_graph import GraphedEncoder


class CohereMegaPipeline:
    """End-to-end graphed ASR pipeline: graphed encoder + K-step graphed decode.

    Shape bucketing (why this pipeline used to be slower than stock)
    ---------------------------------------------------------------
    Both captured graphs are shape-specific: the encoder graph is keyed on
    ``(B, T_mel)`` and the decoder graph on ``(B, prompt_len, S, K)`` where
    ``S = subsample(T_mel)``. Real audio has ~all-distinct lengths, so *every
    clip re-captured both graphs and never replayed* — capture costs 284-2300 ms
    against 9-16 ms for a replay. That is the whole reason cohere-starling ran
    6x slower than stock eager transformers on spgispeech.

    There are two independent places to bucket, and they have very different
    correctness properties:

    ``cross_attn_bucketing`` (default on, **byte-exact**)
        Pads the *encoder output* ``S`` up to a multiple of ``enc_bucket_frames``
        and masks the padding out of cross-attention with ``-inf``. This shares
        the decoder graph across clip lengths. Measured byte-exact: identical
        greedy token ids on 100/100 real leaderboard clips, because the masked
        keys get exactly zero softmax weight and the decoder's self-attention
        already runs at a fixed ``max_cache_len``.

    ``shape_bucketing`` (default **off**, faster but *not* byte-exact)
        Right-pads the *mel* up to a multiple of ``mel_bucket_frames`` so the
        encoder graph is shared too. Padding cannot leak into valid frames --
        the encoder masks it in self-attention, zeroes it before the depthwise
        conv, and re-masks after every strided subsampling conv. But growing the
        post-subsampling length ``S`` retiles the 48-layer conformer's bf16
        reductions, which perturbs the encoder output by ~0.03 (about 2 bf16
        ulps; it falls to ~2e-4 relative in fp32, confirming it is numeric, not
        semantic). Greedy decoding is chaotic, so a single near-tie flip can
        cascade: 12/350 leaderboard clips (3.4%) change their raw transcript.
        Per-dataset WER moves by at most 0.18, in both directions.

        Evidence that it is the ``S`` growth and not padding per se: with
        ``mel_bucket_frames=8`` (which leaves ``S`` unchanged) divergence is
        0/50 clips.

    Measured on an RTX 5090 at n=50/dataset: the default is byte-exact at
    66-96x RTFx (stock eager transformers is 23-33x); ``shape_bucketing=True``
    reaches 138-212x. Both keep GPU memory flat (~4.8/5.0 GiB peak) because the
    captured-graph count is bounded by the bucket count, not the clip count.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        max_cache_len: int = 1024,
        steps_per_replay: int | None = None,
        warmup_iters: int = 4,
        encoder_warmup_iters: int = 3,
        use_graphed_encoder: bool | None = None,
        max_cached_shapes: int = 8,
        shape_bucketing: bool = False,
        mel_bucket_frames: int = 1024,
        max_mel_frames: int | None = None,
        cross_attn_bucketing: bool = True,
        enc_bucket_frames: int = 128,
    ) -> None:
        self.model = model
        self.processor = processor
        self.dtype = getattr(model, "dtype", torch.bfloat16)
        self.max_cache_len = int(max_cache_len)
        self.warmup_iters = int(warmup_iters)
        self.max_cached_shapes = int(max_cached_shapes)

        # Encoder-side (mel) bucketing: shares the encoder graph, perturbs bf16.
        self.shape_bucketing = bool(shape_bucketing)
        self.mel_bucket_frames = max(1, int(mel_bucket_frames))
        # Cap the bucket at the longest mel the processor can emit (it chunks
        # audio at MAX_AUDIO_CLIP_S). Without the cap, a 3501-frame clip would
        # pad to 4096 and burn 17% extra encoder compute for no graph reuse:
        # 3501 is already a canonical shape that every near-max clip shares.
        if max_mel_frames is None:
            hop = int(getattr(processor.feature_extractor, "hop_length", 160))
            max_mel_frames = MAX_AUDIO_CLIP_S * SAMPLE_RATE // hop + 1
        self.max_mel_frames = int(max_mel_frames)
        self._sub_len_cache: dict[int, int] = {}

        # Decoder-side (encoder-output) bucketing: shares the decoder graph,
        # byte-exact. Cheap, so it is on by default regardless of the above.
        self.cross_attn_bucketing = bool(cross_attn_bucketing)
        self.enc_bucket_frames = max(1, int(enc_bucket_frames))
        self.max_enc_frames = self._subsampled_len(self.max_mel_frames)

        # Graphing the encoder only pays off when its input shapes are bucketed;
        # otherwise it captures a fresh graph for nearly every clip, which is the
        # bug this class documents. Default to following ``shape_bucketing``.
        if use_graphed_encoder is None:
            use_graphed_encoder = self.shape_bucketing
        self.encoder = GraphedEncoder(
            model.model.encoder, warmup_iters=encoder_warmup_iters
        ) if use_graphed_encoder else None

        self.steps_per_replay = steps_per_replay
        # Captured decoder graphs are keyed by (B, prompt_len, encoder_len, K).
        # Reusing them matters for mixed-shape serving and repeated short/medium
        # alternation; the previous single decoder re-captured on every shape
        # switch. ``self.decoder`` remains the most recently used decoder for
        # compatibility with benchmark/debug callers.
        self._decoders: OrderedDict[tuple[int, int, int, int], GraphedDecoder] = OrderedDict()
        self.decoder = self._new_decoder(self._steps_for_shape(0))

    @classmethod
    def from_pretrained(
        cls,
        *,
        max_cache_len: int = 1024,
        steps_per_replay: int | None = None,
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
    # shape bucketing
    # ------------------------------------------------------------------ #
    def _subsampled_len(self, t_mel: int) -> int:
        """Encoder output length ``S`` for a mel of ``t_mel`` frames.

        Mirrors ``ParakeetEncoderSubsamplingConv2D._get_output_length`` over the
        strided conv stack, so it tracks the model's own arithmetic rather than
        hardcoding the 8x factor.
        """
        cached = self._sub_len_cache.get(t_mel)
        if cached is not None:
            return cached
        length = int(t_mel)
        for layer in self.model.model.encoder.subsampling.layers:
            if isinstance(layer, torch.nn.Conv2d) and layer.stride != (1, 1):
                pad, k, stride = layer.padding, layer.kernel_size[0], layer.stride[0]
                length = (length + pad[0] + pad[1] - k) // stride + 1
        self._sub_len_cache[t_mel] = length
        return length

    def _bucket_mel_len(self, t_mel: int) -> int:
        """Canonical padded ``T_mel`` for a natural mel length (identity if off)."""
        if not self.shape_bucketing or self.mel_bucket_frames == 1:
            return t_mel
        if t_mel >= self.max_mel_frames:
            return t_mel  # already at/over the cap; nothing to gain
        g = self.mel_bucket_frames
        return min(((t_mel + g - 1) // g) * g, self.max_mel_frames)

    def _maybe_bucket(
        self, input_features: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Right-pad mel + mask up to the canonical bucket ``T_mel`` (zero-copy if
        already on the grid). Padded mask positions are ``0``, so the encoder
        masks them out of self-attention and the conv stack."""
        t_mel = int(input_features.shape[1])
        bucket_t = self._bucket_mel_len(t_mel)
        if bucket_t == t_mel:
            return input_features, attention_mask
        B, F = int(input_features.shape[0]), int(input_features.shape[2])
        feats = torch.zeros(
            (B, bucket_t, F), dtype=input_features.dtype, device=input_features.device
        )
        feats[:, :t_mel].copy_(input_features)
        mask = torch.zeros(
            (B, bucket_t), dtype=attention_mask.dtype, device=attention_mask.device
        )
        mask[:, :t_mel].copy_(attention_mask)
        return feats, mask

    def _prepare_cross(
        self, enc_h: torch.Tensor, s_nat: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Bucket the encoder output on ``S`` and build the cross-attention mask.

        ``s_nat`` is the encoder length the *unbucketed* mel would have produced;
        everything at or past it is padding (either mel-bucket padding that the
        encoder already masked, or the ``S``-axis padding added here).

        Returns ``(enc_h, enc_mask)`` where ``enc_h`` is ``(B, S, H)`` on the
        bucket grid and ``enc_mask`` is a ``(B, 1, 1, S)`` additive mask that is
        ``-inf`` past ``s_nat``. Byte-exact: those keys get exactly zero softmax
        weight, so the decoder attends to precisely the frames it would have seen
        unbucketed and stops at the same natural length.
        """
        B, S, H = enc_h.shape
        s_buc = S
        if self.cross_attn_bucketing and self.enc_bucket_frames > 1:
            g = self.enc_bucket_frames
            s_buc = min(((S + g - 1) // g) * g, max(self.max_enc_frames, S))
        if s_buc > S:
            padded = torch.zeros((B, s_buc, H), dtype=enc_h.dtype, device=enc_h.device)
            padded[:, :S].copy_(enc_h)
            enc_h = padded

        enc_mask = torch.zeros(B, 1, 1, s_buc, device=enc_h.device, dtype=enc_h.dtype)
        if s_nat < s_buc:
            # Hide every padded frame from the decoder. Zeroing the hidden states
            # too keeps a stray non-finite value out of the 0*V term.
            enc_mask[..., s_nat:] = torch.finfo(enc_h.dtype).min
            enc_h[:, s_nat:].zero_()
        return enc_h, enc_mask

    # ------------------------------------------------------------------ #
    def _encode(self, input_features: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.encoder is not None:
            return self.encoder(input_features, attention_mask)
        with torch.inference_mode():
            return self.model.model.encoder(
                input_features=input_features, attention_mask=attention_mask
            ).last_hidden_state

    def _new_decoder(self, steps_per_replay: int) -> GraphedDecoder:
        return GraphedDecoder(
            self.model,
            max_cache_len=self.max_cache_len,
            steps_per_replay=steps_per_replay,
            warmup_iters=self.warmup_iters,
        )

    def _get_decoder(self, dec_in: torch.Tensor, enc_h: torch.Tensor, enc_mask: torch.Tensor) -> GraphedDecoder:
        B, T = dec_in.shape
        S = enc_h.shape[1]
        K = self._steps_for_shape(S)
        key = (B, T, S, K)
        decoder = self._decoders.get(key)
        if decoder is None:
            decoder = self._new_decoder(K)
            decoder.capture(dec_in, enc_h, enc_mask)
            self._decoders[key] = decoder
            while len(self._decoders) > self.max_cached_shapes:
                _, old = self._decoders.popitem(last=False)
                graph = getattr(old, "graph", None)
                if graph is not None:
                    try:
                        graph.reset()
                    except Exception:
                        pass
        else:
            self._decoders.move_to_end(key)
        self.decoder = decoder
        return decoder

    def _steps_for_shape(self, encoder_len: int) -> int:
        """Replay chunk size for this encoder length.

        RTX 5090 sweep on 2026-07-06: short fixture (S=93) prefers K=32;
        medium fixture (S=279) prefers K=8. Explicit constructor values still
        override this auto policy.
        """
        if self.steps_per_replay is not None:
            return max(1, int(self.steps_per_replay))
        return 32 if int(encoder_len) and int(encoder_len) <= 128 else 8

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

        # ``s_nat`` is the encoder length the unbucketed mel would have produced.
        # Everything past it is padding, whichever bucketing stage introduced it.
        s_nat = self._subsampled_len(int(feat.shape[1]))
        feat, amask = self._maybe_bucket(feat, amask)
        enc_h = self._encode(feat, amask)
        enc_h, enc_mask = self._prepare_cross(enc_h, s_nat)

        decoder = self._get_decoder(dec_in, enc_h, enc_mask)
        ids = decoder.decode(dec_in, enc_h, enc_mask, max_new_tokens=max_new_tokens)

        # decode text: full sequence = prompt + generated, per row
        full = torch.cat([dec_in.cpu().expand(ids.shape[0], -1), ids.cpu()], dim=1)
        texts = self.processor.batch_decode(full, skip_special_tokens=True)
        return texts, ids
