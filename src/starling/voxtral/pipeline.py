"""Eager Voxtral Realtime transcription pipeline mirroring stock ``generate``.

Transcription path::

    wav -> processor (prompt ids + mel + num_delay_tokens, opaque output)
        -> embedder once over the whole mel -> all_embeds (1, E, 1280)
        -> prefill: encoder over the first P*4 embeds (own DynamicCache),
           projected audio added to the prompt token embeddings, text
           decoder forward (own DynamicCache)
        -> per step: encoder over the next 4 embeds, project, add to the
           new token's embedding, text decoder forward, greedy argmax
        -> stop at EOS or the stock total-length bound ceil(mel/8)

Audio injection is additive (``inputs_embeds += audio_embeds`` at every
position of the current slice). The per-step shapes are static -- encoder
slice (1, 4, 1280), text step (1, 1, 3072) -- so a CUDA-graphed decode path
can slot in behind :meth:`VoxtralPipeline._audio_step` /
:meth:`VoxtralPipeline._text_step` later; streaming (generator
``input_features`` + delay tokens) is out of scope for v1.

AdaRMSNorm handling: ``num_delay_tokens`` is fixed per request, so each of
the 26 text layers' modulation vectors is constant. The fast path (default)
precomputes them once and serves them from frozen modules instead of
running the time embedding + per-layer ada linears every step. The slow
path calls the stock ``model.forward`` per step with no reimplementation
and is the GPU-box oracle the fast path is byte-checked against.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Union

import numpy as np
import torch
import torch.nn as nn

from .audio import prepare_processor_inputs, read_wav, stock_max_length
from .config import (
    DEFAULT_MAX_CACHE_LEN,
    DOWNSAMPLE_FACTOR,
    EOS_TOKEN_ID,
    SAMPLE_RATE,
)
from .loader import get_components


def generation_cap(
    prompt_len: int, mel_T: int, max_new_tokens: int | None = None
) -> int:
    """Total-sequence-length cap mirroring the stock length logic.

    Stock ``generate`` defaults ``max_length`` to ``ceil(mel/8)`` (prompt +
    generated) and clamps any user ``max_new_tokens``-derived ``max_length``
    down to the same bound. ``None`` reproduces the pure stock default.

    Args:
        prompt_len: Prompt token count P.
        mel_T: Mel-frame count.
        max_new_tokens: Optional user decode budget.

    Returns:
        The total length (prompt + generated) at which to stop.
    """
    bound = stock_max_length(mel_T)
    if max_new_tokens is None:
        return bound
    return min(int(prompt_len) + max(0, int(max_new_tokens)), bound)


def slice_bounds(consumed_text_tokens: int, cur_text_tokens: int) -> tuple[int, int]:
    """Pre-encoder embed slice for a forward covering token range.

    Mirrors ``prepare_inputs_for_generation``: the encoder consumes
    ``DOWNSAMPLE_FACTOR`` embeds per text token, so a forward over
    ``cur`` tokens starting at ``consumed`` reads embeds
    ``[consumed*4, (consumed+cur)*4)``.

    Args:
        consumed_text_tokens: Text tokens already processed (cache length).
        cur_text_tokens: Text tokens in the current forward.

    Returns:
        ``(start, end)`` embed indices into the upfront embedder output.
    """
    start = int(consumed_text_tokens) * DOWNSAMPLE_FACTOR
    return start, start + int(cur_text_tokens) * DOWNSAMPLE_FACTOR


class _FrozenAdaMod(nn.Module):
    """AdaRMSNorm stand-in serving one precomputed modulation vector.

    Stock decoder layers call ``ada_rms_norm(t_cond)`` every forward; with
    ``num_delay_tokens`` fixed per request the result is constant, so the
    fast path swaps each layer's module for this one holding the cached
    ``(1, 3072)`` vector. Same input weights, same output bits, no
    per-step recompute.
    """

    def __init__(self, modulation: torch.Tensor) -> None:
        super().__init__()
        self.modulation = modulation

    def forward(self, _t_cond: torch.Tensor) -> torch.Tensor:
        return self.modulation


class VoxtralPipeline:
    """Eager Voxtral Realtime pipeline owning model + processor.

    Parameters
    ----------
    model : VoxtralRealtimeForConditionalGeneration
        The fully loaded model.
    processor : VoxtralRealtimeProcessor
        Provides prompt ``input_ids``, ``input_features`` mel, and the
        fixed per-request ``num_delay_tokens`` (opaque processor output).
    max_cache_len : int
        Upper bound for prompt + decode budget; guards the future static
        cache (v1 uses stock DynamicCaches).
    use_precomputed_ada : bool
        True (default) selects the fast path with precomputed per-layer
        ada modulation; False selects the exact-slow stock-forward path.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        max_cache_len: int = DEFAULT_MAX_CACHE_LEN,
        use_precomputed_ada: bool = True,
    ) -> None:
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_cache_len = int(max_cache_len)
        self.use_precomputed_ada = bool(use_precomputed_ada)

        comps = get_components(model)
        self._audio_tower = comps["audio_tower"]
        self._projector = comps["multi_modal_projector"]
        self._language_model = comps["language_model"]
        self._lm_head = comps["lm_head"]
        self._embed_tokens = comps["embed_tokens"]
        self._time_embedding = comps["time_embedding"]

    @classmethod
    def from_pretrained(
        cls,
        *,
        attn_impl: str = "eager",
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        max_cache_len: int = DEFAULT_MAX_CACHE_LEN,
        use_precomputed_ada: bool = True,
    ) -> "VoxtralPipeline":
        """Load the model + processor and wrap them in a VoxtralPipeline."""
        from .loader import load_model_and_processor

        model, processor = load_model_and_processor(
            attn_impl=attn_impl, dtype=dtype, device=device
        )
        return cls(
            model,
            processor,
            max_cache_len=max_cache_len,
            use_precomputed_ada=use_precomputed_ada,
        )

    # ------------------------------------------------------------------ #
    # input preparation
    # ------------------------------------------------------------------ #
    def _read_wav_or_array(
        self, wav_or_path: Union[str, Path, np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """Return a mono float32 waveform at 16 kHz from a path or array."""
        if isinstance(wav_or_path, (str, Path)):
            wav, sr = read_wav(str(wav_or_path))
        elif isinstance(wav_or_path, torch.Tensor):
            wav = wav_or_path.detach().cpu().to(torch.float32).numpy()
            sr = SAMPLE_RATE
        else:
            wav = np.ascontiguousarray(wav_or_path, dtype=np.float32)
            sr = SAMPLE_RATE
        if sr != SAMPLE_RATE:
            raise ValueError(f"expected {SAMPLE_RATE} Hz audio, got {sr} Hz")
        wav = np.ascontiguousarray(wav, dtype=np.float32)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        return np.ascontiguousarray(wav, dtype=np.float32)

    def _prepare_batch(
        self, wav: np.ndarray
    ) -> dict[str, Any]:
        """Run the stock processor and move outputs to the model device.

        Float tensors (mel) are cast to the model dtype, mirroring the
        stock doc-pattern ``inputs.to(device, dtype)``; integer tensors
        (ids, masks) move device-only.
        """
        input_ids, mel, num_delay = prepare_processor_inputs(self.processor, wav)
        dev = self._embed_tokens.weight.device
        dt = self._embed_tokens.weight.dtype
        batch: dict[str, Any] = {
            "input_ids": input_ids.to(dev),
            "input_features": mel.to(dev, dt) if mel.is_floating_point() else mel.to(dev),
            "num_delay_tokens": int(num_delay),
        }
        return batch
    # ------------------------------------------------------------------ #
    # fast-path building blocks (static per-step shapes for future graphs)
    # ------------------------------------------------------------------ #
    def _embed_all(self, mel: torch.Tensor) -> torch.Tensor:
        """Run the embedder once over the whole mel: (1, mel_T, 128) -> (1, E, 1280).

        Mirrors ``_prepare_model_inputs`` (plain offline left-pad, no conv
        cache); per-step encoder slices never touch the convs.
        """
        return self._audio_tower.embedder(mel)

    def _pad_slice_to_rows(
        self, all_embeds: torch.Tensor, start: int, end: int
    ) -> tuple[torch.Tensor, int]:
        """Clamp ``[start, end)`` to the embeds, zero-padding short reads.

        Real processor outputs always satisfy ``E % 4 == 0`` and
        ``E >= P*4`` (the streaming pads dominate), so padding only triggers
        on synthetic inputs where stock would fail the projector reshape.

        Returns:
            ``(slice, rows)`` with ``slice`` a multiple-of-4 frame count and
            ``rows`` the projected audio-token rows the caller keeps.
        """
        total = int(all_embeds.shape[1])
        want = max(0, int(end) - int(start))
        have = max(0, min(int(end), total) - max(int(start), 0))
        piece = all_embeds[:, max(int(start), 0) : max(int(start), 0) + have, :]
        if have < want:
            pad = piece.new_zeros((piece.shape[0], want - have, piece.shape[2]))
            piece = torch.cat([piece, pad], dim=1)
        extra = (-piece.shape[1]) % DOWNSAMPLE_FACTOR
        if extra:
            pad = piece.new_zeros((piece.shape[0], extra, piece.shape[2]))
            piece = torch.cat([piece, pad], dim=1)
        return piece, want // DOWNSAMPLE_FACTOR

    def _encode_embeds(
        self, enc_slice: torch.Tensor, enc_cache: Any | None
    ) -> tuple[torch.Tensor, Any]:
        """Encoder transformer over one embed slice via the stock helper.

        Static decode shape: (1, 4, 1280) in -> (1, 1, 3072) audio out.

        Args:
            enc_slice: ``(1, 4k, 1280)`` post-embedder frames.
            enc_cache: Encoder KV cache (None creates a stock DynamicCache).

        Returns:
            ``(audio_embeds, enc_cache)`` with audio ``(1, k, 3072)``.
        """
        out = self.model.get_audio_features(
            encoder_inputs_embeds=enc_slice,
            past_key_values=enc_cache,
            use_cache=True,
        )
        return out.pooler_output, out.past_key_values

    def _text_step(
        self,
        inputs_embeds: torch.Tensor,
        text_cache: Any | None,
        attention_mask: torch.Tensor | None,
        t_cond: torch.Tensor,
    ) -> tuple[torch.Tensor, Any]:
        """Text decoder forward over one slice via the stock module.

        Static decode shape: (1, 1, 3072) in -> (1, 1, 3072) hidden out.

        Args:
            inputs_embeds: ``(1, T, 3072)`` (token embeds + audio).
            text_cache: Decoder KV cache (None creates a stock DynamicCache).
            attention_mask: Prefill mask, or None on decode steps (the
                extended all-ones mask stock builds is equivalent to None).
            t_cond: ``(1, 3072)`` delay conditioning (constant per request).

        Returns:
            ``(hidden_states, text_cache)``.
        """
        out = self._language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=text_cache,
            use_cache=True,
            t_cond=t_cond,
        )
        return out.last_hidden_state, out.past_key_values

    def _precompute_ada(
        self, num_delay_tokens: int, ref_embeds: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Precompute ``t_cond`` + each text layer's ada modulation vector.

        Stock builds ``time_tensor`` on the inputs' device/dtype every
        forward; the value depends only on the fixed ``num_delay_tokens``,
        so one computation per request suffices.
        """
        time_tensor = torch.full(
            (1,),
            int(num_delay_tokens),
            device=ref_embeds.device,
            dtype=ref_embeds.dtype,
        )
        t_cond = self._time_embedding(time_tensor)[None, ...]
        mods = [layer.ada_rms_norm(t_cond) for layer in self._language_model.layers]
        return t_cond, mods

    @contextmanager
    def _frozen_ada(
        self, mods: list[torch.Tensor]
    ) -> Iterator[None]:
        """Swap each text layer's ada module for its precomputed vector."""
        layers = self._language_model.layers
        originals = [layer.ada_rms_norm for layer in layers]
        try:
            for layer, mod in zip(layers, mods):
                layer.ada_rms_norm = _FrozenAdaMod(mod)
            yield
        finally:
            for layer, orig in zip(layers, originals):
                layer.ada_rms_norm = orig

    # ------------------------------------------------------------------ #
    # greedy loops
    # ------------------------------------------------------------------ #
    @staticmethod
    def _greedy_id(logits: torch.Tensor) -> int:
        """Argmax over the last position (greedy decode, no sampling)."""
        return int(logits[0, -1].argmax().item())

    def _check_cache_fit(self, cap: int) -> None:
        """Guard the prompt + budget against the static-cache bound."""
        if cap > self.max_cache_len:
            raise ValueError(
                f"total length cap {cap} exceeds max_cache_len {self.max_cache_len}"
            )

    @torch.inference_mode()
    def _transcribe_fast(self, batch: dict[str, Any], cap: int) -> list[int]:
        """Greedy loop from stock pieces + precomputed ada modulation."""
        input_ids: torch.Tensor = batch["input_ids"]
        mel: torch.Tensor = batch["input_features"]
        num_delay: int = batch["num_delay_tokens"]
        prompt_len = int(input_ids.shape[1])

        all_embeds = self._embed_all(mel)

        # Prefill: encoder over the first P*4 embeds, audio added to prompt.
        pre_slice, pre_rows = self._pad_slice_to_rows(
            all_embeds, *slice_bounds(0, prompt_len)
        )
        audio_embeds, enc_cache = self._encode_embeds(pre_slice, None)
        inputs_embeds = self._embed_tokens(input_ids) + audio_embeds[:, :pre_rows, :].to(
            self._embed_tokens.weight.device, self._embed_tokens.weight.dtype
        )

        t_cond, mods = self._precompute_ada(num_delay, inputs_embeds)
        hidden, text_cache = self._text_step(inputs_embeds, None, None, t_cond)
        with self._frozen_ada(mods):
            next_id = self._greedy_id(self._lm_head(hidden))
            generated = [next_id]
            while prompt_len + len(generated) < cap and next_id != EOS_TOKEN_ID:
                start, end = slice_bounds(prompt_len + len(generated) - 1, 1)
                step_slice, _ = self._pad_slice_to_rows(all_embeds, start, end)
                step_audio, enc_cache = self._encode_embeds(step_slice, enc_cache)
                tok = torch.tensor(
                    [[next_id]], device=input_ids.device, dtype=input_ids.dtype
                )
                step_embeds = self._embed_tokens(tok) + step_audio.to(
                    self._embed_tokens.weight.device,
                    self._embed_tokens.weight.dtype,
                )
                hidden, text_cache = self._text_step(step_embeds, text_cache, None, t_cond)
                next_id = self._greedy_id(self._lm_head(hidden))
                generated.append(next_id)
        return generated

    @torch.inference_mode()
    def _transcribe_slow(self, batch: dict[str, Any], cap: int) -> list[int]:
        """Greedy loop calling the stock ``model.forward`` per step.

        Zero reimplementation: prefill + every decode step go through
        ``VoxtralRealtimeForConditionalGeneration.forward`` (embedding,
        encoder slice, time embedding, ada recompute, text forward,
        lm_head) with stock-created DynamicCaches. Offline the conv
        ``padding_cache`` is inert (the embedder runs once upfront), so it
        is left at the stock default.
        """
        model = self.model
        input_ids: torch.Tensor = batch["input_ids"]
        mel: torch.Tensor = batch["input_features"]
        num_delay: int = batch["num_delay_tokens"]
        prompt_len = int(input_ids.shape[1])

        all_embeds = self._embed_all(mel)
        pre_slice, _ = self._pad_slice_to_rows(all_embeds, *slice_bounds(0, prompt_len))

        out = model(
            input_ids=input_ids,
            encoder_inputs_embeds=pre_slice,
            past_key_values=None,
            encoder_past_key_values=None,
            use_cache=True,
            num_delay_tokens=num_delay,
        )
        text_cache, enc_cache = out.past_key_values, out.encoder_past_key_values
        next_id = self._greedy_id(out.logits)
        generated = [next_id]
        while prompt_len + len(generated) < cap and next_id != EOS_TOKEN_ID:
            start, end = slice_bounds(prompt_len + len(generated) - 1, 1)
            step_slice, _ = self._pad_slice_to_rows(all_embeds, start, end)
            tok = torch.tensor(
                [[next_id]], device=input_ids.device, dtype=input_ids.dtype
            )
            out = model(
                input_ids=tok,
                encoder_inputs_embeds=step_slice,
                past_key_values=text_cache,
                encoder_past_key_values=enc_cache,
                use_cache=True,
                num_delay_tokens=num_delay,
            )
            text_cache, enc_cache = out.past_key_values, out.encoder_past_key_values
            next_id = self._greedy_id(out.logits)
            generated.append(next_id)
        return generated

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def transcribe(
        self,
        wav_or_path: Union[str, Path, np.ndarray, torch.Tensor],
        max_new_tokens: int | None = None,
    ) -> tuple[str, torch.Tensor]:
        """Transcribe one utterance with the eager stock-mirroring loop.

        Args:
            wav_or_path: Path to a 16 kHz wav file, or a mono float32
                waveform at 16 kHz.
            max_new_tokens: Optional decode budget; ``None`` (default)
                reproduces the pure stock length bound ``ceil(mel/8)``.

        Returns:
            ``(transcript_text, generated_token_ids)`` where the ids are
            ``(1, n_new)`` int64 on CPU (generated tokens only, EOS
            included when emitted).
        """
        wav = self._read_wav_or_array(wav_or_path)
        batch = self._prepare_batch(wav)
        prompt_len = int(batch["input_ids"].shape[1])
        mel_T = int(batch["input_features"].shape[-1])
        cap = generation_cap(prompt_len, mel_T, max_new_tokens)
        self._check_cache_fit(cap)
        if cap <= prompt_len:
            generated = []
        elif self.use_precomputed_ada:
            generated = self._transcribe_fast(batch, cap)
        else:
            generated = self._transcribe_slow(batch, cap)
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        ids = torch.tensor(generated, dtype=torch.int64).reshape(1, -1)
        return text, ids

    @torch.inference_mode()
    def transcribe_stock(
        self,
        wav_or_path: Union[str, Path, np.ndarray, torch.Tensor],
        **generate_kwargs: Any,
    ) -> tuple[str, torch.Tensor]:
        """Transcribe via the stock ``model.generate(**processor(...))``.

        Thin wrapper used for golden/parity capture until starling kernels
        land; with no extra kwargs this is pure stock greedy decode
        (default length bound ``ceil(mel/8)``, EOS from generation_config).

        Args:
            wav_or_path: Path to a 16 kHz wav file, or a mono float32
                waveform at 16 kHz.
            generate_kwargs: Forwarded to ``model.generate``.

        Returns:
            ``(transcript_text, generated_token_ids)`` with ``(1, n_new)``
            int64 ids on CPU.
        """
        wav = self._read_wav_or_array(wav_or_path)
        batch = self._prepare_batch(wav)
        prompt_len = int(batch["input_ids"].shape[1])
        gen = self.model.generate(**batch, **generate_kwargs)
        new_ids = gen[:, prompt_len:].cpu()
        text = self.tokenizer.decode(
            new_ids[0].tolist(), skip_special_tokens=True
        )
        return text, new_ids

    def prewarm(self) -> None:
        """Capture the future CUDA-graph decode path (not yet implemented).

        Raises:
            NotImplementedError: The graphed decode (static encoder slice
                (1, 4, 1280) + static text step (1, 1, 3072)) has no GPU
                implementation yet; use :meth:`transcribe` (eager) or
                :meth:`transcribe_stock` until it lands.
        """
        raise NotImplementedError(
            "graphed Voxtral decode is not implemented yet "
            "(static shapes: encoder slice (1, 4, 1280), text step (1, 1, 3072))"
        )
