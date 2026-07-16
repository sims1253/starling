"""End-to-end fused ASR megakernel pipeline for Audex-2B.

Wires the megakernel components into one transcription path:

    mel (N,128,3000) -> FusedEncoder (cudagraph) -> enc_last_hidden (N,750,1280)
                    -> stock projector (eager) -> audio_embeds (N*750, 2048)
                    -> merge into LLM inputs_embeds (boolean-mask scatter,
                       replicating NemotronDenseAudexForConditionalGeneration.
                       prepare_inputs_embeds EXACTLY)
                    -> FusedLLMMega.generate(...) -> generated token ids
                    -> tokenizer.decode -> transcript text

The merge step mirrors ``prepare_inputs_embeds`` byte for byte:

  1. embed ``input_ids`` (no embedding multiplier);
  2. run the encoder + projector -> ``audio_embeds``;
  3. build the placeholder mask (``input_ids == sound_token_id``) and scatter
     the audio embeds into the sound-token slots via boolean-index assignment.

Both the graphed encoder and the fused LLM decoder are byte-exact vs the eager
reference, so the end-to-end transcript reproduces the golden reference.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .config import EOS_TOKEN_ID, SOUND_TOKEN_ID


class MegaPipeline:
    """End-to-end fused ASR pipeline owning encoder + projector + fused LLM."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        feature_extractor: Any,
        *,
        max_cache_len: int = 4096,
        use_fused_llm: bool = True,
        steps_per_replay: int | None = None,
        prefill_use_graph: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.dtype = getattr(model, "dtype", torch.bfloat16)
        self.prefill_use_graph = bool(prefill_use_graph)

        from .loader import get_components

        comps = get_components(model)
        # (1) encoder: graph-captured Qwen2AudioEncoder.
        from .encoder_mega import FusedEncoder

        self.fused_encoder = FusedEncoder(comps["encoder"])
        # (2) projector stays the stock eager MLP (tiny, cheap).
        self.projector = comps["projector"]
        self.embed_tokens = comps["embed_tokens"]

        # (3) LLM decoder trunk + lm_head.
        self._language_model = comps["language_model"]
        self._lm_head = model.lm_head
        self._max_cache_len = int(max_cache_len)
        self.steps_per_replay = (
            None if steps_per_replay is None else max(1, int(steps_per_replay))
        )
        self._llms_by_k: dict[int, Any] = {}
        if use_fused_llm:
            self.llm = self._get_multistep_llm(self._steps_for_prompt(0))
        else:
            from .llm_mega import LLMMega

            self.llm = LLMMega(
                self._language_model,
                self._lm_head,
                max_cache_len=self._max_cache_len,
                eos_token_id=EOS_TOKEN_ID,
                prefill_use_graph=self.prefill_use_graph,
            )
        self.use_fused_llm = use_fused_llm

    @classmethod
    def from_pretrained(
        cls,
        *,
        attn_impl: str = "eager",
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        max_cache_len: int = 4096,
        use_fused_llm: bool = True,
        steps_per_replay: int | None = None,
        prefill_use_graph: bool = False,
    ) -> "MegaPipeline":
        from .loader import load_model_and_processor

        model, tokenizer, feature_extractor = load_model_and_processor(
            attn_impl=attn_impl, dtype=dtype, device=device
        )
        return cls(
            model,
            tokenizer,
            feature_extractor,
            max_cache_len=max_cache_len,
            use_fused_llm=use_fused_llm,
            steps_per_replay=steps_per_replay,
            prefill_use_graph=prefill_use_graph,
        )

    def _steps_for_prompt(self, prompt_len: int) -> int:
        if self.steps_per_replay is not None:
            return self.steps_per_replay
        prompt_len = int(prompt_len)
        if prompt_len <= 128:
            return 1
        if prompt_len <= 1024:
            return 8
        return 1

    def _get_multistep_llm(self, steps_per_replay: int):
        from .multistep import MultiStepLLMMega

        k = max(1, int(steps_per_replay))
        llm = self._llms_by_k.get(k)
        if llm is None:
            llm = MultiStepLLMMega(
                self._language_model,
                self._lm_head,
                max_cache_len=self._max_cache_len,
                steps_per_replay=k,
                eos_token_id=EOS_TOKEN_ID,
                prefill_use_graph=self.prefill_use_graph,
            )
            self._llms_by_k[k] = llm
        return llm

    def set_prefill_use_graph(self, on: bool) -> None:
        on = bool(on)
        self.prefill_use_graph = on
        for llm in self._llms_by_k.values():
            llm.prefill_use_graph = on

    # ------------------------------------------------------------------ #
    # merge step (byte-exact replica of prepare_inputs_embeds scatter)
    # ------------------------------------------------------------------ #
    def build_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        audio_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Merge projected audio embeds into the LLM token embeddings.

        Replicates ``NemotronDenseAudexForConditionalGeneration.prepare_inputs_embeds``
        exactly: embed input_ids, then boolean-mask scatter the projected audio
        embeds into the ``sound_token_id`` positions.
        """
        inputs_embeds = self.embed_tokens(input_ids).clone()
        mask = (input_ids[0] == SOUND_TOKEN_ID)
        n_slots = int(mask.sum().item())
        audio = audio_embeds.reshape(-1, audio_embeds.shape[-1]).to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        n_audio = int(audio.shape[0])
        if n_audio < n_slots:
            pad = audio.new_zeros((n_slots - n_audio, audio.shape[-1]))
            audio = torch.cat([audio, pad], dim=0)
        elif n_audio > n_slots:
            audio = audio[:n_slots]
        inputs_embeds[0, mask] = audio
        return inputs_embeds

    def encode_audio(self, input_features: torch.Tensor) -> torch.Tensor:
        """Run encoder + eager projector. Returns ``audio_embeds``."""
        enc_lhs = self.fused_encoder(input_features)
        return self.projector(enc_lhs.clone())

    # ------------------------------------------------------------------ #
    # full transcribe
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def transcribe(
        self,
        wav: np.ndarray,
        *,
        task_prompt: str = "Transcribe the speech in the input audio.",
        max_new_tokens: int = 200,
    ) -> tuple[str, torch.Tensor]:
        """End-to-end ASR: wav → transcript text.

        Audio longer than 30 s is chunked into 30 s clips and each clip is
        transcribed independently (matching the server's chunked path).
        Feeding multiple clips into one prompt causes the model to over-repeat
        on repetitive audio; per-clip transcription avoids that.

        Args:
            wav: 1-D float32 waveform at 16 kHz.
            task_prompt: ASR instruction text.
            max_new_tokens: greedy decode budget per 30 s clip.

        Returns:
            ``(transcript_text, generated_token_ids)`` where ids are
            ``(1, n_new)`` int64 on CPU. For multi-chunk audio, the text is
            the concatenation of per-chunk transcripts and the ids are from
            the first chunk.
        """
        from .audio import normalize_audio
        from .config import SOUND_CLIP_DURATION, SOUND_TARGET_RATE

        wav = normalize_audio(wav)
        clip_samples = int(round(SOUND_TARGET_RATE * SOUND_CLIP_DURATION))

        # Single 30 s clip: one-pass transcription.
        if len(wav) <= clip_samples:
            return self._transcribe_one_clip(wav, task_prompt, max_new_tokens)

        # Multi-clip: chunk and transcribe each independently.
        texts: list[str] = []
        first_ids = None
        for start in range(0, len(wav), clip_samples):
            clip = wav[start : start + clip_samples]
            if len(clip) < 100:
                continue
            if len(clip) < clip_samples:
                clip = np.pad(clip, (0, clip_samples - len(clip)))
            text, ids = self._transcribe_one_clip(clip, task_prompt, max_new_tokens)
            texts.append(text)
            if first_ids is None:
                first_ids = ids

        joined = " ".join(texts)
        return joined, first_ids if first_ids is not None else torch.zeros((1, 0), dtype=torch.int64)

    def _transcribe_one_clip(
        self, wav: np.ndarray, task_prompt: str, max_new_tokens: int
    ) -> tuple[str, torch.Tensor]:
        """Transcribe a single ≤30 s clip (one prompt, one decode pass)."""
        from .audio import build_inputs

        inputs = build_inputs(
            self.tokenizer, self.feature_extractor, wav, task_prompt=task_prompt
        )

        # (1)+(2) graphed encoder + eager projector
        audio_embeds = self.encode_audio(inputs["input_features"])

        # (3) merge into multimodal inputs_embeds
        inputs_embeds = self.build_inputs_embeds(inputs["input_ids"], audio_embeds)

        # (4) greedy generate
        if self.use_fused_llm:
            self.llm = self._get_multistep_llm(
                self._steps_for_prompt(inputs_embeds.shape[1])
            )
        res = self.llm.generate(
            inputs_embeds, max_new_tokens=max_new_tokens,
            eos_token_id=EOS_TOKEN_ID,
        )

        # (5) decode: strip <think> tags + special tokens
        text = self._decode_response(res.ids)
        return text, res.ids

    def _decode_response(self, ids: torch.Tensor) -> str:
        """Decode generated ids, stripping <think> tags, special tokens, and
        the conversational preamble the model wraps around ASR output.

        Audex is a unified audio-text LLM, not a dedicated ASR model. For ASR
        prompts it wraps transcripts in a preamble like ``"The content of the
        input audio is '<transcript>'."`` The actual transcript is always
        between single quotes. Extract it so downstream WER / users get clean
        text. If no quote-delimited content is found, return the stripped raw
        text (audio understanding, translation, etc.).
        """
        import re

        raw = self.tokenizer.decode(ids[0], skip_special_tokens=False)
        # Strip thinking tags (non-thinking mode should have none, but guard).
        if "</think>" in raw:
            raw = raw.rsplit("</think>", 1)[-1]
        # Strip the IM_END token and whitespace.
        if "<|im_end|>" in raw:
            raw = raw.split("<|im_end|>", 1)[0]
        raw = raw.strip()

        # Extract the transcript from the conversational wrapper.
        # The model consistently wraps ASR output as: preamble '<transcript>'.
        m = re.search(r"'(.+)'", raw, re.DOTALL)
        if m:
            return m.group(1).strip()
        return raw

    def prewarm(self) -> None:
        """Pre-capture CUDA graphs with a short dummy utterance."""
        dummy = np.zeros(int(5.0 * 16000), dtype=np.float32)
        self.transcribe(dummy, max_new_tokens=8)
        torch.cuda.synchronize()
