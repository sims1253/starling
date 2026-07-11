"""End-to-end fused ASR megakernel pipeline for MOSS-Transcribe-preview-2B.

    mel (1, 128, T) -> GraphedAudioEncoder (cudagraph) -> audio_embeds (N, 2048)
                   -> merge into LLM inputs_embeds (masked_scatter)
                   -> MossMultiStepMega.generate(...) -> generated token ids
                   -> tokenizer.batch_decode -> transcript text

The encoder + adapter graph and the LLM decode graph are byte-exact vs the
stock eager reference, so the end-to-end transcript reproduces the golden
reference exactly.

Public API
----------
``MossMegaPipeline(model, processor)``
``MossMegaPipeline.from_pretrained()``
``MossMegaPipeline.transcribe(wav_or_features, input_ids, audio_input_mask, ...) -> (text, ids)``
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from .config import LLM_EOS_TOKEN_ID
from .encoder_graph import GraphedAudioEncoder
from .loader import get_components
from .multistep import MossMultiStepMega


class MossMegaPipeline:
    """End-to-end fused ASR pipeline: graphed encoder + K-step graphed LLM."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        max_cache_len: int = 1024,
        steps_per_replay: Optional[int] = None,
        use_multistep: bool = True,
        encoder_mode: str = "eager",
        compile_decode: bool = True,
    ) -> None:
        from ..flags import get_default_flags

        self.model = model
        self.processor = processor
        self.dtype = getattr(model, "dtype", torch.bfloat16)
        self.max_cache_len = int(max_cache_len)
        self.steps_per_replay = steps_per_replay
        self.use_multistep = bool(use_multistep)
        self.compile_decode = bool(compile_decode)
        comps = get_components(model)
        self.fused_encoder = GraphedAudioEncoder(
            comps["audio_model"], comps["audio_adapter"], mode=encoder_mode
        )
        self.embed_tokens = comps["embed_tokens"]
        self.lm_head = comps["lm_head"]
        self.language_model = comps["language_model"]

        from .fused_decode import FusedMossLLMMega
        from .multistep import FusedMossMultiStepMega

        self._multi_cls = FusedMossMultiStepMega
        self._single_cls = FusedMossLLMMega
        self._llms: dict[int, Any] = {}
        if use_multistep:
            self.llm = self._get_llm(0)
        else:
            self.llm = FusedMossLLMMega(
                comps["language_model"], comps["lm_head"],
                max_cache_len=max_cache_len, compile_decode=compile_decode,
            )

    def _steps_for_shape(self, prompt_len: int) -> int:
        """Select K from prompt length unless the caller forced it."""
        if self.steps_per_replay is not None:
            return max(1, int(self.steps_per_replay))
        return 2 if prompt_len <= 160 else 4

    def _get_llm(self, prompt_len: int):
        if not self.use_multistep:
            return self.llm
        k = self._steps_for_shape(prompt_len)
        llm = self._llms.get(k)
        if llm is None:
            llm = self._multi_cls(
                self.language_model,
                self.lm_head,
                max_cache_len=self.max_cache_len,
                steps_per_replay=k,
                compile_decode=self.compile_decode,
            )
            self._llms[k] = llm
        self.llm = llm
        return llm

    @classmethod
    def from_pretrained(
        cls,
        *,
        max_cache_len: int = 1024,
        steps_per_replay: Optional[int] = None,
        use_multistep: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ) -> "MossMegaPipeline":
        from .loader import load_model_and_processor

        model, processor = load_model_and_processor(dtype=dtype, device=device)
        return cls(
            model, processor,
            max_cache_len=max_cache_len, steps_per_replay=steps_per_replay,
            use_multistep=use_multistep,
        )

    # ------------------------------------------------------------------ #
    def build_inputs_embeds(
        self, input_ids: torch.Tensor, audio_embeds: torch.Tensor,
        audio_input_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Merge audio embeds into the LLM token embeddings (byte-exact)."""
        inputs_embeds = self.embed_tokens(input_ids)
        mask_expanded = audio_input_mask.unsqueeze(-1).expand_as(inputs_embeds)
        return inputs_embeds.masked_scatter(mask_expanded, audio_embeds)

    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def transcribe(
        self,
        audio_data: torch.Tensor,
        audio_data_seqlens: torch.Tensor,
        input_ids: torch.Tensor,
        audio_input_mask: torch.Tensor,
        max_new_tokens: int = 200,
    ) -> tuple[str, torch.Tensor]:
        """End-to-end ASR: mel -> transcript text.

        Args are the processor outputs (audio mel features, seqlens, the
        chat-template input_ids with audio slots, and the audio slot mask).
        """
        # (1) graphed encoder + adapter -> audio_embeds (N, 2048)
        audio_embeds = self.fused_encoder(audio_data, audio_data_seqlens)
        # (2) merge into multimodal inputs_embeds (byte-exact vs stock)
        inputs_embeds = self.build_inputs_embeds(input_ids, audio_embeds, audio_input_mask)
        # (3) K-step graphed greedy generate
        llm = self._get_llm(int(inputs_embeds.shape[1]))
        res = llm.generate(
            inputs_embeds, max_new_tokens=max_new_tokens, eos_token_id=LLM_EOS_TOKEN_ID
        )
        # (4) decode generated ids to text
        text = self.processor.tokenizer.batch_decode(
            res.ids, skip_special_tokens=True
        )[0]
        return text, res.ids

    def prewarm(self, mel_len: int = 743) -> None:
        """Pre-capture encoder + LLM graphs with a dummy utterance.

        ``mel_len`` frames of audio (~4.6s at hop=160/sr=16k) drives both the
        encoder capture (fixed mel length) and the LLM decode graph capture.
        """
        device = self.embed_tokens.weight.device
        audio_data = torch.zeros(128, mel_len, dtype=self.dtype, device=device)
        seqlens = torch.tensor([mel_len], dtype=torch.long, device=device)
        # build a minimal input_ids with one audio slot
        from .config import START_TOKEN_ID, AUDIO_START_TOKEN_ID, AUDIO_END_TOKEN_ID, AUDIO_PLACEHOLDER_ID

        from starling.moss.reference import _get_feat_extract_output_lengths  # noqa
        # num audio tokens for this mel len
        from . import config as C

        class _P:
            pass

        # replicate the processor length math (matches MelConfig):
        #   raw_mel_len -> num_audio_tokens via the processor's formula.
        raw = mel_len
        input_lengths_leave = raw % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        num_audio_tokens = int(
            ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (raw // 100) * 13
        )
        ids = (
            [START_TOKEN_ID, AUDIO_START_TOKEN_ID]
            + [AUDIO_PLACEHOLDER_ID] * num_audio_tokens
            + [AUDIO_END_TOKEN_ID]
        )
        mask = [False, False] + [True] * num_audio_tokens + [False]
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        audio_input_mask = torch.tensor([mask], dtype=torch.bool, device=device)
        self.transcribe(
            audio_data, seqlens, input_ids, audio_input_mask, max_new_tokens=8
        )
        torch.cuda.synchronize()
