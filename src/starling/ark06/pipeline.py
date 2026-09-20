"""End-to-end fused ASR megakernel pipeline for Audio8/ARK-ASR-0.6B.

Thin subclass of the 3B track's :class:`starling.ark.pipeline.MegaPipeline`:
the encoder dims, adapter layout, token ids, and audio front-end are identical,
and the fused LLM decoder derives head_dim / layer count / head counts from the
loaded tensors, so only model loading differs (the 0.6B hub id).

Public API
----------
``MegaPipeline(model, processor, *, steps_per_replay=None, max_cache_len=4096)``
``MegaPipeline.from_pretrained(...)``
``MegaPipeline.transcribe(audio_path_or_array, instruction=..., max_new_tokens=200) -> (text, token_ids)``
"""

from __future__ import annotations

from typing import Optional

import torch

from ..ark.audio import build_inputs_embeds, build_prompt_ids, extract_mel, read_wav
from ..ark.pipeline import MegaPipeline as _ArkMegaPipeline
from .config import build_bad_token_ids
from .loader import load_model_and_processor

__all__ = [
    "MegaPipeline",
    "build_inputs_embeds",
    "build_prompt_ids",
    "extract_mel",
    "read_wav",
]


class MegaPipeline(_ArkMegaPipeline):
    """End-to-end fused ARK-ASR-0.6B pipeline (encoder + fused LLM).

    Inherits transcribe / prewarm / bucketing / graph-mode toggles unchanged
    from the 3B track; only :meth:`from_pretrained` points at the 0.6B loader.

    The 0.6B degenerates into special-token repetition under plain greedy
    decode (verified on stock CPU: ~11 correct words, then a spiral). The ban
    set from :func:`starling.ark06.config.build_bad_token_ids` (the model
    card's ``build_bad_words_ids`` recipe) is computed ONCE here from the
    processor tokenizer and threaded into every decoder the parent builds, so
    banned ids can never win an in-graph argmax. The 3B parent keeps
    ``bad_token_ids=None`` (bit-identical path).
    """

    def __init__(self, *args, **kwargs) -> None:
        # Default ON for the 0.6B: pass suppress_special_tokens=False to opt
        # out (debugging the unsuppressed degeneration). The ban set is
        # computed once from the processor tokenizer after the parent init.
        _suppress = kwargs.pop("suppress_special_tokens", True)
        super().__init__(*args, **kwargs)
        if not _suppress:
            return
        self.bad_token_ids = frozenset(
            build_bad_token_ids(self.processor.tokenizer)
        )
        # Decoders are lazily built per K: the default one was constructed
        # during __init__ without the ban set — propagate it (and the
        # precomputed penalty row) to every cached decoder; _get_llm passes
        # self.bad_token_ids to future ones.
        for llm in self._llms.values():
            llm.bad_token_ids = self.bad_token_ids
            llm._bad_penalty = llm._build_bad_penalty(
                self.bad_token_ids, llm.vocab_size, llm.dtype, llm.device
            )

    @classmethod
    def from_pretrained(
        cls,
        *,
        encoder_mode: str = "cudagraph",
        steps_per_replay: Optional[int] = None,
        max_cache_len: int = 4096,
        attn_impl: str = "eager",
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        shape_bucketing: bool = True,
        mel_bucket_frames: int = 512,
        prefill_use_graph: bool = False,
    ) -> "MegaPipeline":
        """Load the 0.6B model + processor and wrap them in a MegaPipeline."""
        model, processor = load_model_and_processor(
            attn_impl=attn_impl, dtype=dtype, device=device
        )
        return cls(
            model,
            processor,
            encoder_mode=encoder_mode,
            steps_per_replay=steps_per_replay,
            max_cache_len=max_cache_len,
            shape_bucketing=shape_bucketing,
            mel_bucket_frames=mel_bucket_frames,
            prefill_use_graph=prefill_use_graph,
        )
