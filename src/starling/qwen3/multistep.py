"""Qwen3 decoder with shared K-step graph generation."""

from __future__ import annotations
from typing import Any, Optional
import torch
from .config import EOS_TOKEN_ID, EOS_TOKEN_IDS
from .llm_mega import FusedLLMMega, GenerateResult
from .._kernels._compile import torch_compile
from ..multistep import MultiStepDecoder


class MultiStepLLMMega(MultiStepDecoder, FusedLLMMega):
    def __init__(
        self,
        language_model: Any,
        lm_head: Any,
        max_cache_len: int = 4096,
        steps_per_replay: int = 8,
        warmup_iters: int = 3,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        eos_token_id: int = EOS_TOKEN_ID,
        eos_token_ids: tuple[int, ...] | list[int] | None = None,
        compile_decode: bool = True,
        prefill_use_graph: bool = True,
    ) -> None:
        super().__init__(
            language_model,
            lm_head,
            max_cache_len=max_cache_len,
            warmup_iters=warmup_iters,
            device=device,
            dtype=dtype,
            eos_token_id=eos_token_id,
            eos_token_ids=tuple(eos_token_ids) if eos_token_ids is not None else EOS_TOKEN_IDS,
            prefill_use_graph=prefill_use_graph,
        )
        if compile_decode:
            self._decode_step_eager = torch_compile(
                self._decode_step_eager, mode="max-autotune-no-cudagraphs"
            )
        self._init_multistep(steps_per_replay, device)

    @torch.inference_mode()
    def generate(
        self,
        inputs_embeds: torch.Tensor,
        max_new_tokens: int = 200,
        eos_token_id: Optional[int | tuple[int, ...] | list[int]] = None,
        eos_token_ids: Optional[tuple[int, ...] | list[int]] = None,
        tokenizer: Any = None,
        capture: bool = True,
    ) -> GenerateResult:
        if max_new_tokens <= 0:
            return self._finalize([], 0.0, tokenizer)
        if inputs_embeds.shape[0] != 1:
            raise ValueError("Multi-step decoding only supports batch=1.")
        prompt_len = inputs_embeds.shape[1]
        self._validate_token_budget(prompt_len, max_new_tokens)
        first_token = self.prefill(inputs_embeds, use_graph=self.prefill_use_graph)
        if eos_token_ids is None and isinstance(eos_token_id, (list, tuple)):
            eos_token_ids = tuple(eos_token_id)
            eos_token_id = None
        if eos_token_ids is None:
            stop: tuple[int, ...] = (
                (int(eos_token_id),) if eos_token_id is not None else tuple(self.eos_token_ids)
            )
        else:
            stop = tuple(int(t) for t in eos_token_ids)
        ids, elapsed = self._generate_multistep(
            first_token,
            prompt_len,
            max_new_tokens,
            stop,
            capture,
        )
        return self._finalize(ids, elapsed, tokenizer)
