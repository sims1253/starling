"""Higgs decoder with shared K-step graph generation."""

from __future__ import annotations
from typing import Any
import torch
from .config import EOS_TOKEN_IDS
from .fused_decode import FusedLLMMega
from .llm_mega import GenerateResult
from ..multistep import MultiStepDecoder


class MultiStepLLMMega(MultiStepDecoder, FusedLLMMega):
    def __init__(
        self,
        model: Any,
        max_cache_len: int = 1024,
        steps_per_replay: int = 8,
        warmup_iters: int = 3,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        compile_decode: bool = False,
    ) -> None:
        super().__init__(
            model,
            max_cache_len=max_cache_len,
            warmup_iters=warmup_iters,
            device=device,
            dtype=dtype,
            compile_decode=compile_decode,
        )
        self._init_multistep(steps_per_replay, device)
        self.pos_buf = torch.zeros((), dtype=torch.int64, device=device)
        self.cpos_buf = torch.zeros((), dtype=torch.int64, device=device)

    def _reset_to_chunk_start(self, base: int, first_token: torch.Tensor) -> None:
        """Reset all multi-step state to the start of a chunk at position ``base``."""
        self.static_position_ids.fill_(base)
        self.static_cache_position.fill_(base)
        self.pos_buf.fill_(base)
        self.cpos_buf.fill_(base)
        self.valid_len_buf.fill_(base + 1)  # attend to positions [0, base]
        self.static_input_ids.copy_(first_token.reshape(1, 1))
        # Reset the mask: unmask [0, base-1]; the in-graph index_fill_ unmasks one
        # new position per step.
        self.static_attn_mask.fill_(self._neg_val)
        if base > 0:
            self.static_attn_mask.view(-1)[:base] = 0.0

    def _captured_step(self, j: int) -> None:
        """One decode step inside the K-step captured graph (step index ``j``)."""
        # (a) unmask the single new position being written this step.  At step j
        #     the cache writes slot ``base + j`` (= valid_len_buf - 1); attention
        #     must include it.  Single-element index_fill_ (one kernel).
        self._attn_mask_flat.index_fill_(0, (self.valid_len_buf - 1).view(1).long(), 0.0)
        # Sync the scalar buffers into the (1,1)/(1,) static tensors the layer reads.
        self.static_position_ids.copy_(self.pos_buf.view(1, 1))
        self.static_cache_position.copy_(self.cpos_buf.view(1))

        # (b) decode forward (writes static_logits, writes K/V slot cache_position).
        self._decode_step_eager()

        # (c) greedy argmax -> next input + output store (all in-graph).
        tok = self.static_logits[:, -1:, :].argmax(dim=-1)  # (1, 1)
        self.output_ids[j : j + 1].copy_(tok.view(-1))
        self.static_input_ids.copy_(tok)  # feedback for step j+1

        # (d) advance position / cache_position / valid_len for the next step.
        self.pos_buf += 1
        self.cpos_buf += 1
        self.valid_len_buf += 1

    @torch.inference_mode()
    def generate(
        self,
        batch: dict[str, torch.Tensor],
        max_new_tokens: int = 256,
        eos_token_ids=EOS_TOKEN_IDS,
        tokenizer: Any = None,
        capture: bool = True,
    ) -> GenerateResult:
        if max_new_tokens <= 0:
            return self._finalize([], 0.0, tokenizer)
        first_token, prompt_len = self.prefill(batch)
        ids, elapsed = self._generate_multistep(
            first_token, prompt_len, max_new_tokens, eos_token_ids, capture
        )
        return self._finalize(ids, elapsed, tokenizer)
