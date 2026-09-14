"""Granite decoder with shared K-step graph generation."""

from __future__ import annotations
from typing import Any
import torch
from ..config import LLM_EOS_TOKEN_ID
from .llm_mega import FusedLLMMega, GenerateResult
from ..multistep import MultiStepDecoder


class MultiStepLLMMega(MultiStepDecoder, FusedLLMMega):
    def __init__(
        self,
        language_model: Any,
        lm_head: Any,
        max_cache_len: int = 640,
        steps_per_replay: int = 4,
        warmup_iters: int = 3,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(
            language_model,
            lm_head,
            max_cache_len=max_cache_len,
            warmup_iters=warmup_iters,
            device=device,
            dtype=dtype,
        )
        self._init_multistep(steps_per_replay, device)

    @torch.inference_mode()
    def generate(
        self,
        inputs_embeds: torch.Tensor,
        max_new_tokens: int = 100,
        eos_token_id: int = LLM_EOS_TOKEN_ID,
        tokenizer: Any = None,
        capture: bool = True,
    ) -> GenerateResult:
        if max_new_tokens <= 0:
            return self._finalize([], 0.0, tokenizer)
        if inputs_embeds.shape[0] != 1:
            raise ValueError("Multi-step decoding only supports batch=1.")
        prompt_len = inputs_embeds.shape[1]
        self._validate_token_budget(prompt_len, max_new_tokens)
        first_token = self.prefill(inputs_embeds)
        ids, elapsed = self._generate_multistep(
            first_token, prompt_len, max_new_tokens, (eos_token_id,), capture
        )
        return self._finalize(ids, elapsed, tokenizer)

    @torch.inference_mode()
    def bench(
        self,
        inputs_embeds: torch.Tensor,
        max_new_tokens: int = 100,
        eos_token_id: int = LLM_EOS_TOKEN_ID,
        decode_iters: int = 10,
    ) -> Any:
        """Benchmark prefill, per-token decode (K-step replay / K), and total.

        The per-token decode time measures the K-step graph replay divided by K
        so it is directly comparable to the single-step decoder's number.
        """
        from .llm_mega import BenchReport

        T = inputs_embeds.shape[1]
        pos_ids_prefill = torch.arange(T, device=self.device).unsqueeze(0)

        # (a) prefill time.
        def _prefill():
            self._reset_cache_pos(0)
            self.lm(
                inputs_embeds=inputs_embeds,
                position_ids=pos_ids_prefill,
                past_key_values=self.cache,
                use_cache=True,
            )

        prefill_ms = self._cuda_timer(_prefill, warmup=3, iters=10)

        # (b) capture + per-token decode time (K-step replay / K).
        self._reset_cache_pos(0)
        first_tok = self.prefill(inputs_embeds)
        self.capture(first_tok, T)
        self._reset_to_chunk_start(T, first_tok)

        def _k_replay():
            self._ms_graph.replay()
            self._reset_to_chunk_start(T, first_tok)

        replay_ms = self._cuda_timer(_k_replay, warmup=3, iters=decode_iters)
        decode_ms = replay_ms / self.K  # per-token
        decode_tps = 1000.0 / decode_ms if decode_ms > 0 else 0.0

        # (c) full generate (wall clock).
        self._reset_cache_pos(0)
        self._ms_captured = False
        res = self.generate(
            inputs_embeds,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )

        return BenchReport(
            prefill_ms=prefill_ms,
            decode_ms_per_token=decode_ms,
            decode_tok_per_s=decode_tps,
            total_ms=res.total_ms,
            total_tok_per_s=res.tok_per_s,
            notes=(f"decoded {res.n_tokens} tokens; K={self.K}; cache_len={self.max_cache_len}"),
        )
