"""Shared CUDA-graph lifecycle and greedy token harvesting.

Model subclasses own prefill, cache-position updates, and decode arithmetic.
A full replay emits K tokens with one host transfer; a final short chunk runs
only the requested steps so it cannot write past the cache.
"""

from __future__ import annotations

import time

import torch


class MultiStepDecoder:
    """Mixin for decoders exposing fixed buffers and a single-step capture."""

    def _init_multistep(self, steps_per_replay: int, device: str) -> None:
        self.steps_per_replay = max(1, int(steps_per_replay))
        self.K = self.steps_per_replay
        self.output_ids = torch.zeros(self.K, dtype=torch.int64, device=device)
        self.valid_len_buf = torch.zeros((), dtype=torch.int64, device=device)
        self._attn_mask_flat = self.static_attn_mask.view(-1)
        self._ms_graph = None
        self._ms_captured = False

    def _reset_to_chunk_start(self, base: int, first_token: torch.Tensor) -> None:
        self._reset_cache_pos(base)
        self.static_position_ids.fill_(base)
        self.valid_len_buf.fill_(base + 1)
        self.static_input_ids.copy_(first_token.reshape(1, 1))
        self.static_attn_mask.fill_(self._neg_val)
        if base > 0:
            self.static_attn_mask.view(-1)[:base] = 0.0

    def _captured_step(self, j: int) -> None:
        self._attn_mask_flat.index_fill_(0, (self.valid_len_buf - 1).view(1).long(), 0.0)
        self._decode_step_eager()
        tok = self.static_logits[:, -1:, :].argmax(dim=-1)
        self.output_ids[j : j + 1].copy_(tok.view(-1))
        self.static_input_ids.copy_(tok)
        self.static_position_ids += 1
        self.valid_len_buf += 1

    def _run_k_steps(self) -> None:
        for j in range(self.K):
            self._captured_step(j)

    @torch.inference_mode()
    def capture(self, first_token: torch.Tensor, prefill_len: int) -> None:
        if prefill_len < 0 or prefill_len + self.K > self.max_cache_len:
            raise ValueError(
                f"K={self.K} steps overflow cache (prompt={prefill_len}, "
                f"max_cache_len={self.max_cache_len})."
            )
        # Speculative verification also needs the parent's single-step graph.
        super().capture(first_token, prefill_len)
        for _ in range(self.warmup_iters):
            self._reset_to_chunk_start(prefill_len, first_token)
            self._run_k_steps()
        torch.cuda.synchronize()
        self._reset_to_chunk_start(prefill_len, first_token)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._run_k_steps()
        self._reset_to_chunk_start(prefill_len, first_token)
        self._ms_graph = graph
        self._ms_captured = True

    def _validate_token_budget(self, prompt_len: int, max_new_tokens: int) -> None:
        max_safe = self.max_cache_len - prompt_len + 1
        if prompt_len < 1 or prompt_len > self.max_cache_len or max_new_tokens > max_safe:
            raise ValueError(
                f"max_new_tokens={max_new_tokens} overflows cache "
                f"(prompt={prompt_len}, max_cache_len={self.max_cache_len}; "
                f"at most {max_safe} tokens fit)."
            )

    def _generate_multistep(self, first_token, prompt_len, max_new_tokens, eos_token_ids, capture):
        """Reuse a captured graph when present; ``capture`` permits creating one."""
        self._validate_token_budget(prompt_len, max_new_tokens)
        eos = set(eos_token_ids)
        ids = [int(first_token.item())]
        if max_new_tokens == 1 or ids[0] in eos:
            return ids, 0.0

        remaining = max_new_tokens - 1
        if capture and remaining >= self.K and not self._ms_captured:
            self.capture(first_token, prompt_len)
        self._reset_to_chunk_start(prompt_len, first_token)
        start = time.perf_counter()
        while remaining:
            count = min(self.K, remaining)
            if count == self.K and self._ms_captured:
                self._ms_graph.replay()
            else:
                for j in range(count):
                    self._captured_step(j)
            for token in self.output_ids[:count].tolist():
                ids.append(token)
                if token in eos:
                    remaining = 0
                    break
            else:
                remaining -= count
        torch.cuda.synchronize()
        return ids, (time.perf_counter() - start) * 1000.0
