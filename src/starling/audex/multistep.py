"""Multi-step CUDA-graph capture for the Audex-2B Nemotron-Dense LLM decoder.

Ported from ``starling.qwen3.multistep`` (identical design; Audex uses a
different EOS id and inherits the relu2 MLP from :class:`FusedLLMMega`).

Captures **K** consecutive decode steps into one ``torch.cuda.CUDAGraph`` so
the host syncs **once per K tokens** instead of once per token. The greedy
argmax runs INSIDE the captured graph and feeds back as the next step's input
token (all device-side, no sync). The emitted token sequence is byte-exact
with the single-step greedy decoder.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import torch

from .config import EOS_TOKEN_ID
from .llm_mega import FusedLLMMega, GenerateResult


class MultiStepLLMMega(FusedLLMMega):
    """K-step CUDA-graph-captured greedy decoder for the Nemotron-Dense LLM."""

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
        compile_decode: bool = False,
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
            prefill_use_graph=prefill_use_graph,
        )
        self.steps_per_replay = max(1, int(steps_per_replay))
        self.K = self.steps_per_replay
        if compile_decode:
            self._decode_step_eager = torch.compile(
                self._decode_step_eager, mode="max-autotune-no-cudagraphs"
            )

        self.output_ids = torch.zeros(self.K, dtype=torch.int64, device=device)
        self.valid_len_buf = torch.zeros((), dtype=torch.int64, device=device)
        self._attn_mask_flat = self.static_attn_mask.view(-1)
        self._ms_graph: Optional[torch.cuda.CUDAGraph] = None
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
        self._attn_mask_flat.index_fill_(
            0, (self.valid_len_buf - 1).view(1).long(), 0.0
        )
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
        T = int(prefill_len)
        if T + self.K > self.max_cache_len:
            raise ValueError(
                f"K={self.K} steps overflow cache (T={T}, "
                f"max_cache_len={self.max_cache_len})."
            )
        super().capture(first_token, T)
        self._reset_to_chunk_start(T, first_token)
        for _ in range(self.warmup_iters):
            self._run_k_steps()
        torch.cuda.synchronize()
        self._reset_to_chunk_start(T, first_token)
        self._ms_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._ms_graph):
            self._run_k_steps()
        self._reset_to_chunk_start(T, first_token)
        self._ms_captured = True

    @torch.inference_mode()
    def generate(
        self,
        inputs_embeds: torch.Tensor,
        max_new_tokens: int = 200,
        eos_token_id: Optional[int] = None,
        tokenizer: Any = None,
        capture: bool = True,
    ) -> GenerateResult:
        eos = int(eos_token_id) if eos_token_id is not None else self.eos_token_id
        T = inputs_embeds.shape[1]
        max_safe = self.max_cache_len - T + 1
        if max_new_tokens > max_safe:
            raise ValueError(
                f"max_new_tokens={max_new_tokens} overflows cache (T={T}, "
                f"max_cache_len={self.max_cache_len}; max {max_safe})."
            )
        if inputs_embeds.shape[0] != 1:
            raise ValueError("MultiStepLLMMega only supports batch=1.")
        if max_new_tokens <= 0:
            return self._finalize([], 0.0, tokenizer)

        K = self.K
        n_decode = max_new_tokens - 1
        next_token = self.prefill(inputs_embeds, use_graph=self.prefill_use_graph)
        gen_ids = [int(next_token.item())]
        if max_new_tokens <= 1 or n_decode <= 0:
            return self._finalize(gen_ids, 0.0, tokenizer)

        n_chunks = (n_decode + K - 1) // K
        total_steps = n_chunks * K
        if T - 1 + total_steps >= self.max_cache_len:
            raise ValueError(
                f"multi-step decode ({total_steps} steps) overflows cache."
            )

        if capture and not self._ms_captured:
            self.capture(next_token, T)
        self._reset_to_chunk_start(T, next_token)

        t0 = time.perf_counter()
        done = False
        for _chunk in range(n_chunks):
            self._ms_graph.replay()
            out = self.output_ids.tolist()
            for tok in out:
                if len(gen_ids) >= max_new_tokens:
                    done = True
                    break
                gen_ids.append(tok)
                if tok == eos:
                    done = True
                    break
            if done:
                break
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1000.0
        return self._finalize(gen_ids, wall_ms, tokenizer)


__all__ = ["MultiStepLLMMega"]
