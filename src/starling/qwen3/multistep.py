"""Multi-step CUDA-graph capture for the Qwen3 ASR text decoder.

Ported from ``starling.granite.multistep`` (identical design; Qwen3 just uses a
different EOS id and has no logits/embedding multipliers, both handled by the
parent :class:`FusedLLMMega`).

The single-step graph decoder captures ONE decode step per replay and syncs
with the host twice per token (argmax append + EOS check) plus four
host-launched copies. This module captures **K consecutive decode steps** into
one ``torch.cuda.CUDAGraph`` so the host syncs **once per K tokens** instead of
once per token. The greedy argmax runs INSIDE the captured graph and feeds
back as the next step's input token (all device-side, no sync).

The emitted token sequence is byte-exact with the single-step greedy decoder
(greedy = greedy; only the timing of the argmax and sync changes).
"""

from __future__ import annotations

import time
from typing import Any, Optional

import torch

from .config import EOS_TOKEN_ID
from .llm_mega import FusedLLMMega, GenerateResult


class MultiStepLLMMega(FusedLLMMega):
    """K-step CUDA-graph-captured greedy decoder for the Qwen3 LLM.

    Subclasses :class:`FusedLLMMega` and overrides :meth:`capture` /
    :meth:`generate` to capture **K** decode steps per replay instead of one,
    collapsing ~K host<->device syncs per chunk into one.

    Args:
        language_model: The Qwen3 decoder trunk.
        lm_head: ``nn.Linear`` lm_head from the top-level speech model.
        max_cache_len: Fixed K/V cache length.
        steps_per_replay: K -- consecutive decode steps captured per replay.
        warmup_iters: CUDA-graph warmup iterations before capture.
    """

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
            prefill_use_graph=prefill_use_graph,
        )
        self.steps_per_replay = max(1, int(steps_per_replay))
        self.K = self.steps_per_replay
        # torch.compile the fused decode forward (Inductor fuses the RoPE
        # cat+mul+add + attention softmax prep + GQA repeats that the hand loop
        # still emits as separate PyTorch ops). 1.37x faster (4.88->3.55 ms/tok)
        # with byte-identical *decoded tokens* (greedy is robust to the sub-ULP
        # logit noise Inductor introduces; verified 0 token mismatches over full
        # decodes). Set False for the strict per-logit byte-exact fallback.
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
        # (a) unmask the single new position written this step.
        self._attn_mask_flat.index_fill_(
            0, (self.valid_len_buf - 1).view(1).long(), 0.0
        )
        # (b) decode forward (writes static_logits, advances cache by 1).
        self._decode_step_eager()
        # (c) greedy argmax -> next input + output store (all in-graph).
        tok = self.static_logits[:, -1:, :].argmax(dim=-1)  # (1, 1)
        self.output_ids[j : j + 1].copy_(tok.view(-1))
        self.static_input_ids.copy_(tok)
        # (d) advance position + valid_len for the next captured step.
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
                f"K={self.K} captured steps would overflow the static KV cache "
                f"(prompt T={T}, max_cache_len={self.max_cache_len})."
            )
        # Also capture the parent single-step graph for any single-step callers.
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
            raise ValueError(f"MultiStepLLMMega only supports batch=1.")
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
                f"multi-step rounded-up decode ({total_steps} steps across "
                f"{n_chunks} chunks of K={K}) would overflow the cache."
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
