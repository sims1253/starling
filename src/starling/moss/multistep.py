"""Multi-step CUDA-graph capture for the MOSS-Transcribe Qwen3 LLM decoder.

Captures **K** consecutive decode steps into one ``torch.cuda.CUDAGraph`` so the
host syncs once per K tokens instead of once per token.  Greedy argmax happens
INSIDE the captured graph and feeds back as the next step's input token
(device-side, no sync).  Mirrors ``starling.granite.multistep`` (Approach B:
greedy-in-graph, post-hoc EOS trim).

The emitted token sequence is **byte-exact** with the single-step greedy
decoder (greedy is deterministic; only the timing of the argmax + sync changes).
"""

from __future__ import annotations

import time
from typing import Any, Optional

import torch

from .config import LLM_EOS_TOKEN_ID
from .fused_decode import FusedMossLLMMega
from .llm_mega import BenchReport, GenerateResult, MossLLMMega


class MossMultiStepMega(MossLLMMega):
    """K-step CUDA-graph-captured greedy decoder for the MOSS Qwen3 LLM.

    Subclasses the single-step decoder (:class:`MossLLMMega`, the model's own
    layers).  For the ~2x faster fused-Triton path use
    :class:`FusedMossMultiStepMega`.
    """

    def __init__(
        self,
        language_model: Any,
        lm_head: Any,
        max_cache_len: int = 1024,
        steps_per_replay: int = 16,
        warmup_iters: int = 3,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        compile_decode: bool = False,
    ) -> None:
        super().__init__(
            language_model,
            lm_head,
            max_cache_len=max_cache_len,
            warmup_iters=warmup_iters,
            device=device,
            dtype=dtype,
            compile_decode=compile_decode,
        )
        self.steps_per_replay = max(1, int(steps_per_replay))
        self.K = self.steps_per_replay

        # ---- multi-step static buffers (fixed addresses for the graph) -----
        self.output_ids = torch.zeros(self.K, dtype=torch.int64, device=device)
        self.valid_len_buf = torch.zeros((), dtype=torch.int64, device=device)
        self._attn_mask_flat = self.static_attn_mask.view(-1)  # (M,)

        self._ms_graph: Optional[torch.cuda.CUDAGraph] = None
        self._ms_captured = False

    # ------------------------------------------------------------------ #
    def _reset_to_chunk_start(self, base: int, first_token: torch.Tensor) -> None:
        """Reset all multi-step state to the start of a chunk at position ``base``."""
        self._reset_cache_pos(base)
        self.static_position_ids.fill_(base)
        self.static_cache_pos.fill_(base)
        self.valid_len_buf.fill_(base + 1)
        self.static_input_ids.copy_(first_token.reshape(1, 1))
        self.static_attn_mask.fill_(self._neg_val)
        if base > 0:
            self.static_attn_mask.view(-1)[:base] = 0.0

    # ------------------------------------------------------------------ #
    def _captured_step(self, j: int) -> None:
        """One decode step inside the K-step captured graph (step index ``j``)."""
        # (a) unmask the single new position being written this step (slot
        #     base+j = valid_len_buf-1).  Single-element index_fill_.
        self._attn_mask_flat.index_fill_(
            0, (self.valid_len_buf - 1).view(1).long(), 0.0
        )

        # (b) decode forward (writes static_logits, advances cache by 1).
        self._decode_step_eager()

        # (c) greedy argmax -> next input + output store (all in-graph).
        tok = self.static_logits[:, -1:, :].argmax(dim=-1)  # (1, 1)
        self.output_ids[j : j + 1].copy_(tok.view(-1))
        self.static_input_ids.copy_(tok)

        # (d) advance position + cache_position + valid_len for next step.
        self.static_position_ids += 1
        self.static_cache_pos += 1
        self.valid_len_buf += 1

    def _run_k_steps(self) -> None:
        for j in range(self.K):
            self._captured_step(j)

    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def capture(self, first_token: torch.Tensor, prefill_len: int) -> None:
        """Capture K decode steps into a single CUDA graph."""
        T = int(prefill_len)
        if T + self.K > self.max_cache_len:
            raise ValueError(
                f"K={self.K} captured steps would overflow the static KV cache "
                f"(prompt T={T}, max_cache_len={self.max_cache_len}; need "
                f"T + K <= max_cache_len). Reduce K or max_cache_len."
            )

        # (0) capture the single-step graph too (used as a fallback / verify path).
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

    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def generate(
        self,
        inputs_embeds: torch.Tensor,
        max_new_tokens: int = 200,
        eos_token_id: int = LLM_EOS_TOKEN_ID,
        capture: bool = True,
    ) -> GenerateResult:
        """Greedy-generate ``max_new_tokens`` using K-step graph replays."""
        T = inputs_embeds.shape[1]
        max_safe = self.max_cache_len - T + 1
        if max_new_tokens > max_safe:
            raise ValueError(
                f"max_new_tokens={max_new_tokens} would overflow the static KV "
                f"cache (prompt T={T}, max_cache_len={self.max_cache_len})."
            )
        if inputs_embeds.shape[0] != 1:
            raise ValueError(f"batch=1 only, got {inputs_embeds.shape[0]}.")
        if max_new_tokens <= 0:
            return self._finalize([], 0.0)

        K = self.K
        n_decode = max_new_tokens - 1

        next_token = self.prefill(inputs_embeds)
        gen_ids = [int(next_token.item())]

        if max_new_tokens <= 1 or n_decode <= 0:
            return self._finalize(gen_ids, 0.0)

        n_chunks = (n_decode + K - 1) // K
        total_steps = n_chunks * K
        if T - 1 + total_steps >= self.max_cache_len:
            raise ValueError(
                f"multi-step rounded-up decode ({total_steps} steps across "
                f"{n_chunks} chunks of K={K}) would overflow the static KV cache "
                f"(prompt T={T}, max_cache_len={self.max_cache_len})."
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
                if tok == eos_token_id:
                    done = True
                    break
            if done:
                break
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        return self._finalize(gen_ids, (t1 - t0) * 1000.0)

    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def bench(
        self,
        inputs_embeds: torch.Tensor,
        max_new_tokens: int = 200,
        eos_token_id: int = LLM_EOS_TOKEN_ID,
        decode_iters: int = 10,
    ) -> BenchReport:
        """Benchmark prefill, per-token decode (K-step replay / K), and total."""
        T = inputs_embeds.shape[1]
        pos_ids_prefill = torch.arange(T, device=self.device).unsqueeze(0)

        def _prefill():
            self._reset_cache_pos(0)
            ar = torch.arange(self.max_cache_len, device=self.device)
            q = torch.arange(T, device=self.device).unsqueeze(1)
            mask4 = torch.where(
                ar[None, None, None, :] <= q[None, None, :, :], 0.0, self._neg_val
            ).to(self.dtype)
            self.lm(
                inputs_embeds=inputs_embeds,
                position_ids=pos_ids_prefill,
                attention_mask=mask4,
                past_key_values=self.cache,
                use_cache=True,
                cache_position=torch.arange(T, device=self.device),
            )

        prefill_ms = self._cuda_timer(_prefill, warmup=3, iters=10)

        self._reset_cache_pos(0)
        first_tok = self.prefill(inputs_embeds)
        self.capture(first_tok, T)
        self._reset_to_chunk_start(T, first_tok)

        def _k_replay():
            self._ms_graph.replay()
            self._reset_to_chunk_start(T, first_tok)

        replay_ms = self._cuda_timer(_k_replay, warmup=3, iters=decode_iters)
        decode_ms = replay_ms / self.K
        decode_tps = 1000.0 / decode_ms if decode_ms > 0 else 0.0

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
            notes=f"decoded {res.n_tokens} tokens; K={self.K}; cache_len={self.max_cache_len}",
        )


class FusedMossMultiStepMega(FusedMossLLMMega, MossMultiStepMega):
    """K-step graphed decoder built on the fused-Triton single-step path.

    Inherits ``_decode_step_eager`` (the fused hand-iterated layer loop) from
    :class:`FusedMossLLMMega` and all multi-step machinery (capture / generate
    / bench, K-step replay with argmax-in-graph) from :class:`MossMultiStepMega`.
    ~2x faster than the model-forward multistep path, byte-exact.
    """

    pass
