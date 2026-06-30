"""Multi-step CUDA-graph capture for the Higgs-Audio Qwen3 LLM decoder.

The single-step graph decoder (:class:`starling.higgs.llm_mega.LLMMega`)
captures ONE decode step per graph and replays it once per emitted token.  Each
replay is followed by a host<->device sync (``.item()`` for append + EOS check)
plus several host-launched copies.  Over a 100-token decode that is ~100 syncs,
each costing host<->device round-trip latency.

This module captures **K consecutive decode steps** into a single
``torch.cuda.CUDAGraph`` so the host syncs **once per K tokens**.  The greedy
argmax happens INSIDE the captured graph and feeds back as the next step's input
token (all device-side, no sync between steps).

Design (mirrors ``starling.granite.multistep``, adapted for the tf4.51 StaticCache
which has no internal position counter -- position is driven by
``cache_position`` passed to each layer's ``update``):

* **Argmax in-graph.**  Each captured step runs ``decode_step -> argmax -> write
  output_ids[j] -> copy argmax into static_input_ids``.  Step ``j+1`` reads step
  ``j``'s argmax.  No host sync between captured steps.
* **Position / mask / cache_position advance in-graph.**  ``static_position_ids``,
  ``static_cache_position`` and ``valid_len_buf`` are incremented by 1 inside the
  graph after each step.  The 4D mask gains one unmasked position per step via a
  single ``index_fill_``.  The StaticCache writes slot ``cache_position`` each
  step, so advancing it advances the write target.
* **EOS (post-hoc trim).**  The graph always runs K steps; EOS is detected
  host-side by scanning the harvested tokens.  Byte-exact for non-EOS-early-stop
  decodes and correct (greedy up to + including EOS) for early-stop cases.

Correctness: the emitted token sequence is **byte-exact** with the single-step
greedy decoder -- greedy is deterministic, only the timing of the argmax/sync
changes.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import torch

from .config import EOS_TOKEN_IDS
from .fused_decode import FusedLLMMega
from .llm_mega import GenerateResult


class MultiStepLLMMega(FusedLLMMega):
    """K-step CUDA-graph-captured greedy decoder for the Higgs-Audio Qwen3 LLM.

    Subclasses :class:`FusedLLMMega` (fused Triton elementwise kernels) and
    overrides :meth:`capture` / :meth:`generate` to capture **K** decode steps
    per graph replay instead of one.

    Args:
        model: the loaded ``HiggsAudio3Model``.
        max_cache_len: Fixed K/V cache length.
        steps_per_replay: Number of consecutive decode steps captured into one
            graph replay (``K``).
        warmup_iters: CUDA-graph warmup iterations before capture.
        device/dtype: Must match the loaded weights (cuda / bfloat16).
    """

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
        self.steps_per_replay = max(1, int(steps_per_replay))
        self.K = self.steps_per_replay

        # ---- multi-step static buffers (fixed addresses for the graph) -----
        self.output_ids = torch.zeros(self.K, dtype=torch.int64, device=device)
        # Scalar device counters for valid-len and (optionally) cache_position.
        # static_position_ids / static_cache_position are (1,1)/(1,) but we also
        # need scalar buffers we can increment in-graph cleanly.
        self.valid_len_buf = torch.zeros((), dtype=torch.int64, device=device)
        self.pos_buf = torch.zeros((), dtype=torch.int64, device=device)
        self.cpos_buf = torch.zeros((), dtype=torch.int64, device=device)
        self._attn_mask_flat = self.static_attn_mask.view(-1)  # (M,)

        self._ms_graph: Optional[torch.cuda.CUDAGraph] = None
        self._ms_captured = False

    # ------------------------------------------------------------------ #
    # state reset helpers
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # the captured per-step function (runs K times inside the graph)
    # ------------------------------------------------------------------ #
    def _captured_step(self, j: int) -> None:
        """One decode step inside the K-step captured graph (step index ``j``)."""
        # (a) unmask the single new position being written this step.  At step j
        #     the cache writes slot ``base + j`` (= valid_len_buf - 1); attention
        #     must include it.  Single-element index_fill_ (one kernel).
        self._attn_mask_flat.index_fill_(
            0, (self.valid_len_buf - 1).view(1).long(), 0.0
        )
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

    def _run_k_steps(self) -> None:
        for j in range(self.K):
            self._captured_step(j)

    # ------------------------------------------------------------------ #
    # CUDA-graph capture of K decode steps
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def capture(self, first_token: torch.Tensor, prefill_len: int) -> None:
        """Capture K decode steps into a single CUDA graph."""
        T = int(prefill_len)
        if T + self.K > self.max_cache_len:
            raise ValueError(
                f"K={self.K} captured steps would overflow the static KV cache "
                f"(prompt T={T}, max_cache_len={self.max_cache_len}; need "
                f"T + K <= max_cache_len). Reduce K or increase max_cache_len."
            )

        # (0) capture the single-step graph too (parent) so either path works.
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
    # generate (chunked K-step replays)
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def generate(
        self,
        batch: dict[str, torch.Tensor],
        max_new_tokens: int = 256,
        eos_token_ids=EOS_TOKEN_IDS,
        tokenizer: Any = None,
        capture: bool = True,
    ) -> GenerateResult:
        """Greedy-generate ``max_new_tokens`` using K-step graph replays."""
        if max_new_tokens <= 0:
            return self._finalize([], 0.0, tokenizer)

        # (1) prefill -> first token + effective prompt length.
        first_token, T = self.prefill(batch)
        K = self.K
        max_safe = self.max_cache_len - T + 1
        if max_new_tokens > max_safe:
            raise ValueError(
                f"max_new_tokens={max_new_tokens} would overflow the static KV cache "
                f"(effective prompt T_eff={T}, max_cache_len={self.max_cache_len})."
            )
        gen_ids = [int(first_token.item())]
        n_decode = max_new_tokens - 1
        if n_decode <= 0:
            return self._finalize(gen_ids, 0.0, tokenizer)

        # The last chunk runs full K steps even if fewer are needed.
        n_chunks = (n_decode + K - 1) // K
        total_steps = n_chunks * K
        if T - 1 + total_steps >= self.max_cache_len:
            raise ValueError(
                f"multi-step rounded-up decode ({total_steps} steps across "
                f"{n_chunks} chunks of K={K}) would overflow the static KV cache "
                f"(prompt T_eff={T}, max_cache_len={self.max_cache_len})."
            )

        # (2) capture the K-step graph (idempotent).
        if capture and not self._ms_captured:
            self.capture(first_token, T)

        self._reset_to_chunk_start(T, first_token)

        # (3) chunked K-step replay loop.  ONE sync per chunk.
        eos_set = set(eos_token_ids)
        t0 = time.perf_counter()
        done = False
        for _chunk in range(n_chunks):
            self._ms_graph.replay()
            out = self.output_ids.tolist()  # ONE sync for K tokens
            for tok in out:
                if len(gen_ids) >= max_new_tokens:
                    done = True
                    break
                gen_ids.append(tok)
                if tok in eos_set:
                    done = True
                    break
            if done:
                break
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        wall_ms = (t1 - t0) * 1000.0
        return self._finalize(gen_ids, wall_ms, tokenizer)
