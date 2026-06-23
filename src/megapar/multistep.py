"""Multi-step CUDA-graph capture for the Granite-4.0-1b LLM decoder.

The single-step graph decoder (:class:`megapar.llm_mega.LLMMega` /
:class:`megapar.llm_mega.FusedLLMMega`) captures ONE decode step per graph and
replays it once per emitted token.  Each replay is followed by **two
host<->device synchronisations** (``.item()`` for append + EOS check) plus
**four host-launched copies** (input-ids, position-ids, mask fill, mask set).
Over a 100-token decode that is ~200 syncs and ~400 host-launched ops, each
costing 10-30 us of host<->device round-trip latency.

This module captures **K consecutive decode steps** into a single
``torch.cuda.CUDAGraph`` so the host syncs **once per K tokens** instead of
once per token.  The greedy argmax happens INSIDE the captured graph and feeds
back as the next step's input token (all device-side, no sync), exactly like
the sibling parakeet ``GraphedDecoder`` (commit ``2ce54ee``).

Design (Approach B -- "greedy-in-graph, speculated-EOS")
-------------------------------------------------------
* **Argmax in-graph.**  Each captured step runs ``decode_step -> argmax ->
  write output_ids[j] -> copy argmax into static_input_ids``.  Step ``j+1``
  reads step ``j``'s argmax from the same fixed-address buffer.  No host sync
  between captured steps.
* **Position / mask advance in-graph.**  ``static_position_ids`` and a scalar
  ``valid_len_buf`` are incremented by 1 inside the graph after each step.
  The 4D attention mask is rebuilt each step from ``valid_len_buf`` via a
  ``torch.where`` (device-side, no CPU scalar).  Because both advance by
  exactly K per replay and the cache ``cumulative_length`` auto-advances by K
  too, the state is correct for the *next* chunk with **zero host staging** --
  the graph's in-place mutations naturally leave the buffers at the right
  starting values.
* **EOS handling (post-hoc trim).**  The captured graph always runs K steps
  regardless of EOS; if a stream hits EOS mid-chunk the in-graph continuation
  generates post-EOS tokens that are simply discarded on the host (the host
  scans the K harvested tokens for EOS and trims).  Pre-EOS tokens are
  unaffected (greedy = greedy; the math is identical, only the timing of the
  argmax and sync changes).  This is byte-exact for non-EOS-early-stop decodes
  and correct (matching greedy up to and including EOS) for early-stop cases.
* **Chunked generation.**  ``max_new_tokens - 1`` decode steps are served by
  ``ceil(n_decode / K)`` replays.  Each replay does ONE sync (``output_ids
  .tolist()``).  The last chunk may produce a few surplus tokens (the graph
  always runs K steps); these are trimmed to ``max_new_tokens``.

Correctness
-----------
The emitted token sequence is **byte-exact** with the single-step greedy
decoder (and therefore with ``golden/greedy_ids.pt``).  Greedy decoding is
deterministic; the only behavioural change is *when* the argmax runs and *when*
the host syncs, not the arithmetic.  Verified in
:mod:`tests.test_multistep`.

Public API
----------
``MultiStepLLMMega(language_model, lm_head, max_cache_len=640, steps_per_replay=16)``
``MultiStepLLMMega.generate(inputs_embeds, max_new_tokens=100, eos_token_id=...)``
    -> :class:`megapar.llm_mega.GenerateResult`
"""

from __future__ import annotations

import time
from typing import Any, Optional

import torch

from .config import LLM_EOS_TOKEN_ID
from .llm_mega import FusedLLMMega, GenerateResult


class MultiStepLLMMega(FusedLLMMega):
    """K-step CUDA-graph-captured greedy decoder for the Granite LLM.

    Subclasses :class:`FusedLLMMega` (fused Triton elementwise kernels) and
    overrides :meth:`capture` / :meth:`generate` to capture **K** decode steps
    per graph replay instead of one, collapsing ~K host<->device syncs per
    chunk into one.

    Args:
        language_model: The ``GraniteModel`` decoder trunk.
        lm_head: ``nn.Linear`` lm_head from the top-level speech model.
        max_cache_len: Fixed K/V cache length.
        steps_per_replay: Number of consecutive decode steps captured into one
            graph replay (``K``).  Larger ``K`` amortises more syncs but
            captures a bigger graph and needs more cache head-room (the last
            chunk always runs full ``K`` steps even if fewer are needed).
        warmup_iters: CUDA-graph warmup iterations before capture.
        device/dtype: Must match the loaded weights (cuda / bfloat16).
    """

    def __init__(
        self,
        language_model: Any,
        lm_head: Any,
        max_cache_len: int = 640,
        steps_per_replay: int = 16,
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
        self.steps_per_replay = max(1, int(steps_per_replay))
        self.K = self.steps_per_replay

        # ---- multi-step static buffers (fixed addresses for the graph) -----
        M = self.max_cache_len
        # Ring buffer for the K tokens emitted by one replay (harvested by the
        # host with a single .tolist() sync).
        self.output_ids = torch.zeros(self.K, dtype=torch.int64, device=device)
        # Scalar device counter for the current mask valid-length.  Starts at
        # ``base + 1`` (attend to positions [0, base]) and is incremented by 1
        # inside the graph after each step; it auto-advances across chunks.
        self.valid_len_buf = torch.zeros((), dtype=torch.int64, device=device)
        # Flat view of the attention mask for the per-step single-position
        # unmask (avoids rebuilding the full mask each step).
        self._attn_mask_flat = self.static_attn_mask.view(-1)  # (M,)

        self._ms_graph: Optional[torch.cuda.CUDAGraph] = None
        self._ms_captured = False

    # ------------------------------------------------------------------ #
    # state reset helpers
    # ------------------------------------------------------------------ #
    def _reset_to_chunk_start(
        self, base: int, first_token: torch.Tensor
    ) -> None:
        """Reset all multi-step state to the start of a chunk at position ``base``.

        * ``cumulative_length`` (all layers) -> ``base``
        * ``static_position_ids`` -> ``base``
        * ``valid_len_buf`` -> ``base + 1``  (attend to positions [0, base])
        * ``static_input_ids`` -> ``first_token``
        * ``static_attn_mask`` -> positions [0, base-1] unmasked, rest masked

        Called once after capture (to prepare chunk 0) and conceptually before
        every chunk -- but because the graph's in-graph unmask-+increment leave
        the state exactly at the next chunk's start values, only chunk 0 needs
        an explicit reset.
        """
        self._reset_cache_pos(base)
        self.static_position_ids.fill_(base)
        self.valid_len_buf.fill_(base + 1)
        self.static_input_ids.copy_(first_token.reshape(1, 1))
        # Reset the mask: unmask [0, base-1] (from prefill + prior chunks);
        # the in-graph index_fill_ will unmask one new position per step.
        self.static_attn_mask.fill_(self._neg_val)
        if base > 0:
            self.static_attn_mask.view(-1)[:base] = 0.0

    # ------------------------------------------------------------------ #
    # the captured per-step function (runs K times inside the graph)
    # ------------------------------------------------------------------ #
    def _captured_step(self, j: int) -> None:
        """One decode step inside the K-step captured graph (step index ``j``).

        Reads the in-place-mutated static buffers left by step ``j-1``
        (``static_input_ids`` / ``static_position_ids`` / ``valid_len_buf`` /
        cache ``cumulative_length``), computes one greedy decode step, and
        writes the emitted token into ``output_ids[j]``.  ``static_input_ids``
        is chained in-graph (argmax -> copy) so the next captured step reads
        the correct input token without a host sync.  For ``K=1`` this is
        equivalent to the single-step path (same value, byte-identical output).
        """
        # (a) unmask the single new position being written this step.  At step
        #     j the cache writes slot ``base + j`` (= valid_len_buf - 1) and
        #     the attention must include it.  This is a single-element
        #     index_fill_ (one kernel) instead of rebuilding the full M-wide
        #     mask each step.
        self._attn_mask_flat.index_fill_(
            0, (self.valid_len_buf - 1).view(1).long(), 0.0
        )

        # (b) decode forward (reads input_ids/position/mask/cache, writes
        #     static_logits, advances cache cumulative_length by 1).
        self._decode_step_eager()

        # (c) greedy argmax -> next input + output store (all in-graph).
        tok = self.static_logits[:, -1:, :].argmax(dim=-1)  # (1, 1)
        self.output_ids[j : j + 1].copy_(tok.view(-1))  # store step j's token
        self.static_input_ids.copy_(tok)  # feedback for step j+1

        # (d) advance position + valid_len for the next captured step.
        self.static_position_ids += 1
        self.valid_len_buf += 1

    def _run_k_steps(self) -> None:
        """Run K consecutive decode steps (used in warmup + capture)."""
        for j in range(self.K):
            self._captured_step(j)

    # ------------------------------------------------------------------ #
    # CUDA-graph capture of K decode steps
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def capture(self, first_token: torch.Tensor, prefill_len: int) -> None:
        """Capture K decode steps into a single CUDA graph.

        Must be called once after :meth:`prefill` (the prompt K/V must already
        fill cache slots ``[0, prefill_len)``).  ``first_token`` is the token
        produced by the prefill (the input to the first decode step).

        Also captures the parent's single-step ``_graph`` so that
        :class:`megapar.speculative.SpeculativeDecoder` (which uses
        ``llm._graph.replay()`` for single-token verify fallbacks) works with a
        :class:`MultiStepLLMMega` instance unchanged.
        """
        T = int(prefill_len)
        # Guard: K captured steps write cache slots [T, T+K); must fit.
        if T + self.K > self.max_cache_len:
            raise ValueError(
                f"K={self.K} captured steps would overflow the static KV cache "
                f"(prompt T={T}, max_cache_len={self.max_cache_len}; need "
                f"T + K <= max_cache_len). Reduce K or max_cache_len."
            )

        # (0) Capture the single-step graph (parent) for speculative-decoder
        #     compatibility.  This leaves cache at cumulative_length == T.
        super().capture(first_token, T)

        # Prime for multi-step warmup.
        self._reset_to_chunk_start(T, first_token)

        # Warmup advances all state by K per iter; reset after so capture starts
        # cleanly at position T.  Warmup scribbles garbage K/V into slots
        # [T, ...) which is overwritten before it is ever read (masked out by
        # the 4D mask and overwritten by the real decode).
        for _ in range(self.warmup_iters):
            self._run_k_steps()
        torch.cuda.synchronize()
        self._reset_to_chunk_start(T, first_token)

        # Capture K consecutive decode steps into one graph.  Each step reads
        # the in-place-mutated buffers left by the previous step (the output_ids
        # columns + static_input_ids chain are what make one replay == K steps).
        self._ms_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._ms_graph):
            self._run_k_steps()

        # Reset for real generation (capture mutated the buffers).
        self._reset_to_chunk_start(T, first_token)
        self._ms_captured = True

    # ------------------------------------------------------------------ #
    # generate (chunked K-step replays)
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def generate(
        self,
        inputs_embeds: torch.Tensor,
        max_new_tokens: int = 100,
        eos_token_id: int = LLM_EOS_TOKEN_ID,
        tokenizer: Any = None,
        capture: bool = True,
    ) -> GenerateResult:
        """Greedy-generate ``max_new_tokens`` using K-step graph replays.

        Prefill is eager (produces token 0); the remaining ``max_new_tokens -
        1`` decode steps are served by ``ceil(n_decode / K)`` K-step replays,
        each requiring exactly ONE host<->device sync to harvest K tokens.
        """
        T = inputs_embeds.shape[1]
        # Same overflow guard as the single-step path.
        max_safe = self.max_cache_len - T + 1
        if max_new_tokens > max_safe:
            raise ValueError(
                f"max_new_tokens={max_new_tokens} would overflow the static KV "
                f"cache (prompt T={T}, max_cache_len={self.max_cache_len}; at "
                f"most {max_safe} new tokens fit)."
            )
        if inputs_embeds.shape[0] != 1:
            raise ValueError(
                f"MultiStepLLMMega only supports batch=1, got batch="
                f"{inputs_embeds.shape[0]}."
            )
        if max_new_tokens <= 0:
            return self._finalize([], 0.0, tokenizer)

        K = self.K
        n_decode = max_new_tokens - 1  # decode steps after the prefill token

        # (1) prefill -> first token (position T).
        next_token = self.prefill(inputs_embeds)  # (1, 1)
        gen_ids = [int(next_token.item())]

        if max_new_tokens <= 1 or n_decode <= 0:
            return self._finalize(gen_ids, 0.0, tokenizer)

        # The last chunk always runs full K steps even if fewer are needed;
        # ensure the rounded-up step count does not overflow the cache.
        n_chunks = (n_decode + K - 1) // K
        total_steps = n_chunks * K
        if T - 1 + total_steps >= self.max_cache_len:
            raise ValueError(
                f"multi-step rounded-up decode ({total_steps} steps across "
                f"{n_chunks} chunks of K={K}) would overflow the static KV cache "
                f"(prompt T={T}, max_cache_len={self.max_cache_len}). Reduce K "
                f"or max_new_tokens."
            )

        # (2) capture the K-step graph (idempotent).
        if capture and not self._ms_captured:
            self.capture(next_token, T)

        # (3) reset to chunk-0 start (the graph's in-graph increments handle
        #     all subsequent chunk boundaries automatically).
        self._reset_to_chunk_start(T, next_token)

        # (4) chunked K-step replay loop.  ONE sync per chunk.
        t0 = time.perf_counter()
        done = False
        for _chunk in range(n_chunks):
            self._ms_graph.replay()
            # ONE device->host sync for the whole K-step batch.
            out = self.output_ids.tolist()  # list[int] of length K
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

        wall_ms = (t1 - t0) * 1000.0
        return self._finalize(gen_ids, wall_ms, tokenizer)

    # ------------------------------------------------------------------ #
    # benchmark (override to measure K-step replay correctly)
    # ------------------------------------------------------------------ #
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
            notes=(
                f"decoded {res.n_tokens} tokens; K={self.K}; "
                f"cache_len={self.max_cache_len}"
            ),
        )
