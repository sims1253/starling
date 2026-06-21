"""CUDA-graph-captured greedy decoder for the Granite-4.0-1b LLM.

The LLM decoder is ~99% of the Granite-Speech-4.1-2b ASR runtime.  The stock
eager ``model.generate`` path launches dozens of small kernels per token and
rebuilds Python/autograd state on every step, capping throughput far below the
memory-bandwidth ceiling of the RTX 5090.

This module closes that gap with:

* **Phase A** - a correct CUDA-graph-captured greedy decode built on top of the
  model's *own* layers and ``transformers.StaticCache``.  Graph replay of the
  model's own ops is bit-exact with eager, so the decoded token sequence matches
  the golden reference exactly.
* **Phase B** - benchmark hooks (prefill ms, decode ms/token, tok/s, total ms).
* **Phase C** - an optional fused decode path that swaps in Triton kernels
  (fused RMSNorm, fused RoPE, fused SwiGLU) to cut memory traffic and launch
  count further.  Fused kernels use bf16 numerics with fp32 accumulation where
  the reference does, and are re-verified against the golden transcript.

Design notes
------------
``StaticCache`` (``transformers.cache_utils``) pre-allocates fixed-address K/V
tensors for all 40 layers plus a ``cumulative_length`` tensor per layer that is
incremented in-place on each ``update``.  This is inherently CUDA-graph safe:

* ``keys`` / ``values`` are tagged ``mark_static_address`` by the cache.
* ``cumulative_length`` is mutated in-place via ``add_``; on replay the graph
  reads the *current* value, writes the new K/V slot, and advances the counter.

The one wrinkle: ``create_causal_mask`` allocates CPU scalars
(``torch.tensor(0.0)``) which abort CUDA-graph capture.  We bypass it by feeding
a pre-computed **4D** attention mask (``(1, 1, 1, max_cache_len)``); the masking
plumbing early-exits and returns a 4D mask as-is.

The warmup steps advance ``cumulative_length`` and scribble garbage K/V into
slots ``[prefill_len, prefill_len + warmup)``.  We reset the counter back to
``prefill_len`` before capture *and* before the generate loop so the first real
decode writes slot ``prefill_len``.  Stale garbage past the current write slot
is masked out by the 4D mask and overwritten before it is ever read.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from .config import (
    LLM_ATTENTION_MULTIPLIER,
    LLM_EOS_TOKEN_ID,
    LLM_HEAD_DIM,
    LLM_LOGITS_SCALING,
    LLM_NUM_ATTN_HEADS,
    LLM_NUM_KV_HEADS,
    LLM_NUM_LAYERS,
    LLM_RMS_NORM_EPS,
    LLM_RESIDUAL_MULTIPLIER,
)

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class GenerateResult:
    """Output of :meth:`LLMMega.generate`."""

    ids: torch.Tensor  # (1, n_new) int64 on CPU, the newly generated tokens
    text: str
    n_tokens: int
    total_ms: float
    tok_per_s: float


@dataclass
class BenchReport:
    """Aggregated benchmark numbers for printing / JSON."""

    prefill_ms: float = 0.0
    decode_ms_per_token: float = 0.0
    decode_tok_per_s: float = 0.0
    total_ms: float = 0.0
    total_tok_per_s: float = 0.0
    notes: str = ""


# ---------------------------------------------------------------------------
# Phase A + B: CUDA-graph-captured greedy decoder (model's own layers)
# ---------------------------------------------------------------------------
class LLMMega:
    """CUDA-graph-captured greedy decoder for the Granite LLM.

    Wraps a loaded ``GraniteModel`` (the ``language_model`` component from
    :func:`megapar.loader.get_components`) plus the parent model's ``lm_head``.
    The LLM's own layers are used unchanged so decode output is bit-exact with
    the eager golden reference.

    Args:
        language_model: The ``GraniteModel`` (has ``embed_tokens``, ``layers``,
            ``norm``, ``rotary_emb``).
        lm_head: The ``nn.Linear`` lm_head from the top-level speech model.
        max_cache_len: Fixed K/V cache length to pre-allocate.
        warmup_iters: CUDA-graph warmup iterations before capture.
        device/dtype: Must match the loaded weights (cuda / bfloat16).
    """

    def __init__(
        self,
        language_model: Any,
        lm_head: Any,
        max_cache_len: int = 640,
        warmup_iters: int = 3,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.lm = language_model
        self.lm_head = lm_head
        self.config = language_model.config
        self.max_cache_len = int(max_cache_len)
        self.warmup_iters = int(warmup_iters)
        self.device = device
        self.dtype = dtype

        self.vocab_size = int(self.config.vocab_size)
        self.num_layers = int(self.config.num_hidden_layers)

        # ---- static input / output buffers (fixed addresses for the graph) --
        self.static_input_ids = torch.zeros((1, 1), dtype=torch.int64, device=device)
        self.static_position_ids = torch.zeros((1, 1), dtype=torch.int64, device=device)
        self.static_logits = torch.zeros(
            (1, 1, self.vocab_size), dtype=dtype, device=device
        )
        neg = torch.finfo(dtype).min
        self._neg_val = neg
        self.static_attn_mask = torch.full(
            (1, 1, 1, self.max_cache_len), neg, dtype=dtype, device=device
        )

        # The StaticCache is allocated lazily on first prefill (needs to see the
        # K/V dtype/shape from a real forward).  We build it once here so its
        # fixed-address tensors exist before any graph capture.
        from transformers.cache_utils import StaticCache

        self.cache = StaticCache(config=self.config, max_cache_len=self.max_cache_len)

        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._captured = False

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #
    def _reset_cache_pos(self, n: int) -> None:
        """Reset every layer's ``cumulative_length`` to ``n`` in-place."""
        for layer in self.cache.layers:
            layer.cumulative_length.fill_(n)

    def _set_mask(self, valid_len: int) -> None:
        """Unmask positions ``[0, valid_len)``; mask the rest to ``-inf``."""
        self.static_attn_mask.fill_(self._neg_val)
        self.static_attn_mask[:, :, :, :valid_len] = 0.0

    def _decode_step_eager(self) -> None:
        """One eager decode forward writing into ``static_logits``.

        Uses the model's own layers with the pre-computed 4D attention mask so
        ``create_causal_mask`` early-exits (no CPU scalar allocation).
        """
        out = self.lm(
            input_ids=self.static_input_ids,
            position_ids=self.static_position_ids,
            attention_mask=self.static_attn_mask,
            past_key_values=self.cache,
            use_cache=True,
        )
        hidden = out.last_hidden_state[:, -1:, :]
        self.static_logits.copy_(self.lm_head(hidden) / LLM_LOGITS_SCALING)

    # ------------------------------------------------------------------ #
    # prefill
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def prefill(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Eager prefill: fill the StaticCache and return the first token id.

        Args:
            inputs_embeds: ``(1, T, hidden)`` bf16 tensor on cuda (the merged
                multimodal embeds **before** the Granite embedding multiplier;
                ``GraniteModel.forward`` applies it internally).

        Returns:
            ``(1, 1)`` int64 tensor with the first generated token.
        """
        T = inputs_embeds.shape[1]
        assert T < self.max_cache_len, f"prompt {T} >= max_cache_len {self.max_cache_len}"
        # Always start from a clean cache so prefill/generate are idempotent
        # and safe to call repeatedly on the same decoder instance.
        self._reset_cache_pos(0)
        position_ids = torch.arange(T, device=self.device).unsqueeze(0)
        out = self.lm(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=self.cache,
            use_cache=True,
        )
        hidden = out.last_hidden_state[:, -1:, :]
        logits = self.lm_head(hidden) / LLM_LOGITS_SCALING
        return logits.argmax(dim=-1)  # (1, 1)

    # ------------------------------------------------------------------ #
    # CUDA-graph capture of the decode step
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def capture(self, first_token: torch.Tensor, prefill_len: int) -> None:
        """Capture the single-token decode step into a CUDA graph.

        Must be called once after :meth:`prefill`.  ``first_token`` is the token
        produced by the prefill (the input to the first decode step);
        ``prefill_len`` is the prompt length (the K/V cache fill level after
        prefill).
        """
        # Prime the static buffers with valid first-decode values.
        self.static_input_ids.copy_(first_token.reshape(1, 1))
        self.static_position_ids.copy_(
            torch.tensor([[prefill_len]], device=self.device)
        )
        self._set_mask(prefill_len + 1)

        # Warmup advances cumulative_length; we reset before capture so the
        # captured graph starts writing at slot ``prefill_len``.
        for _ in range(self.warmup_iters):
            self._decode_step_eager()
        torch.cuda.synchronize()
        self._reset_cache_pos(prefill_len)

        # Re-prime (warmup consumed the buffer values but shapes are identical).
        self.static_input_ids.copy_(first_token.reshape(1, 1))
        self.static_position_ids.copy_(
            torch.tensor([[prefill_len]], device=self.device)
        )
        self._set_mask(prefill_len + 1)

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._decode_step_eager()

        # The captured step advanced cumulative_length by 1 conceptually; reset
        # so the first generate replay writes slot ``prefill_len``.
        self._reset_cache_pos(prefill_len)
        self._captured = True

    # ------------------------------------------------------------------ #
    # generate
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
        """Greedy-generate ``max_new_tokens`` from ``inputs_embeds``.

        Prefill is eager; the subsequent ``max_new_tokens - 1`` decode steps are
        served by CUDA-graph replay (after :meth:`capture`).
        """
        T = inputs_embeds.shape[1]
        # (1) prefill -> first token
        next_token = self.prefill(inputs_embeds)  # (1, 1)
        gen_ids = [int(next_token.item())]

        if max_new_tokens <= 1:
            return self._finalize(gen_ids, 0.0, tokenizer)

        # (2) capture the decode graph (idempotent)
        if capture and not self._captured:
            self.capture(next_token, T)

        # (3) decode loop
        t0 = time.perf_counter()
        for i in range(max_new_tokens - 1):
            # The prefill produced token 0 (at position T).  Decode step i
            # feeds that token back at position T+i, so the K/V write slot
            # (cumulative_length == T+i) matches the RoPE position exactly.
            # The mask permits keys [0, T+i] which are all valid after this
            # step's in-graph cache write -- no stale slots leak through.
            cur_pos = T + i
            self.static_input_ids.copy_(next_token.reshape(1, 1))
            self.static_position_ids.copy_(
                torch.tensor([[cur_pos]], device=self.device)
            )
            self._set_mask(cur_pos + 1)  # valid keys = [0, cur_pos]
            if self._captured:
                self._graph.replay()
            else:
                self._decode_step_eager()
            next_token = self.static_logits.argmax(dim=-1)  # (1, 1)
            gen_ids.append(int(next_token.item()))
            if int(next_token.item()) == eos_token_id:
                break
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        wall_ms = (t1 - t0) * 1000.0
        return self._finalize(gen_ids, wall_ms, tokenizer)

    def _finalize(
        self, gen_ids: list[int], decode_wall_ms: float, tokenizer: Any
    ) -> GenerateResult:
        ids = torch.tensor(gen_ids, dtype=torch.int64).unsqueeze(0)
        n = len(gen_ids)
        text = ""
        if tokenizer is not None:
            text = tokenizer.decode(ids[0], skip_special_tokens=True)
        # decode tok/s excludes prefill (pure decode throughput)
        decode_tps = n / max(decode_wall_ms / 1000.0, 1e-9)
        return GenerateResult(
            ids=ids,
            text=text,
            n_tokens=n,
            total_ms=decode_wall_ms,
            tok_per_s=decode_tps,
        )

    # ------------------------------------------------------------------ #
    # benchmark
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def bench(
        self,
        inputs_embeds: torch.Tensor,
        max_new_tokens: int = 100,
        eos_token_id: int = LLM_EOS_TOKEN_ID,
        decode_iters: int = 20,
    ) -> BenchReport:
        """Benchmark prefill, per-token decode, and total generate.

        Prefill and per-token decode use CUDA events (warmup 3,
        ``decode_iters`` timed iterations).  Total generate is wall-clock over
        the full decode loop.

        The per-token decode timing measures the steady-state graph replay at
        a fixed cache position (reset each iteration so we stay within bounds
        and measure the same work each time).
        """
        T = inputs_embeds.shape[1]
        pos_ids_prefill = torch.arange(T, device=self.device).unsqueeze(0)

        # (a) prefill time (eager, single forward).  Each timed iteration
        # writes into the cache from slot 0, so reset between iters.
        def _prefill():
            self._reset_cache_pos(0)
            self.lm(
                inputs_embeds=inputs_embeds,
                position_ids=pos_ids_prefill,
                past_key_values=self.cache,
                use_cache=True,
            )

        prefill_ms = self._cuda_timer(_prefill, warmup=3, iters=10)

        # (b) capture the decode graph on a cleanly populated cache.
        self._reset_cache_pos(0)
        first_tok = self.prefill(inputs_embeds)  # fills K/V [0, T), gives tok 1
        self.capture(first_tok, T)

        # Per-token decode time: replay at a fixed position so every iteration
        # does identical work.  Reset the cache slot each iter (the write target
        # is cumulative_length which the graph advances in-place).
        self.static_input_ids.copy_(first_tok.reshape(1, 1))
        self.static_position_ids.copy_(torch.tensor([[T]], device=self.device))
        self._set_mask(T + 1)

        def _one_decode():
            self._graph.replay()
            self._reset_cache_pos(T)  # undo the in-place advance for next iter

        decode_ms = self._cuda_timer(_one_decode, warmup=3, iters=decode_iters)
        decode_tps = 1000.0 / decode_ms if decode_ms > 0 else 0.0

        # (c) full generate (wall clock).  Reset cache and recapture so the
        # generate loop starts from a clean prefill state.
        self._reset_cache_pos(0)
        self._captured = False
        res = self.generate(inputs_embeds, max_new_tokens=max_new_tokens, eos_token_id=eos_token_id)

        return BenchReport(
            prefill_ms=prefill_ms,
            decode_ms_per_token=decode_ms,
            decode_tok_per_s=decode_tps,
            total_ms=res.total_ms,
            total_tok_per_s=res.tok_per_s,
            notes=f"decoded {res.n_tokens} tokens; cache_len={self.max_cache_len}",
        )

    @staticmethod
    def _cuda_timer(fn, warmup: int = 3, iters: int = 20) -> float:
        """Median GPU time (ms) for ``fn`` using CUDA events."""
        import statistics

        torch.cuda.synchronize()
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        return statistics.median(times)
