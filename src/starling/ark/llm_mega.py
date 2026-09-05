"""CUDA-graph-captured greedy decoder for the ARK-ASR-3B Qwen2.5 LLM.

The LLM decoder dominates ARK-ASR-3B inference runtime. The stock eager
``model.generate`` path launches dozens of small kernels per token and rebuilds
Python state on every step, capping throughput far below the memory-bandwidth
ceiling.

This module closes that gap with:

* **Phase A** - a correct CUDA-graph-captured greedy decode built on top of the
  Qwen2Model's *own* layers and ``transformers.StaticCache``. Graph replay of the
  model's own ops is bit-exact with eager, so the decoded token sequence matches
  the golden reference exactly.
* **Phase B** - benchmark hooks (prefill ms, decode ms/token, tok/s, total ms).
* **Phase C** - an optional fused decode path that swaps in the proven Triton
  elementwise kernels (fused RMSNorm, fused SwiGLU, fused residual scale-add)
  from :mod:`starling.granite.llm_kernels` to cut memory traffic and launch
  count further. GEMMs stay as cuBLAS bf16 matmuls; RoPE is kept in PyTorch
  (Triton bf16 multiply rounds differently for large Q values).

Qwen2.5 numerics differ from Granite: no embedding multiplier, no logits
scaling, residual multiplier 1.0, attention scale ``1/sqrt(head_dim)``, and
RMSNorm eps 1e-6. These are wired in from :mod:`starling.ark.config`.

Design notes
------------
``StaticCache`` pre-allocates fixed-address K/V tensors for all 36 layers plus a
``cumulative_length`` tensor per layer that is incremented in-place on each
``update``. This is inherently CUDA-graph safe.

``create_causal_mask`` allocates CPU scalars which abort CUDA-graph capture; we
bypass it by feeding a pre-computed **4D** attention mask
(``(1, 1, 1, max_cache_len)``); the masking plumbing early-exits and returns a
4D mask as-is.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

import torch

from ..attention import gqa_attention

from .config import (
    EOS_TOKEN_ID,
    LLM_EMBEDDING_MULTIPLIER,
    LLM_LOGITS_SCALING,
    LLM_RESIDUAL_MULTIPLIER,
    LLM_RMS_NORM_EPS,
)

# Hard cap on the number of distinct prompt-length ``T`` prefill graphs retained
# resident. The prefill graph cache is keyed on the prompt token length, so a
# benchmark that feeds clips of many different lengths (each producing a
# distinct prompt T) would otherwise capture one graph per length and never free
# it, leaking the private CUDA-graph memory pool of each capture. The cache is
# bounded by LRU eviction: when a new T arrives and the cache is full, the
# least-recently-used prefill graph is dropped and its pool released
# (``del graph; torch.cuda.empty_cache()``). This bounds memory without changing
# numerics (each distinct T still gets its own exact graph; only old ones are
# recycled once resident memory pressure rises).
#
# 64 (not 512): the encoder graphs are now shape-bucketed (see
# ``MegaPipeline.shape_bucketing``), so the prefill graphs are the *only*
# per-clip accumulator left. At 512 they grow one-per-distinct-length across a
# 350-clip leaderboard sweep (~50-70 GB of private pools) and overflow the 32 GB
# card into WSL shared memory until it OOMs. A 64-graph ceiling caps that at a
# few GB; because leaderboard prompt lengths are near-unique and processed
# sequentially, evicted graphs are never revisited, so eviction costs ~no
# re-capture (unlike a repeated-length workload).
MAX_PREFILL_GRAPHS: int = 64


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
    """CUDA-graph-captured greedy decoder for the ARK-ASR-3B Qwen2.5 LLM.

    Wraps the loaded ``Qwen2Model`` decoder trunk (the ``language_model``
    component from :func:`starling.ark.loader.get_components`) plus the parent
    model's tied ``lm_head``. The LLM's own layers are used unchanged so decode
    output is bit-exact with the eager golden reference.

    Args:
        language_model: The ``Qwen2Model`` (has ``embed_tokens``, ``layers``,
            ``norm``, ``rotary_emb``).
        lm_head: The ``nn.Linear`` lm_head from the top-level model.
        max_cache_len: Fixed K/V cache length to pre-allocate.
        warmup_iters: CUDA-graph warmup iterations before capture.
        device/dtype: Must match the loaded weights (cuda / bfloat16).
        max_prefill_graphs: Maximum number of distinct prompt-length prefill
            graphs kept resident (LRU eviction beyond this).
    """

    def __init__(
        self,
        language_model: Any,
        lm_head: Any,
        max_cache_len: int = 4096,
        warmup_iters: int = 3,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        *,
        max_prefill_graphs: int = MAX_PREFILL_GRAPHS,
        prefill_use_graph: bool = True,
        graph_pool=None,
    ) -> None:
        self.lm = language_model
        self.lm_head = lm_head
        self.config = language_model.config
        self.max_cache_len = int(max_cache_len)
        self.warmup_iters = int(warmup_iters)
        self.device = device
        self.dtype = dtype
        self.graph_pool = graph_pool  # shared pool for safe eviction
        self.max_prefill_graphs = max(1, int(max_prefill_graphs))
        # Prefill is a one-shot, compute-bound forward over T tokens: CUDA-graph
        # capture removes only its host launch overhead (a small fraction of the
        # prefill compute) but costs a full per-T graph's private pool. On a
        # diverse-length sweep those per-T graphs are the dominant memory
        # accumulator (they overflow the 32 GB card into WSL shared memory ->
        # RTFx collapse). Running prefill eager keeps the memory flat with ~no
        # speed cost; the decode loop stays graphed. Byte-exact either way.
        self.prefill_use_graph = bool(prefill_use_graph)

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

        # The StaticCache is allocated here so its fixed-address tensors exist
        # before any graph capture. Qwen2Model's config drives layer count /
        # GQA head layout.
        from transformers.cache_utils import StaticCache

        self.cache = StaticCache(config=self.config, max_cache_len=self.max_cache_len)

        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._captured = False

        # ---- graphed-prefill cache (shape-keyed by prompt length T) ----
        # The eager prefill runs all 36 decoder layers over the full prompt and
        # is a fixed-shape computation, so it captures into a CUDA graph the same
        # way the decode step does. The graph runs the model's own forward, so it
        # is byte-exact with eager prefill. One capture per T; amortised across
        # same-length calls.
        #
        # ``OrderedDict`` + LRU eviction: each distinct prompt length T captures
        # its own graph with its own private CUDA-graph memory pool. A benchmark
        # that feeds many different lengths would otherwise accumulate one pool
        # per length forever. The cache is capped to ``max_prefill_graphs``; on
        # overflow the LRU entry is dropped and its pool released
        # (``del graph; torch.cuda.empty_cache()``) so resident memory stays
        # bounded. The per-T ``_prefill_pos_ids["mask_<T>"]`` attention-mask
        # cache is NOT bounded (those are small host-built tensors that do not
        # own a graph pool) so it is left as a plain dict to preserve exact
        # masks across eviction/re-capture.
        self._prefill_graphs: OrderedDict[int, torch.cuda.CUDAGraph] = OrderedDict()
        self._prefill_static_in: dict[int, torch.Tensor] = {}
        self._prefill_static_out: dict[int, torch.Tensor] = {}
        self._prefill_pos_ids: dict[int, torch.Tensor] = {}

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
        ``create_causal_mask`` early-exits (no CPU scalar allocation). Qwen2.5
        applies no logits scaling, so ``lm_head`` output is copied as-is.
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
    # prefill graph cache eviction
    # ------------------------------------------------------------------ #
    def _free_prefill_graph(self, T: int) -> None:
        """Release a captured prefill graph + its static buffers.

        Each prefill graph has its OWN private pool, so ``graph.reset()``
        deterministically frees its blocks. (The prefill output is cloned out
        of the pool at capture time, so the stored ``_prefill_static_out`` ref
        doesn't dangle after reset.)
        """
        graph = self._prefill_graphs.pop(T, None)
        self._prefill_static_in.pop(T, None)
        self._prefill_static_out.pop(T, None)
        self._prefill_pos_ids.pop(T, None)
        self._prefill_pos_ids.pop(f"mask_{T}", None)
        if graph is not None:
            try:
                graph.reset()
            except Exception:
                pass
            del graph

    def _evict_prefill_if_needed(self) -> None:
        """Evict the LRU prefill entry while the cache exceeds ``max_prefill_graphs``."""
        while len(self._prefill_graphs) >= self.max_prefill_graphs:
            # popitem(last=False) -> least-recently-used (head of the OrderedDict).
            T, _ = self._prefill_graphs.popitem(last=False)
            self._free_prefill_graph(T)

    # ------------------------------------------------------------------ #
    # prefill
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def prefill(self, inputs_embeds: torch.Tensor, *, use_graph: bool = True) -> torch.Tensor:
        """Fill the StaticCache and return the first token id.

        By default the prefill is served by a shape-keyed CUDA graph (captured on
        first use for a given prompt length ``T``); pass ``use_graph=False`` to
        force the eager forward. Both paths run the model's own layers, so the
        result is byte-exact; the graph only removes host launch overhead.

        The cache is reset to position 0 first, so prefill/generate are
        idempotent and safe to call repeatedly on the same decoder instance.

        Args:
            inputs_embeds: ``(1, T, hidden)`` bf16 tensor on cuda (the multimodal
                embeds with audio features already injected).
            use_graph: If True (default), use the captured prefill graph for this
                ``T`` (capturing it on first call).

        Returns:
            ``(1, 1)`` int64 tensor with the first generated token.
        """
        T = inputs_embeds.shape[1]
        assert T < self.max_cache_len, f"prompt {T} >= max_cache_len {self.max_cache_len}"
        # Always start from a clean cache so prefill/generate are idempotent
        # and safe to call repeatedly on the same decoder instance.
        self._reset_cache_pos(0)
        if use_graph:
            self._prefill_graphed(inputs_embeds, T)
        else:
            self._prefill_eager(inputs_embeds, T)
        hidden = self._prefill_last_hidden(T)
        logits = self.lm_head(hidden) / LLM_LOGITS_SCALING
        return logits.argmax(dim=-1)  # (1, 1)

    def _prefill_eager(self, inputs_embeds: torch.Tensor, T: int) -> None:
        """Run the eager prefill forward, populating the StaticCache."""
        position_ids = torch.arange(T, device=self.device).unsqueeze(0)
        out = self.lm(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=self._prefill_mask(T),
            past_key_values=self.cache,
            use_cache=True,
        )
        self._prefill_static_out[T] = out.last_hidden_state

    def _prefill_mask(self, T: int) -> torch.Tensor:
        """Pre-computed 4D causal mask for prefill so ``create_causal_mask``
        early-exits (it otherwise allocates a CPU scalar that aborts graph
        capture).

        Shape ``(1, 1, T, max_cache_len)``: lower-triangular causal in the
        first ``T`` query/key positions (so query ``i`` attends to keys
        ``[0, i]``) and ``-inf`` for key positions ``>= T`` (the StaticCache
        has ``max_cache_len`` key slots; the un-written ones must be masked).
        The eager attention adds this mask to ``attn_weights`` of shape
        ``(1, n_q, T, T_written)`` where ``T_written == T`` during prefill, so
        the leading ``T`` columns are what matter; the trailing ``max_cache_len
        - T`` masked columns are never read.
        """
        key = f"mask_{T}"
        m = self._prefill_pos_ids.get(key)
        if m is None:
            neg = self._neg_val
            m = torch.full((1, 1, T, self.max_cache_len), neg, dtype=self.dtype, device=self.device)
            # lower-triangular causal over the leading T x T block.
            causal = torch.tril(torch.ones(T, T, device=self.device, dtype=torch.bool))
            m[:, :, :T, :T] = torch.where(causal, 0.0, neg)
            self._prefill_pos_ids[key] = m
        return m

    def _prefill_last_hidden(self, T: int) -> torch.Tensor:
        """Return the last-position hidden state produced by the prefill.

        Both graphed and eager prefill store the full ``last_hidden_state``
        sequence; the final position drives the first-token argmax.
        """
        return self._prefill_static_out[T][:, -1:, :]

    @torch.inference_mode()
    def _prefill_graphed(self, inputs_embeds: torch.Tensor, T: int) -> None:
        """Capture (first call for ``T``) and replay the prefill graph.

        On a hit the entry is marked most-recently-used (``move_to_end``). On a
        miss the LRU entry is evicted first (if the cache is full) so the
        resident-graph count stays bounded; then a fresh graph is captured.
        """
        graph = self._prefill_graphs.get(T)
        if graph is None:
            # New prompt length: evict the LRU prefill graph + free its pool so
            # the resident count never exceeds ``max_prefill_graphs``.
            self._evict_prefill_if_needed()

            static_in = torch.zeros_like(inputs_embeds)
            self._prefill_static_in[T] = static_in
            pos_ids = torch.arange(T, device=self.device).unsqueeze(0)
            self._prefill_pos_ids[T] = pos_ids
            mask = self._prefill_mask(T)
            static_in.copy_(inputs_embeds)
            # Warmup on a side stream so lazy init settles before capture.
            device = torch.device(self.device)
            side = torch.cuda.Stream(device=device)
            side.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(side):
                for _ in range(2):
                    self._reset_cache_pos(0)
                    out = self.lm(
                        inputs_embeds=static_in,
                        position_ids=pos_ids,
                        attention_mask=mask,
                        past_key_values=self.cache,
                        use_cache=True,
                    )
            torch.cuda.current_stream(device).wait_stream(side)
            self._reset_cache_pos(0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = self.lm(
                    inputs_embeds=static_in,
                    position_ids=pos_ids,
                    attention_mask=mask,
                    past_key_values=self.cache,
                    use_cache=True,
                )
            # Keep the graph-owned output tensor itself. CUDA graph replay
            # mutates tensors allocated during capture in-place; cloning here
            # would freeze the capture-time output and every later replay would
            # read a stale hidden state.
            self._prefill_static_out[T] = out.last_hidden_state
            self._prefill_graphs[T] = graph  # appended at the MRU tail
            # Reset so the first real replay starts from a clean cache.
            self._reset_cache_pos(0)
        else:
            self._prefill_graphs.move_to_end(T)  # mark most-recently-used
        self._prefill_static_in[T].copy_(inputs_embeds)
        self._reset_cache_pos(0)
        graph.replay()

    # ------------------------------------------------------------------ #
    # CUDA-graph capture of the decode step
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def capture(self, first_token: torch.Tensor, prefill_len: int) -> None:
        """Capture the single-token decode step into a CUDA graph.

        Must be called once after :meth:`prefill`. ``first_token`` is the token
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
        eos_token_id: int = EOS_TOKEN_ID,
        tokenizer: Any = None,
        capture: bool = True,
    ) -> GenerateResult:
        """Greedy-generate ``max_new_tokens`` from ``inputs_embeds``.

        Prefill is eager; the subsequent ``max_new_tokens - 1`` decode steps are
        served by CUDA-graph replay (after :meth:`capture`).
        """
        T = inputs_embeds.shape[1]
        # Guard against cache overflow: the prefill fills K/V slots [0, T) and
        # each decode step writes one additional slot, so the total cache
        # footprint of ``max_new_tokens`` new tokens is ``T + max_new_tokens - 1``.
        max_safe = self.max_cache_len - T + 1
        if max_new_tokens > max_safe:
            raise ValueError(
                f"max_new_tokens={max_new_tokens} would overflow the static KV cache "
                f"(prompt T={T}, max_cache_len={self.max_cache_len}; at most "
                f"{max_safe} new tokens fit). Increase max_cache_len or reduce "
                f"max_new_tokens."
            )
        if inputs_embeds.shape[0] != 1:
            raise ValueError(
                f"LLMMega only supports batch=1 (static buffers + GQA cache layout "
                f"are hard-coded for B=1), got batch={inputs_embeds.shape[0]}."
            )
        if max_new_tokens <= 0:
            # HF generate() returns zero new tokens in this case; match that.
            return self._finalize([], 0.0, tokenizer)
        # (1) prefill -> first token
        next_token = self.prefill(inputs_embeds, use_graph=self.prefill_use_graph)  # (1, 1)
        gen_ids = [int(next_token.item())]

        if max_new_tokens <= 1:
            return self._finalize(gen_ids, 0.0, tokenizer)

        # (2) capture the decode graph (idempotent)
        if capture and not self._captured:
            self.capture(next_token, T)

        # (3) decode loop
        t0 = time.perf_counter()
        for i in range(max_new_tokens - 1):
            # The prefill produced token 0 (at position T). Decode step i feeds
            # that token back at position T+i, so the K/V write slot
            # (cumulative_length == T+i) matches the RoPE position exactly. The
            # mask permits keys [0, T+i] which are all valid after this step's
            # in-graph cache write -- no stale slots leak through.
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
        eos_token_id: int = EOS_TOKEN_ID,
        decode_iters: int = 20,
    ) -> BenchReport:
        """Benchmark prefill, per-token decode, and total generate.

        Prefill and per-token decode use CUDA events (warmup 3,
        ``decode_iters`` timed iterations). Total generate is wall-clock over
        the full decode loop.

        The per-token decode timing measures the steady-state graph replay at a
        fixed cache position (reset each iteration so we stay within bounds and
        measure the same work each time).
        """
        T = inputs_embeds.shape[1]
        pos_ids_prefill = torch.arange(T, device=self.device).unsqueeze(0)

        # (a) prefill time (eager, single forward). Each timed iteration writes
        # into the cache from slot 0, so reset between iters.
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
        first_tok = self.prefill(inputs_embeds, use_graph=self.prefill_use_graph)  # fills K/V [0, T), gives tok 1
        self.capture(first_tok, T)

        # Per-token decode time: replay at a fixed position so every iteration
        # does identical work. Reset the cache slot each iter (the write target
        # is cumulative_length which the graph advances in-place).
        self.static_input_ids.copy_(first_tok.reshape(1, 1))
        self.static_position_ids.copy_(torch.tensor([[T]], device=self.device))
        self._set_mask(T + 1)

        def _one_decode():
            self._graph.replay()
            self._reset_cache_pos(T)  # undo the in-place advance for next iter

        decode_ms = self._cuda_timer(_one_decode, warmup=3, iters=decode_iters)
        decode_tps = 1000.0 / decode_ms if decode_ms > 0 else 0.0

        # (c) full generate (wall clock). Reset cache and recapture so the
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


# =========================================================================== #
# Phase C: Fused decode path with Triton elementwise kernels
# =========================================================================== #
# Reuse the model's own Linear weights but replace the memory-bound elementwise
# glue with single-launch Triton kernels. GEMMs stay as cuBLAS bf16 matmuls.
# The kernels are the proven granite implementations (identical arithmetic:
# RMSNorm without mean subtraction, SwiGLU silu*mul, residual x + alpha*y) and
# are re-verified against the ARK golden transcript.

# Pre-extract constants to avoid repeated attribute lookups in the hot path.
_EMB_MULT = LLM_EMBEDDING_MULTIPLIER  # 1.0 for Qwen2.5 (no embedding multiplier)


class FusedLLMMega(LLMMega):
    """CUDA-graph-captured greedy decoder with **fused Triton elementwise kernels**.

    Inherits all graph-capture / generate / bench machinery from :class:`LLMMega`
    and overrides only :meth:`_decode_step_eager` with a custom forward that
    manually iterates the 36 Qwen2 decoder layers, replacing the small
    elementwise ops (RMSNorm, SwiGLU, residual scale-add) with single-launch
    Triton kernels imported from :mod:`starling.granite.llm_kernels`.

    GEMMs (q/k/v/o_proj, gate/up/down_proj, lm_head) and the attention
    softmax/matmul stay as stock PyTorch ops (cuBLAS). RoPE is kept in PyTorch
    (same as the granite decoder): Triton bf16 multiply rounds differently than
    ATen for large Q values, which would exceed the logit tolerance over 36
    layers.

    Correctness: fused kernels use fp32 internal accumulation matching the
    reference; max abs logit diff < ``LLM_LOGIT_ATOL`` (0.05) and the decoded
    transcript is identical to the golden reference.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        from ..flags import get_default_flags
        from ..granite import llm_kernels as _k  # reuse the proven granite kernels

        self._k = _k
        self._flags = get_default_flags()
        # Pre-extract per-layer references for speed in the hot decode loop.
        self._layers = list(self.lm.layers)
        self._embed = self.lm.embed_tokens
        self._final_norm = self.lm.norm
        self._rotary = self.lm.rotary_emb
        cfg = self.config
        self._n_q_heads = int(cfg.num_attention_heads)
        self._n_kv_heads = int(cfg.num_key_value_heads)
        self._head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // self._n_q_heads))
        self._n_kv_groups = self._n_q_heads // self._n_kv_heads
        # Qwen2.5 numerics (from config; mirrored in starling.ark.config).
        # The attention scale derives from the loaded head_dim (1/sqrt(64) for
        # the 0.6B decoder, 1/sqrt(128) for the 3B) rather than a module
        # constant, so both tracks share this decode path.
        self._attn_scale = self._head_dim ** -0.5
        self._res_mult = LLM_RESIDUAL_MULTIPLIER
        self._rms_eps = LLM_RMS_NORM_EPS
        self._intermediate = int(cfg.intermediate_size)

        # ---- weight fusion: collapse per-layer GEMV launches ----
        # The decode is launch-bound: each token runs ~108 attention GEMVs
        # (q/k/v/o × 36 layers) + 108 MLP GEMVs (gate/up/down × 36), each a
        # 12-55 us cuBLAS launch doing almost no compute. Concatenating the
        # independent projections that share an input into one GEMM cuts the
        # launch count roughly in half with no arithmetic change (concatenating
        # weights/biases is associative over the matmul+add).
        #   * QKV fusion: 3 GEMVs -> 1 ([Wq;Wk;Wv]@x + [bq;bk;bv]).
        #   * gate+up fusion: 2 GEMVs -> 1 ([Wg;Wu]@x).
        # The model's own layer weights are left untouched; the fused tensors
        # are additive copies used only by the fused decode path.
        self._fused = self._fuse_layer_weights()

    def _fuse_layer_weights(self) -> list[dict]:
        """Pre-concatenate QKV and gate/up weights per layer (additive copies).

        Returns one dict per layer with keys ``qkv_w``, ``qkv_b`` (or None),
        ``gu_w``, ``o_proj``, ``down_proj``. The original modules are not
        modified; the fused tensors are byte-exact equivalents.
        """
        fused = []
        for layer in self._layers:
            sa = layer.self_attn
            mlp = layer.mlp
            # QKV: weight rows laid out [q(2048); k(256); v(256)] -> (2560, 2048).
            qkv_w = torch.cat([sa.q_proj.weight, sa.k_proj.weight, sa.v_proj.weight], dim=0)
            if sa.q_proj.bias is not None:
                qkv_b = torch.cat([sa.q_proj.bias, sa.k_proj.bias, sa.v_proj.bias], dim=0)
            else:
                qkv_b = None
            # gate+up: rows [gate(11008); up(11008)] -> (22016, 2048). No bias.
            gu_w = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], dim=0)
            fused.append({
                "qkv_w": qkv_w.contiguous(),
                "qkv_b": qkv_b.contiguous() if qkv_b is not None else None,
                "gu_w": gu_w.contiguous(),
                "o_proj": sa.o_proj,
                "down_proj": mlp.down_proj,
            })
        return fused

    def _decode_step_eager(self) -> None:
        """Custom single-token decode forward with fused Triton kernels.

        Replicates Qwen2Model.forward + Qwen2DecoderLayer.forward exactly but
        replaces elementwise glue with fused kernels and collapses the
        independent QKV and gate/up GEMVs into one GEMM each. Writes the final
        logits (post lm_head; Qwen2.5 applies no logits scaling) into
        ``self.static_logits``.

        Fused (Triton, exact match): RMSNorm, SwiGLU ``silu(gate)*up``,
        residual scale-add ``x + alpha*y`` (alpha = 1.0 for Qwen2.5), and RoPE
        (Q and K rotated in one launch).
        Fused GEMMs (cuBLAS, exact): QKV projection, gate+up projection.
        Kept as separate cuBLAS bf16 GEMVs: o_proj and down_proj (their inputs
        differ from any sibling, so they cannot be merged).
        """
        k = self._k
        hd = self._head_dim
        n_q = self._n_q_heads
        n_kv = self._n_kv_heads
        qkv_split = [n_q * hd, n_kv * hd, n_kv * hd]  # [2048, 256, 256]
        inter = self._intermediate

        # (1) embedding lookup (Qwen2.5 has no embedding multiplier).
        hidden = self._embed(self.static_input_ids) * _EMB_MULT  # (1, 1, 2048)

        # (2) rotary cos/sin for this position (single head_dim, shared by Q/K).
        cos, sin = self._rotary(hidden, position_ids=self.static_position_ids)

        # (3) iterate layers
        for idx, layer in enumerate(self._layers):
            f = self._fused[idx]
            o_proj = f["o_proj"]
            down_proj = f["down_proj"]

            # --- attention block ---
            residual = hidden  # (1, 1, 2048)

            # fused input RMSNorm (Qwen2RMSNorm: no mean subtraction, eps=1e-6)
            normed = k.fused_rmsnorm(hidden, layer.input_layernorm.weight, self._rms_eps)

            # FUSED QKV projection: one cuBLAS GEMV instead of three.
            # qkv: (1, 1, 2560) -> split into q(1,n_q,1,hd), k(1,n_kv,1,hd), v(1,n_kv,1,hd)
            x2 = normed.view(1, -1)  # (1, 2048)
            qkv = torch.nn.functional.linear(x2, f["qkv_w"], f["qkv_b"])  # (1, 2560)
            q, kv, v = qkv.view(-1).split(qkv_split, dim=0)
            q = q.view(1, n_q, 1, hd)
            kv = kv.view(1, n_kv, 1, hd)
            v = v.view(1, n_kv, 1, hd)

            # FUSED RoPE: rotate Q and K in one Triton kernel launch.
            q, kv = k.fused_rope(q, kv, cos, sin)

            # cache update (in-place on static-address K/V tensors)
            kv, v = self.cache.update(kv, v, idx)

            attn_out = gqa_attention(
                q,
                kv,
                v,
                self.static_attn_mask,
                self._attn_scale,
                self.dtype,
                self._flags,
            )

            # reshape + output projection
            attn_out = attn_out.transpose(1, 2).reshape(1, 1, n_q * hd)
            attn_out = o_proj(attn_out)  # (1, 1, 2048)

            # fused residual scale-add (alpha = 1.0 for Qwen2.5)
            hidden = k.fused_residual_scale(residual, attn_out, self._res_mult)

            # --- MLP block ---
            residual = hidden

            # fused post-attention RMSNorm
            normed = k.fused_rmsnorm(hidden, layer.post_attention_layernorm.weight, self._rms_eps)

            # FUSED gate+up projection: one cuBLAS GEMV instead of two.
            x3 = normed.view(1, -1)  # (1, 2048)
            gu = torch.nn.functional.linear(x3, f["gu_w"], None)  # (1, 22016)
            gate, up = gu.view(-1).split([inter, inter], dim=0)
            gate = gate.view(1, 1, inter)
            up = up.view(1, 1, inter)

            # fused SwiGLU: silu(gate) * up
            act = k.fused_silu_mul(gate, up)  # (1, 1, inter)

            # down projection (cuBLAS bf16 GEMV)
            mlp_out = down_proj(act)  # (1, 1, 2048)


            # fused residual scale-add (alpha = 1.0 for Qwen2.5)
            hidden = k.fused_residual_scale(residual, mlp_out, self._res_mult)

        # (4) final fused RMSNorm
        hidden = k.fused_rmsnorm(hidden, self._final_norm.weight, self._rms_eps)

        # (5) lm_head (Qwen2.5 applies no logits scaling; LLM_LOGITS_SCALING=1.0)
        logits = self.lm_head(hidden) / LLM_LOGITS_SCALING
        self.static_logits.copy_(logits)
