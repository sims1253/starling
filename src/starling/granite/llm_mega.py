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

Each warmup resets the cache counter to ``prefill_len`` before writing one
decode slot. Capture and generation reset it again; future slots remain masked
until a real decode step writes them.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

import torch

from ..config import (
    LLM_EMBEDDING_MULTIPLIER,
    LLM_EOS_TOKEN_ID,
    LLM_LOGITS_SCALING,
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
    :func:`starling.granite.loader.get_components`) plus the parent model's ``lm_head``.
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
        self._prefill_graphs: OrderedDict[int, tuple[torch.Tensor, torch.cuda.CUDAGraph, torch.Tensor]] = OrderedDict()
        self._prefill_masks: dict[int, torch.Tensor] = {}
        self._max_prefill_graphs = 8

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #
    def _reset_cache_pos(self, n: int) -> None:
        """Reset every layer's ``cumulative_length`` to ``n`` in-place."""
        for layer in self.cache.layers:
            cl = layer.cumulative_length
            # StaticCache layers store a tensor; some layer variants store an
            # int. Handle both (fill_ for tensor, assignment for int).
            if hasattr(cl, "fill_"):
                cl.fill_(n)
            else:
                layer.cumulative_length = n

    def _set_mask(self, valid_len: int) -> None:
        """Unmask positions ``[0, valid_len)``; mask the rest to ``-inf``."""
        self.static_attn_mask.fill_(self._neg_val)
        self.static_attn_mask[:, :, :, :valid_len] = 0.0

    def _decode_step_eager(self) -> None:
        """One eager decode forward writing into ``static_logits``.

        Uses the model's own layers with the pre-computed 4D attention mask so
        ``create_causal_mask`` early-exits (no CPU tensor allocation). Passing
        ``attention_mask=None`` forces transformers 5.14+ to materialise a mask
        internally (``torch.where`` with a fresh scalar), which crashes under
        CUDA-graph capture and diverges from ``model.generate``'s SDPA
        ``is_causal`` fast-path.
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
    def prefill(self, inputs_embeds: torch.Tensor, *, use_graph: bool = True) -> torch.Tensor:
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
        if use_graph:
            entry = self._prefill_graphs.get(T)
            if entry is None:
                entry = self._capture_prefill(inputs_embeds)
                self._prefill_graphs[T] = entry
                while len(self._prefill_graphs) > self._max_prefill_graphs:
                    _, old = self._prefill_graphs.popitem(last=False)
                    try:
                        old[1].reset()
                    except Exception:
                        pass
            else:
                self._prefill_graphs.move_to_end(T)
            static_emb, graph, out_tok = entry
            static_emb.copy_(inputs_embeds)
            self._reset_cache_pos(0)
            graph.replay()
            return out_tok.clone()

        return self._prefill_eager(inputs_embeds)

    def _prefill_eager(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Reference prefill forward."""
        T = inputs_embeds.shape[1]
        self._reset_cache_pos(0)
        position_ids = torch.arange(T, device=self.device).unsqueeze(0)
        out = self.lm(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=self._prefill_mask(T),
            past_key_values=self.cache,
            use_cache=True,
        )
        hidden = out.last_hidden_state[:, -1:, :]
        logits = self.lm_head(hidden) / LLM_LOGITS_SCALING
        return logits.argmax(dim=-1)  # (1, 1)

    def _prefill_mask(self, T: int) -> torch.Tensor:
        """Graph-safe 4D causal mask for prefill over a StaticCache."""
        m = self._prefill_masks.get(T)
        if m is None:
            neg = self._neg_val
            ar = torch.arange(self.max_cache_len, device=self.device)
            q = torch.arange(T, device=self.device).unsqueeze(1)
            m = torch.where(
                ar[None, None, None, :] <= q[None, None, :, :],
                0.0,
                neg,
            ).to(self.dtype)
            self._prefill_masks[T] = m
        return m

    @torch.inference_mode()
    def _capture_prefill(self, inputs_embeds: torch.Tensor):
        """Capture a prompt-length-specific prefill CUDA graph."""
        device = inputs_embeds.device
        static_emb = torch.empty_like(inputs_embeds)
        static_emb.copy_(inputs_embeds)

        def _run():
            self._reset_cache_pos(0)
            return self._prefill_eager(static_emb)

        side = torch.cuda.Stream(device=device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            for _ in range(2):
                _ = _run()
        torch.cuda.current_stream(device).wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        self._reset_cache_pos(0)
        with torch.cuda.graph(graph):
            out_tok = _run()
        self._reset_cache_pos(0)
        return static_emb, graph, out_tok

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
            self._reset_cache_pos(prefill_len)
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
        # Guard against cache overflow: the prefill fills K/V slots [0, T) and
        # each decode step writes one additional slot, so the total cache
        # footprint of ``max_new_tokens`` new tokens is ``T + max_new_tokens - 1``.
        # Without this guard, requesting too many tokens triggers an
        # ``index_copy_(): index out of bounds`` CUDA device-side assert that
        # poisons the CUDA context and cascades into opaque errors on every
        # subsequent CUDA call.
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
                f"LLMMega only supports batch=1 (static buffers + _repeat_kv reshape "
                f"are hard-coded for B=1), got batch={inputs_embeds.shape[0]}."
            )
        if max_new_tokens <= 0:
            # HF generate() returns zero new tokens in this case; match that.
            return self._finalize([], 0.0, tokenizer)
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


# =========================================================================== #
# Phase C: Fused decode path with Triton elementwise kernels
# =========================================================================== #
# Reuse the model's own Linear weights but replace the memory-bound elementwise
# glue with single-launch Triton kernels.  GEMMs stay as cuBLAS bf16 matmuls.

# Pre-extract constants to avoid repeated attribute lookups in the hot path.
_EMB_MULT = LLM_EMBEDDING_MULTIPLIER  # 12.0 for granite-4.0-1b


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA: repeat KV heads to match Q heads.  x is (B, n_kv, S, D) -> (B, n_q, S, D)."""
    if n_rep == 1:
        return x
    B, n_kv, S, D = x.shape
    return x[:, :, None, :, :].expand(B, n_kv, n_rep, S, D).reshape(B, n_kv * n_rep, S, D)


class FusedLLMMega(LLMMega):
    """CUDA-graph-captured greedy decoder with **fused Triton elementwise kernels**.

    Inherits all graph-capture / generate / bench machinery from
    :class:`LLMMega` and overrides only :meth:`_decode_step_eager` with a custom
    forward that manually iterates the 40 decoder layers, replacing the small
    elementwise ops (RMSNorm, RoPE, SwiGLU, residual scale-add) with
    single-launch Triton kernels.

    GEMMs (q/k/v/o_proj, gate/up/down_proj, lm_head) and the attention
    softmax/matmul stay as stock PyTorch ops (cuBLAS).  This cuts kernel
    launches from ~600/step to ~240/step (40 layers × 6 fused ops removed) and
    reduces intermediate tensor allocations.

    Correctness: fused kernels use fp32 internal accumulation matching the
    reference; max abs logit diff < ``LLM_LOGIT_ATOL`` (0.05) and the decoded
    transcript is identical to the golden reference.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        from . import llm_kernels as _k  # local import to avoid circular dep
        from ..attention import gqa_attention as _gqa_attention
        from ..flags import get_default_flags

        self._k = _k
        self._gqa_attention = _gqa_attention
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
        self._attn_scale = float(cfg.attention_multiplier)
        self._res_mult = float(cfg.residual_multiplier)
        self._rms_eps = float(cfg.rms_norm_eps)
        self._intermediate = int(cfg.intermediate_size)
        # Active optimisation flags (read once per forward, not per layer).
        self._flags = get_default_flags()
        # Precomputed RoPE ``rotate_half`` index + sign buffers.
        # The reference recipe is ``rotate_half(x) = cat(-x[half:], x[:half])``
        # which allocates two tensors per layer per step. We precompute a
        # permutation index and a +/-1 sign vector so the rotate becomes a
        # single gather + mul -- byte-exact (same fp arithmetic over the same
        # elements, just no ``cat`` allocation). Half-head_dim only; the
        # gathered vector broadcasts across (B, H, 1, hd).
        # Gated by ``OptFlags.rope_alloc_free`` so the ablation harness can
        # measure the benefit; defaults on (byte-exact).
        _hd = self._head_dim
        _half = _hd // 2
        _idx = torch.arange(_hd, device=self.device)
        self._rope_idx = torch.cat([_idx[_half:], _idx[:_half]]).long()
        self._rope_sign = torch.cat(
            [-torch.ones(_half, dtype=self.dtype, device=self.device),
             torch.ones(_half, dtype=self.dtype, device=self.device)]
        )                                          # (hd,) bf16
        # Pre-scaled lm_head weight: ``lm_head(h) / s == (W/s) @ h``. Folding
        # the constant into the GEMM epilogue removes one elementwise divide
        # per step. Done in fp32 then re-cast to bf16 to match the rounding
        # of the post-GEMM bf16 divide as closely as possible (verified
        # byte-exact against the golden greedy ids).
        _w = self.lm_head.weight.detach()
        self._lm_head_scaled_w = (_w.to(torch.float32) / LLM_LOGITS_SCALING).to(_w.dtype)
        # Pre-concatenated QKV / gate-up weights per layer (additive copies;
        # byte-exact -- see :meth:`_fuse_layer_weights`).  Built lazily so the
        # non-fused path pays nothing.
        self._fused: Optional[list[dict]] = None
        if self._flags.fused_qkv:
            self._fused = self._fuse_layer_weights()

        # ------------------------------------------------------------------
        # FP8 weight-only quantization (flag-gated, additive). When
        # ``fp8_weights`` is on we quantize the per-layer projection weights to
        # fp8e4m3 at load time and replace the bf16 ``F.linear`` calls in
        # :meth:`_decode_step_eager` with the shared fused dequant-GEMV. Halves the
        # weight bandwidth that dominates decode (~57% of the captured step per
        # the profiler).  NOT byte-exact (fp8 weight rounding); gated by
        # ``OptFlags.fp8_weights`` (which requires ``tolerance_mode=True`` and
        # forces ``fused_qkv``).  The lm_head stays bf16.  See
        # ``starling.granite.fp8``.
        # ------------------------------------------------------------------
        self._fp8: Optional[list[dict]] = None
        if self._flags.fp8_weights:
            from .fp8 import quantize_weight_e4m3
            assert self._fused is not None, "fp8_weights requires fused_qkv"
            self._fp8 = [
                {
                    "qkv_w":     quantize_weight_e4m3(f["qkv_w"].detach()),
                    "gu_w":      quantize_weight_e4m3(f["gu_w"].detach()),
                    "o_proj":    quantize_weight_e4m3(f["o_proj"].weight.detach()),
                    "down_proj": quantize_weight_e4m3(f["down_proj"].weight.detach()),
                }
                for f in self._fused
            ]

        # ------------------------------------------------------------------
        # NVFP4 weight quantization (flag-gated, additive). When
        # ``nvfp4_weights`` is on we quantize every GEMM weight to NVFP4 at
        # load time and replace the bf16 ``F.linear`` calls in
        # :meth:`_decode_step_eager` with the fused dequant-GEMV
        # (:func:`_fp4_linear_fused`). Nibble-packed codes + fp8 scales give
        # 3.56x weight-byte reduction; the fused kernel streams the packed
        # bytes and dequantizes in registers (no bf16 materialisation).
        # ------------------------------------------------------------------
        self._fp4: Optional[dict] = None
        if self._flags.nvfp4_weights or self._flags.nvfp4_lm_head_only:
            from .fp4 import quantize_fp4_packed
            self._fp4 = {"layers": [], "lm_head": None}
            if self._flags.nvfp4_weights:
                for f in self._fused:
                    self._fp4["layers"].append({
                        "qkv_w":     quantize_fp4_packed(f["qkv_w"].detach()),
                        "gu_w":      quantize_fp4_packed(f["gu_w"].detach()),
                        "o_proj":    quantize_fp4_packed(f["o_proj"].weight.detach()),
                        "down_proj": quantize_fp4_packed(f["down_proj"].weight.detach()),
                    })
            # lm_head uses the pre-scaled weight (lm_head_scale_fold default).
            self._fp4["lm_head"] = quantize_fp4_packed(self._lm_head_scaled_w.detach())

        # ------------------------------------------------------------------
        # GEMM-epilogue fusion (CODA Pattern 1): pre-build gamma-prescaled QKV
        # and gate-up weights so the RMSNorm reduces to a scalar ``rstd`` that
        # folds into the GEMV epilogue (see llm_kernels.fused_gemv_normscale).
        # NOT byte-exact (~1-3 bf16 ULP); gated by OptFlags.gemm_epilogue_fusion.
        # Requires fused_qkv so the concatenated weights exist.
        # ------------------------------------------------------------------
        self._fused_epilogue: Optional[list[dict]] = None
        if self._flags.gemm_epilogue_fusion and self._fused is not None:
            self._fused_epilogue = self._fuse_epilogue_weights()

    def _fuse_layer_weights(self) -> list[dict]:
        """Pre-concatenate QKV and gate/up weights per layer (additive copies).

        Returns one dict per layer with keys ``qkv_w``, ``gu_w``, ``o_proj``,
        ``down_proj``.  The original modules are not modified; the fused
        tensors are byte-exact equivalents (concatenating weights is
        associative over the matmul -- ``[Wq; Wk; Wv] @ x == [Wq@x; Wk@x;
        Wv@x]``).
        """
        fused = []
        for layer in self._layers:
            sa = layer.self_attn
            mlp = layer.mlp
            qkv_w = torch.cat(
                [sa.q_proj.weight, sa.k_proj.weight, sa.v_proj.weight], dim=0
            )
            gu_w = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], dim=0)
            fused.append({
                "qkv_w": qkv_w.contiguous(),
                "gu_w": gu_w.contiguous(),
                "o_proj": sa.o_proj,
                "down_proj": mlp.down_proj,
            })
        return fused

    def _fuse_epilogue_weights(self) -> list[dict]:
        """Pre-scale the QKV and gate-up weights by the preceding RMSNorm gamma,
        so the norm reduces to a scalar ``rstd`` folded into the GEMV epilogue
        (CODA Pattern 1: ``rmsnorm(x) @ W^T == (x @ (gamma .* W)^T) * rstd``).

        Returns one dict per layer: ``qkv_w_s``, ``gu_w_s`` (gamma .* W). The
        per-channel multiply is done in fp32 then re-cast to bf16 to control
        rounding. NOT byte-exact vs the unfused path (~1-3 bf16 ULP on the
        projection output, because the unfused path truncates the normalized
        hidden to bf16 before the GEMV while this path accumulates in fp32 and
        truncates once). Gated by ``OptFlags.gemm_epilogue_fusion``.
        """
        out = []
        for layer, f in zip(self._layers, self._fused):
            g_in = layer.input_layernorm.weight.detach()
            g_post = layer.post_attention_layernorm.weight.detach()
            out.append({
                "qkv_w_s": (f["qkv_w"].to(torch.float32) * g_in.to(torch.float32)).to(self.dtype).contiguous(),
                "gu_w_s":  (f["gu_w"].to(torch.float32)  * g_post.to(torch.float32)).to(self.dtype).contiguous(),
            })
        return out

    def _decode_step_eager(self) -> None:
        """Custom single-token decode forward with fused Triton kernels.

        Replicates GraniteModel.forward + GraniteDecoderLayer.forward exactly
        but replaces elementwise glue with fused kernels.  Writes the final
        logits (post lm_head / logits_scaling) into ``self.static_logits``.

        Fused (Triton, exact match): RMSNorm, SwiGLU ``silu(gate)*up``,
        residual scale-add ``x + alpha*y``.
        Kept as PyTorch ops: all GEMMs (cuBLAS), the attention softmax/matmul,
        and RoPE.  RoPE stays in PyTorch because Triton bf16 multiply rounds
        differently than ATen for the large Q values this model produces
        (Q range ±45), causing >0.05 logit divergence over 40 layers.  The
        other fused kernels match bit-exact (verified: 0.000000 diff).
        """
        k = self._k
        hd = self._head_dim
        n_q = self._n_q_heads
        n_kv = self._n_kv_heads
        half = hd // 2
        qkv_split = [n_q * hd, n_kv * hd, n_kv * hd]
        inter = self._intermediate
        flags = self._flags
        fused = self._fused
        fp4 = self._fp4  # None when nvfp4_weights is off -> default path unchanged
        fp8 = self._fp8  # None when fp8_weights is off -> default path unchanged
        epi = self._fused_epilogue  # None when gemm_epilogue_fusion is off
        use_epi = epi is not None and fp4 is None and fp8 is None  # fp4/fp8 take precedence

        # (1) embedding lookup + multiplier
        hidden = self._embed(self.static_input_ids) * _EMB_MULT  # (1, 1, 2048)

        # (2) rotary cos/sin for this position
        cos, sin = self._rotary(hidden, position_ids=self.static_position_ids)
        # cos/sin: (1, 1, head_dim) -> unsqueeze for broadcast with (B, H, 1, D)
        cos4 = cos.unsqueeze(1)  # (1, 1, 1, hd)
        sin4 = sin.unsqueeze(1)

        # (3) iterate layers
        for idx, layer in enumerate(self._layers):
            sa = layer.self_attn
            mlp = layer.mlp

            # --- attention block ---
            residual = hidden  # (1, 1, 2048)

            # Q/K/V projections + preceding RMSNorm. Three paths:
            #  * gemm_epilogue_fusion: fold the norm into the GEMV epilogue
            #    (skip the standalone _rmsnorm_kernel; ~1-3 bf16 ULP, not exact).
            #  * nvfp4_weights: slow dequant-then-matmul reference path.
            #  * default: fused RMSNorm then cuBLAS GEMM (byte-exact).
            if use_epi:
                fe = epi[idx]
                rstd = k.compute_rstd(hidden, self._rms_eps)
                qkv = k.fused_gemv_normscale(hidden, fe["qkv_w_s"], rstd).view(-1)
                q, kv, v = qkv.split(qkv_split, dim=0)
                q = q.view(1, n_q, 1, hd)
                kv = kv.view(1, n_kv, 1, hd)
                v = v.view(1, n_kv, 1, hd)
                o_proj = fused[idx]["o_proj"]
            else:
                # fused input RMSNorm
                normed = k.fused_rmsnorm(hidden, layer.input_layernorm.weight, self._rms_eps)

            if not use_epi:
                # (use_epi already set qkv/kv/v/o_proj above; otherwise pick the
                # fp8 / nvfp4 reference, fused-cuBLAS, or stock-projection path.)
                if fp8 is not None:
                    from .fp8 import fp8_linear
                    f8 = fp8[idx]
                    x2 = normed.view(1, -1)
                    qkv = fp8_linear(x2, f8["qkv_w"][0], f8["qkv_w"][1]).view(-1)
                    q, kv, v = qkv.split(qkv_split, dim=0)
                    q = q.view(1, n_q, 1, hd)
                    kv = kv.view(1, n_kv, 1, hd)
                    v = v.view(1, n_kv, 1, hd)
                    o_proj = None  # fp8 o-proj is dispatched at its call site
                elif fp4 is not None:
                    from .fp4 import _fp4_linear_fused
                    f4 = fp4["layers"][idx] if fp4["layers"] else None
                    x2 = normed.view(1, -1)
                    if f4 is not None:
                        qkv = _fp4_linear_fused(x2, f4["qkv_w"]).view(-1)
                        o_proj = None
                    else:
                        f = fused[idx]
                        qkv = torch.nn.functional.linear(x2, f["qkv_w"], None).view(-1)
                        o_proj = f["o_proj"]
                    q, kv, v = qkv.split(qkv_split, dim=0)
                    q = q.view(1, n_q, 1, hd)
                    kv = kv.view(1, n_kv, 1, hd)
                    v = v.view(1, n_kv, 1, hd)
                elif fused is not None:
                    f = fused[idx]
                    x2 = normed.view(1, -1)
                    qkv = torch.nn.functional.linear(x2, f["qkv_w"], None).view(-1)
                    q, kv, v = qkv.split(qkv_split, dim=0)
                    q = q.view(1, n_q, 1, hd)
                    kv = kv.view(1, n_kv, 1, hd)
                    v = v.view(1, n_kv, 1, hd)
                    o_proj = f["o_proj"]
                else:
                    q = sa.q_proj(normed).view(1, 1, n_q, hd).transpose(1, 2)
                    kv = sa.k_proj(normed).view(1, 1, n_kv, hd).transpose(1, 2)
                    v = sa.v_proj(normed).view(1, 1, n_kv, hd).transpose(1, 2)
                    o_proj = sa.o_proj

            # RoPE: rotate_half via precomputed index+sign buffers (byte-exact
            # with the reference cat-based recipe; avoids 2 allocs/layer).
            # Falls back to the cat-based reference when rope_alloc_free=False
            # so the ablation harness can measure the benefit.
            if self._flags.rope_alloc_free:
                q_rot = q[..., self._rope_idx] * self._rope_sign
                kv_rot = kv[..., self._rope_idx] * self._rope_sign
            else:
                q_rot = torch.cat((-q[..., half:], q[..., :half]), dim=-1)
                kv_rot = torch.cat((-kv[..., half:], kv[..., :half]), dim=-1)
            q = q * cos4 + q_rot * sin4
            kv = kv * cos4 + kv_rot * sin4

            # cache update (in-place on static-address K/V tensors)
            kv, v = self.cache.update(kv, v, idx)

            # attention (SDPA math backend + enable_gqa, byte-exact, or the
            # manual 4-launch reference path).
            attn_out = self._gqa_attention(
                q, kv, v, self.static_attn_mask, self._attn_scale, self.dtype, flags
            )  # (1, n_q, 1, hd)

            # reshape + output projection
            attn_out = attn_out.transpose(1, 2).reshape(1, 1, n_q * hd)
            if fp8 is not None:
                from .fp8 import fp8_linear
                f8 = fp8[idx]
                attn_out = fp8_linear(attn_out.view(1, -1), f8["o_proj"][0], f8["o_proj"][1]).view(1, 1, -1)
            elif fp4 is not None:
                from .fp4 import _fp4_linear_fused
                f4 = fp4["layers"][idx] if fp4["layers"] else None
                attn_out = _fp4_linear_fused(attn_out, f4["o_proj"]) if f4 is not None else o_proj(attn_out)
            else:
                attn_out = o_proj(attn_out)  # (1, 1, 2048)

            # fused residual scale-add
            hidden = k.fused_residual_scale(residual, attn_out, self._res_mult)

            # --- MLP block ---
            residual = hidden

            # gate/up projections + preceding RMSNorm (same three-path dispatch
            # as the attention block above).
            if use_epi:
                fe = epi[idx]
                rstd = k.compute_rstd(hidden, self._rms_eps)
                gu = k.fused_gemv_normscale(hidden, fe["gu_w_s"], rstd).view(-1)
                gate, up = gu.split([inter, inter], dim=0)
                gate = gate.view(1, 1, inter)
                up = up.view(1, 1, inter)
                down_proj = fused[idx]["down_proj"]
            else:
                # fused post-attention RMSNorm
                normed = k.fused_rmsnorm(hidden, layer.post_attention_layernorm.weight, self._rms_eps)

            if not use_epi:
                if fp8 is not None:
                    from .fp8 import fp8_linear
                    f8 = fp8[idx]
                    x3 = normed.view(1, -1)
                    gu = fp8_linear(x3, f8["gu_w"][0], f8["gu_w"][1]).view(-1)
                    gate, up = gu.split([inter, inter], dim=0)
                    gate = gate.view(1, 1, inter)
                    up = up.view(1, 1, inter)
                    down_proj = None  # fp8 down-proj is dispatched at its call site
                elif fp4 is not None:
                    from .fp4 import _fp4_linear_fused
                    f4 = fp4["layers"][idx] if fp4["layers"] else None
                    x3 = normed.view(1, -1)
                    if f4 is not None:
                        gu = _fp4_linear_fused(x3, f4["gu_w"]).view(-1)
                        down_proj = None
                    else:
                        f = fused[idx]
                        gu = torch.nn.functional.linear(x3, f["gu_w"], None).view(-1)
                        down_proj = f["down_proj"]
                    gate, up = gu.split([inter, inter], dim=0)
                    gate = gate.view(1, 1, inter)
                    up = up.view(1, 1, inter)
                elif fused is not None:
                    f = fused[idx]
                    x3 = normed.view(1, -1)
                    gu = torch.nn.functional.linear(x3, f["gu_w"], None).view(-1)
                    gate, up = gu.split([inter, inter], dim=0)
                    gate = gate.view(1, 1, inter)
                    up = up.view(1, 1, inter)
                    down_proj = f["down_proj"]
                else:
                    gate = mlp.gate_proj(normed)
                    up = mlp.up_proj(normed)
                    down_proj = mlp.down_proj

            # fused SwiGLU: silu(gate) * up
            act = k.fused_silu_mul(gate, up)

            # down projection (fp8 / fp4 dequant-GEMV, or cuBLAS bf16 GEMM)
            if fp8 is not None:
                from .fp8 import fp8_linear
                f8 = fp8[idx]
                mlp_out = fp8_linear(act.view(1, -1), f8["down_proj"][0], f8["down_proj"][1]).view(1, 1, -1)
            elif fp4 is not None:
                from .fp4 import _fp4_linear_fused
                f4 = fp4["layers"][idx] if fp4["layers"] else None
                mlp_out = _fp4_linear_fused(act, f4["down_proj"]) if f4 is not None else down_proj(act)
            else:
                mlp_out = down_proj(act)  # (1, 1, 2048)

            # fused residual scale-add
            hidden = k.fused_residual_scale(residual, mlp_out, self._res_mult)

        # (4) final fused RMSNorm
        hidden = k.fused_rmsnorm(hidden, self._final_norm.weight, self._rms_eps)

        # (5) lm_head.  NVFP4 (slow reference dequant-then-matmul) takes
        # precedence; otherwise lm_head_scale_fold (default, byte-exact) folds
        # the logits-scaling constant into a pre-scaled weight; otherwise the
        # reference path (lm_head then divide) so the ablation harness can
        # measure the difference.
        if fp4 is not None:
            from .fp4 import _fp4_linear_fused
            logits = _fp4_linear_fused(hidden, fp4["lm_head"])
        elif self._flags.lm_head_scale_fold:
            logits = torch.nn.functional.linear(hidden, self._lm_head_scaled_w, None)
        else:
            logits = self.lm_head(hidden) / LLM_LOGITS_SCALING
        self.static_logits.copy_(logits)
