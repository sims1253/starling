"""Batched (B > 1) inference pipeline for Granite-Speech-4.1-2b.

The batch=1 pipeline (:class:`starling.pipeline.MegaPipeline`) keeps the RTX 5090
~10% busy during LLM decode: each of the ~280 GEMVs per token is launch-latency
bound, not bandwidth bound, so the tensor cores sit idle.  Batching ``B``
independent audio streams turns those tiny GEMVs into real GEMMs that saturate
the tensor cores, and reads the 4.4 GB of weights *once for B tokens* instead of
once per token.  Aggregate throughput (RTFx = sum(audio_seconds) / wall_time)
therefore scales with ``B`` until the GPU saturates.

Design
------
* **Encoder + projector: per-stream (batch=1), byte-exact.**  The conformer's
  BatchNorm (running_var ~4e-10) amplifies any batch-size-dependent reduction
  difference ~316x per block, so a batched ``(B, T, 160)`` encoder forward is
  *not* byte-identical to the single forward (measured 5.2 max-abs diff in the
  encoder hidden).  Encoding each stream at batch=1 with the existing
  :class:`FusedEncoder` is byte-exact with the batch=1 path; the encoder is only
  ~12 ms/stream so even B=16 adds <200 ms, dwarfed by the decode win.
* **LLM decode: batched, CUDA-graph-captured, byte-exact per stream.**  The
  Granite decoder trunk and ``transformers.StaticCache`` support batch>1
  natively (the cache lazily allocates ``(B, n_kv, max_cache_len, head_dim)``
  and advances a single ``cumulative_length`` per layer -- perfect for
  lock-step decode).  We feed a pre-computed 4D attention mask so
  ``create_causal_mask`` early-exits (no CPU-scalar allocation that would abort
  graph capture).  Each stream's Q/K/V only ever touch its *own* KV cache rows,
  so cross-stream independence is guaranteed; verified the batched greedy output
  matches batch=1 for 80/80 tokens on identical inputs.
* **Per-stream EOS handling.**  Streams finish at different times.  A
  ``finished`` bool mask tracks them; finished streams keep feeding the pad
  token (their KV is written but never read for output) while still-active
  streams continue.  All streams stay lock-step (shared ``cumulative_length``),
  which is what makes a single CUDA graph valid for the whole batch.

Public API
----------
``BatchedPipeline(model, processor, *, max_batch_size, encoder_mode, max_cache_len)``
``BatchedPipeline.transcribe_batch(list_input_features, list_input_ids, ...) -> list[str]``
``BatchedLLMMega(language_model, lm_head, max_cache_len, max_batch_size)``
``BatchedLLMMega.generate(inputs_embeds, prompt_lengths=None, max_new_tokens, eos_token_id)``
    -> :class:`BatchedGenerateResult`
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from .config import (
    AUDIO_TOKEN_ID,
    LLM_EOS_TOKEN_ID,
    LLM_LOGITS_SCALING,
    LLM_PAD_TOKEN_ID,
)
from .encoder_mega import FusedEncoder
from .loader import get_components


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class BatchedGenerateResult:
    """Output of :meth:`BatchedLLMMega.generate`."""

    ids_list: list[torch.Tensor]
    """B CPU int64 tensors, each ``(n_b,)`` -- the per-stream generated tokens
    (truncated at that stream's EOS)."""
    n_tokens_per_stream: list[int]
    total_tokens: int
    n_streams: int
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    """Decode-loop wall time (excludes prefill)."""
    total_ms: float = 0.0
    """Prefill + decode wall time."""
    max_new_tokens: int = 0

    @property
    def decode_tok_per_s(self) -> float:
        return self.total_tokens / max(self.decode_ms / 1000.0, 1e-9)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_streams": self.n_streams,
            "total_tokens": self.total_tokens,
            "prefill_ms": round(self.prefill_ms, 3),
            "decode_ms": round(self.decode_ms, 3),
            "total_ms": round(self.total_ms, 3),
            "decode_tok_per_s": round(self.decode_tok_per_s, 1),
            "max_new_tokens": self.max_new_tokens,
        }


# =========================================================================== #
# Batched CUDA-graph-captured greedy decoder
# =========================================================================== #
class BatchedLLMMega:
    """Batched CUDA-graph-captured greedy decoder for the Granite LLM.

    Processes ``B = max_batch_size`` independent streams in lock-step.  The
    decode step is the model's own forward (``language_model(...)``) captured
    into a single CUDA graph, mirroring :class:`starling.llm_mega.LLMMega` but
    with a batch dimension.  Output is byte-exact per stream vs the batch=1
    decoder (verified 80/80 tokens on identical inputs).

    Args:
        language_model: The ``GraniteModel`` decoder trunk.
        lm_head: ``nn.Linear`` lm_head from the top-level speech model.
        max_cache_len: Fixed K/V cache length.
        max_batch_size: Number of streams (``B``).  Static buffers, the cache,
            and the captured graph are all sized for exactly this ``B``.
        warmup_iters: CUDA-graph warmup iterations before capture.
        device/dtype: Must match the loaded weights (cuda / bfloat16).
    """

    def __init__(
        self,
        language_model: Any,
        lm_head: Any,
        max_cache_len: int = 640,
        max_batch_size: int = 8,
        warmup_iters: int = 3,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.lm = language_model
        self.lm_head = lm_head
        self.config = language_model.config
        self.max_cache_len = int(max_cache_len)
        self.max_batch_size = int(max_batch_size)
        self.warmup_iters = int(warmup_iters)
        self.device = device
        self.dtype = dtype

        self.vocab_size = int(self.config.vocab_size)
        B = self.max_batch_size

        # ---- static input / output buffers (fixed addresses for the graph) --
        self.static_input_ids = torch.zeros((B, 1), dtype=torch.int64, device=device)
        self.static_position_ids = torch.zeros((B, 1), dtype=torch.int64, device=device)
        self.static_logits = torch.zeros(
            (B, 1, self.vocab_size), dtype=dtype, device=device
        )
        neg = torch.finfo(dtype).min
        self._neg_val = neg
        # Batched 4D attention mask: (B, 1, 1, max_cache_len).  Broadcasts over
        # the query-head axis when added to scores (B, n_q, 1, max_cache_len).
        # For the no-padding case every row is identical (== the batch=1 mask).
        self.static_attn_mask = torch.full(
            (B, 1, 1, self.max_cache_len), neg, dtype=dtype, device=device
        )

        # StaticCache allocates lazily on first prefill (infers batch B from the
        # first key_states it sees).  Build it now so its fixed-address tensors
        # exist before any graph capture.
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

    def _fill_shared_mask(self, valid_len: int) -> None:
        """Shared causal mask: positions ``[0, valid_len)`` valid for all streams."""
        self.static_attn_mask.fill_(self._neg_val)
        self.static_attn_mask[:, :, :, :valid_len] = 0.0

    def _fill_batched_mask(
        self, prompt_lengths: torch.Tensor, cur_pos: int, pad_len: int
    ) -> None:
        """Per-stream mask for right-padded prompts.

        Stream ``b`` at absolute decode position ``cur_pos`` may attend to:
          * real prompt positions ``[0, prompt_lengths[b])``;
          * real decode positions ``[pad_len, cur_pos]``.
        The right-padding hole ``[prompt_lengths[b], pad_len)`` (pad KV) is
        masked to ``-inf`` so it never leaks into a real stream's attention.
        For the no-padding case (``prompt_lengths[b] == pad_len`` for all b)
        this reduces to ``valid = [0, cur_pos]`` for every stream.
        """
        M = self.max_cache_len
        pos = torch.arange(M, device=self.device)  # (M,)
        prompt_valid = pos.unsqueeze(0) < prompt_lengths.unsqueeze(1)  # (B, M)
        decode_valid = (pos >= pad_len) & (pos <= cur_pos)  # (M,)
        valid = prompt_valid | decode_valid.unsqueeze(0)  # (B, M)
        m = torch.zeros((self.max_batch_size, M), dtype=self.dtype, device=self.device)
        m.masked_fill_(~valid, self._neg_val)
        self.static_attn_mask.copy_(m.view(self.max_batch_size, 1, 1, M))

    def _decode_step_eager(self) -> None:
        """One eager decode forward writing into ``static_logits``.

        Uses the model's own layers with the pre-computed 4D attention mask so
        ``create_causal_mask`` early-exits (no CPU-scalar allocation).  Identical
        to :meth:`LLMMega._decode_step_eager` but the batch dim is ``B``.
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
    def prefill(
        self,
        inputs_embeds: torch.Tensor,
        prompt_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Eager batched prefill: fill the StaticCache, return first tokens.

        Args:
            inputs_embeds: ``(B, T, hidden)`` bf16 (merged multimodal embeds,
                *before* the Granite embedding multiplier; the model applies it
                internally, matching the batch=1 path).
            prompt_lengths: optional ``(B,)`` real prompt lengths when the
                ``inputs_embeds`` were right-padded to a common ``T``.  ``None``
                means no padding (every stream is exactly ``T`` long).

        Returns:
            ``(B, 1)`` int64 tensor with each stream's first generated token.
        """
        B, T = inputs_embeds.shape[:2]
        assert T < self.max_cache_len, (
            f"prompt {T} >= max_cache_len {self.max_cache_len}"
        )
        self._reset_cache_pos(0)
        position_ids = torch.arange(T, device=self.device).unsqueeze(0).expand(B, T)
        if prompt_lengths is None:
            attn_mask = None
        else:
            # 2D (B, T) padding mask: 1 for real prompt, 0 for right-pad.  The
            # model folds it into its causal mask so pad positions never attend
            # and are never attended-to during prefill.
            attn_mask = (
                torch.arange(T, device=self.device).unsqueeze(0)
                < prompt_lengths.to(self.device).long().unsqueeze(1)
            )
        out = self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_values=self.cache,
            use_cache=True,
        )
        hidden = out.last_hidden_state  # (B, T, H)
        if prompt_lengths is None:
            # No padding: last position is real for every stream.
            last_hidden = hidden[:, -1:, :]
        else:
            # Right-padded: gather each stream's LAST REAL prompt position
            # (position prompt_lengths[b] - 1), not the trailing pad slot.
            last_idx = (prompt_lengths.long() - 1).view(B, 1, 1).expand(
                B, 1, hidden.shape[-1]
            )
            last_hidden = hidden.gather(1, last_idx)  # (B, 1, H)
        logits = self.lm_head(last_hidden) / LLM_LOGITS_SCALING
        return logits.argmax(dim=-1)  # (B, 1)

    # ------------------------------------------------------------------ #
    # CUDA-graph capture of the decode step
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def capture(
        self,
        first_tokens: torch.Tensor,
        prefill_len: int,
        prompt_lengths: Optional[torch.Tensor] = None,
    ) -> None:
        """Capture the single-token batched decode step into a CUDA graph.

        Must be called once after :meth:`prefill`.  ``first_tokens`` is the
        ``(B, 1)`` token tensor produced by prefill (the input to the first
        decode step); ``prefill_len`` is the padded prompt length ``T``.
        """
        # Finished streams (first token == EOS) feed the pad token so their KV
        # never carries a real token into the captured graph's warmup state.
        finished0 = first_tokens.view(-1) == LLM_EOS_TOKEN_ID
        primed = torch.where(
            finished0.view(-1, 1),
            torch.full_like(first_tokens, LLM_PAD_TOKEN_ID),
            first_tokens,
        )

        def _prime(pos: int) -> None:
            self.static_input_ids.copy_(primed)
            self.static_position_ids.fill_(pos)
            if prompt_lengths is None:
                self._fill_shared_mask(pos + 1)
            else:
                self._fill_batched_mask(prompt_lengths, pos, prefill_len)

        # Warmup advances cumulative_length; reset before capture so the
        # captured graph starts writing at slot ``prefill_len``.
        _prime(prefill_len)
        for _ in range(self.warmup_iters):
            self._decode_step_eager()
        torch.cuda.synchronize()
        self._reset_cache_pos(prefill_len)

        _prime(prefill_len)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._decode_step_eager()

        self._reset_cache_pos(prefill_len)
        self._captured = True

    # ------------------------------------------------------------------ #
    # generate
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def generate(
        self,
        inputs_embeds: torch.Tensor,
        prompt_lengths: Optional[torch.Tensor] = None,
        max_new_tokens: int = 200,
        eos_token_id: int = LLM_EOS_TOKEN_ID,
    ) -> BatchedGenerateResult:
        """Greedy-generate up to ``max_new_tokens`` for ``B`` streams at once.

        Prefill is eager; the subsequent decode steps are served by CUDA-graph
        replay (after :meth:`capture`).  Streams emit EOS independently; once a
        stream finishes it feeds the pad token and its (ignored) logits stop
        being collected.
        """
        B, T = inputs_embeds.shape[:2]
        if B != self.max_batch_size:
            raise ValueError(
                f"inputs_embeds batch {B} != max_batch_size {self.max_batch_size}; "
                f"construct BatchedLLMMega(max_batch_size={B}) for this batch."
            )
        max_safe = self.max_cache_len - T + 1
        if max_new_tokens > max_safe:
            raise ValueError(
                f"max_new_tokens={max_new_tokens} would overflow the static KV cache "
                f"(prompt T={T}, max_cache_len={self.max_cache_len}; at most "
                f"{max_safe} new tokens fit)."
            )

        device = self.device
        pad_full = torch.full(
            (B, 1), LLM_PAD_TOKEN_ID, dtype=torch.int64, device=device
        )

        # Resolve prompt_lengths (None == no right-padding).
        no_pad = prompt_lengths is None
        if no_pad:
            prompt_lengths_t = torch.full(
                (B,), T, dtype=torch.long, device=device
            )
        else:
            prompt_lengths_t = prompt_lengths.to(device).long()

        # Per-stream RoPE position offset for the padded case.  A padded stream
        # (real prompt T_b < T) must decode at RoPE position T_b, T_b+1, ... even
        # though its KV is written to the *shared* cache slots [T, T+1, ...].
        # RoPE makes attention depend on the RELATIVE (q_pos - k_pos), so feeding
        # position_id = cur_pos - (T - T_b) makes stream b's relative positions
        # identical to batch=1, yielding a byte-exact greedy match.  Zero for the
        # no-padding case (== constant cur_pos for every stream).
        pad_offset = (T - prompt_lengths_t).to(device)  # (B,)

        # (1) prefill -> first tokens (timed separately from decode).
        t_pf0 = time.perf_counter()
        next_token = self.prefill(inputs_embeds, None if no_pad else prompt_lengths_t)
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t_pf0) * 1000.0

        finished = torch.zeros(B, dtype=torch.bool, device=device)
        gen: list[list[int]] = [[] for _ in range(B)]
        first_list = next_token.view(-1).tolist()
        for b in range(B):
            gen[b].append(first_list[b])
            if first_list[b] == eos_token_id:
                finished[b] = True

        if max_new_tokens <= 1 or bool(finished.all()):
            return self._finalize(gen, prefill_ms, 0.0, max_new_tokens)

        # (2) capture the decode graph (idempotent).
        if not self._captured:
            self.capture(next_token, T, None if no_pad else prompt_lengths_t)

        # Finished streams feed pad from the very first decode step.
        next_token = torch.where(finished.view(B, 1), pad_full, next_token)

        # (3) decode loop.
        t_dec0 = time.perf_counter()
        for i in range(max_new_tokens - 1):
            cur_pos = T + i
            self.static_input_ids.copy_(next_token.view(B, 1))
            if no_pad:
                self.static_position_ids.fill_(cur_pos)
                self._fill_shared_mask(cur_pos + 1)
            else:
                # Per-stream RoPE positions: stream b decodes at the cache slot
                # ``cur_pos`` but uses RoPE position ``cur_pos - pad_offset[b]``.
                self.static_position_ids.copy_(
                    (cur_pos - pad_offset).view(B, 1)
                )
                self._fill_batched_mask(prompt_lengths_t, cur_pos, T)
            if self._captured:
                self._graph.replay()
            else:
                self._decode_step_eager()
            new_tok = self.static_logits[:, -1:, :].argmax(dim=-1)  # (B, 1)
            new_list = new_tok.view(-1).tolist()
            for b in range(B):
                if not finished[b]:
                    gen[b].append(new_list[b])
                    if new_list[b] == eos_token_id:
                        finished[b] = True
            next_token = torch.where(finished.view(B, 1), pad_full, new_tok)
            if bool(finished.all()):
                break
        torch.cuda.synchronize()
        decode_ms = (time.perf_counter() - t_dec0) * 1000.0
        return self._finalize(gen, prefill_ms, decode_ms, max_new_tokens)

    def _finalize(
        self,
        gen: list[list[int]],
        prefill_ms: float,
        decode_ms: float,
        max_new_tokens: int,
    ) -> BatchedGenerateResult:
        ids_list = [torch.tensor(g, dtype=torch.int64) for g in gen]
        n_per = [len(g) for g in gen]
        return BatchedGenerateResult(
            ids_list=ids_list,
            n_tokens_per_stream=n_per,
            total_tokens=sum(n_per),
            n_streams=len(gen),
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            total_ms=prefill_ms + decode_ms,
            max_new_tokens=max_new_tokens,
        )

    def reset(self) -> None:
        """Drop the captured graph + cache so a new shape can be captured."""
        self._graph = None
        self._captured = False
        from transformers.cache_utils import StaticCache

        self.cache = StaticCache(config=self.config, max_cache_len=self.max_cache_len)


# =========================================================================== #
# Fused batched decoder (Triton elementwise kernels, batch-agnostic)
# =========================================================================== #
class BatchedFusedLLMMega(BatchedLLMMega):
    """Batched decoder that swaps the model's own forward for a manual
    per-layer loop using the fused Triton elementwise kernels
    (:mod:`starling.llm_kernels`).

    Inherits all graph-capture / generate / prefill machinery from
    :class:`BatchedLLMMega` and overrides only :meth:`_decode_step_eager` with a
    custom forward that mirrors :class:`starling.llm_mega.FusedLLMMega` but with a
    batch dimension ``B``.  The fused kernels (RMSNorm, SwiGLU, residual
    scale-add) reshape to ``(M, N)`` where ``M = B`` for single-token decode, so
    they are batch-agnostic; GEMMs (q/k/v/o, gate/up/down, lm_head) stay as
    cuBLAS bf16 matmuls that now become real GEMMs at B > 1.

    Correctness is byte-exact with :class:`BatchedLLMMega` (and therefore with
    batch=1) because the fused kernels match the model's own bf16 arithmetic
    bit-for-bit (verified 0.0 diff in :mod:`starling.llm_kernels`).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        from . import llm_kernels as _k  # local import to avoid circular dep
        from .llm_mega import _EMB_MULT, _repeat_kv  # noqa: F401 (re-exported)

        self._k = _k
        self._emb_mult = _EMB_MULT
        self._repeat_kv = staticmethod(_repeat_kv)
        # Pre-extract per-layer references for speed in the hot decode loop.
        self._layers = list(self.lm.layers)
        self._embed = self.lm.embed_tokens
        self._final_norm = self.lm.norm
        self._rotary = self.lm.rotary_emb
        cfg = self.config
        self._n_q_heads = int(cfg.num_attention_heads)
        self._n_kv_heads = int(cfg.num_key_value_heads)
        self._head_dim = int(
            getattr(cfg, "head_dim", cfg.hidden_size // self._n_q_heads)
        )
        self._n_kv_groups = self._n_q_heads // self._n_kv_heads
        self._attn_scale = float(cfg.attention_multiplier)
        self._res_mult = float(cfg.residual_multiplier)
        self._rms_eps = float(cfg.rms_norm_eps)

    def _decode_step_eager(self) -> None:
        """Custom batched single-token decode forward with fused Triton kernels.

        Mirrors :meth:`FusedLLMMega._decode_step_eager` exactly but with
        ``B = max_batch_size``.  Writes the final logits (post lm_head /
        logits_scaling) into ``self.static_logits`` (B, 1, vocab).
        """
        k = self._k
        B = self.max_batch_size
        hd = self._head_dim
        n_q = self._n_q_heads
        n_kv = self._n_kv_heads
        half = hd // 2

        # (1) embedding lookup + Granite embedding multiplier.
        hidden = self._embed(self.static_input_ids) * self._emb_mult  # (B, 1, 2048)

        # (2) rotary cos/sin for this position (per-stream position_ids).
        cos, sin = self._rotary(hidden, position_ids=self.static_position_ids)
        cos4 = cos.unsqueeze(1)  # (B, 1, 1, hd)
        sin4 = sin.unsqueeze(1)

        # (3) iterate layers.
        for idx, layer in enumerate(self._layers):
            sa = layer.self_attn
            mlp = layer.mlp

            # --- attention block ---
            residual = hidden  # (B, 1, 2048)
            normed = k.fused_rmsnorm(
                hidden, layer.input_layernorm.weight, self._rms_eps
            )

            q = sa.q_proj(normed).view(B, 1, n_q, hd).transpose(1, 2)    # (B, n_q, 1, hd)
            kv = sa.k_proj(normed).view(B, 1, n_kv, hd).transpose(1, 2)  # (B, n_kv, 1, hd)
            v = sa.v_proj(normed).view(B, 1, n_kv, hd).transpose(1, 2)

            # RoPE (PyTorch -- matches the reference bf16 arithmetic exactly;
            # the Triton rope kernel is batch=1 only).
            q_rot = torch.cat((-q[..., half:], q[..., :half]), dim=-1)
            kv_rot = torch.cat((-kv[..., half:], kv[..., :half]), dim=-1)
            q = q * cos4 + q_rot * sin4
            kv = kv * cos4 + kv_rot * sin4

            # cache update (in-place on static-address K/V tensors, batch-B).
            kv, v = self.cache.update(kv, v, idx)
            kv_r = self._repeat_kv(kv, self._n_kv_groups)  # (B, n_q, max_len, hd)
            v_r = self._repeat_kv(v, self._n_kv_groups)

            # attention: scores = Q @ K^T * scale + mask, softmax, @ V
            scores = torch.matmul(q, kv_r.transpose(2, 3)) * self._attn_scale
            scores = scores + self.static_attn_mask  # (B,1,1,max_len) broadcast
            attn = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(
                self.dtype
            )
            attn_out = torch.matmul(attn, v_r)  # (B, n_q, 1, hd)

            attn_out = attn_out.transpose(1, 2).reshape(B, 1, n_q * hd)
            attn_out = sa.o_proj(attn_out)
            hidden = k.fused_residual_scale(residual, attn_out, self._res_mult)

            # --- MLP block ---
            residual = hidden
            normed = k.fused_rmsnorm(
                hidden, layer.post_attention_layernorm.weight, self._rms_eps
            )
            gate = mlp.gate_proj(normed)  # (B, 1, 4096)
            up = mlp.up_proj(normed)
            act = k.fused_silu_mul(gate, up)
            mlp_out = mlp.down_proj(act)
            hidden = k.fused_residual_scale(residual, mlp_out, self._res_mult)

        # (4) final fused RMSNorm + lm_head + logits scaling.
        hidden = k.fused_rmsnorm(hidden, self._final_norm.weight, self._rms_eps)
        logits = self.lm_head(hidden) / LLM_LOGITS_SCALING
        self.static_logits.copy_(logits)


# =========================================================================== #
# Batched end-to-end pipeline
# =========================================================================== #
class BatchedPipeline:
    """Batched ASR pipeline: B independent audio streams -> B transcripts.

    The encoder + projector run **per stream** (batch=1) for byte-exactness with
    the batch=1 path (the conformer's BatchNorm makes a batched encoder forward
    non-byte-exact).  Only the LLM decode is batched -- that is where the
    launch-bound GEMVs become saturating GEMMs.

    Parameters
    ----------
    model : GraniteSpeechForConditionalGeneration
        Fully loaded top-level speech model (lm_head lives on it).
    processor : GraniteSpeech processor
    max_batch_size : int
        Number of streams ``B`` the decoder is sized for.  ``transcribe_batch``
        must receive exactly this many streams (extra streams can be padded with
        copies of the first; their transcripts are discarded).
    encoder_mode : str
        Forwarded to :class:`FusedEncoder`.  ``"cudagraph"`` (default) is the
        byte-exact, zero-launch-overhead encoder.
    max_cache_len : int
        LLM static KV cache length.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        max_batch_size: int = 8,
        encoder_mode: str = "cudagraph",
        max_cache_len: int = 640,
        use_fused_llm: bool = True,
        flags: Any = None,
    ) -> None:
        from .flags import OptFlags, get_default_flags

        if flags is None:
            flags = get_default_flags()
        elif isinstance(flags, dict):
            flags = OptFlags(**flags)
        self.flags = flags

        self.model = model
        self.processor = processor
        self.dtype = getattr(model, "dtype", torch.bfloat16)
        self.max_batch_size = int(max_batch_size)

        comps = get_components(model)
        # (1) fused encoder (per-stream, batch=1, byte-exact).
        self.fused_encoder = FusedEncoder(comps["encoder"], mode=encoder_mode)
        # Raw encoder for the batched-encoder fast path (tolerance mode only --
        # a batched conformer forward is NOT byte-exact due to BatchNorm).
        self._raw_encoder = comps["encoder"]
        # (2) stock eager BLIP2 projector (per-stream).
        self.projector = comps["projector"]
        # embed_tokens used by the merge step.
        self.embed_tokens = comps["language_model"].get_input_embeddings()

        # (3) batched LLM decoder trunk + lm_head from the top-level model.
        # BatchedFusedLLMMega (fused Triton elementwise kernels) is the default
        # for maximum throughput; BatchedLLMMega (model's own forward) is the
        # simpler fallback.  Both are byte-exact per stream vs batch=1.
        # ``quantized_weights`` (tolerance mode) selects the weight-only INT8
        # decoder (:class:`starling.quant.BatchedQuantLLMMega`).
        if flags.quantized_weights:
            from .quant import BatchedQuantLLMMega

            self.llm = BatchedQuantLLMMega(
                comps["language_model"],
                model.lm_head,
                max_cache_len=max_cache_len,
                max_batch_size=self.max_batch_size,
            )
        else:
            llm_cls = BatchedFusedLLMMega if use_fused_llm else BatchedLLMMega
            self.llm = llm_cls(
                comps["language_model"],
                model.lm_head,
                max_cache_len=max_cache_len,
                max_batch_size=self.max_batch_size,
            )
        self.use_fused_llm = use_fused_llm

    # ------------------------------------------------------------------ #
    # merge step (byte-exact replica of get_merged_audio_embeddings)
    # ------------------------------------------------------------------ #
    def build_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        audio_embeds: torch.Tensor,
        input_features_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Merge projected audio embeds into the LLM token embeddings (per stream).

        Identical logic to :meth:`MegaPipeline.build_inputs_embeds`.
        """
        is_audio_index = input_ids == AUDIO_TOKEN_ID
        llm_input_ids = torch.where(is_audio_index, 0, input_ids)
        inputs_embeds = self.embed_tokens(llm_input_ids)

        af = audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        if input_features_mask is not None:
            af = af[input_features_mask]

        special_audio_mask = is_audio_index.unsqueeze(-1).expand_as(inputs_embeds)
        return inputs_embeds.masked_scatter(special_audio_mask, af)

    # ------------------------------------------------------------------ #
    # per-stream audio encoding (byte-exact with the batch=1 path)
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def encode_stream(
        self,
        input_features: torch.Tensor,
        input_ids: torch.Tensor,
        input_features_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode one stream: fused encoder -> projector -> merge -> inputs_embeds.

        Returns ``(1, T, 2048)`` bf16 -- byte-identical to what the batch=1
        :class:`MegaPipeline` produces for the same input.
        """
        feats = input_features
        if feats.dtype != self.dtype:
            feats = feats.to(self.dtype)
        enc_hidden = self.fused_encoder(feats)
        audio_embeds = self.projector(enc_hidden)
        return self.build_inputs_embeds(input_ids, audio_embeds, input_features_mask)

    @torch.inference_mode()
    def _encode_batched_encoder(
        self,
        feats_list: list[torch.Tensor],
        ids_list: list[torch.Tensor],
        mask_list: list[Optional[torch.Tensor]],
    ) -> list[torch.Tensor]:
        """Batched-encoder fast path (tolerance mode only, NOT byte-exact).

        Runs all B streams through the conformer encoder in ONE batched forward
        (``(B, T, 160)``) instead of B per-stream forwards.  The projector +
        merge still run per-stream (they are byte-exact at any batch).  This
        trades ~5.2 max-abs encoder-hidden divergence (BatchNorm
        ``running_var ~4e-10`` amplifies batch-dependent reduction diffs
        ~316x/block) for a roughly B x encoder-launch reduction.

        Returns a list of B ``(1, T, 2048)`` merged ``inputs_embeds``.
        """
        B = len(feats_list)
        # Stack features into (B, T, 160). All streams must share the same T
        # for a batched encoder forward (the caller pads to max_batch_size with
        # copies, so this holds in the common uniform-length case).
        feats = torch.cat(
            [f.to(self.dtype) if f.dtype != self.dtype else f for f in feats_list],
            dim=0,
        )  # (B, T, 160)
        enc_out = self._raw_encoder(feats, return_dict=True)  # (B, T, 1024)
        enc_hidden = enc_out.last_hidden_state
        ies = []
        for b in range(B):
            eh = enc_hidden[b : b + 1]  # (1, T, 1024)
            audio_embeds = self.projector(eh)
            ies.append(
                self.build_inputs_embeds(
                    ids_list[b], audio_embeds, mask_list[b]
                )
            )
        return ies

    # ------------------------------------------------------------------ #
    # full batched transcribe
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def transcribe_batch(
        self,
        list_of_input_features: list[torch.Tensor],
        list_of_input_ids: list[torch.Tensor],
        list_of_input_features_mask: Optional[list[Optional[torch.Tensor]]] = None,
        max_new_tokens: int = 200,
    ) -> list[str]:
        """Process B audio streams in parallel. Returns B transcripts.

        Each stream is encoded independently (byte-exact), then the B merged
        ``inputs_embeds`` are right-padded to a common prompt length (if needed)
        and decoded together by :class:`BatchedLLMMega`.
        """
        res = self.run_batch(
            list_of_input_features,
            list_of_input_ids,
            list_of_input_features_mask,
            max_new_tokens=max_new_tokens,
        )
        tok = self.processor.tokenizer
        return [
            tok.decode(ids, skip_special_tokens=True) for ids in res.ids_list
        ]

    @torch.inference_mode()
    def run_batch(
        self,
        list_of_input_features: list[torch.Tensor],
        list_of_input_ids: list[torch.Tensor],
        list_of_input_features_mask: Optional[list[Optional[torch.Tensor]]] = None,
        max_new_tokens: int = 200,
    ) -> BatchedGenerateResult:
        """Like :meth:`transcribe_batch` but returns the raw :class:`BatchedGenerateResult`."""
        n = len(list_of_input_features)
        if list_of_input_features_mask is None:
            list_of_input_features_mask = [None] * n
        if n > self.max_batch_size:
            raise ValueError(
                f"got {n} streams but pipeline is sized for max_batch_size="
                f"{self.max_batch_size}"
            )

        # Pad the batch up to max_batch_size with copies of stream 0 so the
        # decoder always sees exactly B streams (its static buffers / cache /
        # graph are sized for B).  Padded transcripts are discarded.
        padded = n < self.max_batch_size
        feats_list = list(list_of_input_features)
        ids_list = list(list_of_input_ids)
        mask_list = list(list_of_input_features_mask)
        if padded:
            for _ in range(self.max_batch_size - n):
                feats_list.append(list_of_input_features[0])
                ids_list.append(list_of_input_ids[0])
                mask_list.append(list_of_input_features_mask[0])

        # (1) encode.  Per-stream (byte-exact) by default; batched-encoder fast
        #     path (tolerance mode) when the flag is set.
        if getattr(self.flags, "batched_encoder", False):
            ies = self._encode_batched_encoder(
                feats_list, ids_list, mask_list
            )
        else:
            ies = [
                self.encode_stream(f, i, m)
                for f, i, m in zip(feats_list, ids_list, mask_list)
            ]
        Ts = [int(ie.shape[1]) for ie in ies]
        T_max = max(Ts)

        # (2) right-pad each stream's inputs_embeds to T_max (zero-pad; pad
        #     positions are masked during prefill + decode).  When all streams
        #     share the same prompt length (the common case) no padding happens.
        no_pad = all(T == T_max for T in Ts)
        if no_pad:
            inputs_embeds = torch.cat(ies, dim=0)  # (B, T_max, 2048)
            prompt_lengths = None
        else:
            padded_ies = []
            for ie, T in zip(ies, Ts):
                if T < T_max:
                    ie = torch.nn.functional.pad(
                        ie, (0, 0, 0, T_max - T)
                    )  # right-pad seq dim with zeros
                padded_ies.append(ie)
            inputs_embeds = torch.cat(padded_ies, dim=0)  # (B, T_max, 2048)
            prompt_lengths = torch.tensor(Ts, dtype=torch.long, device="cuda")

        # (3) batched greedy decode.
        max_cache_len = int(getattr(self.llm, "max_cache_len", 640))
        budget = max(1, min(max_new_tokens, max_cache_len - T_max - 1))
        res = self.llm.generate(
            inputs_embeds,
            prompt_lengths=prompt_lengths,
            max_new_tokens=budget,
            eos_token_id=LLM_EOS_TOKEN_ID,
        )

        # (4) drop the padded-stream transcripts.
        if padded:
            res.ids_list = res.ids_list[:n]
            res.n_tokens_per_stream = res.n_tokens_per_stream[:n]
            res.total_tokens = sum(res.n_tokens_per_stream)
            res.n_streams = n
        return res


def main() -> int:
    """Quick CLI smoke: batched-transcribe 4 copies of the sample audio."""
    import time

    from .audio import build_inputs, load_sample_audio
    from .golden import load_golden, load_golden_text
    from .loader import load_model_and_processor

    B = 4
    print(f"[batched] loading model + building BatchedPipeline(B={B}) ...")
    t0 = time.perf_counter()
    model, proc = load_model_and_processor(attn_impl="eager")
    pipe = BatchedPipeline(model, proc, max_batch_size=B, encoder_mode="cudagraph")
    print(f"[batched] built in {time.perf_counter() - t0:.1f}s")

    wav, sr = load_sample_audio()
    inputs = build_inputs(proc, wav)
    audio_seconds = wav.shape[1] / sr
    feats = [inputs["input_features"]] * B
    ids = [inputs["input_ids"]] * B
    masks = [inputs.get("input_features_mask")] * B

    t0 = time.perf_counter()
    texts = pipe.transcribe_batch(feats, ids, masks, max_new_tokens=100)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0

    print(f"\n[batched] B={B} audio streams in {ms:.1f} ms "
          f"(aggregate RTFx = {B * audio_seconds / (ms / 1000.0):.2f}x)")
    for i, t in enumerate(texts):
        print(f"[batched] stream {i}: {t[:120]!r}")

    # correctness vs golden
    golden_text = load_golden_text().strip()
    golden_resp = golden_text.split("ASSISTANT:", 1)[1].strip()
    golden_gen = load_golden("greedy_ids.pt")[0, 271:]
    print(f"\n[batched] golden response (first 120): {golden_resp[:120]!r}")
    print(f"[batched] all streams match golden text: "
          f"{all(t.strip() == golden_resp for t in texts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
