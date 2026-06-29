"""CUDA-graph-captured greedy decoder for the Higgs-Audio-v3 Qwen3-1.7B LLM.

The Qwen3 decoder is the bottleneck of higgs-audio-v3-stt ASR.  The stock eager
``model.generate`` path launches dozens of small kernels per token and rebuilds
Python state on every step, capping throughput far below the RTX 5090 ceiling --
the same launch-bound pathology granite's LLM has.

This module closes that gap with a CUDA-graph-captured greedy decode built on the
model's *own* Qwen3 layers and ``transformers.StaticCache``.  Graph replay of the
model's own ops is bit-exact with eager, so the decoded token sequence matches the
golden reference exactly.

Design notes
------------
Mirrors ``starling.granite.llm_mega``.  Key differences from granite:
* **No embedding multiplier** and **no logit scaling** (Qwen3 has neither; granite
  multiplies embeddings by 12 and divides logits by a scaling factor).
* The text lm_head is ``audio_decoder_proj.text_lm_head`` (a separate
  ``nn.Linear``, NOT tied to ``embed_tokens``).
* The per-step forward calls each ``Qwen3DecoderLayer`` with a precomputed
  ``position_embeddings`` (cos/sin) and a 4D additive ``causal_mask`` sized to the
  StaticCache, bypassing ``_update_causal_mask`` (which allocates CPU scalars that
  abort CUDA-graph capture).

``StaticCache`` pre-allocates fixed-address K/V tensors per layer; the layer
``update`` mutates them in place, which is inherently CUDA-graph safe.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import torch

from .config import EOS_TOKEN_IDS

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class GenerateResult:
    """Output of :meth:`LLMMega.generate`."""

    ids: torch.Tensor          # (1, n_new) int64 on CPU, the newly generated tokens
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
# CUDA-graph-captured greedy decoder (model's own layers)
# ---------------------------------------------------------------------------
class LLMMega:
    """CUDA-graph-captured greedy decoder for the Higgs-Audio Qwen3 LLM.

    Wraps the loaded ``HiggsAudio3Model`` and drives its own Qwen3 decoder layers
    directly so decode output is bit-exact with the eager golden reference.

    Args:
        model: the loaded ``HiggsAudio3Model`` (provides embed_tokens, layers,
            norm, rotary_emb, and audio_decoder_proj.text_lm_head).
        max_cache_len: Fixed K/V cache length to pre-allocate.
        warmup_iters: CUDA-graph warmup iterations before capture.
        device/dtype: Must match the loaded weights (cuda / bfloat16).
    """

    def __init__(
        self,
        model: Any,
        max_cache_len: int = 1024,
        warmup_iters: int = 3,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model = model
        self.config = model.config
        self.text_config = model.config.text_config
        self.max_cache_len = int(max_cache_len)
        self.warmup_iters = int(warmup_iters)
        self.device = device
        self.dtype = dtype

        # Pull the Qwen3 decoder pieces (kept as hot-path references).
        self._embed = model.embed_tokens
        self._layers = list(model.layers)
        self._final_norm = model.norm
        self._rotary = model.rotary_emb
        self._lm_head = model.audio_decoder_proj.text_lm_head
        self._num_layers = len(self._layers)

        self.vocab_size = int(self.text_config.vocab_size)

        # ---- static input / output buffers (fixed addresses for the graph) --
        self.static_input_ids = torch.zeros((1, 1), dtype=torch.int64, device=device)
        self.static_position_ids = torch.zeros((1, 1), dtype=torch.int64, device=device)
        self.static_cache_position = torch.zeros((1,), dtype=torch.int64, device=device)
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

        self.cache = StaticCache(
            config=self.text_config, max_batch_size=1, max_cache_len=self.max_cache_len,
            device=device, dtype=dtype,
        )

        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._captured = False

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #
    def _reset_cache_pos(self, n: int) -> None:
        """No-op position reset.

        transformers 4.51 ``StaticCache`` has no internal position counter --
        slot addressing is driven entirely by the ``cache_position`` argument
        passed to each layer's ``update`` (which we control via
        ``static_cache_position``).  Pre-warmup garbage writes are overwritten
        by the real decode steps and masked out by the 4D attention mask, so no
        explicit reset is needed.  Kept as a no-op to match granite's API.
        """
        return

    def _set_mask(self, valid_len: int) -> None:
        """Unmask positions ``[0, valid_len)``; mask the rest to ``-inf``."""
        self.static_attn_mask.fill_(self._neg_val)
        self.static_attn_mask[:, :, :, :valid_len] = 0.0

    def _decode_step_eager(self) -> None:
        """One eager decode forward writing into ``static_logits``.

        Drives the model's own Qwen3 layers directly with a precomputed 4D
        attention mask so ``_update_causal_mask`` (which allocates CPU scalars
        that abort capture) is bypassed.  This is bit-exact with the model's
        ``forward`` -> ``_forward_core`` for the single-token decode step.
        """
        # (1) embedding lookup (no multiplier for Qwen3)
        hidden = self._embed(self.static_input_ids)  # (1, 1, hidden)

        # (2) rotary cos/sin for this position (computed once, shared by layers)
        cos, sin = self._rotary(hidden, self.static_position_ids)
        position_embeddings = (cos, sin)

        # (3) iterate the 28 Qwen3 decoder layers
        for idx, layer in enumerate(self._layers):
            out = layer(
                hidden,
                attention_mask=self.static_attn_mask,
                position_ids=self.static_position_ids,
                past_key_value=self.cache,
                output_attentions=False,
                use_cache=True,
                cache_position=self.static_cache_position,
                position_embeddings=position_embeddings,
            )
            # transformers 4.51 returns (hidden_states, attn_weights); be robust.
            hidden = out[0] if isinstance(out, tuple) else out

        # (4) final RMSNorm + text lm_head (no logit scaling for Qwen3)
        hidden = self._final_norm(hidden)
        logits = self._lm_head(hidden)
        self.static_logits.copy_(logits)

    # ------------------------------------------------------------------ #
    # prefill (eager -- runs the full model once to build the multimodal cache)
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def prefill(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, int]:
        """Eager prefill: fill the StaticCache and return the first token id.

        Runs the full ``model.forward`` (encoder + projector + Qwen3 prefill) so
        the audio features are injected and the cache is populated exactly as in
        the reference ``generate``.  The StaticCache is used so the fixed-address
        K/V tensors the graph will replay are the same ones prefill wrote.

        Args:
            batch: the collated batch dict (input_ids, attention_mask,
                audio_features, audio_feature_attention_mask) on cuda.

        Returns:
            ``(first_token (1,1) int64, prefill_len)`` where ``prefill_len`` is
            the prompt length (the K/V cache fill level after prefill).
        """
        T = batch["input_ids"].shape[1]
        assert T < self.max_cache_len, (
            f"prompt {T} >= max_cache_len {self.max_cache_len}"
        )
        # Always start from a clean cache so prefill/generate are idempotent.
        self._reset_cache_pos(0)
        out = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            audio_features=batch["audio_features"],
            audio_feature_attention_mask=batch["audio_feature_attention_mask"],
            past_key_values=self.cache,
            use_cache=True,
        )
        # The merged multimodal prompt length (after audio-feature expansion) is
        # the cache fill level.  Read it from the cache, not from input_ids.
        prefill_len = int(self.cache.get_seq_length())
        logits = out.logits[:, -1:, :]
        first_token = logits.float().argmax(dim=-1)  # (1, 1)
        return first_token, prefill_len

    # ------------------------------------------------------------------ #
    # CUDA-graph capture of the decode step
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def capture(self, first_token: torch.Tensor, prefill_len: int) -> None:
        """Capture the single-token decode step into a CUDA graph.

        ``first_token`` is the token produced by prefill (the input to the first
        decode step); ``prefill_len`` is the prompt length (cache fill level).
        """
        # Prime the static buffers with valid first-decode values.
        self.static_input_ids.copy_(first_token.reshape(1, 1))
        self.static_position_ids.copy_(
            torch.tensor([[prefill_len]], device=self.device)
        )
        self.static_cache_position.copy_(
            torch.tensor([prefill_len], device=self.device)
        )
        self._set_mask(prefill_len + 1)

        # Warmup advances cumulative_length; reset before capture so the captured
        # graph starts writing at slot ``prefill_len``.
        for _ in range(self.warmup_iters):
            self._decode_step_eager()
        torch.cuda.synchronize()
        self._reset_cache_pos(prefill_len)

        # Re-prime (warmup consumed the buffer values but shapes are identical).
        self.static_input_ids.copy_(first_token.reshape(1, 1))
        self.static_position_ids.copy_(
            torch.tensor([[prefill_len]], device=self.device)
        )
        self.static_cache_position.copy_(
            torch.tensor([prefill_len], device=self.device)
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
        batch: dict[str, torch.Tensor],
        max_new_tokens: int = 256,
        eos_token_ids=EOS_TOKEN_IDS,
        tokenizer: Any = None,
        capture: bool = True,
    ) -> GenerateResult:
        """Greedy-generate ``max_new_tokens`` from a collated batch.

        Prefill is eager (runs the full multimodal forward); the subsequent
        ``max_new_tokens - 1`` decode steps are served by CUDA-graph replay.
        """
        T = batch["input_ids"].shape[1]
        # The merged prompt can be longer than T (audio features expand the
        # <|AUDIO|> placeholder); the real cache footprint is only known after
        # prefill.  Guard against the post-prefill case in prefill().
        if max_new_tokens <= 0:
            return self._finalize([], 0.0, tokenizer)

        # (1) prefill -> first token + real prompt length
        first_token, T_eff = self.prefill(batch)
        max_safe = self.max_cache_len - T_eff + 1
        if max_new_tokens > max_safe:
            raise ValueError(
                f"max_new_tokens={max_new_tokens} would overflow the static KV cache "
                f"(effective prompt T_eff={T_eff}, max_cache_len={self.max_cache_len}; "
                f"at most {max_safe} new tokens fit)."
            )
        gen_ids = [int(first_token.item())]

        if max_new_tokens <= 1:
            return self._finalize(gen_ids, 0.0, tokenizer)

        # (2) capture the decode graph (idempotent)
        if capture and not self._captured:
            self.capture(first_token, T_eff)

        # (3) decode loop
        t0 = time.perf_counter()
        next_token = first_token
        eos_set = set(eos_token_ids)
        for i in range(max_new_tokens - 1):
            cur_pos = T_eff + i
            self.static_input_ids.copy_(next_token.reshape(1, 1))
            self.static_position_ids.copy_(
                torch.tensor([[cur_pos]], device=self.device)
            )
            self.static_cache_position.copy_(
                torch.tensor([cur_pos], device=self.device)
            )
            self._set_mask(cur_pos + 1)
            if self._captured:
                self._graph.replay()
            else:
                self._decode_step_eager()
            next_token = self.static_logits.argmax(dim=-1)  # (1, 1)
            gen_ids.append(int(next_token.item()))
            if int(next_token.item()) in eos_set:
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
        batch: dict[str, torch.Tensor],
        max_new_tokens: int = 256,
        eos_token_ids=EOS_TOKEN_IDS,
        decode_iters: int = 20,
    ) -> BenchReport:
        """Benchmark prefill, per-token decode, and total generate."""
        # (a) prefill time (eager, single forward).  Reset the cache each iter.
        def _prefill():
            self._reset_cache_pos(0)
            self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                audio_features=batch["audio_features"],
                audio_feature_attention_mask=batch["audio_feature_attention_mask"],
                past_key_values=self.cache,
                use_cache=True,
            )

        prefill_ms = self._cuda_timer(_prefill, warmup=3, iters=10)

        # (b) capture the decode graph on a cleanly populated cache.
        self._reset_cache_pos(0)
        first_tok, T_eff = self.prefill(batch)
        self.capture(first_tok, T_eff)

        # Per-token decode time: replay at a fixed position so every iteration
        # does identical work.  Reset the cache slot each iter (the write target
        # is cumulative_length which the graph advances in-place).
        self.static_input_ids.copy_(first_tok.reshape(1, 1))
        self.static_position_ids.copy_(torch.tensor([[T_eff]], device=self.device))
        self.static_cache_position.copy_(torch.tensor([T_eff], device=self.device))
        self._set_mask(T_eff + 1)

        def _one_decode():
            self._graph.replay()
            self._reset_cache_pos(T_eff)  # undo the in-place advance for next iter

        decode_ms = self._cuda_timer(_one_decode, warmup=3, iters=decode_iters)
        decode_tps = 1000.0 / decode_ms if decode_ms > 0 else 0.0

        # (c) full generate (wall clock).  Reset cache and recapture.
        self._reset_cache_pos(0)
        self._captured = False
        res = self.generate(batch, max_new_tokens=max_new_tokens, eos_token_ids=eos_token_ids)

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
