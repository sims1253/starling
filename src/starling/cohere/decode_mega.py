"""CUDA-graph-captured greedy decode for CohereLabs/cohere-transcribe-03-2026.

This is the seq2seq (Whisper-style) counterpart of ``starling.parakeet.decode_mega``
/ ``starling.granite.llm_mega``. The per-token autoregressive decode loop is the
launch-bound bottleneck (~10% GPU-busy, hundreds of tiny per-layer kernels), so
the whole step is captured into a single ``torch.cuda.CUDAGraph`` and served by
``graph.replay()``.

cohere-transcribe is the repo's first **encoder-decoder** model, so the decode
step has TWO attention blocks per layer (vs the encoder+LLM models' one):
  * **self-attention** (causal) over the decoder's own growing KV cache
  * **cross-attention** (bidirectional) over the frozen encoder output

Both KV caches live in an ``EncoderDecoderCache``. We use ``StaticCache`` for
both halves so the K/V tensors have FIXED shape (``(B, n_heads, max_cache_len,
head_dim)`` for self, ``(B, n_heads, S, head_dim)`` for cross) — fixed shapes
are a hard requirement for CUDA-graph capture (a ``DynamicCache``'s growing K
tensor breaks the captured graph's shape contract).

The capture-safe mask trick (same family as granite/moss): the stock
``create_causal_mask`` / ``create_bidirectional_mask`` do CPU-scalar shape
branching that aborts capture. Passing a **ready 4D additive mask** makes both
early-exit (``_preprocess_mask_arguments`` returns a 4D mask as-is), so the whole
decoder forward is capture-safe. We build a per-step 4D causal mask of shape
``(B,1,1,max_cache_len)`` with key ``j`` valid iff ``j <= query_position``.

Multi-step capture (K steps per replay)
---------------------------------------
Mirroring ``starling.parakeet.decode_mega``: instead of capturing ONE decode
step and replaying it N times (N host syncs), the graph captures K consecutive
decode steps into one ``torch.cuda.CUDAGraph``. This is sound because every
step's state lives in static buffers mutated in place:

* ``self_token`` (B,1)       -- decoder input token (chained in-graph: argmax feeds next step)
* ``output_ring`` (B,K)      -- step ``j`` writes its emitted token into column ``j``
* the ``EncoderDecoderCache`` -- the self-attn ``StaticCache`` K/V tensors are
  written in place at advancing positions; cross-attn cache is frozen after prefill.

``last_token`` is chained IN GRAPH (the argmax of step ``j`` is copied into the
``self_token`` buffer that step ``j+1`` reads), so no host sync is needed between
captured steps. Each step's per-step causal mask is a separate static buffer
``mask_j`` (fixed at capture time, since the query position ``T+j`` is known at
capture). The host loop replays the K-step graph ``ceil(max_out / K)`` times and
syncs once per replay.

Prefill is eager (the prefill builds the prompt KV + fills the cross-attn cache
once); steps 1+ run graphed. This mirrors the parakeet "prefill eager, decode
graphed" pattern. Each real decode re-runs the eager prefill into the captured
cache and rewinds the write heads to ``T`` (no cached state is carried over from
the capture utterance), so one captured graph serves many utterances of the same
``(B, prompt_len, S)`` shape.
"""

from __future__ import annotations

import torch

try:
    from torch._dynamo import mark_static_address as _mark_static
except Exception:  # pragma: no cover - older torch
    def _mark_static(t):  # type: ignore[misc]
        return t


def _mark_many(tensors) -> None:
    for t in tensors:
        try:
            _mark_static(t)
        except Exception:
            pass


class GraphedDecoder:
    """Capture the per-step seq2seq decode into a CUDA graph; decode many inputs.

    The graph is fixed to a ``prompt_len`` (the decoder prompt length, which
    determines the captured per-step positions). The encoder length ``S`` is NOT
    fixed at capture — the cross-attn K/V are recomputed in the (eager) prefill
    for each new utterance and written into the cross-attn StaticCache, so the
    captured decode steps read whatever cross K/V the prefill left there. This
    means one captured graph serves utterances of any encoder length, as long as
    the *prompt* length matches (it is a constant 10 for this model's chat
    format).

    Args:
        model: a loaded ``CohereAsrForConditionalGeneration`` on cuda (eval, bf16).
        max_cache_len: max decoder positions (prompt + max_new_tokens).
        steps_per_replay: K — number of consecutive decode steps captured into
            one graph replay (default ``8``). ``1`` reproduces one-step-per-replay.
        warmup_iters: side-stream warmup iterations before graph capture.
    """

    def __init__(
        self,
        model,
        *,
        max_cache_len: int = 1024,
        steps_per_replay: int = 8,
        warmup_iters: int = 4,
    ) -> None:
        self.model = model
        self.decoder = model.model.decoder
        self.proj_out = model.proj_out
        self.cfg = model.config
        self.max_cache_len = int(max_cache_len)
        self.steps_per_replay = max(1, int(steps_per_replay))
        self.warmup_iters = int(warmup_iters)
        self.eos_token_id = int(self.cfg.eos_token_id)
        self.pad_token_id = int(self.cfg.pad_token_id)
        self.n_layers = int(self.cfg.num_hidden_layers)
        self._captured = False

    # ------------------------------------------------------------------ #
    # prefill (eager) -- fills the cross-attn cache once, returns first token
    # ------------------------------------------------------------------ #
    def _make_cache(self, S):
        from transformers.cache_utils import EncoderDecoderCache, StaticCache

        return EncoderDecoderCache(
            StaticCache(config=self.cfg, max_cache_len=self.max_cache_len),
            StaticCache(config=self.cfg, max_cache_len=S),
        )

    def _prefill(self, dec_in, enc_h, enc_mask, cache):
        """Run the eager prefill into ``cache``; returns first token (B,1).

        CRITICAL for multi-utterance correctness: the decoder's cross-attention
        recomputes K/V from ``encoder_hidden_states`` ONLY when
        ``cache.is_updated[layer]`` is False; otherwise it reuses stale cached K/V
        (see CohereAsrCrossAttention.forward). The captured graph plus
        ``_reset_cache`` leave ``is_updated`` True, so without resetting it here
        the second-and-later utterances' prefill would silently reuse the
        *capture-time* cross K/V and decode the wrong utterance (every clip after
        the first returned clip 0's transcript). We also rewind the cross-cache
        ``cumulative_length`` to 0 so ``StaticLayer.update``'s ``index_copy_``
        overwrites slots [0, S) (after capture it sits at S, and max_cache_len ==
        S, so a second write would be out-of-bounds). Both are no-ops on a fresh
        cache (the capture path), so the first utterance is unaffected.

        The SELF-attn cache needs the identical rewind, for the same reason.
        ``CohereAsrSelfAttention`` calls ``past_key_values.update(k, v, layer_idx)``
        without a ``cache_position``, so ``StaticLayer.update`` writes at the
        layer's ``cumulative_length`` — which the previous decode left at
        ``T + n_generated``. Without rewinding, this prefill scribbles the prompt
        K/V into slots ``[T+n, T+n+T)`` while :meth:`_reset_cache` points the
        decode back at slots ``[0, T)``, which still hold the *previous*
        utterance's prompt. Only layer 0 survives that (its prompt K/V depend on
        the token ids alone); layers ``i>0`` derive theirs from layer ``i-1``'s
        cross-attention output and so carry the previous clip's encoder state.
        """
        # force cross-attention to recompute K/V from THIS utterance's enc_h
        if getattr(cache, "is_updated", None) is not None:
            cache.is_updated = {i: False for i in range(self.n_layers)}
        # rewind both write heads so this prefill lands at [0, T) / [0, S)
        zero = None
        for cache_half in (cache.self_attention_cache, cache.cross_attention_cache):
            for layer in cache_half.layers:
                if getattr(layer, "is_initialized", False):
                    if zero is None:
                        zero = torch.zeros(
                            (1,), dtype=layer.cumulative_length.dtype,
                            device=layer.cumulative_length.device,
                        )
                    layer.cumulative_length.copy_(zero)

        B, T = dec_in.shape
        device = enc_h.device
        dtype = enc_h.dtype
        neg = torch.finfo(dtype).min
        ar = torch.arange(self.max_cache_len, device=device)

        pos = torch.arange(T, device=device).unsqueeze(0)
        q = torch.arange(T, device=device).unsqueeze(1)
        cmask = torch.where(
            ar[None, None, None, :] <= q[None, None, :, :], 0.0, neg
        ).to(dtype)
        out = self.decoder(
            input_ids=dec_in,
            attention_mask=cmask,
            position_ids=pos,
            encoder_hidden_states=enc_h,
            encoder_attention_mask=enc_mask,
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.arange(T, device=device),
        )
        h = out.last_hidden_state[:, -1:, :]
        nxt = self.proj_out(h).argmax(dim=-1)  # (B,1)
        return nxt

    def _set_self_cache_pos(self, pos: int) -> None:
        """Set every self-attn StaticCache layer's cumulative_length to ``pos``.

        Used to rewind the in-graph-advanced counter to the start of the next
        K-step block's slot range before each replay, so the captured steps write
        into the correct K/V slots. The K/V *contents* at slots [0, pos) are left
        intact (the previous replays wrote them correctly).
        """
        fill = torch.full((1,), pos, dtype=torch.long, device=self.device)
        for layer in self._cache.self_attention_cache.layers:
            layer.cumulative_length.copy_(fill)

    def _reset_cache(self, cache, first_token):
        """Rewind the decode state to the post-prefill position for a new decode.

        Rewinds the self-attn write head to ``T``, reseeds the decoder input
        token, and marks the cross-attn cache populated (so the captured steps
        reuse the K/V that :meth:`_prefill` just wrote rather than recomputing
        them, which would not be capture-safe).

        NOTHING in either cache is restored from a snapshot. Both halves hold
        state that belongs to the CURRENT utterance:

        * cross-attn K/V were just recomputed from this utterance's ``enc_h``.
        * self-attn prompt slots ``[0, T)`` were just written by this utterance's
          prefill — and they are utterance-specific, because layer ``i>0`` derives
          its prompt K/V from layer ``i-1``'s cross-attention output. Restoring a
          capture-time snapshot here made every clip that *reused* a captured
          graph decode with the capture clip's prompt state. That stayed hidden
          while each distinct clip length captured its own decoder; sharing one
          graph across lengths (shape bucketing) exposed it.

        Decode slots ``[T, ...)`` need no clearing: each captured step's causal
        mask only admits keys ``<= cur_pos``, and every such slot was written by
        this utterance's prefill or by an earlier step of this decode.
        """
        fill = torch.full((1,), self._T, dtype=torch.long, device=self.device)
        for layer in cache.self_attention_cache.layers:
            layer.cumulative_length.copy_(fill)
        cache.is_updated = {i: True for i in range(self.n_layers)}
        self.self_token.copy_(first_token)
        self.cur_pos.fill_(self._T)

    # ------------------------------------------------------------------ #
    # the captured per-step compute
    # ------------------------------------------------------------------ #
    def _alloc(self, B, T, S, device, dtype):
        K = self.steps_per_replay
        self._B = B
        self._T = T
        self._S = S
        self.K = K
        self.device = device
        self.dtype = dtype
        neg = torch.finfo(dtype).min
        self._neg = neg
        self._ar = torch.arange(self.max_cache_len, device=device)
        # decoder input token (chained in-graph); written by argmax each step.
        self.self_token = torch.zeros(B, 1, dtype=torch.long, device=device)
        # current query position (advances in-graph each step, mirroring the
        # StaticCache cumulative_length). Seed = T (first decode position).
        self.cur_pos = torch.full((1,), T, dtype=torch.long, device=device)
        # output ring: step j writes its emitted token into column j
        self.output_ring = torch.zeros(B, K, dtype=torch.long, device=device)
        self.output_ring_cpu = torch.empty(B, K, dtype=torch.long, pin_memory=True)
        self.pad_const = torch.full((B, 1), self.pad_token_id, dtype=torch.long, device=device)
        # STATIC encoder input buffers. The captured graph reads encoder_hidden_states
        # / encoder_attention_mask from these (NOT from call args, which a CUDA graph
        # freezes at capture time). decode() copies each new utterance's enc_h /
        # enc_mask into them before replay, so one captured graph serves many
        # utterances of the same (B, T, S) shape. Without this, every replay re-reads
        # the capture-time enc_h and every clip after the first decodes clip 0.
        # NOTE: the encoder hidden dim (e.g. 1280) differs from the decoder's
        # hidden_size (e.g. 1024) -- cross-attention projects between them -- so
        # size from the actual enc_h, not the decoder config.
        enc_hidden_dim = int(self._capture_enc_h_shape[-1])
        self.static_enc_h = torch.zeros(B, S, enc_hidden_dim, dtype=dtype, device=device)
        self.static_enc_mask = torch.zeros(B, 1, 1, S, dtype=dtype, device=device)
        _mark_many([self.output_ring, self.self_token, self.cur_pos, self.static_enc_h, self.static_enc_mask])

    def _step_fn(self, j):
        """The captured per-step compute; writes its token to ``output_ring[:, j]``.

        Reads the in-place-advancing ``cur_pos`` to build this step's causal mask
        ``(B,1,1,max_cache)`` (key k valid iff k <= cur_pos) and position_ids, so
        the K captured steps advance correctly without per-step baked masks. After
        the forward, ``cur_pos`` is incremented by 1 in-graph for the next step.

        Encoder hidden states / mask come from ``self.static_enc_h`` /
        ``self.static_enc_mask`` (populated by decode() per utterance) — NEVER from
        call args, which a captured graph would freeze.
        """
        cp = self.cur_pos  # (1,) scalar-long
        cm = torch.where(self._ar <= cp, 0.0, self._neg).to(self.dtype).view(
            1, 1, 1, self.max_cache_len
        ).expand(self._B, 1, 1, self.max_cache_len)
        p = cp.view(1, 1).expand(self._B, 1)
        out = self.decoder(
            input_ids=self.self_token,
            attention_mask=cm,
            position_ids=p,
            encoder_hidden_states=self.static_enc_h,
            encoder_attention_mask=self.static_enc_mask,
            past_key_values=self._cache,
            use_cache=True,
        )
        h = out.last_hidden_state[:, -1:, :]
        nx = self.proj_out(h).argmax(dim=-1)  # (B,1)
        self.output_ring[:, j].copy_(nx.squeeze(1))
        # chain in-graph: finished elements feed pad (KV ignored downstream)
        finished = (nx == self.eos_token_id)
        self.self_token.copy_(torch.where(finished, self.pad_const, nx))
        # advance the query position for the next captured step
        self.cur_pos.add_(1)

    # ------------------------------------------------------------------ #
    # capture
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def capture(self, dec_in, enc_h, enc_mask):
        """Prefill (eager), allocate buffers for this ``(B, prompt_len)`` shape,
        capture the K-step graph.

        ``dec_in`` / ``enc_h`` / ``enc_mask`` are a representative prompt + encoder
        output of the target ``(B, prompt_len)`` shape (used to drive warmup +
        capture). The graph is ``B``- and prompt-length-specific. The encoder
        length ``S`` is captured into the cross-attn StaticCache size, so the
        graph is also ``S``-specific; re-capture if ``S`` changes (the pipeline
        captures per shape).
        """
        B, T = dec_in.shape
        S = enc_h.shape[1]
        device = enc_h.device
        dtype = enc_h.dtype
        K = self.steps_per_replay

        # ONE cache for the lifetime of this graph; prefill + graph reference it.
        self._cache = self._make_cache(S)
        nxt0 = self._prefill(dec_in, enc_h, enc_mask, self._cache)
        self._capture_enc_h_shape = enc_h.shape
        self._alloc(B, T, S, device, dtype)
        # seed the static encoder buffers with the capture utterance so the
        # captured graph reads valid enc_h/enc_mask during warmup + capture.
        self.static_enc_h.copy_(enc_h)
        self.static_enc_mask.copy_(enc_mask)

        def _warm_block():
            self._reset_cache(self._cache, nxt0)
            for j in range(K):
                self._step_fn(j)

        # warmup on a side stream (stabilises cudnn/cublas autotune)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(self.warmup_iters):
                _warm_block()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        # capture K consecutive decode steps into one graph
        self._reset_cache(self._cache, nxt0)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            for j in range(K):
                self._step_fn(j)
        self._reset_cache(self._cache, nxt0)  # capture mutated state; restore
        self._captured = True
        self._captured_B = B
        self._captured_S = S
        return self

    # ------------------------------------------------------------------ #
    # decode
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def decode(
        self,
        dec_in,
        enc_h,
        enc_mask,
        *,
        max_new_tokens: int = 300,
    ) -> torch.Tensor:
        """Prefill for this utterance, then replay the K-step graph until done.

        Args:
            dec_in: ``(B, T)`` decoder prompt (must match the captured prompt len).
            enc_h: ``(B, S, 1280)`` encoder hidden states (must match captured S).
            enc_mask: ``(B, 1, 1, S)`` additive bidirectional mask.
            max_new_tokens: cap on generated tokens per row.

        Returns ``(B, n_gen)`` int64 on CPU — the *generated* tokens (prompt
        excluded). Each row ends with ``eos_token_id``; rows that finish early are
        padded with ``pad_token_id`` so all rows share the same length. The first
        column is the prefill output token; subsequent columns come from replays.
        """
        B, T = dec_in.shape
        K = self.K
        assert T == self._T, f"prompt len {T} != captured {self._T}; re-capture"
        assert B == self._captured_B, f"batch {B} != captured {self._captured_B}"
        max_out = max_new_tokens

        # prefill for THIS utterance into the SAME captured cache object
        nxt0 = self._prefill(dec_in, enc_h, enc_mask, self._cache)
        # rewind the write heads to the post-prefill position. Both caches keep
        # the state THIS prefill just wrote — see _reset_cache.
        self._reset_cache(self._cache, nxt0)
        # bind THIS utterance's encoder hidden states / mask into the static
        # buffers the captured graph reads. Without this, every replay re-reads
        # the capture-time enc_h and decodes the wrong utterance.
        self.static_enc_h.copy_(enc_h)
        self.static_enc_mask.copy_(enc_mask)
        # bookkeeping on CPU (the ring is synced to CPU each replay)
        results = [nxt0.squeeze(1).cpu()]
        finished = (results[0] == self.eos_token_id)

        # the K captured steps form a template running at positions
        # [base, base+1, ..., base+K-1]; `base` advances by K each replay.
        # Before each replay we set cur_pos / cumulative_length / self_token to
        # the block's base so the captured steps continue the sequence and write
        # into the right K/V slots. The mask is built dynamically from cur_pos,
        # so it is correct at any base (no per-step baked masks).
        T = self._T
        n_done = 1
        base = T  # the first decode step queries position T
        # seed the first replay: self_token = prefill token, pos already = T
        while n_done < max_out:
            self.cur_pos.fill_(base)
            self._set_self_cache_pos(base)
            self.graph.replay()
            self.output_ring_cpu.copy_(self.output_ring, non_blocking=False)
            ring = self.output_ring_cpu  # (B, K) — one sync per replay
            for j in range(K):
                col = ring[:, j]
                emitted = torch.where(finished, self.pad_token_id, col)
                results.append(emitted)
                finished = finished | (col == self.eos_token_id)
                n_done += 1
                if bool(finished.all()) or n_done >= max_out:
                    break
            if bool(finished.all()):
                break
            base += K
            # seed next replay's self_token with this block's last emitted token
            self.self_token[:, 0] = ring[:, -1].to(self.self_token.device)
        gen = torch.stack(results, dim=1).cpu()  # (B, n_gen)
        return gen


def greedy_decode_graphed(
    model,
    input_features: torch.Tensor,
    attention_mask: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 300,
    steps_per_replay: int = 8,
    warmup_iters: int = 4,
    max_cache_len: int = 1024,
) -> torch.Tensor:
    """Convenience wrapper: encode + capture + decode a single batch.

    Returns ``(B, n_gen)`` generated token ids (prompt excluded), byte-exact
    with the eager ``starling.cohere.reference.greedy_generate`` path.
    """
    gd = GraphedDecoder(
        model, max_cache_len=max_cache_len, steps_per_replay=steps_per_replay,
        warmup_iters=warmup_iters,
    )
    with torch.inference_mode():
        enc_h = model.model.encoder(
            input_features=input_features, attention_mask=attention_mask
        ).last_hidden_state
        B, S = enc_h.shape[:2]
        neg = torch.finfo(enc_h.dtype).min
        enc_mask = torch.zeros(B, 1, 1, S, device=enc_h.device, dtype=enc_h.dtype)
        gd.capture(decoder_input_ids, enc_h, enc_mask)
        return gd.decode(
            decoder_input_ids, enc_h, enc_mask, max_new_tokens=max_new_tokens
        )
