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
graphed" pattern. The captured decode state is fully resettable (the post-prefill
cache snapshot is restored before each real decode), so one captured graph serves
many utterances of the same ``(prompt_len)`` shape.
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
        """Run the eager prefill into ``cache``; returns first token (B,1)."""
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

    def _snapshot_cache(self, cache):
        """Save the post-prefill cache state for later resets."""
        self_cache = cache.self_attention_cache
        cross_cache = cache.cross_attention_cache
        self._snap_self_keys = [l.keys.clone() for l in self_cache.layers]
        self._snap_self_vals = [l.values.clone() for l in self_cache.layers]
        self._snap_self_cum = [l.cumulative_length.clone() for l in self_cache.layers]
        self._snap_cross_keys = [l.keys.clone() for l in cross_cache.layers]
        self._snap_cross_vals = [l.values.clone() for l in cross_cache.layers]
        self._snap_cross_cum = [l.cumulative_length.clone() for l in cross_cache.layers]

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
        """Restore the FULL post-prefill cache snapshot + reset decoder input token.

        Used once at capture time (to undo warmup/capture mutations) — NOT per
        replay in :meth:`decode`.
        """
        self_cache = cache.self_attention_cache
        cross_cache = cache.cross_attention_cache
        for l, k, v, c in zip(
            self_cache.layers, self._snap_self_keys, self._snap_self_vals, self._snap_self_cum
        ):
            l.keys.copy_(k)
            l.values.copy_(v)
            l.cumulative_length.copy_(c)
        for l, k, v, c in zip(
            cross_cache.layers, self._snap_cross_keys, self._snap_cross_vals, self._snap_cross_cum
        ):
            l.keys.copy_(k)
            l.values.copy_(v)
            l.cumulative_length.copy_(c)
        cache.is_updated = {i: True for i in range(self.n_layers)}
        self.self_token.copy_(first_token)
        self.cur_pos.fill_(self._T)

    # ------------------------------------------------------------------ #
    # the captured per-step compute
    # ------------------------------------------------------------------ #
    def _alloc(self, B, T, device, dtype):
        K = self.steps_per_replay
        self._B = B
        self._T = T
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
        self.pad_const = torch.full((B, 1), self.pad_token_id, dtype=torch.long, device=device)
        _mark_many([self.output_ring, self.self_token, self.cur_pos])

    def _step_fn(self, j, enc_h, enc_mask):
        """The captured per-step compute; writes its token to ``output_ring[:, j]``.

        Reads the in-place-advancing ``cur_pos`` to build this step's causal mask
        ``(B,1,1,max_cache)`` (key k valid iff k <= cur_pos) and position_ids, so
        the K captured steps advance correctly without per-step baked masks. After
        the forward, ``cur_pos`` is incremented by 1 in-graph for the next step.
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
            encoder_hidden_states=enc_h,
            encoder_attention_mask=enc_mask,
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
        self._snapshot_cache(self._cache)
        self._alloc(B, T, device, dtype)

        def _warm_block():
            self._reset_cache(self._cache, nxt0)
            for j in range(K):
                self._step_fn(j, enc_h, enc_mask)

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
                self._step_fn(j, enc_h, enc_mask)
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
        # reset to post-prefill snapshot once (undo nothing here, but ensures
        # clean state: self K/V at slots 0..T-1, cumulative_length=T, cross full)
        self._reset_cache(self._cache, nxt0)
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
            ring = self.output_ring.cpu()  # (B, K) — one sync per replay
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
