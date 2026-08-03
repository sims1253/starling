"""CUDA-graph-captured greedy RNN-T decode for parakeet-unified-en-0.6b.

Same I/O and byte-exact token output as :mod:`decode_eager`, but the per-step
compute is captured into one ``torch.cuda.CUDAGraph`` and served by
``graph.replay()``. The stock greedy RNN-T loop is launch-bound (hundreds of
tiny per-step prediction+joint kernels, each ~us of host launch latency); one
graph replay per K steps collapses those into a single replay.

The design mirrors the sibling TDT :class:`starling.parakeet.decode_mega.GraphedDecoder`
(static buffers, K-step capture, ring buffer + single-sync-per-replay, in-graph
last_token chaining, device-side blank-skip ``torch.where``). The RNNT-specific
rewrite is in :meth:`GraphedDecoder._step_fn`: instead of a TDT step
(prediction + joint-with-duration + duration table), each captured step is one
**RNN-T emission attempt**:

1. prediction net step: ``embed(last_token)`` -> one-step LSTM (state in static
   ``h_buf``/``c_buf``) -> ``pred (B,1,H)``;
2. gather the acoustic frame at ``frame_idx``;
3. joint: ``Linear(enc) + Linear(pred)`` -> ReLU -> ``Linear -> (B, V+1)``;
4. argmax -> ``tok``;
5. device-side branching (no host sync):
   * ``tok != blank`` **and** ``sym_count < max_symbols``: emit ``tok`` (write to
     ring), advance the LSTM state, ``last_token <- tok``, ``sym_count += 1``,
     frame pointer UNCHANGED;
   * ``tok == blank`` **or** ``sym_count >= max_symbols``: emit nothing (write
     blank to ring), FREEZE the LSTM state (blank-skip), ``last_token <- blank``,
     ``sym_count <- 0``, ``frame_idx += 1``.
6. finished rows (``frame_idx >= valid_lengths``) are frozen in-graph
   (``last_token <- blank``) so they keep emitting blank + advancing until the
   host stops the loop.

The max_symbols_per_step cap is implemented device-side (``sym_count >=
max_symbols`` forces a blank), so the captured loop is byte-exact with the eager
oracle's ``while not_blank and symbols < max_symbols`` guard.

No eager step-0 prefill
-----------------------
Unlike the TDT megakernel, the RNNT prediction net has no host-side
``cache.is_initialized`` branch (we drive ``nn.LSTM`` directly with explicit
state tensors). So step 0 -- zero ``h_buf``/``c_buf``/``sym_count``/``frame_idx``
and ``last_token = blank`` (SOS) -- is capture-safe and runs inside the graph.
The host just resets the static buffers before each decode and replays.

Multi-step capture (K steps per replay)
---------------------------------------
The graph captures ``K = steps_per_replay`` consecutive emission attempts. Every
step's state (``last_token`` / ``frame_idx`` / ``h_buf`` / ``c_buf`` /
``sym_count``) lives in static buffers mutated IN PLACE, so step ``j+1`` of one
replay reads step ``j``'s in-place mutations from the same fixed addresses.
``last_token`` and ``sym_count`` are chained IN GRAPH (finished/blank freeze)
so no host sync is needed between captured steps; each step writes its emitted
token and post-step cumulative ``frame_idx`` into ``output_ring[:, j]`` /
``frame_ring[:, j]``.

The host loop replays the K-step graph ``ceil(total_steps / K)`` times and does
ONE device->host sync per replay of the stacked ``(2, B, K)`` ring
``[output_ring, frame_ring]``: it scatters the K tokens into ``output``
(padding finished/blank positions with ``blank_id``), and stops when every batch
element's ``frame_idx >= valid_length``.
"""

from __future__ import annotations

from typing import List, Tuple

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
    """Capture the per-step greedy RNN-T decode into one CUDA graph; decode many.

    The graph is shape-specific (``B``, ``T_enc`` fixed at :meth:`capture` time).
    :meth:`capture` allocates buffers + warmup + capture once; each
    :meth:`decode` resets the decoder state and replays the captured graph until
    every batch element is finished.

    Args:
        decoder: a loaded :class:`~starling.parakeet_unified.modeling.RNNTDecoder`
            on cuda (eval mode).
        joint: a loaded :class:`~starling.parakeet_unified.modeling.RNNTJoint`.
        blank_id: the RNN-T blank token id (1024).
        vocab_size: BPE vocab size (1024); ``blank_id == vocab_size``.
        max_symbols: ``max_symbols_per_step`` guard (10).
        pred_hidden: prediction-net hidden size (640).
        n_layers: prediction-net LSTM layer count (2).
        warmup_iters: side-stream warmup iterations before graph capture.
        steps_per_replay: number of consecutive emission steps captured into ONE
            graph replay (default ``32``). ``1`` reproduces one-step-per-replay
            with byte-identical output.
    """

    def __init__(
        self,
        decoder,
        joint,
        *,
        blank_id: int,
        vocab_size: int,
        max_symbols: int,
        pred_hidden: int,
        n_layers: int,
        warmup_iters: int = 4,
        steps_per_replay: int = 32,
        graph_pool=None,
    ) -> None:
        self.dec = decoder
        self.joint = joint
        self.dec.prediction.dec_rnn.lstm.flatten_parameters()
        self.blank_id = int(blank_id)
        self.vocab_size = int(vocab_size)
        self.max_symbols = int(max_symbols)
        self.pred_hidden = int(pred_hidden)
        self.n_layers = int(n_layers)
        self.warmup_iters = int(warmup_iters)
        self.steps_per_replay = max(1, int(steps_per_replay))
        self.graph_pool = graph_pool  # shared torch.cuda.graph_pool_handle() or None

        self._captured = False
        self._B: int | None = None
        self._T_enc: int | None = None

    # ------------------------------------------------------------------ #
    # buffer allocation
    # ------------------------------------------------------------------ #
    def _alloc(self, B: int, T_enc: int, device, dtype) -> None:
        K = self.steps_per_replay
        H = self.pred_hidden
        nl = self.n_layers
        self._B = B
        self._T_enc = T_enc
        self.K = K
        self.device = device
        self.dtype = dtype
        # Total emission-steps budget: each of T_enc frames consumes up to
        # max_symbols non-blank steps + 1 blank step. +16 slack.
        max_out = (self.max_symbols + 1) * T_enc + 16
        self.max_out = max_out

        self.pooler = torch.zeros((B, T_enc, 1024), dtype=dtype, device=device)
        self.valid_lengths = torch.zeros((B,), dtype=torch.long, device=device)
        self.frame_idx = torch.zeros((B,), dtype=torch.long, device=device)
        self.last_token = torch.full((B,), self.blank_id, dtype=torch.long, device=device)
        self.sym_count = torch.zeros((B,), dtype=torch.long, device=device)
        self.arange_B = torch.arange(B, device=device)
        self.h_buf = torch.zeros((nl, B, H), dtype=dtype, device=device)
        self.c_buf = torch.zeros((nl, B, H), dtype=dtype, device=device)
        # output holds the raw step stream: emitted token, or blank_id for
        # non-emits (natural blank, force-blank cap, finished padding). The
        # tokenizer's ids_to_text filters blank_id, so blank == "no token".
        # Host-side accumulation buffer. The CUDA graph writes only the K-step
        # ring buffers; after each replay those rings are synced to CPU for
        # EOS/finish checks, so keeping the final output on CPU avoids a
        # CPU->GPU scatter that would be copied back to CPU for tokenization.
        self.output = torch.full((B, max_out), self.blank_id, dtype=torch.long)
        self.ring_pair = torch.zeros((2, B, K), dtype=torch.long, device=device)
        self.output_ring = self.ring_pair[0]
        self.frame_ring = self.ring_pair[1]
        self.ring_pair_cpu = torch.empty((2, B, K), dtype=torch.long, pin_memory=True)
        self.valid_lengths_cpu = torch.empty((B,), dtype=torch.long, pin_memory=True)
        self.blank_cpu = torch.full((B,), self.blank_id, dtype=torch.long)
        self.blank_const = torch.full((B,), self.blank_id, dtype=torch.long, device=device)
        self.zero_const = torch.zeros((B,), dtype=torch.long, device=device)
        self.one_const = torch.ones((B,), dtype=torch.long, device=device)
        _mark_many([
            self.pooler, self.valid_lengths, self.frame_idx, self.last_token,
            self.sym_count, self.arange_B, self.h_buf, self.c_buf,
            self.ring_pair, self.output_ring, self.frame_ring, self.blank_const,
            self.zero_const, self.one_const,
        ])

    # ------------------------------------------------------------------ #
    # the captured per-step compute
    # ------------------------------------------------------------------ #
    def _step_fn(self, ring_col: int = 0) -> None:
        """One RNN-T emission attempt; writes its outputs to ring column ``ring_col``.

        The K-step graph calls this K times with ``ring_col`` in ``0..K-1``; each
        call reads the in-place-mutated static buffers left by the previous call,
        computes one emission attempt, and writes the emitted token (or blank for
        non-emit) + the post-step cumulative frame_idx into the K-step rings.
        ``last_token`` and ``sym_count`` are chained IN GRAPH so the next in-graph
        step reads the correct values without a host sync.
        """
        T_enc = self._T_enc
        # (1) prediction net step: embed(last_token) -> one LSTM step
        lt = self.last_token.unsqueeze(1)                          # (B,1)
        emb = self.dec.prediction.embed(lt)                        # (B,1,H)
        lstm_out, (hn, cn) = self.dec.prediction.dec_rnn.lstm(
            emb, (self.h_buf, self.c_buf)
        )                                                          # (B,1,H)
        # (2) gather the acoustic frame at the current pointer
        idx = self.frame_idx.clamp(max=T_enc - 1)
        enc_frame = self.pooler[self.arange_B, idx]                # (B, 1024)
        # (3) joint: inlined joint forward (T=1, U=1) -> (B, V+1)
        e = self.joint.enc(enc_frame)                              # (B, H)
        p = self.joint.pred(lstm_out[:, 0, :])                     # (B, H)
        z = e + p                                                  # (B, H)
        logits = self.joint.joint_net.linear(self.joint.joint_net.relu(z))  # (B, V+1)
        tok = logits.argmax(dim=-1)                                # (B,)
        # (4) device-side emission logic (NO host sync).
        blank_mask = tok == self.blank_id                          # natural blank
        force_blank = self.sym_count >= self.max_symbols           # cap reached
        emit_blank = blank_mask | force_blank                      # (B,) bool
        not_blank = ~emit_blank
        # token to record: emit tok if a real emission, else blank (no token)
        rec_tok = torch.where(not_blank, tok, self.blank_const)    # (B,)
        # frame advances ONLY on emit_blank (1 step)
        self.frame_idx.add_(emit_blank.long())
        # symbol counter: +1 on emit, reset to 0 on blank/force-blank
        self.sym_count = torch.where(
            not_blank, self.sym_count + 1, self.zero_const
        )
        # LSTM state: ALWAYS advance (standard RNN-T runs the prediction net on
        # every step, including after a blank -- ``last_token`` is the input to
        # the next step's prediction). The TDT blank-skip freeze does NOT apply
        # to RNNT (the eager oracle in decode_eager advances the LSTM
        # unconditionally on every loop iteration).
        self.h_buf.copy_(hn)
        self.c_buf.copy_(cn)
        # write this step's (token, post-step cumulative frame_idx) to the rings
        self.output_ring[:, ring_col].copy_(rec_tok)
        self.frame_ring[:, ring_col].copy_(self.frame_idx)
        # chain last_token IN GRAPH for the next captured step.
        # RNN-T greedy semantics (mirrors decode_eager.greedy_decode): on a real
        # emission ``last_token <- tok``; on blank ``last_token`` is LEFT
        # UNCHANGED (the eager oracle does NOT reset last_token on a blank -- the
        # next step's prediction net re-runs on the same last emitted token, NOT
        # on blank; only the SOS start token is blank). Finished rows are frozen
        # to blank (they keep emitting blank + advancing past the end).
        finished_now = self.frame_idx >= self.valid_lengths
        next_last = torch.where(not_blank, tok, self.last_token)
        self.last_token.copy_(torch.where(finished_now, self.blank_const, next_last))

    # ------------------------------------------------------------------ #
    # capture / decode
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def capture(
        self,
        pooler: torch.Tensor,
        valid_lengths: torch.Tensor,
        *,
        steps_per_replay: int | None = None,
    ) -> "GraphedDecoder":
        """Allocate buffers for this ``(B, T_enc)`` shape and capture the graph.

        ``pooler`` / ``valid_lengths`` are a representative encoder output of the
        target shape (used to drive warmup); the graph itself is shape-only and
        is reused by :meth:`decode` for any same-shape input.
        """
        if steps_per_replay is not None:
            self.steps_per_replay = max(1, int(steps_per_replay))
        K = self.steps_per_replay
        B, T_enc, _ = pooler.shape
        device = pooler.device
        dtype = pooler.dtype
        self._alloc(B, T_enc, device, dtype)
        self.pooler.copy_(pooler)
        self.valid_lengths.copy_(valid_lengths)

        # save the pre-capture (zeroed) reset point
        h_s = self.h_buf.clone()
        c_s = self.c_buf.clone()
        fi_s = self.frame_idx.clone()
        lt_s = self.last_token.clone()
        sc_s = self.sym_count.clone()

        def _reset():
            self.h_buf.copy_(h_s)
            self.c_buf.copy_(c_s)
            self.frame_idx.copy_(fi_s)
            self.last_token.copy_(lt_s)
            self.sym_count.copy_(sc_s)

        # warmup on a side stream (stabilises cudnn/cublas autotune). Run the
        # full K-step block each warmup iter so capture records the exact
        # in-graph chained sequence the replay loop will exercise.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(self.warmup_iters):
                for j in range(K):
                    self._step_fn(j)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        _reset()

        # capture K consecutive emission steps into one graph. Each step reads
        # the in-place-mutated static buffers left by the previous step.
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            for j in range(K):
                self._step_fn(j)
        _reset()  # capture mutated the buffers; restore for real decodes
        self._captured = True
        return self

    @torch.inference_mode()
    def _run_loop(
        self,
        pooler: torch.Tensor,
        valid_lengths: torch.Tensor,
    ) -> Tuple[int, torch.Tensor]:
        """Reset state, replay the K-step graph until all finished; return ``(out_step, output)``.

        ``out_step`` is the raw step count written (emits + blanks + finished
        padding up to the all-finished column). ``output`` is the ``(B, max_out)``
        static buffer; per-element emitted tokens = ``output[b, :out_step]`` with
        ``blank_id`` entries filtered out.
        """
        K = self.K
        self.pooler.copy_(pooler)
        self.valid_lengths.copy_(valid_lengths)
        self.frame_idx.zero_()
        self.last_token.fill_(self.blank_id)
        self.sym_count.zero_()
        self.h_buf.zero_()
        self.c_buf.zero_()
        self.output.fill_(self.blank_id)

        self.valid_lengths_cpu.copy_(self.valid_lengths, non_blocking=False)
        valid_lengths_cpu = self.valid_lengths_cpu                # (B,)
        blank_cpu = self.blank_cpu                                # (B,)
        step = 0
        out_step = 0

        if self._captured:
            while step < self.max_out:
                # one K-step replay: output_ring / frame_ring are filled and
                # last_token / sym_count are chained in-graph for the next replay
                self.graph.replay()
                # ONE device->host sync for the whole K-step batch
                self.ring_pair_cpu.copy_(self.ring_pair, non_blocking=False)
                info = self.ring_pair_cpu                        # (2, B, K)
                ring_cpu = info[0]                                 # (B, K) tokens
                fring_cpu = info[1]                                # (B, K) frame_idx
                kk = min(K, self.max_out - step)
                if kk <= 0:
                    break
                # finished mask per kept column: frame_idx >= valid_length
                fin = fring_cpu[:, :kk] >= valid_lengths_cpu[:, None]   # (B, kk)
                # scatter: blank where finished (suppresses any stale emit from
                # a row that crossed its boundary mid-K); else the ring token
                # (which is already blank for non-emit steps, so emits + blanks
                # are both recorded correctly).
                self.output[:, step:step + kk] = torch.where(
                    fin, blank_cpu[:, None], ring_cpu[:, :kk]
                )
                # stop once EVERY batch element is finished (first such column)
                col_all_done = fin.all(dim=0)                      # (kk,)
                done_idx = col_all_done.nonzero(as_tuple=False).flatten()
                if done_idx.numel() > 0:
                    j_break = int(done_idx[0].item())
                    out_step = step + j_break + 1
                    break
                out_step = step + kk
                step += K
        else:  # pragma: no cover - capture is always called before decode
            raise RuntimeError("GraphedDecoder not captured; call .capture() first")

        return out_step, self.output

    @torch.inference_mode()
    def decode(self, pooler: torch.Tensor, valid_lengths: torch.Tensor) -> List[List[int]]:
        """Decode one (already-encoded) batch; returns ``B`` token-id lists.

        Each list is the non-blank emitted tokens for that utterance, in emission
        order -- byte-identical to
        :func:`starling.parakeet_unified.decode_eager.greedy_decode`.
        """
        B = self._B
        T_enc = self._T_enc
        assert pooler.shape == (B, T_enc, 1024), (
            f"pooler {tuple(pooler.shape)} != captured {(B, T_enc, 1024)}; "
            "re-capture for this shape"
        )
        out_step, output = self._run_loop(pooler, valid_lengths)
        results: List[List[int]] = []
        for b in range(B):
            row = output[b, :out_step].tolist()
            results.append([int(t) for t in row if t != self.blank_id])
        return results


def greedy_decode_graphed(
    encoder_hidden: torch.Tensor,
    enc_lengths: torch.Tensor,
    decoder,
    joint,
    *,
    blank_id: int,
    vocab_size: int,
    max_symbols_per_step: int,
    pred_hidden: int,
    n_layers: int,
    warmup_iters: int = 4,
    steps_per_replay: int = 16,
) -> List[List[int]]:
    """CUDA-graph-captured greedy RNN-T decode (byte-exact with eager).

    Convenience wrapper: capture + decode a single batch. For repeated decodes
    of the same shape, reuse a :class:`GraphedDecoder` directly so the one-off
    capture cost is amortised.
    """
    gd = GraphedDecoder(
        decoder, joint,
        blank_id=blank_id, vocab_size=vocab_size,
        max_symbols=max_symbols_per_step,
        pred_hidden=pred_hidden, n_layers=n_layers,
        warmup_iters=warmup_iters, steps_per_replay=steps_per_replay,
    )
    gd.capture(encoder_hidden, enc_lengths)
    return gd.decode(encoder_hidden, enc_lengths)


__all__ = ["GraphedDecoder", "greedy_decode_graphed"]
