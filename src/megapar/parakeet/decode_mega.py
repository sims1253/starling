"""CUDA-graph-captured greedy TDT decode for nvidia/parakeet-tdt-0.6b-v3.

Same I/O and byte-exact output as :mod:`decode_eager`, but the per-step compute
(``ALGORITHM.md`` steps 1-5) is captured into a single ``torch.cuda.CUDAGraph``
and served by ``graph.replay()`` each step. The stock decode loop is ~80% wall
and launch-bound (~10% GPU-busy, see ``outputs/profile_analysis.md``); replaying
one graph per step collapses the hundreds of tiny per-step kernel launches into
a single replay, removing the launch overhead.

Two entry points
----------------
* :class:`GraphedDecoder` -- capture the graph ONCE for a fixed ``(B, T_enc)``
  shape, then :meth:`decode` many encoder-feature tensors. This is the
  production-realistic shape: amortise capture, replay per utterance. The
  benchmark uses it so the timed decode loop excludes one-off capture cost.
* :func:`greedy_decode_graphed` -- a thin convenience wrapper that captures +
  decodes a single batch (used by the byte-exactness test).

Static-buffer strategy
----------------------
Every tensor the captured graph reads or writes lives at a fixed GPU address for
the whole decode, tagged with ``torch._dynamo.mark_static_address`` so the graph
keeps referencing them across replays:

* ``pooler`` (B, T_enc, 640)         -- encoder output, read by gather each step
* ``frame_idx`` (B,)                 -- per-element encoder frame pointer (advances in place)
* ``last_token`` (B,)                -- decoder input token for this step
* ``static_token`` (B,)              -- graph writes the chosen token here
* ``h_buf`` / ``c_buf`` (2,B,640)    -- LSTM hidden / cell state (advanced in place)
* ``cc_buf`` (B,1,640)               -- frozen decoder-output cache (blank-skip)
* ``arange_B``, ``dur_table``, ``ones_b``, ``valid_lengths``, ``output`` -- constants / sink

The host loop replays the graph, then does ONE device->host sync per step of a
small ``(2, B)`` stack ``[static_token, frame_idx]``: the host reads the emitted
tokens, writes them into ``output`` (padding finished elements with ``pad_id``),
updates ``last_token`` (``finished -> blank`` so a finished element's decoder
state freezes), and stops when ``all(frame_idx >= valid_lengths)``.

Why the decoder is replicated manually (the blank-skip + graph interaction)
-------------------------------------------------------------------------
``model.decoder.forward`` implements blank-skip with a **host-side** branch::

    if cache.is_initialized and blank_mask.all():   # blank_mask.all() is a device
        return cache.cache                          # tensor used as a Python bool
                                                    # -> host sync -> aborts capture

That ``if`` short-circuits cleanly only while the cache is *uninitialized*; once
the cache is initialized (every step after the first), evaluating
``blank_mask.all()`` triggers a host sync which CUDA-graph capture forbids
(``cudaErrorStreamCaptureUnsupported``). So the model's own ``decoder.forward``
is **not** graph-capturable past step 0.

We therefore replicate the decoder step with the model's own submodules
(``embedding`` -> ``lstm`` -> ``decoder_projector``) plus a **device-side**
``torch.where`` blank-skip freeze (validated bit-exact, 0.000e+00 diff, against
``model.decoder.forward`` for all-blank / mixed / all-nonblank batches). This
keeps the whole step capture-safe.

The very first decode step (``last_token == blank`` start token, zero cache) must
run the LSTM unconditionally (the eager init path does NOT freeze even on a blank
token, because the cache is uninitialized); so step 0 runs eager and steps 1+ run
graphed. This mirrors the sibling ``llm_mega.py`` "prefill eager, decode graphed"
pattern.
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
    """Capture the per-step TDT decode into one CUDA graph; decode many inputs.

    The graph is shape-specific (``B``, ``T_enc`` fixed at :meth:`capture` time).
    :meth:`capture` runs step 0 eager + warmup + capture once; each
    :meth:`decode` resets the decoder state, re-runs step 0 for the new
    utterance, then replays the captured graph until finished.

    Args:
        model: a loaded ``ParakeetForTDT`` on cuda (eval mode, bf16).
        warmup_iters: side-stream warmup iterations before graph capture.
    """

    def __init__(self, model, *, warmup_iters: int = 4) -> None:
        cfg = model.config
        self.model = model
        self.dec = model.decoder
        self.joint = model.joint
        self.blank_id = int(cfg.blank_token_id)
        self.vocab_size = int(cfg.vocab_size)
        self.max_symbols = int(cfg.max_symbols_per_step)
        self.hid = int(cfg.decoder_hidden_size)   # 640
        self.nl = int(cfg.num_decoder_layers)     # 2
        self.durations = list(cfg.durations)
        self.warmup_iters = int(warmup_iters)

        self._captured = False
        self._B: int | None = None
        self._T_enc: int | None = None

    # ------------------------------------------------------------------ #
    # buffer allocation + the captured step
    # ------------------------------------------------------------------ #
    def _alloc(self, B: int, T_enc: int, device) -> None:
        self._B = B
        self._T_enc = T_enc
        self.device = device
        max_out = self.max_symbols * T_enc + 16
        self.max_out = max_out
        self.pooler = torch.zeros((B, T_enc, self.hid), dtype=torch.bfloat16, device=device)
        self.valid_lengths = torch.zeros((B,), dtype=torch.long, device=device)
        self.frame_idx = torch.zeros((B,), dtype=torch.long, device=device)
        self.last_token = torch.full((B,), self.blank_id, dtype=torch.long, device=device)
        self.static_token = torch.zeros((B,), dtype=torch.long, device=device)
        self.arange_B = torch.arange(B, device=device)
        self.h_buf = torch.zeros((self.nl, B, self.hid), dtype=torch.bfloat16, device=device)
        self.c_buf = torch.zeros((self.nl, B, self.hid), dtype=torch.bfloat16, device=device)
        self.cc_buf = torch.zeros((B, 1, self.hid), dtype=torch.bfloat16, device=device)
        self.ones_b = torch.ones((B,), dtype=torch.long, device=device)
        self.dur_table = torch.tensor(self.durations, device=device, dtype=torch.long)
        self.output = torch.full((B, max_out), self.pad_id, dtype=torch.long, device=device)
        _mark_many([
            self.pooler, self.valid_lengths, self.frame_idx, self.last_token,
            self.static_token, self.arange_B, self.h_buf, self.c_buf, self.cc_buf,
            self.ones_b, self.dur_table, self.output,
        ])

    def _step_fn(self) -> None:
        """The captured per-step compute (manual decoder; no host branch)."""
        B = self._B
        T_enc = self._T_enc
        # decoder: embedding -> lstm -> projector + device-side blank-skip freeze
        lt = self.last_token.unsqueeze(1)                          # (B,1)
        emb = self.dec.embedding(lt)                               # (B,1,640)
        lstm_out, (hn, cn) = self.dec.lstm(emb, (self.h_buf, self.c_buf))
        proj = self.dec.decoder_projector(lstm_out)                # (B,1,640)
        advance = (self.last_token != self.blank_id)               # (B,) True=advance
        adv_out = advance.view(B, 1, 1)
        adv_h = advance.view(1, B, 1)
        decoder_out = torch.where(adv_out, proj, self.cc_buf)      # freeze blank elems
        h_new = torch.where(adv_h, hn, self.h_buf)
        c_new = torch.where(adv_h, cn, self.c_buf)
        # joint -> combined logits (token[:8193] | dur[8193:])
        idx = self.frame_idx.clamp(max=T_enc - 1)
        enc_frame = self.pooler[self.arange_B, idx]                # (B,640)
        logits = self.joint(
            encoder_hidden_states=enc_frame[:, None, None, :],
            decoder_hidden_states=decoder_out[:, None, :, :],
        ).squeeze(1).squeeze(1)                                    # (B,8198)
        tok = logits[:, :self.vocab_size].argmax(dim=-1)
        dur_idx = logits[:, self.vocab_size:].argmax(dim=-1)
        dur = self.dur_table[dur_idx]
        blank_mask = (tok == self.blank_id)
        dur = torch.where(blank_mask & (dur == 0), self.ones_b, dur)
        self.frame_idx.add_(dur)
        # write back cache state (in place on the static buffers)
        self.h_buf.copy_(h_new)
        self.c_buf.copy_(c_new)
        self.cc_buf.copy_(decoder_out)
        self.static_token.copy_(tok)

    def _step0_eager(self) -> torch.Tensor:
        """Eager step 0 (init path: zero cache, NO blank-skip freeze).

        Reads the current ``pooler`` / decoder buffers; returns the chosen token.
        """
        T_enc = self._T_enc
        lt = self.last_token.unsqueeze(1)
        emb = self.dec.embedding(lt)
        lstm_out, (hn, cn) = self.dec.lstm(emb, (self.h_buf, self.c_buf))
        proj = self.dec.decoder_projector(lstm_out)
        decoder_out = proj                                         # no freeze at step 0
        idx = self.frame_idx.clamp(max=T_enc - 1)
        enc_frame = self.pooler[self.arange_B, idx]
        logits = self.joint(
            encoder_hidden_states=enc_frame[:, None, None, :],
            decoder_hidden_states=decoder_out[:, None, :, :],
        ).squeeze(1).squeeze(1)
        tok = logits[:, :self.vocab_size].argmax(dim=-1)
        dur_idx = logits[:, self.vocab_size:].argmax(dim=-1)
        dur = self.dur_table[dur_idx]
        bm = (tok == self.blank_id)
        dur = torch.where(bm & (dur == 0), self.ones_b, dur)
        self.frame_idx.add_(dur)
        self.h_buf.copy_(hn)
        self.c_buf.copy_(cn)
        self.cc_buf.copy_(decoder_out)
        self.static_token.copy_(tok)
        return tok

    # ------------------------------------------------------------------ #
    # capture / decode
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def capture(self, pooler: torch.Tensor, valid_lengths: torch.Tensor,
                pad_id: int) -> "GraphedDecoder":
        """Allocate buffers for this ``(B, T_enc)`` shape and capture the graph.

        ``pooler`` / ``valid_lengths`` are a representative encoder output of the
        target shape (used to drive warmup + capture); the graph itself is
        shape-only and is reused by :meth:`decode` for any same-shape input.
        """
        B, T_enc, _ = pooler.shape
        self.pad_id = pad_id
        device = pooler.device
        self._alloc(B, T_enc, device)
        self.pooler.copy_(pooler)
        self.valid_lengths.copy_(valid_lengths)

        # step 0 eager (init) on the representative input
        self._step0_eager()
        finished = (self.frame_idx >= self.valid_lengths)
        if bool(finished.all()):
            # utterance so short it ends in one step; no graph needed
            self._captured = False
            return self

        # save the post-step-0 reset point
        h_s = self.h_buf.clone(); c_s = self.c_buf.clone(); cc_s = self.cc_buf.clone()
        fi_s = self.frame_idx.clone(); lt_s = self.last_token.clone()

        def _reset():
            self.h_buf.copy_(h_s); self.c_buf.copy_(c_s); self.cc_buf.copy_(cc_s)
            self.frame_idx.copy_(fi_s); self.last_token.copy_(lt_s)

        # warmup on a side stream (stabilises cudnn/cublas autotune)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(self.warmup_iters):
                self._step_fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        _reset()

        # capture the per-step graph
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._step_fn()
        _reset()  # capture mutated the buffers; restore for real decodes
        self._captured = True
        return self

    @torch.inference_mode()
    def _run_loop(
        self,
        pooler: torch.Tensor,
        valid_lengths: torch.Tensor,
        *,
        collect_meta: bool = False,
    ):
        """Reset state, run step 0 + the replay loop; return ``out_step``.

        Excludes ``processor.batch_decode`` so callers (and the benchmark) can
        time the GPU decode loop only, matching the baseline's decode split.

        When ``collect_meta=True``, additionally returns two per-batch-element
        lists ``meta_tokens`` and ``meta_frames``: for each emitted token (in
        emission order, including the leading blank start token), the token id
        and the **cumulative local encoder-frame index** of the decoder pointer
        *after* emitting that token (i.e. the sum of that token's and all prior
        tokens' durations). This frame position is what frame-aligned chunk
        stitching needs (see :mod:`chunking`). The emitted token sequence and
        ``out_step`` are byte-identical to the ``collect_meta=False`` path -- the
        only difference is the extra bookkeeping.
        """
        self.pooler.copy_(pooler)
        self.valid_lengths.copy_(valid_lengths)
        self.frame_idx.zero_()
        self.last_token.fill_(self.blank_id)
        self.h_buf.zero_()
        self.c_buf.zero_()
        self.cc_buf.zero_()
        self.output.fill_(self.pad_id)
        self.output[:, 0] = self.blank_id

        B = self._B
        tok0 = self._step0_eager()
        self.output[:, 1] = tok0
        self.last_token.copy_(tok0)
        out_step = 2
        finished = (self.frame_idx >= self.valid_lengths)

        # meta bookkeeping: leading blank start token (frame 0) + step-0 token
        # (frame = decoder pointer after step 0's duration advance).
        if collect_meta:
            fi_step0 = self.frame_idx.cpu()                # (B,)
            tok0_cpu = tok0.cpu()                           # (B,)
            meta_tokens = [
                [int(self.blank_id), int(tok0_cpu[b].item())] for b in range(B)
            ]
            meta_frames = [
                [0, int(fi_step0[b].item())] for b in range(B)
            ]
            elem_done = [bool(finished[b].item()) for b in range(B)]

        if self._captured and not bool(finished.all()):
            valid_lengths_cpu = self.valid_lengths.cpu()
            for step in range(out_step, self.max_out):
                self.graph.replay()
                info = torch.stack([self.static_token, self.frame_idx], dim=0).cpu()
                tok_cpu = info[0]
                fi_cpu = info[1]
                fin = fi_cpu >= valid_lengths_cpu
                self.output[:, step] = torch.where(
                    fin, torch.full_like(tok_cpu, self.pad_id), tok_cpu
                )
                self.last_token.copy_(
                    torch.where(
                        self.frame_idx >= self.valid_lengths,
                        torch.full_like(self.static_token, self.blank_id),
                        self.static_token,
                    )
                )
                if collect_meta:
                    for b in range(B):
                        if not elem_done[b]:
                            meta_tokens[b].append(int(tok_cpu[b].item()))
                            meta_frames[b].append(int(fi_cpu[b].item()))
                            if bool(fin[b].item()):
                                elem_done[b] = True
                out_step = step + 1
                if bool(fin.all()):
                    break
        if collect_meta:
            return out_step, meta_tokens, meta_frames
        return out_step

    @torch.inference_mode()
    def decode(self, pooler: torch.Tensor, valid_lengths: torch.Tensor,
               processor) -> list[str]:
        """Decode one (already-encoded) batch; returns ``B`` text strings."""
        B = self._B
        T_enc = self._T_enc
        assert pooler.shape == (B, T_enc, self.hid), (
            f"pooler {tuple(pooler.shape)} != captured {(B, T_enc, self.hid)}; "
            "re-capture for this shape"
        )
        out_step = self._run_loop(pooler, valid_lengths)
        out_lists = [self.output[b, :out_step].tolist() for b in range(B)]
        return processor.batch_decode(out_lists, skip_special_tokens=True)

    @torch.inference_mode()
    def decode_with_durations(
        self,
        pooler: torch.Tensor,
        valid_lengths: torch.Tensor,
        processor,
    ) -> tuple[list[str], list[list[int]], list[list[int]]]:
        """Decode one batch AND return per-token cumulative encoder-frame positions.

        Identical to :meth:`decode` (same emitted token ids, same ``out_step``,
        byte-identical text via ``processor.batch_decode``), but additionally
        returns the raw per-step metadata needed for frame-aligned chunk
        stitching (see :mod:`chunking`):

        Returns ``(texts, meta_tokens, meta_frames)`` where for each batch
        element ``b``:

        * ``meta_tokens[b]`` -- the emitted token ids **in emission order**,
          including the leading blank start token and any mid-sequence blanks.
          (Pad padding and trailing special tokens are NOT included; they are
          irrelevant for stitching. ``processor.batch_decode`` with
          ``skip_special_tokens=True`` strips blanks, so the decoded ``text`` is
          unaffected by their presence.)
        * ``meta_frames[b]`` -- the matching **cumulative local encoder-frame
          index** of the decoder pointer *after* emitting each token (i.e. the
          running sum of that token's and all prior tokens' TDT durations). The
          first entry (leading blank) is ``0``; subsequent entries are
          non-decreasing (TDT durations are in ``[0, 1, 2, 3, 4]`` with a forced
          ``>=1`` advance on blank-with-duration-0).

        Because the encoder-frame index is the absolute position of each token
        *within the chunk*, a caller that knows the chunk's audio start sample
        can convert these to global positions and dedup overlap regions between
        adjacent chunks. ``texts`` is byte-identical to :meth:`decode`.
        """
        B = self._B
        T_enc = self._T_enc
        assert pooler.shape == (B, T_enc, self.hid), (
            f"pooler {tuple(pooler.shape)} != captured {(B, T_enc, self.hid)}; "
            "re-capture for this shape"
        )
        out_step, meta_tokens, meta_frames = self._run_loop(
            pooler, valid_lengths, collect_meta=True
        )
        out_lists = [self.output[b, :out_step].tolist() for b in range(B)]
        texts = processor.batch_decode(out_lists, skip_special_tokens=True)
        return texts, meta_tokens, meta_frames


def greedy_decode_graphed(
    model,
    input_features: torch.Tensor,
    attention_mask: torch.Tensor,
    processor,
    *,
    warmup_iters: int = 4,
) -> list[str]:
    """CUDA-graph-captured greedy TDT decode (byte-exact with eager / stock).

    Convenience wrapper: precompute encoder features, capture the graph, decode.
    For repeated decodes of the same shape, reuse a :class:`GraphedDecoder`
    directly so the one-off capture cost is amortised.

    Args:
        model: a loaded ``ParakeetForTDT`` on cuda (eval mode, bf16).
        input_features: ``(B, T_mel, 128)`` mel features on cuda.
        attention_mask: ``(B, T_mel)`` feature attention mask on cuda.
        processor: the matching ``AutoProcessor`` (for ``batch_decode``).
        warmup_iters: side-stream warmup iterations before graph capture.

    Returns:
        list of ``B`` decoded text strings (``skip_special_tokens=True``).
    """
    pad_id = processor.tokenizer.pad_token_id
    with torch.inference_mode():
        enc = model.get_audio_features(
            input_features=input_features, attention_mask=attention_mask
        )
        pooler = enc.pooler_output.contiguous()
        valid_lengths = enc.attention_mask.to(torch.long).sum(-1).contiguous()
    gd = GraphedDecoder(model, warmup_iters=warmup_iters)
    gd.capture(pooler, valid_lengths, pad_id)
    return gd.decode(pooler, valid_lengths, processor)
