"""CUDA-graph-captured greedy TDT decode for nvidia/parakeet-tdt-0.6b-v3.

Same I/O and byte-exact output as :mod:`decode_eager`, but the per-step compute
(``ALGORITHM.md`` steps 1-5) is captured into a single ``torch.cuda.CUDAGraph``
and served by ``graph.replay()`` each step. The stock decode loop is ~80% wall
and launch-bound (~10% GPU-busy, see ``outputs/profile_analysis.md``); replaying
one graph per step collapses the hundreds of tiny per-step kernel launches into
a single replay, removing the launch overhead.

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

    if cache.is_initialized and blank_mask.all():   # <-- blank_mask.all() is a
        return cache.cache                          #     device tensor used as a
                                                    #     Python bool -> host sync
    ...                                             #     -> aborts stream capture

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


def greedy_decode_graphed(
    model,
    input_features: torch.Tensor,
    attention_mask: torch.Tensor,
    processor,
    *,
    warmup_iters: int = 4,
) -> list[str]:
    """CUDA-graph-captured greedy TDT decode (byte-exact with eager / stock).

    Args:
        model: a loaded ``ParakeetForTDT`` on cuda (eval mode, bf16).
        input_features: ``(B, T_mel, 128)`` mel features on cuda.
        attention_mask: ``(B, T_mel)`` feature attention mask on cuda.
        processor: the matching ``AutoProcessor`` (for ``batch_decode``).
        warmup_iters: side-stream warmup iterations before graph capture.

    Returns:
        list of ``B`` decoded text strings (``skip_special_tokens=True``).
    """
    cfg = model.config
    blank_id = int(cfg.blank_token_id)
    vocab_size = int(cfg.vocab_size)
    max_symbols = int(cfg.max_symbols_per_step)
    hid = int(cfg.decoder_hidden_size)        # 640
    nl = int(cfg.num_decoder_layers)          # 2
    pad_id = processor.tokenizer.pad_token_id

    B = input_features.shape[0]
    device = input_features.device
    dec = model.decoder
    joint = model.joint

    with torch.inference_mode():
        # ---- 1. precompute encoder features ONCE ----
        enc = model.get_audio_features(
            input_features=input_features, attention_mask=attention_mask
        )
        pooler = enc.pooler_output.contiguous()              # (B, T_enc, 640)
        valid_lengths = enc.attention_mask.to(torch.long).sum(-1).contiguous()
        T_enc = pooler.shape[1]
        max_out = max_symbols * T_enc + 16

        # ---- 2. static buffers (fixed addresses for the graph) ----
        frame_idx = torch.zeros((B,), dtype=torch.long, device=device)
        last_token = torch.full((B,), blank_id, dtype=torch.long, device=device)
        static_token = torch.zeros((B,), dtype=torch.long, device=device)
        arange_B = torch.arange(B, device=device)
        h_buf = torch.zeros((nl, B, hid), dtype=torch.bfloat16, device=device)
        c_buf = torch.zeros((nl, B, hid), dtype=torch.bfloat16, device=device)
        cc_buf = torch.zeros((B, 1, hid), dtype=torch.bfloat16, device=device)
        ones_b = torch.ones((B,), dtype=torch.long, device=device)
        dur_table = torch.tensor(list(cfg.durations), device=device, dtype=torch.long)
        output = torch.full((B, max_out), pad_id, dtype=torch.long, device=device)
        output[:, 0] = blank_id
        _mark_many([
            pooler, valid_lengths, frame_idx, last_token, static_token, arange_B,
            h_buf, c_buf, cc_buf, ones_b, dur_table, output,
        ])

        # ---- 3. the captured per-step compute (manual decoder; no host branch) ----
        def step_fn():
            # decoder: embedding -> lstm -> projector + device-side blank-skip freeze
            lt = last_token.unsqueeze(1)                          # (B,1)
            emb = dec.embedding(lt)                               # (B,1,640)
            lstm_out, (hn, cn) = dec.lstm(emb, (h_buf, c_buf))    # (B,1,640),(nl,B,640)
            proj = dec.decoder_projector(lstm_out)                # (B,1,640)
            advance = (last_token != blank_id)                    # (B,) True=advance
            adv_out = advance.view(B, 1, 1)
            adv_h = advance.view(1, B, 1)
            decoder_out = torch.where(adv_out, proj, cc_buf)      # freeze blank elems
            h_new = torch.where(adv_h, hn, h_buf)
            c_new = torch.where(adv_h, cn, c_buf)
            # joint -> combined logits (token[:8193] | dur[8193:])
            idx = frame_idx.clamp(max=T_enc - 1)
            enc_frame = pooler[arange_B, idx]                     # (B,640)
            logits = joint(
                encoder_hidden_states=enc_frame[:, None, None, :],
                decoder_hidden_states=decoder_out[:, None, :, :],
            ).squeeze(1).squeeze(1)                               # (B,8198)
            tok = logits[:, :vocab_size].argmax(dim=-1)
            dur_idx = logits[:, vocab_size:].argmax(dim=-1)
            dur = dur_table[dur_idx]
            blank_mask = (tok == blank_id)
            dur = torch.where(blank_mask & (dur == 0), ones_b, dur)
            frame_idx.add_(dur)
            # write back cache state (in place on the static buffers)
            h_buf.copy_(h_new)
            c_buf.copy_(c_new)
            cc_buf.copy_(decoder_out)
            static_token.copy_(tok)

        # ---- 4. step 0 EAGER (init path: zero cache, NO blank-skip freeze) ----
        lt = last_token.unsqueeze(1)
        emb = dec.embedding(lt)
        lstm_out, (hn, cn) = dec.lstm(emb, (h_buf, c_buf))
        proj = dec.decoder_projector(lstm_out)
        decoder_out0 = proj                                       # no freeze at step 0
        idx = frame_idx.clamp(max=T_enc - 1)
        enc_frame = pooler[arange_B, idx]
        logits = joint(
            encoder_hidden_states=enc_frame[:, None, None, :],
            decoder_hidden_states=decoder_out0[:, None, :, :],
        ).squeeze(1).squeeze(1)
        tok0 = logits[:, :vocab_size].argmax(dim=-1)
        dur_idx = logits[:, vocab_size:].argmax(dim=-1)
        dur = dur_table[dur_idx]
        bm = (tok0 == blank_id)
        dur = torch.where(bm & (dur == 0), ones_b, dur)
        frame_idx.add_(dur)
        h_buf.copy_(hn)
        c_buf.copy_(cn)
        cc_buf.copy_(decoder_out0)
        static_token.copy_(tok0)
        output[:, 1] = tok0
        last_token.copy_(tok0)
        out_step = 2
        finished = (frame_idx >= valid_lengths)

        if bool(finished.all()):
            # utterance exhausted in a single step (very short audio)
            graph = None
        else:
            # ---- 5. save the post-step-0 reset point ----
            h_s, c_s, cc_s = h_buf.clone(), c_buf.clone(), cc_buf.clone()
            fi_s, lt_s = frame_idx.clone(), last_token.clone()

            def _reset():
                h_buf.copy_(h_s); c_buf.copy_(c_s); cc_buf.copy_(cc_s)
                frame_idx.copy_(fi_s); last_token.copy_(lt_s)

            # ---- 6. warmup on a side stream (stabilises cudnn/cublas autotune) ----
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(warmup_iters):
                    step_fn()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            _reset()

            # ---- 7. capture the per-step graph ----
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                step_fn()
            _reset()  # capture mutated the buffers; restore for the real loop

        # ---- 8. host replay loop (steps 1+) ----
        if graph is not None:
            valid_lengths_cpu = valid_lengths.cpu()
            for step in range(out_step, max_out):
                graph.replay()
                # one device->host sync: read emitted token + new frame_idx
                info = torch.stack([static_token, frame_idx], dim=0).cpu()
                tok_cpu = info[0]
                fi_cpu = info[1]
                fin = fi_cpu >= valid_lengths_cpu
                output[:, step] = torch.where(
                    fin, torch.full_like(tok_cpu, pad_id), tok_cpu
                )
                # next-step last_token: finished -> blank (freeze), else token
                last_token.copy_(
                    torch.where(
                        frame_idx >= valid_lengths,
                        torch.full_like(static_token, blank_id),
                        static_token,
                    )
                )
                out_step = step + 1
                if bool(fin.all()):
                    break

    out_lists = [output[b, :out_step].tolist() for b in range(B)]
    return processor.batch_decode(out_lists, skip_special_tokens=True)
