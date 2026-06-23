"""Hand-rolled eager greedy TDT decode for nvidia/parakeet-tdt-0.6b-v3.

Implements the per-step TDT decode loop EXACTLY as distilled in
``ALGORITHM.md`` (the verified spec). This is the correctness reference and the
launch-pad for the CUDA-graph-captured decoder in :mod:`decode_mega`.

The loop is the model's own components (``get_audio_features`` + the model's own
``decoder`` with its built-in blank-skip cache + the model's own ``joint``), so
output is byte-exact with the stock ``model.generate`` greedy path. What we drop
is the ``generation_utils`` machinery (sampling/scoring/beam bookkeeping) that
this deterministic greedy TDT path does not need.

Algorithm summary (see ALGORITHM.md for the full derivation):
  1. precompute encoder features once (``get_audio_features``)
  2. per step: gather the current encoder frame, run the decoder on the last
     token (blank-skip freeze is built into the model's decoder/cache), joint,
     argmax token + duration, apply the TDT frame-advance rule
     (``blank & dur==0 -> dur=1``).
  3. stop per element when ``frame_idx >= valid_lengths``.
"""

from __future__ import annotations

import torch

# Architecture constants (parakeet-tdt-0.6b-v3, verified empirically).
_JOINT_OUT = 8198  # = vocab_size (8193) + num_durations (5)


def greedy_decode(
    model,
    input_features: torch.Tensor,
    attention_mask: torch.Tensor,
    processor,
) -> list[str]:
    """Eager greedy TDT decode (byte-exact with stock generate).

    Args:
        model: a loaded ``ParakeetForTDT`` on cuda (eval mode, bf16).
        input_features: ``(B, T_mel, 128)`` mel features on cuda.
        attention_mask: ``(B, T_mel)`` feature attention mask on cuda.
        processor: the matching ``AutoProcessor`` (for ``batch_decode``).

    Returns:
        list of ``B`` decoded text strings (``skip_special_tokens=True``).
    """
    # Import lazily so importing this module never pays the HF import cost.
    from transformers.models.parakeet.generation_parakeet import (
        ParakeetRNNTDecoderCache,
    )

    cfg = model.config
    blank_id = int(cfg.blank_token_id)            # 8192
    vocab_size = int(cfg.vocab_size)              # 8193
    durations = torch.tensor(
        list(cfg.durations), device=input_features.device, dtype=torch.long
    )                                              # [0,1,2,3,4]
    max_symbols = int(cfg.max_symbols_per_step)   # 10
    pad_id = processor.tokenizer.pad_token_id

    B = input_features.shape[0]
    device = input_features.device

    with torch.inference_mode():
        # ---- 1. precompute encoder features ONCE ----
        enc = model.get_audio_features(
            input_features=input_features, attention_mask=attention_mask
        )
        pooler = enc.pooler_output                # (B, T_enc, 640)
        # enc.attention_mask is int32; sum gives the per-utterance valid frame count
        valid_lengths = enc.attention_mask.to(torch.long).sum(-1)  # (B,)
        T_enc = pooler.shape[1]

        # Output buffer: stock path sizes ~ max_symbols_per_step * T_enc; add a
        # small margin. The finished mask stops us well before this in practice.
        max_out = max_symbols * T_enc + 16

        # ---- 2. fresh decoder cache (model's own; blank-skip is built in) ----
        dec_cache = ParakeetRNNTDecoderCache(config=cfg)
        arange = torch.arange(B, device=device)
        frame_idx = torch.zeros((B,), dtype=torch.long, device=device)
        last_token = torch.full((B,), blank_id, dtype=torch.long, device=device)
        finished = torch.zeros((B,), dtype=torch.bool, device=device)
        # The stock path prepends decoder_start_token_id (== blank); col 0 is it.
        output = torch.full((B, max_out), pad_id, dtype=torch.long, device=device)
        output[:, 0] = blank_id
        out_ptr = torch.ones((B,), dtype=torch.long, device=device)

        for _ in range(1, max_out):
            if bool(finished.all()):
                break

            # (1) gather current encoder frame (clamp guards past-end reads)
            idx = frame_idx.clamp(max=T_enc - 1)
            enc_frame = pooler[arange, idx]                 # (B, 640)

            # (2) decoder on last token (blank-skip freeze is internal)
            decoder_input = last_token.unsqueeze(1)         # (B, 1)
            decoder_out = model.decoder(decoder_input, cache=dec_cache)  # (B,1,640)

            # (3) joint -> combined logits (token[:8193] | dur[8193:])
            logits = model.joint(
                encoder_hidden_states=enc_frame[:, None, None, :],
                decoder_hidden_states=decoder_out[:, None, :, :],
            ).squeeze(1).squeeze(1)                          # (B, 8198)

            # (4) greedy token + duration index
            token = logits[:, :vocab_size].argmax(dim=-1)    # (B,)
            dur_idx = logits[:, vocab_size:].argmax(dim=-1)  # (B,)
            dur = durations[dur_idx]                         # (B,)

            # (5) TDT frame-advance: blank & dur==0 -> force dur=1
            blank_mask = token == blank_id
            dur = torch.where(
                blank_mask & (dur == 0), torch.ones_like(dur), dur
            )
            frame_idx = frame_idx + dur

            # (6) emit token for unfinished elements; advance output pointer
            emit_rows = (~finished).nonzero(as_tuple=False).squeeze(-1)
            if emit_rows.numel() > 0:
                output[emit_rows, out_ptr[emit_rows]] = token[emit_rows]
                out_ptr[emit_rows] += 1

            # (7) next-step last_token: finished -> blank (freeze), else token
            last_token = torch.where(
                finished,
                torch.full_like(last_token, blank_id),
                token,
            )
            # (8) update finished mask
            finished = finished | (frame_idx >= valid_lengths)

    # Trim each row to its real emitted length and decode.
    out_lists = [output[b, : int(out_ptr[b].item())].tolist() for b in range(B)]
    return processor.batch_decode(out_lists, skip_special_tokens=True)
