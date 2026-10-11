"""Capture golden references for nvidia/parakeet-tdt-0.6b-v3 on fixtures.

Runs the byte-exact eager greedy TDT decode (:func:`greedy_decode` below,
which is byte-exact with stock ``model.generate``) on
``tests/fixtures/{short,medium,long}.wav`` and persists:

  * ``parakeet_tdt_{name}_ids.pt``  -- raw emitted token id stream (with blanks)
  * ``parakeet_tdt_{name}_text.txt``-- the detokenized transcript text
  * ``parakeet_tdt_{name}_meta.pt`` -- shape/length metadata
  * ``parakeet_tdt_{name}_mel.pt``  -- the (T_mel,128) bf16 mel input features
  * ``parakeet_tdt_{name}_enc.pt``  -- the (T_enc,640) projected encoder output

The mel/enc intermediates let the ggml port validate COMPONENT-by-COMPONENT
(mel extractor parity, then encoder parity, then full TDT decode parity) instead
of only end-to-end text. The token-id stream is the strict gate (greedy TDT is
deterministic); the text is the loose gate (matches the existing oracle contract).

Correctness basis
-----------------
The eager greedy loop (:func:`greedy_decode`) calls the model's own
``get_audio_features`` + ``decoder`` (with its built-in blank-skip cache) +
``joint`` -- the identical modules stock ``model.generate`` runs -- and follows
the TDT frame-advance rule (``blank & dur==0 -> dur=1``). It is byte-exact with
stock generate's emitted token stream.

Usage:  uv run --group convert python scripts/goldens/parakeet_tdt_golden.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[2]


def load_wav(path: Path) -> np.ndarray:
    wav, sr = sf.read(str(path))
    if sr != 16000:
        import librosa

        wav = librosa.resample(wav.astype("float32"), orig_sr=sr, target_sr=16000)
    if wav.ndim > 1:
        wav = wav.mean(1)
    return np.ascontiguousarray(wav, dtype=np.float32)


def greedy_decode(
    model,
    input_features: torch.Tensor,
    attention_mask: torch.Tensor,
    processor,
) -> list[str]:
    """Eager greedy TDT decode (byte-exact with stock generate).

    Per step: gather the current encoder frame, run the model's decoder on
    the last token (blank-skip freeze is built into its cache), joint, argmax
    token + duration, apply the TDT frame-advance rule; stop per element
    when ``frame_idx >= valid_lengths``.

    Args:
        model: a loaded ``ParakeetForTDT`` (eval mode, bf16).
        input_features: ``(B, T_mel, 128)`` mel features.
        attention_mask: ``(B, T_mel)`` feature attention mask.
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


def main() -> int:
    from transformers import AutoModelForTDT, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16  # production numerics; matches the starling path
    model_id = "nvidia/parakeet-tdt-0.6b-v3"

    print(f"[golden] loading {model_id} ...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForTDT.from_pretrained(model_id, dtype=dtype, device_map=device)
    model.eval()
    print("[golden] models ready")

    gdir = REPO / "golden"
    gdir.mkdir(exist_ok=True)

    for name in ("short", "medium", "long"):
        audio = load_wav(REPO / "tests" / "fixtures" / f"{name}.wav")
        seconds = audio.shape[0] / 16000

        # processor: CPU mel -> bf16 features + bool attention mask (the stock path)
        inp = processor([audio], sampling_rate=16000)
        inp = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inp.items()}
        inp["input_features"] = inp["input_features"].to(dtype)
        t_mel = int(inp["input_features"].shape[1])

        torch.cuda.synchronize() if device == "cuda" else None
        t0 = time.perf_counter()
        with torch.inference_mode():
            texts = greedy_decode(
                model,
                input_features=inp["input_features"],
                attention_mask=inp.get("attention_mask"),
                processor=processor,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        text = texts[0]

        # Capture the encoder output (pooler_output, projected to 640) for
        # component-level parity: the ggml port must reproduce it before the TDT
        # decode loop. Run it once more under inference_mode (cheap vs the decode).
        with torch.inference_mode():
            enc = model.get_audio_features(
                input_features=inp["input_features"],
                attention_mask=inp.get("attention_mask"),
            )
            enc_out = enc.pooler_output[0].contiguous()        # (T_enc, 640)
            enc_mask = enc.attention_mask[0].contiguous()      # (T_enc,)
            valid_len = int(enc_mask.to(torch.long).sum().item())

        # Capture the greedy token stream by re-running greedy_decode but asking
        # for the raw ids. greedy_decode returns text; rebuild the id stream by
        # re-decoding via the stock path ONCE (deterministic) so we store the
        # exact emitted ids including blanks. Use model.generate for the id
        # stream (it is byte-exact with greedy_decode; both run the same TDT
        # greedy math). Keep it small: B=1.
        with torch.inference_mode():
            gen_out = model.generate(**inp, return_dict_in_generate=True)
            ids = gen_out.sequences[0].contiguous()            # (T_out,) incl. blank

        mel_out = inp["input_features"][0].contiguous().cpu()  # (T_mel,128) bf16
        mel_mask = (
            inp["attention_mask"][0].contiguous().cpu()
            if "attention_mask" in inp
            else torch.ones(t_mel, dtype=torch.bool)
        )

        torch.save(ids.cpu(), gdir / f"parakeet_tdt_{name}_ids.pt")
        torch.save(
            {
                "audio_seconds": float(seconds),
                "t_mel": t_mel,
                "t_enc": int(enc_out.shape[0]),
                "valid_len": valid_len,
                "n_emitted": int(ids.numel()),
                "vocab_size": int(model.config.vocab_size),
                "blank_id": int(model.config.blank_token_id),
                "durations": list(model.config.durations),
                "max_symbols_per_step": int(model.config.max_symbols_per_step),
            },
            gdir / f"parakeet_tdt_{name}_meta.pt",
        )
        torch.save(mel_out, gdir / f"parakeet_tdt_{name}_mel.pt")
        torch.save(mel_mask, gdir / f"parakeet_tdt_{name}_mel_mask.pt")
        torch.save(enc_out.cpu(), gdir / f"parakeet_tdt_{name}_enc.pt")
        (gdir / f"parakeet_tdt_{name}_text.txt").write_text(text)
        print(
            f"[golden] {name}: {seconds:.2f}s audio, mel {t_mel} frames, "
            f"enc {tuple(enc_out.shape)} (valid {valid_len}), "
            f"{int(ids.numel())} emitted tokens in {ms:.0f}ms"
        )
        print(f"[golden]   text: {text[:160]!r}")
        print(f"[golden]   ids[:12]: {ids[:12].tolist()}")
    print(f"\n[golden] saved -> {gdir}/parakeet_tdt_*_ids.pt + _text.txt + _meta.pt + _mel.pt + _enc.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
