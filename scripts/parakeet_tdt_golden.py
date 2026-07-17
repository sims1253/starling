"""Capture golden references for nvidia/parakeet-tdt-0.6b-v3 on fixtures.

Runs the byte-exact eager greedy TDT decode (:mod:`starling.parakeet.decode_eager`,
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
The eager greedy loop (``decode_eager.greedy_decode``) calls the model's own
``get_audio_features`` + ``decoder`` (with its built-in blank-skip cache) +
``joint`` -- the identical modules stock ``model.generate`` runs -- and follows
the verified TDT frame-advance rule (``ALGORITHM.md``). It is byte-exact with
stock generate's emitted token stream; ``scripts/parakeet_unified_golden.py``
uses the same methodology.

Usage:  uv run python scripts/parakeet_tdt_golden.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def load_wav(path: Path) -> np.ndarray:
    wav, sr = sf.read(str(path))
    if sr != 16000:
        import librosa

        wav = librosa.resample(wav.astype("float32"), orig_sr=sr, target_sr=16000)
    if wav.ndim > 1:
        wav = wav.mean(1)
    return np.ascontiguousarray(wav, dtype=np.float32)


def main() -> int:
    from transformers import AutoModelForTDT, AutoProcessor

    from starling.parakeet.decode_eager import greedy_decode

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
