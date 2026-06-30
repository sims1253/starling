"""Capture golden references for cohere-transcribe-03-2026 on short/medium/long fixtures.

Runs the stock encoder + a manual eager greedy decode over an
``EncoderDecoderCache`` (the HF ``generate`` path agrees byte-for-byte; the
manual loop calls the identical model modules — verified in the test suite).

Usage:  uv run python scripts/cohere_golden.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def load_wav(path: Path) -> tuple[torch.Tensor, int]:
    import numpy as np

    wav, sr = sf.read(str(path))
    if sr != 16000:
        import librosa

        wav = librosa.resample(wav.astype("float32"), orig_sr=sr, target_sr=16000)
        sr = 16000
    if wav.ndim > 1:
        wav = wav.mean(1)
    return torch.from_numpy(wav.astype("float32")), sr


def main() -> int:
    from starling.cohere.loader import load_model_and_processor
    from starling.cohere.reference import encode, greedy_generate

    print("[golden] loading model ...")
    model, proc = load_model_and_processor()

    gdir = REPO / "golden"
    gdir.mkdir(exist_ok=True)

    for name in ("short", "medium", "long"):
        wav, sr = load_wav(REPO / "tests" / "fixtures" / f"{name}.wav")
        seconds = wav.shape[0] / sr
        inp = proc(wav.numpy(), sampling_rate=sr, language="en", return_tensors="pt")
        feat = inp["input_features"].to(torch.bfloat16).cuda()
        amask = inp["attention_mask"].cuda()
        dec_in = inp["decoder_input_ids"].cuda()
        print(f"\n[golden] {name}: {seconds:.2f}s, mel {tuple(feat.shape)}, "
              f"dec_in {tuple(dec_in.shape)}")

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            enc_h, enc_mask = encode(model, feat, amask)
            ids = greedy_generate(model, enc_h, enc_mask, dec_in, max_new_tokens=300)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        # decode text: the full sequence (prompt + generated) per row
        full = torch.cat([dec_in.cpu().expand(ids.shape[0], -1), ids.cpu()], dim=1)
        texts = proc.batch_decode(full, skip_special_tokens=True)
        n_gen = ids.shape[1]
        print(f"[golden] {name}: B={ids.shape[0]} x {n_gen} gen tokens in {ms:.0f}ms "
              f"(RTFx {seconds/(ms/1000):.1f}x)")
        for b, t in enumerate(texts):
            print(f"[golden]   row {b}: {t[:160]}")
        text = texts[0] if ids.shape[0] == 1 else " || ".join(texts)
        torch.save(ids.cpu(), gdir / f"cohere_{name}_ids.pt")
        # also save the encoder output shape + dec_in for tests
        torch.save(
            {"enc_h_shape": list(enc_h.shape), "dec_in": dec_in.cpu()},
            gdir / f"cohere_{name}_meta.pt",
        )
        (gdir / f"cohere_{name}_text.txt").write_text(text)
    print(f"\n[golden] saved -> {gdir}/cohere_*_ids.pt + .txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
