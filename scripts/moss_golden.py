"""Capture golden references for MOSS-Transcribe on the short/medium/long fixtures.

Runs the stock audio encoder + adapter + a manual eager greedy decode (the HF
``generate`` path is broken on this transformers dev build's strict kwarg
validation, but the manual loop calls the identical model modules, so the
output IS the byte-exact stock reference).
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

    wav, sr = sf.read(str(path))
    if sr != 16000:
        import librosa

        wav = librosa.resample(wav.astype("float32"), orig_sr=sr, target_sr=16000)
        sr = 16000
    if wav.ndim > 1:
        wav = wav.mean(1)
    return torch.from_numpy(wav.astype("float32")), sr


def main() -> int:
    from starling.moss.loader import load_model_and_processor
    from starling.moss.reference import audio_features, build_inputs_embeds, greedy_generate

    print("[golden] loading model ...")
    model, proc = load_model_and_processor()

    gdir = REPO / "golden"
    gdir.mkdir(exist_ok=True)

    for name in ("short", "medium", "long"):
        wav, sr = load_wav(REPO / "tests" / "fixtures" / f"{name}.wav")
        seconds = wav.shape[0] / sr
        inp = proc(wav.numpy())
        inp = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in inp.items()}
        print(f"\n[golden] {name}: {seconds:.2f}s, input_ids {inp['input_ids'].shape}, "
              f"audio {inp['audio_data'].shape}, {int(inp['audio_input_mask'].sum())} audio tokens")

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            feats = audio_features(model, inp["audio_data"], inp["audio_data_seqlens"])
            emb = build_inputs_embeds(
                model, inp["input_ids"], feats, inp["audio_input_mask"]
            )
            ids = greedy_generate(model, emb, max_new_tokens=200, max_cache_len=2048)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        text = proc.tokenizer.decode(ids[0], skip_special_tokens=True)
        print(f"[golden] {name}: {ids.shape[1]} tokens in {ms:.0f}ms "
              f"({ids.shape[1]/(ms/1000):.0f} tok/s, RTFx {seconds/(ms/1000):.1f}x)")
        print(f"[golden] transcript: {text[:200]}")
        torch.save(ids.cpu(), gdir / f"moss_{name}_ids.pt")
        (gdir / f"moss_{name}_text.txt").write_text(text)
    print(f"\n[golden] saved -> {gdir}/moss_*_ids.pt + .txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
