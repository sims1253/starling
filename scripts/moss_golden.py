"""Capture golden references for MOSS-Transcribe on the short/medium/long fixtures.

Runs the stock audio encoder + adapter + a manual eager greedy decode (the HF
``generate`` path is broken on this transformers dev build's strict kwarg
validation, but the manual loop calls the identical model modules, so the
output IS the byte-exact stock reference).

The capture device defaults to CUDA when available and falls back to CPU
(``STARLING_GOLDEN_DEVICE=cpu|cuda`` overrides). The chosen device, library
versions, and repository revision are recorded in
``golden/moss_reference_provenance.json`` so a parity run can prove WHICH
reference it was measured against (issue #167: the reference record must be
reproducible, not anecdotal).

Goldens are never regenerated silently: existing files are only overwritten
with ``--force`` (a reference update is a separately reviewed change).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

FIXTURES = ("short", "medium", "long")


def load_wav(path: Path) -> tuple[torch.Tensor, int]:

    wav, sr = sf.read(str(path))
    if sr != 16000:
        import librosa

        wav = librosa.resample(wav.astype("float32"), orig_sr=sr, target_sr=16000)
        sr = 16000
    if wav.ndim > 1:
        wav = wav.mean(1)
    return torch.from_numpy(wav.astype("float32")), sr


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing golden files (default: refuse)")
    args = ap.parse_args()

    from starling.moss.loader import load_model_and_processor
    from starling.moss.reference import audio_features, build_inputs_embeds, greedy_generate

    import os

    env_dev = os.environ.get("STARLING_GOLDEN_DEVICE")
    if env_dev in ("cpu", "cuda"):
        device = env_dev
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    existing = [p for n in FIXTURES for p in
                (REPO / "golden" / f"moss_{n}_ids.pt", REPO / "golden" / f"moss_{n}_text.txt")
                if p.exists()]
    if existing and not args.force:
        print("[golden] REFUSING to overwrite existing goldens (no --force):")
        for p in existing:
            print(f"  {p}")
        return 3

    print(f"[golden] loading model ... (device={device})")
    model, proc = load_model_and_processor(device=device)

    gdir = REPO / "golden"
    gdir.mkdir(exist_ok=True)

    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                             capture_output=True, text=True, check=True
                             ).stdout.strip()

        def _dirty(*paths: str) -> bool:
            return bool(subprocess.run(
                ["git", "status", "--porcelain", *paths], cwd=REPO,
                capture_output=True, text=True, check=True).stdout.strip())

        # The reference MODULE (src/starling/moss) defines the numbers; the
        # capture script only records them. Track the two separately so a
        # dirty capture script is not mistaken for a dirty reference path.
        reference_dirty = _dirty("src/starling/moss")
        script_dirty = _dirty("scripts/moss_golden.py")
    except Exception:
        rev, reference_dirty, script_dirty = "unknown", False, False

    provenance = {
        "model_id": "OpenMOSS-Team/MOSS-Transcribe-preview-2B",
        "capture_script": "scripts/moss_golden.py",
        "reference_path": "starling.moss.reference (eager greedy, exact-width DynamicCache)",
        "device": device,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "repo_revision": rev,
        "reference_module_dirty": reference_dirty,
        "capture_script_dirty": script_dirty,
        "max_new_tokens": 200,
        "max_cache_len": 2048,
        "eos_token_id": 151645,
        "fixtures": {},
    }

    for name in FIXTURES:
        wav, sr = load_wav(REPO / "tests" / "fixtures" / f"{name}.wav")
        seconds = wav.shape[0] / sr
        inp = proc(wav.numpy())
        inp = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inp.items()}
        print(f"\n[golden] {name}: {seconds:.2f}s, input_ids {inp['input_ids'].shape}, "
              f"audio {inp['audio_data'].shape}, {int(inp['audio_input_mask'].sum())} audio tokens")

        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            feats = audio_features(model, inp["audio_data"], inp["audio_data_seqlens"])
            emb = build_inputs_embeds(
                model, inp["input_ids"], feats, inp["audio_input_mask"]
            )
            ids = greedy_generate(model, emb, max_new_tokens=200, max_cache_len=2048)
        if device == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        text = proc.tokenizer.decode(ids[0], skip_special_tokens=True)
        print(f"[golden] {name}: {ids.shape[1]} tokens in {ms:.0f}ms "
              f"({ids.shape[1]/(ms/1000):.0f} tok/s, RTFx {seconds/(ms/1000):.1f}x)")
        print(f"[golden] transcript: {text[:200]}")
        torch.save(ids.cpu(), gdir / f"moss_{name}_ids.pt")
        (gdir / f"moss_{name}_text.txt").write_text(text)
        provenance["fixtures"][name] = {
            "n_tokens": int(ids.shape[1]),
            "ids_sha256": _sha256(gdir / f"moss_{name}_ids.pt"),
            "text_sha256": _sha256(gdir / f"moss_{name}_text.txt"),
        }
    (gdir / "moss_reference_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"\n[golden] saved -> {gdir}/moss_*_ids.pt + .txt + moss_reference_provenance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
