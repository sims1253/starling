"""WER sweep over quantized parakeet-tdt GGUFs (see docs/quantization.md).

Runs the in-tree starling-ggml engine (libstarling_ggml via ctypes) on the
deterministic LibriSpeech fixtures and reports WER/CER per tier against the
ground-truth transcripts in ``benchmarks/wer.py``. Each ``label=path`` pair
is evaluated in-process (the engine re-reads ``STARLING_GGML_PARAKEET_MODEL``
on every load).

Usage::

    uv run python benchmarks/wer_quant.py \\
        --models f32=models/parakeet-tdt-0.6b-v3-f32.gguf \\
                 q8_0=models/parakeet-tdt-0.6b-v3-q8_0.gguf \\
                 q4_k_m=models/parakeet-tdt-0.6b-v3-q4_k_m.gguf \\
        --tiers short,medium,long --json /tmp/wer_quant.json

The fixtures repeat one LibriSpeech utterance, so absolute WER here is not a
leaderboard number; the point is the DELTA between the f32 baseline and each
quantization (and catching gross breakage, e.g. a mis-quantized tensor class).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests" / "fixtures"))

import make_fixtures as mkfx  # noqa: E402

from engines import StarlingGgmlParakeet  # noqa: E402
from wer import REFERENCE_TRANSCRIPTS, cer_pct, wer_pct  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", required=True,
                    help="label=path pairs (or plain paths, labelled by stem)")
    ap.add_argument("--tiers", default="short,medium,long")
    ap.add_argument("--snr-db", type=float, default=None,
                    help="add deterministic gaussian noise at this SNR to "
                         "every tier (harder inputs discriminate quant levels; "
                         "the reference text is unchanged)")
    ap.add_argument("--json", default=None, help="optional JSON output path")
    args = ap.parse_args()

    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    fixtures = mkfx.load_fixtures()
    if args.snr_db is not None:
        import numpy as np
        rng = np.random.default_rng(0)

        def noised(audio: "np.ndarray") -> "np.ndarray":
            signal = float(np.sqrt(np.mean(audio ** 2)))
            noise = rng.standard_normal(audio.shape).astype(np.float32)
            noise *= signal / (10 ** (args.snr_db / 20))
            return np.clip(audio + noise, -1.0, 1.0).astype(np.float32)
    else:
        def noised(audio):
            return audio

    rows = []
    for spec in args.models:
        label, sep, path = spec.partition("=")
        if not sep:
            label, path = Path(spec).stem, spec
        p = Path(path).expanduser()
        if not p.exists():
            print(f"[skip] {label}: {p} does not exist")
            continue

        os.environ["STARLING_GGML_PARAKEET_MODEL"] = str(p.resolve())
        eng = StarlingGgmlParakeet()
        if not eng.available:
            print(f"[skip] {label}: engine unavailable "
                  f"(build/libstarling_ggml.so missing or model rejected)")
            continue

        row = {"model": label, "path": str(p), "mb": round(p.stat().st_size / 1e6, 1),
               "wer": {}, "cer": {}}
        try:
            eng.load()
            for tier in tiers:
                hyp = eng.transcribe(noised(fixtures[tier]))[0]
                ref = REFERENCE_TRANSCRIPTS[tier]
                row["wer"][tier] = round(wer_pct(ref, hyp), 2)
                row["cer"][tier] = round(cer_pct(ref, hyp), 2)
        finally:
            eng.close()
        rows.append(row)
        print(f"[done] {label}: " + " ".join(f"wer_{t}={row['wer'][t]}" for t in tiers))

    if not rows:
        print("no models evaluated")
        return 1

    hdr = f"{'model':<16} {'MB':>7} " + " ".join(f"{'wer_' + t:>10}" for t in tiers)
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['model']:<16} {r['mb']:>7.1f} "
              + " ".join(f"{r['wer'][t]:>10.2f}" for t in tiers))

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
