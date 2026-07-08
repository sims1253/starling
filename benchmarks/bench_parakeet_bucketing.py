"""Regression bench for the parakeet shape-bucketing RTFx fix.

Measures per-clip latency for DIVERSE clip lengths (the leaderboard scenario,
where each clip is a different length -> without bucketing, per-shape graph
re-capture) and compares ``shape_bucketing`` ON vs OFF. Also verifies
text-byte-exactness: each fixture's bucketed transcript must equal its
unbucketed transcript.

Expected on the RTX 5090: bucketing ON recovers ~20x on the diverse-length
scenario (the unbucketed leaderboard collapse), while keeping text byte-exact.

  uv run python benchmarks/bench_parakeet_bucketing.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from starling.parakeet.gpu_lock import with_gpu_lock  # noqa: E402

FX_ROOT = REPO if (REPO / "tests" / "fixtures" / "short.wav").exists() else Path(
    "/home/m0hawk/Documents/starling"
)


def _synth_clip(seconds: float, sr: int = 16000) -> np.ndarray:
    n = int(seconds * sr)
    t = np.arange(n, dtype=np.float32) / sr
    sig = 0.08 * (np.sin(2 * np.pi * 220.0 * t) + 0.5 * np.sin(2 * np.pi * 553.0 * t))
    return np.ascontiguousarray(sig, dtype=np.float32)


def _load_wav(name: str):
    import soundfile as sf

    wav, sr = sf.read(str(FX_ROOT / "tests" / "fixtures" / f"{name}.wav"))
    if wav.ndim > 1:
        wav = wav.mean(1)
    return np.ascontiguousarray(wav.astype(np.float32)), sr


def _time_one(pipe, audio):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    texts, timing = pipe.transcribe_with_timing([audio])
    torch.cuda.synchronize()
    total = (time.perf_counter() - t0) * 1000.0
    return texts[0], total, timing


def _run_scenario(pipe, clips, label):
    print(f"\n=== {label} ===", flush=True)
    print(f"{'i':>3} {'dur_s':>6} {'total_ms':>9} "
          f"{'mel_ms':>7} {'enc_ms':>7} {'dec_ms':>7}", flush=True)
    totals, texts = [], []
    for i, audio in enumerate(clips):
        text, total, timing = _time_one(pipe, audio)
        totals.append(total)
        texts.append(text)
        print(f"{i:>3} {len(audio)/16000:>6.1f} {total:>9.1f} "
              f"{timing['mel_ms']:>7.1f} {timing['encoder_ms']:>7.1f} "
              f"{timing['decode_ms']:>7.1f}", flush=True)
    med = float(np.median(totals))
    print(f"[{label}] median={med:.1f}ms mean={np.mean(totals):.1f}ms", flush=True)
    return texts, med


def main() -> int:
    from starling.parakeet.pipeline import MegaParakeetPipeline

    diverse_secs = [2.1, 3.7, 5.3, 7.9, 9.4, 11.8, 14.2, 18.6, 23.1, 29.5]
    diverse_clips = [_synth_clip(s) for s in diverse_secs]

    # Also load real fixtures to check text-byte-exactness bucketed vs unbucketed.
    real_clips = {n: _load_wav(n)[0] for n in ["short", "medium", "long"]}

    with with_gpu_lock(
        session="parakeet-profile", model="parakeet-tdt-0.6b-v3",
        eta_min=8, note="parakeet bucketing profile",
    ):
        # ---- bucketing OFF (baseline: the current leaderboard behaviour) --------
        print("[profile] loading pipeline (shape_bucketing=False) ...", flush=True)
        pipe_off = MegaParakeetPipeline(shape_bucketing=False)
        texts_off_div, med_off_div = _run_scenario(
            pipe_off, diverse_clips, "DIVERSE, bucketing OFF"
        )
        texts_off_rep, _ = _run_scenario(
            pipe_off, [diverse_clips[5]] * len(diverse_clips), "REPEATED, bucketing OFF"
        )
        texts_off_real = {n: pipe_off.transcribe([a])[0] for n, a in real_clips.items()}
        del pipe_off
        torch.cuda.empty_cache()

        # ---- bucketing ON (the fix) ---------------------------------------------
        print("\n[profile] loading pipeline (shape_bucketing=True, default) ...", flush=True)
        pipe_on = MegaParakeetPipeline()  # bucketing is the default
        print(f"[profile] mel_bucket_frames={pipe_on.mel_bucket_frames}", flush=True)
        texts_on_div, med_on_div = _run_scenario(
            pipe_on, diverse_clips, "DIVERSE, bucketing ON"
        )
        texts_on_real = {n: pipe_on.transcribe([a])[0] for n, a in real_clips.items()}

        # ---- text-byte-exactness check ------------------------------------------
        print("\n=== Text byte-exactness (bucketing ON vs OFF) ===", flush=True)
        all_match = True
        for n in ["short", "medium", "long"]:
            match = texts_on_real[n] == texts_off_real[n]
            all_match = all_match and match
            print(f"  {n:>6}: {'MATCH' if match else 'DIFFER'}", flush=True)
            if not match:
                print(f"    off: {texts_off_real[n]!r}", flush=True)
                print(f"    on : {texts_on_real[n]!r}", flush=True)

        # ---- summary -------------------------------------------------------------
        print("\n=== SUMMARY ===", flush=True)
        print(f"diverse median (bucketing OFF): {med_off_div:8.1f} ms", flush=True)
        print(f"diverse median (bucketing ON ): {med_on_div:8.1f} ms", flush=True)
        print(f"speedup on diverse lengths    : {med_off_div/max(med_on_div,1e-6):7.1f}x", flush=True)
        print(f"text byte-exact ON vs OFF     : {'YES (all 3 fixtures)' if all_match else 'NO'}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
