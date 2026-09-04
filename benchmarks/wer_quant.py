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

# The in-process model sweep is CPU-sound only: on GPU builds the K-step
# replay caches outlive eng.close() and would replay against the previous
# model's freed weight buffers. Pin unless the caller overrides.
os.environ.setdefault("STARLING_GGML_DEVICE", "cpu")
sys.path.insert(0, str(REPO_ROOT / "tests" / "fixtures"))

import make_fixtures as mkfx  # noqa: E402

from engines import StarlingGgmlParakeet  # noqa: E402
from wer import REFERENCE_TRANSCRIPTS, cer_pct, wer_pct  # noqa: E402


def _bootstrap_ci(values: list[float], n: int = 1000, seed: int = 0):
    """95% percentile bootstrap CI of the mean (None when too few samples)."""
    if len(values) < 5:
        return None
    import numpy as np
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    means = rng.choice(arr, size=(n, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return round(float(lo), 2), round(float(hi), 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", required=True,
                    help="label=path pairs (or plain paths, labelled by stem)")
    ap.add_argument("--tiers", default="short,medium,long")
    ap.add_argument("--snr-db", type=float, default=None,
                    help="add deterministic gaussian noise at this SNR to "
                         "every tier (harder inputs discriminate quant levels; "
                         "the reference text is unchanged)")
    ap.add_argument("--mls-de", type=int, default=None, metavar="N",
                    help="additionally evaluate N German clips from the MLS "
                         "de_de test split (streamed, human transcripts; "
                         "v3 is multilingual so quant levels must hold DE)")
    ap.add_argument("--fleurs-eval", action="append", default=[], metavar="CFG:N",
                    help="additionally evaluate mean WER over N test clips "
                         "of google/fleurs CFG (repeatable), e.g. de_de:60")
    ap.add_argument("--corpus", action="append", default=[], metavar="DIR",
                    help="evaluate wavs+txt sidecars from a fleurs_download "
                         "corpus dir; one mean-WER column per "
                         "<config>_<split> prefix")
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

    fleurs_evals: dict[str, list[tuple]] = {}
    if args.fleurs_eval:
        from fleurs_util import fleurs_clips
        for spec in args.fleurs_eval:
            cfg, _, n = spec.partition(":")
            fleurs_evals[cfg] = [
                (audio, text) for _, audio, text in fleurs_clips({cfg: int(n)}, split="test")
            ]
            print(f"[fleurs-eval] {cfg}: {len(fleurs_evals[cfg])} test clips")

    corpus_evals: dict[str, list[tuple]] = {}
    if args.corpus:
        import numpy as np
        import soundfile as sf
        for cdir in args.corpus:
            for wav in sorted(Path(cdir).expanduser().glob("*.wav")):
                txt = wav.with_suffix(".txt")
                if not txt.exists():
                    continue
                prefix = wav.stem.rsplit("_", 1)[0]  # <config>_<split>
                audio, sr = sf.read(str(wav), dtype="float32", always_2d=False)
                assert sr == 16000, f"{wav} at {sr} Hz"
                corpus_evals.setdefault(prefix, []).append(
                    (np.ascontiguousarray(audio), txt.read_text().strip()))
        for prefix, clips_ in corpus_evals.items():
            print(f"[corpus] {prefix}: {len(clips_)} clips")

    rows = []
    de_clips: list[tuple] = []
    if args.mls_de:
        import numpy as np
        from datasets import load_dataset
        ds = load_dataset("facebook/multilingual_librispeech", "german",
                          split="test", streaming=True)
        for i, ex in enumerate(ds):
            if i >= args.mls_de:
                break
            aud = ex["audio"]
            if hasattr(aud, "get_all_samples"):  # datasets>=5 lazy decoder
                s = aud.get_all_samples()
                array, sr = s.data.numpy(), int(s.sample_rate)  # [ch, n]
            else:
                array, sr = aud["array"], int(aud["sampling_rate"])
            array = np.asarray(array, dtype=np.float32)
            if array.ndim == 2:  # mono arrives as [1, n]
                array = array[0] if array.shape[0] in (1, 2) else array.reshape(-1)
            if sr != 16000:  # MLS opus decodes at 48 kHz
                import torch
                import torchaudio.functional as AF
                array = AF.resample(torch.from_numpy(array), sr, 16000).numpy()
                sr = 16000
            assert sr == 16000, f"MLS clip at {sr} Hz"
            de_clips.append((np.ascontiguousarray(array), ex["transcript"]))
        print(f"[mls-de] {len(de_clips)} German clips "
              f"(~{sum(len(a) for a, _ in de_clips) / 16000 / 60:.1f} min)")

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
               "wer": {}, "cer": {}, "wer_ci": {}}
        per_clip: dict[str, list[float]] = {}
        try:
            eng.load()
            for tier in tiers:
                hyp = eng.transcribe(noised(fixtures[tier]))[0]
                ref = REFERENCE_TRANSCRIPTS[tier]
                row["wer"][tier] = round(wer_pct(ref, hyp), 2)
                row["cer"][tier] = round(cer_pct(ref, hyp), 2)
            if de_clips:
                wers = [wer_pct(ref, eng.transcribe(audio)[0])
                        for audio, ref in de_clips]
                row["wer"]["mls_de"] = round(sum(wers) / len(wers), 2)
                per_clip["mls_de"] = wers
            for cfg, clips_ in fleurs_evals.items():
                if not clips_:  # a config that failed to stream; warned above
                    continue
                wers = [wer_pct(ref, eng.transcribe(audio)[0])
                        for audio, ref in clips_]
                row["wer"][f"fleurs_{cfg}"] = round(sum(wers) / len(wers), 2)
                per_clip[f"fleurs_{cfg}"] = wers
            for prefix, clips_ in corpus_evals.items():
                wers = [wer_pct(ref, eng.transcribe(audio)[0])
                        for audio, ref in clips_]
                row["wer"][prefix] = round(sum(wers) / len(wers), 2)
                per_clip[prefix] = wers
            for col, wers in per_clip.items():
                import zlib
                row["wer_ci"][col] = _bootstrap_ci(
                    wers, seed=zlib.crc32(col.encode()) % 2**31)
        finally:
            eng.close()
        rows.append(row)
        print(f"[done] {label}: " + " ".join(f"wer_{t}={row['wer'][t]}" for t in tiers))

    if not rows:
        print("no models evaluated")
        return 1

    cols = list(tiers)
    for r in rows:  # extra columns (mls_de, fleurs_*) in first-seen order
        for k in r["wer"]:
            if k not in cols:
                cols.append(k)
    hdr = f"{'model':<16} {'MB':>7} " + " ".join(f"{'wer_' + t:>10}" for t in cols)
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['model']:<16} {r['mb']:>7.1f} "
              + " ".join(f"{r['wer'].get(t, float('nan')):>10.2f}" for t in cols))

    ci_cols = [c for c in cols if any(r["wer_ci"].get(c) for r in rows)]
    if ci_cols:
        print("\n95% bootstrap CIs (mean [lo–hi]):\n")
        for c in ci_cols:
            print(f"  {c}:")
            for r in rows:
                ci = r["wer_ci"].get(c)
                if ci:
                    print(f"    {r['model']:<16} {r['wer'][c]:>6.2f} "
                          f"[{ci[0]:.2f}–{ci[1]:.2f}]")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.json}")
    if de_clips or fleurs_evals:
        # datasets>=5 streaming leaves a thread that crashes interpreter
        # finalization; flush everything, then exit hard.
        sys.stdout.flush()
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
