"""Download FLEURS clips to a plain on-disk corpus (wav + txt sidecars).

Reads the dataset's parquet shards directly over HTTP range requests
(``hf://datasets/...`` via fsspec + pyarrow), so only the first row groups
of each ~2 GB shard are fetched instead of the whole file, and memory stays
bounded by the batch size. This replaces the datasets-library streaming
path, whose per-config memory growth (multi-GB accumulation) made long
multi-config runs unrunable next to a loaded model.

Corpus layout (consumed by imatrix_collect --wavs / wer_quant --corpus):

    <out>/<config>_<split>_<i>.wav   16 kHz mono PCM_16
    <out>/<config>_<split>_<i>.txt   the normalized transcript

Usage::

    uv run python benchmarks/fleurs_download.py --out models/fleurs_train \\
        --fleurs-25 12
    uv run python benchmarks/fleurs_download.py --out models/fleurs_test \\
        --split test --fleurs de_de:50 --fleurs en_us:50

A configuration is complete only when every requested WAV and transcript
exists. Increasing the clip budget fetches the larger corpus.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fleurs_util import PARAKEET_V3_LANGS  # noqa: E402


def fetch_config(config: str, split: str, n: int, out_dir: Path) -> bool:
    """Fetch the first n clips of one config. Returns True when complete."""
    import fsspec
    import pyarrow.parquet as pq
    import soundfile as sf

    url = (f"hf://datasets/google/fleurs/parquet-data/{config}/"
           f"{split}-00000-of-00001.parquet")
    with fsspec.open(url, "rb") as f:
        pf = pq.ParquetFile(f)
        got = 0
        for batch in pf.iter_batches(batch_size=4,
                                     columns=["audio", "transcription"]):
            for row in batch.to_pylist():
                raw = row["audio"]["bytes"]
                audio, sr = sf.read(io.BytesIO(raw), dtype="float32",
                                    always_2d=False)
                if audio.ndim > 1:
                    audio = audio[:, 0]
                text = (row.get("transcription") or "").strip()
                if audio.size < 1600 or not text:  # <0.1 s or empty: skip
                    continue
                sf.write(str(out_dir / f"{config}_{split}_{got}.wav"),
                         audio, sr, subtype="PCM_16")
                (out_dir / f"{config}_{split}_{got}.txt").write_text(text + "\n")
                got += 1
                if got >= n:
                    return True
    return got >= n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=False, type=Path)
    ap.add_argument("--split", default="train")
    ap.add_argument("--fleurs", action="append", default=[], metavar="CFG:N")
    ap.add_argument("--fleurs-25", type=int, default=None, metavar="N")
    ap.add_argument("--_fetch", nargs=4, default=None,
                    help=argparse.SUPPRESS)  # config split n outdir
    args = ap.parse_args()

    if args._fetch:
        ok = fetch_config(args._fetch[0], args._fetch[1], int(args._fetch[2]),
                          Path(args._fetch[3]))
        raise SystemExit(0 if ok else 1)
    if args.out is None:
        ap.error("--out is required")

    budget: dict[str, int] = {}
    for spec in args.fleurs:
        cfg, _, n = spec.partition(":")
        budget[cfg] = int(n)
    if args.fleurs_25 is not None:
        for cfg in PARAKEET_V3_LANGS:
            budget[cfg] = max(budget.get(cfg, 0), args.fleurs_25)

    if not budget or any(n <= 0 for n in budget.values()):
        ap.error("request at least one configuration with a positive clip count")

    args.out.mkdir(parents=True, exist_ok=True)
    import subprocess
    failed = []
    for cfg, n in budget.items():
        if all((args.out / f"{cfg}_{args.split}_{i}{suffix}").is_file()
               for i in range(n) for suffix in (".wav", ".txt")):
            print(f"[skip] {cfg} ({args.split}): done", flush=True)
            continue
        print(f"[fetch] {cfg} ({args.split}) x{n} ...", flush=True)
        try:
            r = subprocess.run(
                [sys.executable, __file__, "--_fetch", cfg, args.split, str(n),
                 str(args.out)], timeout=1800)
            ok = r.returncode == 0
        except subprocess.TimeoutExpired:
            ok = False
        if not ok:
            failed.append(cfg)
            print(f"[fail] {cfg} (timeout or rc!=0)", flush=True)
    if failed:
        print(f"FAILED configs: {failed} (re-run to retry)")
        return 1
    n_clips = len(list(args.out.glob("*.wav")))
    print(f"corpus ready: {n_clips} wavs in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
