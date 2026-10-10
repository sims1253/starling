"""Paced streaming workloads with reference text (issues #226, #310, #357).

Builds deterministic dictation-like takes from the public LibriSpeech dummy
validation split (``hf-internal-testing/librispeech_asr_dummy``, clean, 73
utterances, CC BY 4.0) so streaming runs replay identical audio with known
reference words and utterance time spans:

- ``short``: one ~10 s utterance (first-partial and stop latency).
- ``medium``: ~60 s of consecutive utterances with short pauses.
- ``long``: ~6 min of consecutive utterances with seeded 0.3–1.5 s pauses and
  a 4–8 s "thinking pause" after every eighth utterance.

The audio is not committed. The manifest (``manifest.json`` next to the WAVs)
records the source parquet hash, the pause schedule, every utterance span and
reference, and each WAV's sha256; ``stream_replay.py`` refuses a workload
whose WAV no longer matches its manifest.

    python benchmarks/experiments/stream_workload.py --out build/stream-workload

Needs ``pyarrow`` and ``soundfile`` (``uv run --with pyarrow ...``) and the
dataset parquet in the Hugging Face cache, or ``--parquet <path>``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
DATASET = "hf-internal-testing/librispeech_asr_dummy"
PARQUET_GLOB = (
    "datasets--hf-internal-testing--librispeech_asr_dummy/snapshots/*/clean/"
    "validation-00000-of-00001.parquet"
)
SEED = 357
WORKLOAD_VERSION = 1


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _find_parquet() -> Path:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    found = sorted(hub.glob(PARQUET_GLOB))
    if not found:
        sys.exit(f"{DATASET} parquet not in {hub}; download it or pass --parquet")
    return found[-1]


def _load(parquet: Path) -> list[dict]:
    import pyarrow.parquet as pq
    import soundfile as sf

    rows = pq.read_table(parquet).to_pylist()
    out = []
    for r in sorted(rows, key=lambda r: r["id"]):
        audio, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        if sr != SAMPLE_RATE or audio.ndim != 1:
            sys.exit(f"{r['id']}: expected mono {SAMPLE_RATE} Hz, got {sr} Hz")
        out.append({"id": r["id"], "text": r["text"], "audio": audio})
    return out


def _write_wav(path: Path, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def _take(utts: list[dict], pauses: list[float]) -> tuple[np.ndarray, list[dict]]:
    """Concatenate utterances with the given trailing pauses (silence)."""
    parts, spans, at = [], [], 0
    for u, pause in zip(utts, pauses):
        spans.append({"id": u["id"], "start_s": round(at / SAMPLE_RATE, 3),
                      "end_s": round((at + len(u["audio"])) / SAMPLE_RATE, 3),
                      "text": u["text"], "pause_after_s": pause})
        parts.append(u["audio"])
        at += len(u["audio"])
        gap = np.zeros(int(round(pause * SAMPLE_RATE)), dtype=np.float32)
        parts.append(gap)
        at += len(gap)
    return np.concatenate(parts), spans


def build(parquet: Path, out: Path) -> dict:
    utts = _load(parquet)
    rng = random.Random(SEED)
    out.mkdir(parents=True, exist_ok=True)
    takes = {}

    # short: the utterance closest to 10 s, with a 0.5 s trailing pause.
    short = min(utts, key=lambda u: abs(len(u["audio"]) / SAMPLE_RATE - 10.0))
    plans = {"short": ([short], [0.5])}

    # medium: consecutive utterances (dataset order) until >= 60 s.
    med, pauses, dur = [], [], 0.0
    for u in utts:
        if dur >= 60.0:
            break
        p = round(rng.uniform(0.4, 1.2), 2)
        med.append(u)
        pauses.append(p)
        dur += len(u["audio"]) / SAMPLE_RATE + p
    plans["medium"] = (med, pauses)

    # long: from the start again until >= 360 s, with a thinking pause after
    # every eighth utterance.
    lng, pauses, dur = [], [], 0.0
    for i, u in enumerate(utts):
        if dur >= 360.0:
            break
        p = round(rng.uniform(4.0, 8.0) if (i + 1) % 8 == 0 else rng.uniform(0.3, 1.5), 2)
        lng.append(u)
        pauses.append(p)
        dur += len(u["audio"]) / SAMPLE_RATE + p
    if dur < 300.0:
        sys.exit(f"long take only {dur:.0f} s; the source split is too small")
    plans["long"] = (lng, pauses)

    for name, (us, ps) in plans.items():
        audio, spans = _take(us, ps)
        wav = out / f"{name}.wav"
        _write_wav(wav, audio)
        takes[name] = {
            "wav": wav.name,
            "sha256": _sha256(wav),
            "duration_s": round(len(audio) / SAMPLE_RATE, 3),
            "reference": " ".join(s["text"] for s in spans),
            "utterances": spans,
        }
    manifest = {
        "version": WORKLOAD_VERSION,
        "source": {"dataset": DATASET, "config": "clean", "split": "validation",
                   "license": "CC BY 4.0", "parquet_sha256": _sha256(parquet)},
        "seed": SEED,
        "sample_rate": SAMPLE_RATE,
        "takes": takes,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--parquet", type=Path, default=None)
    args = ap.parse_args(argv)
    manifest = build(args.parquet or _find_parquet(), args.out)
    for name, t in manifest["takes"].items():
        print(f"{name}: {t['duration_s']:.1f} s, {len(t['utterances'])} utterances, "
              f"sha256 {t['sha256'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
