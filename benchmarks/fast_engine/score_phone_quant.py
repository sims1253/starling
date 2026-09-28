#!/usr/bin/env python3
"""Score Pixel ``starling-bench`` transcripts with the held-out WER gate.

Each log is one fresh model process. The first WAV in each process is a warmup;
the remaining WAVs must cover the sorted corpus exactly once and in order.
The fast engine's own load log establishes the selected Vulkan device. The
generic ``backend=`` header reports the ggml backend and is not sufficient.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wer import wer_pct  # noqa: E402

BENCH_LINE = re.compile(r"^(.+\.wav) run=0 audio=([0-9.]+)s time=([0-9.]+)ms rtf=([0-9.]+)$")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_logs(logs: list[Path], engine_logs: list[Path], names: list[str],
               fast_device: str, warmup_name: str) -> dict[str, dict]:
    if not logs or len(logs) != len(engine_logs):
        raise ValueError("supply one engine log for every benchmark log")
    parsed: dict[str, dict] = {}
    seen_order: list[str] = []
    for log, engine_log in zip(logs, engine_logs):
        engine_text = engine_log.read_text()
        if f"[fast] device '{fast_device}'" not in engine_text or \
                f"parakeet engine: fast/vulkan '{fast_device}'" not in engine_text:
            raise ValueError(f"{engine_log}: actual fast Vulkan device is not {fast_device!r}")
        lines = log.read_text().splitlines()
        if not lines or not re.fullmatch(r"load [0-9.]+ ms\s+backend=\S+", lines[0]):
            raise ValueError(f"{log}: missing or malformed load header")
        warmups = 0
        for offset in range(1, len(lines), 2):
            if offset + 1 >= len(lines):
                raise ValueError(f"{log}: truncated final transcript")
            match = BENCH_LINE.fullmatch(lines[offset])
            if not match or not lines[offset + 1].startswith("  "):
                raise ValueError(f"{log}:{offset + 1}: malformed benchmark result")
            name = Path(match.group(1)).name
            if name == warmup_name:
                warmups += 1
                if offset != 1:
                    raise ValueError(f"{log}: warmup is not first")
                continue
            if name in parsed:
                raise ValueError(f"{log}: duplicate clip {name}")
            seen_order.append(name)
            parsed[name] = {"hypothesis": lines[offset + 1][2:],
                            "audio_seconds": float(match.group(2)),
                            "time_ms": float(match.group(3))}
        if warmups != 1:
            raise ValueError(f"{log}: expected one warmup, found {warmups}")
    if seen_order != names:
        missing = sorted(set(names) - set(seen_order))
        extra = sorted(set(seen_order) - set(names))
        raise ValueError(f"clips are missing or out of manifest order: missing={missing[:4]}, extra={extra[:4]}")
    return parsed


def score(args: argparse.Namespace) -> dict:
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("schema") != "quant-wer-noninferiority-v1":
        raise ValueError("unsupported protocol")
    names = sorted(wav.name for wav in args.corpus.glob("*.wav"))
    if not names:
        raise ValueError("empty corpus")
    parsed = parse_logs(args.logs, args.engine_logs, names, args.fast_device,
                        args.warmup_name)
    row = {"model": args.label, "path": str(args.model),
           "mb": round(args.model.stat().st_size / 1e6, 1),
           "wer": {}, "cer": {}, "wer_ci": {}, "clips": {},
           "provenance": {
               "model_sha256": file_sha256(args.model),
               "model_bytes": args.model.stat().st_size,
               "source_sha256": file_sha256(args.source),
               "imatrix_sha256": file_sha256(args.imatrix),
               "engine_sha256": file_sha256(args.binary),
               "scorer_sha256": file_sha256(Path(__file__).resolve().parents[1] / "wer.py"),
               "protocol_sha256": file_sha256(args.protocol),
               "device": f"{args.fast_device}/fast-vulkan",
               "engine_source_commit": args.source_commit,
               "bench_stdout_sha256": {p.name: file_sha256(p) for p in args.logs},
               "bench_stderr_sha256": {p.name: file_sha256(p) for p in args.engine_logs},
           }}
    for name in names:
        wav = args.corpus / name
        txt = wav.with_suffix(".txt")
        if not txt.is_file():
            raise ValueError(f"missing reference: {txt}")
        reference = txt.read_text().strip()
        if not reference:
            raise ValueError(f"empty reference: {txt}")
        audio, rate = sf.read(wav, dtype="float32", always_2d=False)
        if rate != 16000 or audio.ndim != 1:
            raise ValueError(f"{wav}: expected 16 kHz mono WAV")
        audio_hash = hashlib.sha256(np.ascontiguousarray(audio).tobytes()).hexdigest()
        measured = parsed[name]
        if abs(measured["audio_seconds"] - len(audio) / rate) > 0.01:
            raise ValueError(f"{name}: benchmark audio duration differs from WAV")
        cohort = wav.stem.rsplit("_", 1)[0]
        row["clips"].setdefault(cohort, []).append({
            "id": name, "audio_sha256": audio_hash, "reference": reference,
            "hypothesis": measured["hypothesis"],
            "wer": wer_pct(reference, measured["hypothesis"]),
            "time_ms": measured["time_ms"],
        })
    required = protocol["cohorts"]
    if {k: len(v) for k, v in row["clips"].items()} != required:
        raise ValueError("corpus cohorts do not match the sealed protocol")
    for cohort, clips in row["clips"].items():
        row["wer"][cohort] = round(sum(c["wer"] for c in clips) / len(clips), 2)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--imatrix", required=True, type=Path)
    ap.add_argument("--binary", required=True, type=Path)
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--protocol", required=True, type=Path)
    ap.add_argument("--log", dest="logs", action="append", required=True, type=Path)
    ap.add_argument("--engine-log", dest="engine_logs", action="append", required=True, type=Path)
    ap.add_argument("--fast-device", required=True)
    ap.add_argument("--warmup-name", default="short.wav")
    ap.add_argument("--source-commit", required=True)
    ap.add_argument("--json", required=True, type=Path)
    args = ap.parse_args()
    try:
        for path in [args.model, args.source, args.imatrix, args.binary,
                     args.protocol, *args.logs, *args.engine_logs]:
            if not path.is_file():
                raise ValueError(f"missing input: {path}")
        row = score(args)
    except (OSError, ValueError) as exc:
        ap.error(str(exc))
    args.json.write_text(json.dumps([row], indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"wer": row["wer"], "clips": {k: len(v) for k, v in row["clips"].items()}},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
