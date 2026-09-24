#!/usr/bin/env python3
"""Export FLEURS test clips from a local Hugging Face parquet cache to WAV.

Writes <out>/<cfg>_<i>.wav (16 kHz mono int16) and <out>/refs.json mapping
file name -> reference transcript, for engine comparisons (wer_engines.py)
and for pushing a fixed evaluation set to a phone with adb.

Usage:
  export_fleurs.py --cfg en_us --n 100 --out /tmp/fleurs_en
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import wave

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="en_us")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--google--fleurs/snapshots"))
    a = ap.parse_args()

    files = sorted(glob.glob(f"{a.cache}/*/parquet-data/{a.cfg}/{a.split}-*.parquet"))
    if not files:
        raise SystemExit(f"no parquet for {a.cfg}/{a.split} under {a.cache}")
    os.makedirs(a.out, exist_ok=True)
    refs: dict[str, str] = {}
    for f in files:
        if len(refs) >= a.n:
            break
        t = pq.read_table(f)
        cols = t.column_names
        text_col = next((c for c in ("transcription", "raw_transcription") if c in cols), None)
        if text_col is None:
            raise SystemExit(f"{f}: no transcription column (have {cols})")
        for row in t.to_pylist():
            if len(refs) >= a.n:
                break
            audio = row["audio"]
            data, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=True)
            data = data.mean(axis=1)
            if sr != 16000:
                raise SystemExit(f"unexpected sample rate {sr}")
            text = (row.get(text_col) or "").strip()
            if not text or data.size < 1600:
                continue
            name = f"{a.cfg}_{len(refs):04d}.wav"
            pcm = np.clip(np.round(data * 32767.0), -32768, 32767).astype(np.int16)
            with wave.open(os.path.join(a.out, name), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(pcm.tobytes())
            refs[name] = text
    with open(os.path.join(a.out, "refs.json"), "w") as fh:
        json.dump(refs, fh, ensure_ascii=False, indent=1)
    print(f"wrote {len(refs)} clips to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
