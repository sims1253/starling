#!/usr/bin/env python3
"""WER, speed and memory of quantized GGUFs for any Starling ASR engine.

The model-agnostic sibling of fast_engine/wer_engines.py (Parakeet/MOSS
engine variants): every ``label=path`` GGUF is transcribed in its own fresh
process through the flat C API, over a clip directory written by fast_engine/export_fleurs.py
(WAVs + refs.json). Per GGUF it reports corpus WER (fast_engine/wer.py
normalization), mean per-clip WER, wall time, real-time factor, load time,
peak RSS and file size, and keeps every hypothesis in the JSON so two runs
can be compared clip by clip (compare_wer.py). With STARLING_IMATRIX set,
each process also collects an importance matrix (written at exit).

Usage:
  wer_models.py --lib build/host/libstarling_ggml.so --model qwen3_06 \
      --clips /tmp/fleurs_en --json out.json \
      bf16=models/qwen3-asr-0.6b-bf16.gguf q4=models/qwen3-asr-0.6b-q4.gguf
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import resource
import subprocess
import sys
import time
import wave

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fast_engine"))
from wer import normalize  # noqa: E402

# starling_ggml_model (cpp/include/starling_ggml.h).
KINDS = {"parakeet": 1, "moss": 2, "ark": 3, "granite": 6, "qwen3": 7,
         "ark06": 10, "qwen3_06": 12}


def peak_rss_mb() -> float:
    # ru_maxrss is KiB on Linux, bytes on macOS.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024.0 * 1024.0) if sys.platform == "darwin" else rss / 1024.0


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def read_wav(path: str):
    import numpy as np
    with wave.open(path, "rb") as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit(f"{path}: need 16 kHz mono PCM16")
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return (data.astype(np.float32) / 32768.0).copy()


def worker(a) -> int:
    """One GGUF in this process; prints one JSON line on stdout."""
    refs = json.load(open(os.path.join(a.clips, "refs.json")))
    names = sorted(refs)[: a.limit or None]
    lib = ctypes.CDLL(a.lib)
    lib.starling_ggml_load.restype = ctypes.c_void_p
    lib.starling_ggml_load.argtypes = [ctypes.c_int, ctypes.c_char_p]
    lib.starling_ggml_last_error.restype = ctypes.c_char_p
    lib.starling_ggml_last_error.argtypes = [ctypes.c_void_p]
    lib.starling_ggml_transcribe_pcm.restype = ctypes.c_void_p
    lib.starling_ggml_transcribe_pcm.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64, ctypes.c_int]
    lib.starling_ggml_free_string.argtypes = [ctypes.c_void_p]
    lib.starling_ggml_free.argtypes = [ctypes.c_void_p]

    t0 = time.perf_counter()
    ctx = lib.starling_ggml_load(KINDS[a.model], a.worker.encode())
    load_s = time.perf_counter() - t0
    if not ctx:
        print(json.dumps({"error": lib.starling_ggml_last_error(None).decode()}))
        return 1
    hyps, secs, audio_s = [], [], 0.0
    for n in names:
        pcm = read_wav(os.path.join(a.clips, n))
        audio_s += len(pcm) / 16000.0
        t = time.perf_counter()
        p = lib.starling_ggml_transcribe_pcm(
            ctx, pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(pcm), 16000)
        secs.append(time.perf_counter() - t)
        if not p:
            hyps.append(None)
            continue
        hyps.append(ctypes.cast(p, ctypes.c_char_p).value.decode("utf-8", "replace"))
        lib.starling_ggml_free_string(p)
    lib.starling_ggml_free(ctx)
    print(json.dumps({"names": names, "hyps": hyps, "secs": secs, "audio_s": audio_s,
                      "load_s": load_s,
                      "peak_rss_mb": peak_rss_mb()}))
    return 0


def score(refs: dict, names: list[str], hyps: list) -> tuple[float, float]:
    errs = words = 0
    per_clip = []
    for n, h in zip(names, hyps):
        r = normalize(refs[n]).split()
        e = edit_distance(r, normalize(h or "").split())
        errs += e
        words += len(r)
        per_clip.append(100.0 * e / max(len(r), 1))
    return 100.0 * errs / max(words, 1), sum(per_clip) / max(len(per_clip), 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--model", required=True, choices=sorted(KINDS))
    ap.add_argument("--clips", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json", help="write per-GGUF results incl. hypotheses here")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("ggufs", nargs="*", help="label=path")
    a = ap.parse_args()
    if a.worker:
        return worker(a)
    if not a.ggufs:
        ap.error("give at least one label=path")

    refs = json.load(open(os.path.join(a.clips, "refs.json")))
    results = {}
    for spec in a.ggufs:
        label, _, path = spec.partition("=")
        cmd = [sys.executable, __file__, "--lib", a.lib, "--model", a.model,
               "--clips", a.clips, "--limit", str(a.limit), "--worker", path]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        line = (proc.stdout.strip().splitlines() or ["{}"])[-1]
        r = json.loads(line) if line.startswith("{") else {"error": proc.stderr[-2000:]}
        if "error" in r or proc.returncode != 0:
            print(f"{label}: FAILED {r.get('error', '')[-500:]}", flush=True)
            results[label] = {"path": path, "error": r.get("error")}
            continue
        wer, mean_clip = score(refs, r["names"], r["hyps"])
        total = sum(r["secs"])
        r.update(path=path, size_mb=os.path.getsize(path) / 1e6, wer=wer,
                 mean_clip_wer=mean_clip, wall_s=total, rtf=total / max(r["audio_s"], 1e-9),
                 failed=sum(h is None for h in r["hyps"]))
        results[label] = r
        print(f"{label:24s} WER {wer:6.2f}%  clipWER {mean_clip:6.2f}%  wall {total:8.2f}s  "
              f"RTF {r['rtf']:.4f}  load {r['load_s']:5.2f}s  RSS {r['peak_rss_mb']:7.0f} MB  "
              f"file {r['size_mb']:7.0f} MB" + (f"  FAILED {r['failed']}" if r["failed"] else ""),
              flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"model": a.model, "clips": a.clips, "results": results}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
