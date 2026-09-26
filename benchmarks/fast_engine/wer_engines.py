#!/usr/bin/env python3
"""WER / latency comparison of Starling engines on a directory of WAV clips.

Every engine variant is a (name, env) pair loaded from the same
libstarling_ggml, e.g. the ggml reference and the fast engine with and
without packed-f16 GEMMs. Reports corpus WER (benchmarks/wer.py
normalization), total transcription time, and how many transcripts differ
from the first variant.

Usage:
  wer_engines.py --lib build/libstarling_ggml.so --model parakeet --gguf M.gguf \
      --clips DIR [--variants ggml fast fast-f16] [--limit N]
The clip directory holds WAVs plus refs.json (see export_fleurs.py).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from wer import normalize  # noqa: E402

KINDS = {"parakeet": 1, "moss": 2}
VARIANTS = {
    "ggml": {"STARLING_ENGINE": "ggml"},
    "fast": {"STARLING_ENGINE": "fast", "STARLING_FAST_F16": "0"},
    "fast-f16": {"STARLING_ENGINE": "fast", "STARLING_FAST_F16": "1"},
    # W4 decode GEMVs through unpackUnorm4x8 (the PowerVR default).
    "fast-w4u": {"STARLING_ENGINE": "fast", "STARLING_FAST_F16": "0", "STARLING_FAST_W4U": "1"},
    "fast-w4": {"STARLING_ENGINE": "fast", "STARLING_FAST_F16": "0", "STARLING_FAST_W4U": "0"},
    # Source-quantized packed weights (#319): STARLING_FAST_PACKED names the
    # .pack file; being in the variant table means the other variants run
    # without it.
    "fast-packed": {"STARLING_ENGINE": "fast", "STARLING_FAST_F16": "0",
                    "STARLING_FAST_PACKED": ""},
}
VARIANT_KEYS = sorted({k for env in VARIANTS.values() for k in env})
# fast-packed takes the .pack path from this process's environment (empty = unset).
VARIANTS["fast-packed"]["STARLING_FAST_PACKED"] = os.environ.get("STARLING_FAST_PACKED", "")


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return (data.astype(np.float32) / 32768.0).copy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--model", default="parakeet", choices=sorted(KINDS))
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--variants", nargs="+", default=["ggml", "fast", "fast-f16"], choices=sorted(VARIANTS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--show-diffs", action="store_true")
    a = ap.parse_args()
    if "fast-packed" in a.variants and not VARIANTS["fast-packed"]["STARLING_FAST_PACKED"]:
        raise SystemExit("fast-packed needs STARLING_FAST_PACKED=<file.pack> in the environment "
                         "(without it the variant is a plain fast run)")

    refs = json.load(open(os.path.join(a.clips, "refs.json")))
    names = sorted(refs)[: a.limit or None]
    clips = [(n, read_wav(os.path.join(a.clips, n))) for n in names]
    audio_s = sum(len(c) for _, c in clips) / 16000.0

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

    outputs: dict[str, list[str]] = {}
    for v in a.variants:
        # Each variant starts from a clean slate: keys set by an earlier
        # variant must not leak into this one.
        for k in VARIANT_KEYS:
            os.environ.pop(k, None)
        for k, val in VARIANTS[v].items():
            if val:
                os.environ[k] = val
        ctx = lib.starling_ggml_load(KINDS[a.model], a.gguf.encode())
        if not ctx:
            print(f"{v}: load failed: {lib.starling_ggml_last_error(None).decode()}")
            continue
        # Warm up once (pipeline compilation / graph capture) outside the clock.
        w = clips[0][1]
        p = lib.starling_ggml_transcribe_pcm(ctx, w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(w), 16000)
        lib.starling_ggml_free_string(p)
        texts = []
        failed = 0
        t0 = time.perf_counter()
        for _, pcm in clips:
            p = lib.starling_ggml_transcribe_pcm(
                ctx, pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(pcm), 16000)
            if not p:
                failed += 1
                texts.append("")
                continue
            texts.append(ctypes.cast(p, ctypes.c_char_p).value.decode("utf-8", "replace"))
            lib.starling_ggml_free_string(p)
        dt = time.perf_counter() - t0
        lib.starling_ggml_free(ctx)
        errs = words = 0
        for (n, _), hyp in zip(clips, texts):
            r, h = normalize(refs[n]).split(), normalize(hyp).split()
            errs += edit_distance(r, h)
            words += len(r)
        outputs[v] = texts
        base_name = next(iter(outputs))   # first variant that loaded
        ndiff = sum(x != y for x, y in zip(outputs[base_name], texts))
        print(f"{v:9s} WER {100.0 * errs / max(words, 1):6.2f}%  time {dt:7.2f}s  "
              f"RTF {dt / audio_s:.4f}  clips {len(clips)}  differ-from-{base_name} {ndiff}"
              + (f"  FAILED {failed}" if failed else ""),
              flush=True)
    if a.show_diffs and len(outputs) > 1:
        base_name, base = next(iter(outputs.items()))
        for v, texts in outputs.items():
            for (n, _), x, y in zip(clips, base, texts):
                if x != y:
                    print(f"[{v}] {n}\n   {base_name}: {x}\n   {v}: {y}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
