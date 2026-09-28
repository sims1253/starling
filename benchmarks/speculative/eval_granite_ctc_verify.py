"""Full raw-WAV Granite greedy/CTC timing through the native C entry points.

Example (the CTC head is optional, so use a GGUF converted with it):
  STARLING_GGML_DEVICE=cpu STARLING_GGML_THREADS=4 taskset -c 8-15 \
    python benchmarks/speculative/eval_granite_ctc_verify.py \
      --library /tmp/granite-build/libstarling_ggml.so \
      --gguf /tmp/granite-with-ctc.gguf --wav tests/fixtures/2086-149220-0033.wav
"""

from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import soundfile as sf


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--wav", required=True, type=Path)
    parser.add_argument("--max-k", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.max_k <= 16 or not 0 <= args.repeats <= 20:
        parser.error("--max-k must be 1..16 and --repeats must be 0..20")

    pcm, rate = sf.read(args.wav, dtype="float32")
    if rate != 16000 or pcm.ndim != 1 or len(pcm) == 0:
        parser.error("WAV must contain nonempty mono 16 kHz PCM")
    pcm = np.ascontiguousarray(pcm)
    print(json.dumps({
        "kind": "artifact", "gguf_sha256": sha256(args.gguf),
        "library_sha256": sha256(args.library), "wav_sha256": sha256(args.wav),
        "wav_samples": len(pcm), "device_env": os.getenv("STARLING_GGML_DEVICE"),
        "threads_env": os.getenv("STARLING_GGML_THREADS"),
        "max_k": args.max_k, "repeats": args.repeats,
    }), flush=True)

    library = C.CDLL(str(args.library.resolve()))
    library.starling_ggml_granite_load.argtypes = [C.c_char_p, C.POINTER(C.c_char_p)]
    library.starling_ggml_granite_load.restype = C.c_void_p
    library.starling_ggml_granite_free.argtypes = [C.c_void_p]
    library.starling_ggml_granite_decode.argtypes = [
        C.c_void_p, C.POINTER(C.c_float), C.c_int64, C.POINTER(C.c_char_p)]
    library.starling_ggml_granite_decode.restype = C.c_void_p
    library.starling_ggml_granite_decode_ctc.argtypes = [
        C.c_void_p, C.POINTER(C.c_float), C.c_int64, C.c_int32,
        C.POINTER(C.c_char_p)]
    library.starling_ggml_granite_decode_ctc.restype = C.c_void_p
    library.starling_ggml_free_string.argtypes = [C.c_void_p]
    library.starling_ggml_backend_name.restype = C.c_char_p

    error = C.c_char_p()
    handle = library.starling_ggml_granite_load(
        os.fsencode(args.gguf.resolve()), C.byref(error))
    if not handle:
        raise RuntimeError(f"Granite load failed: {error.value!r}")
    try:
        print(json.dumps({"kind": "backend", "name":
                          library.starling_ggml_backend_name().decode()}), flush=True)
        pointer = pcm.ctypes.data_as(C.POINTER(C.c_float))

        def run(kind: str, phase: str, repeat: int) -> str:
            error.value = None
            start = time.perf_counter()
            if kind == "ctc":
                result = library.starling_ggml_granite_decode_ctc(
                    handle, pointer, len(pcm), args.max_k, C.byref(error))
            else:
                result = library.starling_ggml_granite_decode(
                    handle, pointer, len(pcm), C.byref(error))
            elapsed_ms = (time.perf_counter() - start) * 1000
            if not result:
                raise RuntimeError(f"{kind} {phase} failed: {error.value!r}")
            try:
                text = C.cast(result, C.c_char_p).value.decode("utf-8")
            finally:
                library.starling_ggml_free_string(result)
            record = {"kind": "run", "mode": kind, "phase": phase,
                      "repeat": repeat, "full_ms": round(elapsed_ms, 3),
                      "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                      "text_chars": len(text)}
            print(json.dumps(record), flush=True)
            return text

        reference = run("greedy", "warm", 0)
        if run("ctc", "warm", 0) != reference:
            raise AssertionError("CTC warm transcript differs from greedy")
        for repeat in range(1, args.repeats + 1):
            order = ("greedy", "ctc") if repeat % 2 else ("ctc", "greedy")
            for kind in order:
                if run(kind, "paired", repeat) != reference:
                    raise AssertionError(f"{kind} repeat {repeat} transcript differs")
    finally:
        library.starling_ggml_granite_free(handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
