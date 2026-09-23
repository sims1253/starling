"""Compare overlapping Parakeet windows with each CPU engine in its own process."""

from __future__ import annotations

import argparse
import array
import ctypes
import json
import os
import re
import statistics
import subprocess
import sys
import time
import wave
from pathlib import Path


def windows(samples: int, window: int, advance: int) -> list[tuple[int, int]]:
    result = []
    first = 0
    while first + window <= samples:
        result.append((first, window))
        first += advance
    if first < samples:
        result.append((first, samples - first))
    return result


def reference_run(args: argparse.Namespace, raw: bytes,
                  plan: list[tuple[int, int]]) -> dict:
    integer = array.array("h")
    integer.frombytes(raw)
    pcm = (ctypes.c_float * len(integer))(*(sample / 32768.0 for sample in integer))
    os.environ["STARLING_GGML_DEVICE"] = "cpu"
    os.environ["STARLING_GGML_THREADS"] = str(args.threads)
    lib = ctypes.CDLL(str(args.library.resolve()))
    lib.starling_ggml_backend_name.restype = ctypes.c_char_p
    lib.starling_ggml_load.argtypes = [ctypes.c_int, ctypes.c_char_p]
    lib.starling_ggml_load.restype = ctypes.c_void_p
    lib.starling_ggml_transcribe_pcm.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64, ctypes.c_int
    ]
    lib.starling_ggml_transcribe_pcm.restype = ctypes.c_void_p
    lib.starling_ggml_free_string.argtypes = [ctypes.c_void_p]
    lib.starling_ggml_free.argtypes = [ctypes.c_void_p]
    lib.starling_ggml_last_error.argtypes = [ctypes.c_void_p]
    lib.starling_ggml_last_error.restype = ctypes.c_char_p
    ctx = lib.starling_ggml_load(1, os.fsencode(args.gguf))
    if not ctx:
        raise RuntimeError(lib.starling_ggml_last_error(None).decode())
    try:
        backend = lib.starling_ggml_backend_name().decode()
        if "cpu" not in backend.lower():
            raise RuntimeError(f"refusing non-CPU backend: {backend}")

        def infer(first: int, count: int) -> str:
            ptr = ctypes.cast(ctypes.byref(pcm, first * ctypes.sizeof(ctypes.c_float)),
                              ctypes.POINTER(ctypes.c_float))
            output = lib.starling_ggml_transcribe_pcm(ctx, ptr, count, 16000)
            if not output:
                raise RuntimeError(lib.starling_ggml_last_error(ctx).decode())
            transcript = ctypes.string_at(output).decode("utf-8")
            lib.starling_ggml_free_string(output)
            return transcript

        infer(*plan[0])  # Warm model kernels and shape caches outside the timer.
        start = time.perf_counter()
        cpu_start = time.process_time()
        texts = [infer(first, count) for first, count in plan]
        seconds = time.perf_counter() - start
        cpu_seconds = time.process_time() - cpu_start
        return {"backend": backend, "seconds": seconds,
                "cpu_seconds": cpu_seconds, "texts": texts}
    finally:
        lib.starling_ggml_free(ctx)


def direct_run(args: argparse.Namespace, plan: list[tuple[int, int]]) -> dict:
    command = [str(args.direct.resolve()), str(args.gguf.resolve()), str(args.wav.resolve()),
               str(args.threads), "--benchmark-window-s", str(args.window_s),
               "--benchmark-advance-s", str(args.advance_s), "--iterations", "1"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    texts = [line.split(" transcript=", 1)[1] for line in result.stdout.splitlines()
             if line.startswith("window=")]
    match = re.search(r"session_median_s=([0-9.]+)", result.stdout)
    cpu_match = re.search(r"session_cpu_s=([0-9.]+)", result.stdout)
    if not match or not cpu_match or len(texts) != len(plan):
        raise RuntimeError(f"invalid direct benchmark output: {result.stdout}")
    return {"backend": "direct-cpu", "seconds": float(match.group(1)),
            "cpu_seconds": float(cpu_match.group(1)), "texts": texts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--window-s", type=float, default=12.0)
    parser.add_argument("--advance-s", type=float, default=9.0)
    parser.add_argument("--runs", type=int, default=2, help="paired rounds, alternating engine order")
    parser.add_argument("--allow-text-mismatch", action="store_true",
                        help="report timing despite text differences (default: exit 1)")
    parser.add_argument("--baseline-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.threads < 1 or args.runs < 1 or not 0 < args.advance_s <= args.window_s:
        parser.error("threads and runs must be positive, with 0 < advance <= window")
    with wave.open(str(args.wav), "rb") as source:
        if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (16000, 1, 2):
            parser.error("--wav must be 16 kHz mono PCM16")
        raw = source.readframes(source.getnframes())
    sample_count = len(raw) // 2
    window = round(args.window_s * 16000)
    advance = round(args.advance_s * 16000)
    if window < 1 or advance < 1:
        parser.error("window and advance must contain at least one sample")
    plan = windows(sample_count, window, advance)
    if not plan:
        parser.error("empty WAV")
    if args.baseline_worker:
        print(json.dumps(reference_run(args, raw, plan)))
        return

    os.environ["OMP_PROC_BIND"] = "true"
    worker_command = [sys.executable, str(Path(__file__).resolve()),
                      "--gguf", str(args.gguf.resolve()), "--wav", str(args.wav.resolve()),
                      "--library", str(args.library.resolve()), "--direct", str(args.direct.resolve()),
                      "--threads", str(args.threads), "--window-s", str(args.window_s),
                      "--advance-s", str(args.advance_s), "--baseline-worker"]
    pairs = []
    for index in range(args.runs):
        def worker() -> dict:
            completed = subprocess.run(worker_command, capture_output=True, text=True, check=True)
            return json.loads(completed.stdout)

        if index % 2 == 0:
            reference, direct = worker(), direct_run(args, plan)
        else:
            direct, reference = direct_run(args, plan), worker()
        mismatches = [i for i, (a, b) in enumerate(zip(reference["texts"], direct["texts"]))
                      if a != b]
        pairs.append((reference["seconds"], direct["seconds"],
                      reference["cpu_seconds"], direct["cpu_seconds"], mismatches))
        print(f"pair={index} ggml_s={reference['seconds']:.4f} direct_s={direct['seconds']:.4f} "
              f"ggml_over_direct={reference['seconds'] / direct['seconds']:.3f}x "
              f"ggml_cpu_s={reference['cpu_seconds']:.4f} "
              f"direct_cpu_s={direct['cpu_seconds']:.4f} "
              f"window_text_mismatches={mismatches}", flush=True)
    ratios = [row[0] / row[1] for row in pairs]
    cpu_ratios = [row[2] / row[3] for row in pairs]
    print(f"audio_s={sample_count / 16000:.2f} windows={len(plan)} "
          f"full_windows={sum(count == window for _, count in plan)} "
          f"threads={args.threads} ggml_backend=CPU direct_backend=direct-cpu "
          f"median_ggml_s={statistics.median(x[0] for x in pairs):.4f} "
          f"median_direct_s={statistics.median(x[1] for x in pairs):.4f} "
          f"median_pair_ratio={statistics.median(ratios):.3f}x "
          f"median_cpu_ratio={statistics.median(cpu_ratios):.3f}x")
    if any(row[4] for row in pairs) and not args.allow_text_mismatch:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
