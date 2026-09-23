"""CPU-only end-to-end reference for the Q4 kernel prototype.

Runs Starling's current native ggml engine through its C ABI on fixed synthetic
16 kHz audio. The kernel benchmark in this directory uses the same CPU build.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import os
import statistics
import time
import wave
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--model", choices=("parakeet", "moss"), required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--wav", type=Path, help="16 kHz mono PCM16 WAV; otherwise use a fixed test waveform")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--ids", action="store_true", help="print Parakeet token IDs for parity")
    args = parser.parse_args()
    if args.threads < 1 or args.seconds <= 0 or args.iterations < 1:
        parser.error("threads, seconds, and iterations must be positive")

    # Set before the library can construct its process-global backend. The
    # benchmark refuses any backend whose reported name is not CPU.
    os.environ["STARLING_GGML_DEVICE"] = "cpu"
    os.environ["STARLING_GGML_THREADS"] = str(args.threads)
    if args.model == "parakeet":
        os.environ["STARLING_PARAKEET_TIMING"] = "1"
    else:
        os.environ["STARLING_MOSS_TIMING"] = "1"

    lib = ctypes.CDLL(str(args.library.resolve()))
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
    lib.starling_ggml_backend_name.restype = ctypes.c_char_p

    if args.wav:
        with wave.open(str(args.wav), "rb") as source:
            if source.getframerate() != 16000 or source.getnchannels() != 1 or source.getsampwidth() != 2:
                parser.error("--wav must be 16 kHz mono PCM16")
            raw = source.readframes(source.getnframes())
        import array
        samples = array.array("h")
        samples.frombytes(raw)
        count = len(samples)
        pcm = (ctypes.c_float * count)(*(sample / 32768.0 for sample in samples))
    else:
        count = round(args.seconds * 16000)
        pcm = (ctypes.c_float * count)()
        for i in range(count):
            t = i / 16000
            envelope = 0.12 + 0.08 * math.sin(2 * math.pi * 2.3 * t) ** 2
            pcm[i] = envelope * (
                math.sin(2 * math.pi * 180 * t)
                + 0.35 * math.sin(2 * math.pi * 310 * t)
            )

    start = time.perf_counter()
    ctx = lib.starling_ggml_load(1 if args.model == "parakeet" else 2,
                                 os.fsencode(args.gguf))
    if not ctx:
        raise RuntimeError(lib.starling_ggml_last_error(None).decode())
    load_s = time.perf_counter() - start
    try:
        backend = lib.starling_ggml_backend_name().decode()
        if "cpu" not in backend.lower():
            raise RuntimeError(f"refusing non-CPU backend: {backend}")
        times: list[float] = []
        outputs: list[str] = []
        for i in range(args.iterations + 1):
            start = time.perf_counter()
            result = lib.starling_ggml_transcribe_pcm(ctx, pcm, count, 16000)
            elapsed = time.perf_counter() - start
            if not result:
                raise RuntimeError(lib.starling_ggml_last_error(ctx).decode())
            text = ctypes.string_at(result).decode("utf-8")
            lib.starling_ggml_free_string(result)
            if i > 0:
                times.append(elapsed)
                outputs.append(text)
        if len(set(outputs)) != 1:
            raise RuntimeError("repeated transcriptions differ")
        print(f"model={args.model} backend={backend} threads={args.threads} "
              f"audio_s={count / 16000:.2f} load_s={load_s:.3f}")
        print(f"warm_median_s={statistics.median(times):.3f} "
              f"runs_s={[round(x, 3) for x in times]}")
        print(f"transcript={outputs[0]!r}")
        if args.ids:
            if args.model != "parakeet":
                parser.error("--ids is supported only for Parakeet")
            lib.starling_ggml_parakeet_decode_ids_pub.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64,
                ctypes.POINTER(ctypes.c_int64),
            ]
            lib.starling_ggml_parakeet_decode_ids_pub.restype = ctypes.c_void_p
            count_ids = ctypes.c_int64()
            ptr = lib.starling_ggml_parakeet_decode_ids_pub(
                ctx, pcm, count, ctypes.byref(count_ids)
            )
            if not ptr:
                raise RuntimeError(lib.starling_ggml_last_error(ctx).decode())
            ids = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int64))
            print("ids=" + ",".join(str(ids[i]) for i in range(count_ids.value)))
            libc = ctypes.CDLL(None)
            libc.free.argtypes = [ctypes.c_void_p]
            libc.free(ptr)
    finally:
        lib.starling_ggml_free(ctx)


if __name__ == "__main__":
    main()
