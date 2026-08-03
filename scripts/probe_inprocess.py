"""Quick in-process probe: ctypes-bind libparakeet.so, time transcription.

Measures the TRUE compute floor (no HTTP, no WAV encode/decode, no subprocess)
so we know how much of the ggml engine's latency is wrapper overhead vs the
ggml decode itself. Run inside one Bash call.
"""
import ctypes
import time
from pathlib import Path

import numpy as np
import soundfile as sf

SO = "/home/m0hawk/Documents/parakeet.cpp/build-cuda/libparakeet.so"
MODEL = "/home/m0hawk/asr-bench/models/tdt-0.6b-v3-f16.gguf"
FIX = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
GOLDEN = Path(__file__).resolve().parents[1] / "golden"


def main() -> int:
    lib = ctypes.CDLL(SO)
    lib.parakeet_capi_abi_version.restype = ctypes.c_int
    lib.parakeet_capi_load.argtypes = [ctypes.c_char_p]
    lib.parakeet_capi_load.restype = ctypes.c_void_p
    lib.parakeet_capi_transcribe_pcm.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib.parakeet_capi_transcribe_pcm.restype = ctypes.POINTER(ctypes.c_char)
    lib.parakeet_capi_free_string.argtypes = [ctypes.POINTER(ctypes.c_char)]
    lib.parakeet_capi_last_error.argtypes = [ctypes.c_void_p]
    lib.parakeet_capi_last_error.restype = ctypes.c_char_p
    lib.parakeet_capi_free.argtypes = [ctypes.c_void_p]

    print(f"[probe] ABI version: {lib.parakeet_capi_abi_version()}")
    t0 = time.perf_counter()
    ctx = lib.parakeet_capi_load(MODEL.encode())
    load_s = time.perf_counter() - t0
    if not ctx:
        print(f"[probe] load FAILED: {lib.parakeet_capi_last_error(ctx).decode()}")
        return 1
    print(f"[probe] model loaded in {load_s:.1f}s")

    for name in ("short", "medium", "long"):
        wav, sr = sf.read(str(FIX / f"{name}.wav"))
        if wav.ndim > 1:
            wav = wav.mean(1)
        wav = np.ascontiguousarray(wav, dtype=np.float32)
        audio_s = wav.shape[0] / 16000

        # warmup (encoder graph capture / ggml autotune)
        ptr = lib.parakeet_capi_transcribe_pcm(
            ctx, wav.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(wav), 16000, 0)
        if ptr:
            lib.parakeet_capi_free_string(ptr)

        # timed reps
        ts = []
        last = ""
        wav_ptr = wav.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        for _ in range(5):
            t0 = time.perf_counter()
            ptr = lib.parakeet_capi_transcribe_pcm(
                ctx, wav_ptr, len(wav), 16000, 0)
            ts.append((time.perf_counter() - t0) * 1000.0)
            if ptr:
                last = ctypes.cast(ptr, ctypes.c_char_p).value.decode()
                lib.parakeet_capi_free_string(ptr)
        ts.sort()
        med = ts[len(ts) // 2]
        golden = (GOLDEN / f"parakeet_tdt_{name}_text.txt").read_text().strip()
        match = last.strip() == golden
        print(f"[probe] {name}: {audio_s:.1f}s audio, median {med:.1f}ms "
              f"(min {min(ts):.1f}) RTFx {audio_s/(med/1000):.0f}x "
              f"match_golden={match}")
        if not match:
            print(f"  golden: {golden[:80]!r}\n  got:    {last[:80]!r}")

    lib.parakeet_capi_free(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
