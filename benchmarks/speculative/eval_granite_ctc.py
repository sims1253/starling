#!/usr/bin/env python3
"""Compare native Granite CTC draft IDs with pinned Python reference IDs.

Build the native shared library with STARLING_GGML_SHARED=ON. Convert the
de575db6 source snapshot with --include-ctc-head. This runs the opt-in CTC
probe only; it does not claim speculative decoder speed or greedy parity.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()

    reference = json.loads(args.reference.read_text())
    audio_hash = hashlib.sha256(args.wav.read_bytes()).hexdigest()
    if audio_hash != reference["audio_sha256"]:
        parser.error(f"audio hash differs from reference: {audio_hash}")
    audio, sample_rate = sf.read(args.wav, dtype="float32")
    if sample_rate != 16000 or audio.ndim != 1:
        parser.error("Granite CTC parity expects mono 16 kHz WAV")
    audio = np.ascontiguousarray(audio)

    lib = ctypes.CDLL(str(args.library.resolve()))
    err = ctypes.c_char_p()
    lib.starling_ggml_granite_load.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p)]
    lib.starling_ggml_granite_load.restype = ctypes.c_void_p
    lib.starling_ggml_granite_free.argtypes = [ctypes.c_void_p]
    lib.starling_ggml_granite_ctc_draft.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int32), ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_char_p),
    ]
    lib.starling_ggml_granite_ctc_draft.restype = ctypes.c_bool
    handle = lib.starling_ggml_granite_load(
        str(args.gguf.resolve()).encode(), ctypes.byref(err))
    if not handle:
        raise RuntimeError((err.value or b"model load failed").decode())
    try:
        capacity = max(1, len(audio) // 160 + 1)
        ids = (ctypes.c_int32 * capacity)()
        count = ctypes.c_int32()
        ok = lib.starling_ggml_granite_ctc_draft(
            handle, audio.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(audio), ids, capacity, ctypes.byref(count), ctypes.byref(err))
        if not ok:
            raise RuntimeError((err.value or b"draft extraction failed").decode())
        native = list(ids[:count.value])
    finally:
        lib.starling_ggml_granite_free(handle)

    expected = reference["python_ctc_ids"]
    first_difference = next((i for i, (a, b) in enumerate(zip(native, expected))
                             if a != b), None)
    if first_difference is None and len(native) != len(expected):
        first_difference = min(len(native), len(expected))
    print(json.dumps({
        "model_revision": reference["model_revision"],
        "audio_sha256": audio_hash,
        "native_count": len(native),
        "python_count": len(expected),
        "exact": first_difference is None,
        "first_difference": first_difference,
        "native_ids": native,
    }, indent=2))
    return 0 if first_difference is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
