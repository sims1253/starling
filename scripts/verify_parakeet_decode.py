#!/usr/bin/env python3
"""Verify the IN-TREE parakeet engine (libstarling_ggml) byte-exactness.

Two checks:
  (A) TEXT byte-exact vs golden parakeet_tdt_*_text.txt  (the parity gate;
      blanks drop in detokenization, so this is independent of blank placement).
  (B) ID STREAM: multistep/GPU ids == serial-loop ids (the in-tree serial loop
      is the byte-identical REFERENCE per plans/wave-c-...md req #2; the HF
      golden _ids.pt differs from the C++ port by one blank, so it is NOT the
      reference for ids). Run once under STARLING_GGML_TDT_SERIAL=1 to WRITE the
      reference ids, then again under the GPU path to COMPARE.

Usage:
  # 1. capture the serial-loop reference ids (CPU, byte-identical reference):
  STARLING_GGML_DEVICE=cpu STARLING_GGML_TDT_SERIAL=1 STARLING_GGML_REF_WRITE=1 \
      uv run python scripts/verify_parakeet_decode.py
  # 2. compare the GPU/multistep path against it:
  STARLING_GGML_PARAKEET_MODEL=... uv run python scripts/verify_parakeet_decode.py
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests" / "fixtures"))
sys.path.insert(0, str(REPO / "src"))
import make_fixtures as mkfx  # noqa: E402

FIXTURES = mkfx.load_fixtures()
GOLDEN = REPO / "golden"
REF_DIR = REPO / "build" / "decode_ref"

MODEL = Path(os.environ.get(
    "STARLING_GGML_PARAKEET_MODEL",
    str(Path.home() / "Documents" / "parakeet.cpp" / "models" / "tdt-0.6b-v3-f16.gguf"),
))
LIB = REPO / "build" / "libstarling_ggml.so"
WRITE_REF = bool(os.environ.get("STARLING_GGML_REF_WRITE"))
BLANK = 8192

PARAKEET_TDT = 1
_c_float_p = ctypes.POINTER(ctypes.c_float)


def main() -> int:
    assert LIB.exists(), f"missing {LIB}"
    assert MODEL.exists(), f"missing model {MODEL}"
    lib = ctypes.CDLL(str(LIB))
    lib.starling_ggml_load.restype = ctypes.c_void_p
    lib.starling_ggml_load.argtypes = [ctypes.c_int, ctypes.c_char_p]
    lib.starling_ggml_free.argtypes = [ctypes.c_void_p]
    lib.starling_ggml_free.restype = None
    lib.starling_ggml_free_string.argtypes = [ctypes.POINTER(ctypes.c_char)]
    lib.starling_ggml_parakeet_decode_pub.restype = ctypes.POINTER(ctypes.c_char)
    lib.starling_ggml_parakeet_decode_pub.argtypes = [ctypes.c_void_p, _c_float_p, ctypes.c_int64]
    lib.starling_ggml_parakeet_decode_ids_pub.restype = ctypes.POINTER(ctypes.c_int64)
    lib.starling_ggml_parakeet_decode_ids_pub.argtypes = [
        ctypes.c_void_p, _c_float_p, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64)]

    ctx = lib.starling_ggml_load(PARAKEET_TDT, str(MODEL).encode())
    assert ctx, "load failed"

    mode = "WRITE-REF" if WRITE_REF else "COMPARE"
    dev = os.environ.get("STARLING_GGML_DEVICE", "(auto)")
    serial = bool(os.environ.get("STARLING_GGML_TDT_SERIAL"))
    print(f"== verify_parakeet_decode mode={mode} device={dev} serial={serial} ==")

    if WRITE_REF:
        REF_DIR.mkdir(parents=True, exist_ok=True)
    all_ok = True
    try:
        for name in ["short", "medium", "long"]:
            pcm = np.ascontiguousarray(FIXTURES[name], dtype=np.float32)
            ptr = pcm.ctypes.data_as(_c_float_p)
            tp = lib.starling_ggml_parakeet_decode_pub(ctx, ptr, pcm.size)
            text = ctypes.cast(tp, ctypes.c_char_p).value.decode("utf-8", "replace")
            lib.starling_ggml_free_string(ctypes.cast(tp, ctypes.POINTER(ctypes.c_char)))
            n = ctypes.c_int64(0)
            ip = lib.starling_ggml_parakeet_decode_ids_pub(ctx, ptr, pcm.size, ctypes.byref(n))
            ids = np.ctypeslib.as_array(ip, shape=(n.value,)).copy()

            g_text = (GOLDEN / f"parakeet_tdt_{name}_text.txt").read_text()
            text_ok = (text == g_text)
            status = [f"text={'OK' if text_ok else 'MISMATCH'}",
                      f"len={len(ids)} blanks={int((ids==BLANK).sum())}"]
            if not text_ok:
                all_ok = False
                status += [f"golden={g_text!r}", f"got={text!r}"]

            if WRITE_REF:
                np.save(REF_DIR / f"{name}.npy", ids)
                status.append("(ref written)")
            else:
                ref_path = REF_DIR / f"{name}.npy"
                if ref_path.exists():
                    ref = np.load(ref_path)
                    ids_ok = np.array_equal(ids, ref)
                    status.append(f"ids_vs_serial={'OK' if ids_ok else 'MISMATCH'}")
                    if not ids_ok:
                        all_ok = False
                        m = min(len(ids), len(ref))
                        d = next((i for i in range(m) if ids[i] != ref[i]), m)
                        status += [f"first diff@{d}: got={ids[d] if d<len(ids) else None} "
                                   f"ref={ref[d] if d<len(ref) else None} "
                                   f"(lens got={len(ids)} ref={len(ref)})"]
                else:
                    status.append("ids_vs_serial=NO-REF")
            print(f"[{name}] " + " | ".join(status))
    finally:
        lib.starling_ggml_free(ctx)

    print("RESULT:", "OK" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
