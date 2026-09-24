#!/usr/bin/env python3
"""Parakeet parity: ggml engine vs the Vulkan fast engine on the same build.

Loads the model twice from one libstarling_ggml (STARLING_ENGINE=ggml, then
=fast), runs every WAV through both and reports

  * encoder output difference (joint-projected encoder, max/mean abs, cosine)
  * token-stream equality (non-blank tokens; blank cadence is reported apart)
  * transcript equality

Usage:
  parity_parakeet.py --lib build-fast/libstarling_ggml.so --gguf MODEL.gguf a.wav ...
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import wave

import numpy as np

KIND_PARAKEET = 1


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit(f"{path}: expected 16 kHz mono 16-bit PCM")
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return (data.astype(np.float32) / 32768.0).copy()


class Engine:
    def __init__(self, lib: ctypes.CDLL, gguf: str, engine: str):
        os.environ["STARLING_ENGINE"] = engine
        self.lib = lib
        self.ctx = lib.starling_ggml_load(KIND_PARAKEET, gguf.encode())
        if not self.ctx:
            msg = lib.starling_ggml_last_error(None)
            raise SystemExit(f"load ({engine}) failed: {msg.decode() if msg else 'unknown error'}")
        self.name = engine

    def encode(self, pcm: np.ndarray) -> np.ndarray:
        t = ctypes.c_int(0)
        ptr = self.lib.starling_ggml_parakeet_encode_pub(
            self.ctx, pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(pcm), ctypes.byref(t))
        if not ptr:
            raise SystemExit(f"encode ({self.name}): {self.lib.starling_ggml_last_error(self.ctx).decode()}")
        n = t.value * 640
        out = np.ctypeslib.as_array(ptr, shape=(n,)).copy().reshape(t.value, 640)
        self.lib.starling_ggml_free_string(ctypes.cast(ptr, ctypes.c_char_p))
        return out

    def ids(self, pcm: np.ndarray) -> list[int]:
        n = ctypes.c_int64(0)
        ptr = self.lib.starling_ggml_parakeet_decode_ids_pub(
            self.ctx, pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(pcm), ctypes.byref(n))
        if not ptr:
            raise SystemExit(f"ids ({self.name}): {self.lib.starling_ggml_last_error(self.ctx).decode()}")
        out = ptr[: n.value]
        self.lib.starling_ggml_free_string(ctypes.cast(ptr, ctypes.c_char_p))
        return out

    def text(self, pcm: np.ndarray) -> str:
        ptr = self.lib.starling_ggml_transcribe_pcm(
            self.ctx, pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(pcm), 16000)
        if not ptr:
            raise SystemExit(f"text ({self.name}): {self.lib.starling_ggml_last_error(self.ctx).decode()}")
        s = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8", "replace")
        self.lib.starling_ggml_free_string(ctypes.cast(ptr, ctypes.c_char_p))
        return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--ref-engine", default="ggml")
    ap.add_argument("wavs", nargs="+")
    a = ap.parse_args()

    lib = ctypes.CDLL(a.lib)
    lib.starling_ggml_load.restype = ctypes.c_void_p
    lib.starling_ggml_load.argtypes = [ctypes.c_int, ctypes.c_char_p]
    lib.starling_ggml_last_error.restype = ctypes.c_char_p
    lib.starling_ggml_last_error.argtypes = [ctypes.c_void_p]
    lib.starling_ggml_parakeet_encode_pub.restype = ctypes.POINTER(ctypes.c_float)
    lib.starling_ggml_parakeet_encode_pub.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64, ctypes.POINTER(ctypes.c_int)]
    lib.starling_ggml_parakeet_decode_ids_pub.restype = ctypes.POINTER(ctypes.c_int64)
    lib.starling_ggml_parakeet_decode_ids_pub.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64)]
    lib.starling_ggml_transcribe_pcm.restype = ctypes.c_void_p
    lib.starling_ggml_transcribe_pcm.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64, ctypes.c_int]
    lib.starling_ggml_free_string.argtypes = [ctypes.c_char_p]

    ref = Engine(lib, a.gguf, a.ref_engine)
    fast = Engine(lib, a.gguf, "fast")
    blank = 8192
    n_text_equal = n_tok_equal = 0
    for path in a.wavs:
        pcm = read_wav(path)
        er, ef = ref.encode(pcm), fast.encode(pcm)
        d = np.abs(er - ef)
        cos = float((er * ef).sum() / (np.linalg.norm(er) * np.linalg.norm(ef) + 1e-30))
        ir, if_ = ref.ids(pcm), fast.ids(pcm)
        tr = [i for i in ir if i != blank]
        tf = [i for i in if_ if i != blank]
        xr, xf = ref.text(pcm), fast.text(pcm)
        n_tok_equal += tr == tf
        n_text_equal += xr == xf
        print(f"{os.path.basename(path)}: T'={er.shape[0]} enc max|d|={d.max():.4g} mean|d|={d.mean():.3g} "
              f"(|ref| mean {np.abs(er).mean():.3g}) cos={cos:.7f}  tokens {'==' if tr == tf else '!='} "
              f"({len(tr)} vs {len(tf)}; blanks {len(ir) - len(tr)} vs {len(if_) - len(tf)})  "
              f"text {'==' if xr == xf else '!='}")
        if xr != xf:
            print(f"   ref : {xr}\n   fast: {xf}")
    print(f"tokens equal {n_tok_equal}/{len(a.wavs)}, text equal {n_text_equal}/{len(a.wavs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
