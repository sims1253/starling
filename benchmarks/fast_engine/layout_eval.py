#!/usr/bin/env python3
"""Desktop weight-only WER eval for candidate Pixel layouts (#319, #317).

Pipeline per candidate layout spec (a rules file for starling-layout-quant):

  1. `pack`   MOSS source safetensors -> .pack (skipped when cached)
  2. `eval`   .pack + q4e8 GGUF -> an eval GGUF whose packed tensors are
              re-encoded into their bit-exact ggml block type when one
              exists (w4g32sym* -> Q4_0, w4g32asym -> Q4_1, w8g32sym ->
              Q8_0: same f16 scales and codes, zero store rounding); other
              layouts dequantize to bf16 (the engine accepts bf16 weights;
              f16/f32 trip ggml asserts — see the research log). The bf16
              store costs ~0.4% per weight, material only for 8-bit
              candidates, so prefer native-typed layouts when comparing
              W8-family candidates.
  3. `wer`    the ggml engine on the eval GGUF over the FLEURS clip sets
              (en_us 100 + de_de 100 + one tail language).

Results are cached by (model, source sha256 from the pack header, rules
content, rounding, clip refs) so the #317 loop never re-scores a layout.

Usage:
  layout_eval.py --rules models/rebuild-w4e8.rules --name w4e8-rebuild \
      [--lib build-vk/libstarling_ggml.so] [--tool build-cpu/starling-layout-quant] \
      [--source SNAP] [--imatrix models/moss-full.imx] \
      [--gguf-ref models/...q4e8-fullimx.gguf] \
      [--clips en=/tmp/fleurs_en de=/tmp/fleurs_de_de tail=/tmp/fleurs_ta_in]
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from wer import normalize  # noqa: E402

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".layout_eval_cache")
DEFAULT_SOURCE = os.path.expanduser(
    "~/.cache/huggingface/hub/models--OpenMOSS-Team--MOSS-Transcribe-preview-2B/snapshots"
    "/c98175cb20e48bd9be4e95f6c85f2af18899f780/model-00000-of-00001.safetensors")


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def read_pack_header(path: str) -> dict:
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"SFPK":
            raise SystemExit(f"{path}: not a pack file")
        ver = int.from_bytes(f.read(4), "little")
        if ver != 1:
            raise SystemExit(f"{path}: unsupported version {ver}")
        source = f.read(64).split(b"\0")[0].decode()
        rounding = f.read(15).split(b"\0")[0].decode()
        n = int.from_bytes(f.read(4), "little")
    return {"source": source, "rounding": rounding, "tensors": n}


def run_wer(lib: str, gguf: str, clips_dir: str) -> tuple[float, int]:
    """One ggml-engine pass over a clip dir; returns (wer%, n_clips)."""
    env = {"STARLING_ENGINE": "ggml", "STARLING_GGML_THREADS": "8",
           "PATH": os.environ.get("PATH", ""), "HOME": os.environ["HOME"]}
    lib = os.path.abspath(lib)
    libc = ctypes.CDLL(lib)
    libc.starling_ggml_load.restype = ctypes.c_void_p
    libc.starling_ggml_load.argtypes = [ctypes.c_int, ctypes.c_char_p]
    libc.starling_ggml_transcribe_pcm.restype = ctypes.c_void_p
    libc.starling_ggml_transcribe_pcm.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64, ctypes.c_int]
    libc.starling_ggml_free_string.argtypes = [ctypes.c_void_p]
    libc.starling_ggml_free.argtypes = [ctypes.c_void_p]
    for k, v in os.environ.items():
        env.setdefault(k, v)
    env.update({"STARLING_ENGINE": "ggml"})
    refs = json.load(open(os.path.join(clips_dir, "refs.json")))
    clips = []
    for name in sorted(refs):
        with wave.open(os.path.join(clips_dir, name)) as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        clips.append((name, pcm.astype(np.float32) / 32768.0))
    ctx = libc.starling_ggml_load(2, gguf.encode())
    if not ctx:
        libc.starling_ggml_last_error.restype = ctypes.c_char_p
        libc.starling_ggml_last_error.argtypes = [ctypes.c_void_p]
        raise SystemExit(f"load failed: {libc.starling_ggml_last_error(None).decode()}")
    # warm-up (pipeline compilation) outside the WER pass
    p = libc.starling_ggml_transcribe_pcm(
        ctx, clips[0][1].ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(clips[0][1]), 16000)
    libc.starling_ggml_free_string(p)
    errs = words = 0
    for name, pcm in clips:
        p = libc.starling_ggml_transcribe_pcm(
            ctx, pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(pcm), 16000)
        hyp = ctypes.cast(p, ctypes.c_char_p).value.decode("utf-8", "replace") if p else ""
        libc.starling_ggml_free_string(p)
        r, h = normalize(refs[name]).split(), normalize(hyp).split()
        errs += edit_distance(r, h)
        words += len(r)
    libc.starling_ggml_free(ctx)
    return 100.0 * errs / max(words, 1), len(clips)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", required=True, help="starling-layout-quant rules file")
    ap.add_argument("--name", required=True, help="cache/report label")
    ap.add_argument("--lib", default="build-vk/libstarling_ggml.so")
    ap.add_argument("--tool", default="build-cpu/starling-layout-quant")
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--imatrix", default="models/moss-full.imx")
    ap.add_argument("--gguf-ref",
                   default="models/moss-transcribe-preview-2b-q4e8-fullimx.gguf",
                   help="reference GGUF for the eval build: the q4e8 model, so the"
                        " F32 1-D tensors and engine graph fusions match the baseline")
    ap.add_argument("--work", default="/tmp/layout_eval")
    ap.add_argument("--clips", nargs="+", default=[
        "en=/tmp/fleurs_en", "de=/tmp/fleurs_de_de", "tail=/tmp/fleurs_ta_in"])
    a = ap.parse_args()

    os.makedirs(a.work, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    clips = {}
    for c in a.clips:
        k, v = c.split("=", 1)
        clips[k] = v

    rules = open(a.rules).read()
    pack = os.path.join(a.work, f"{a.name}.pack")
    if not os.path.exists(pack):
        cmd = [a.tool, "pack", "--source", a.source, "--out", pack, "--rules", a.rules,
               "--threads", str(os.cpu_count() or 8)]
        if a.imatrix:
            cmd += ["--imatrix", a.imatrix]
        print(f"[layout-eval] packing {a.name} ...", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:], sep="\n")
            return 1
    hdr = read_pack_header(pack)

    # Cache key: model, source hash, descriptor (rules), rounding, clip refs.
    key_mat = ["moss", hdr["source"], rules, hdr["rounding"], os.path.basename(a.gguf_ref)]
    for k in sorted(clips):
        refs = json.load(open(os.path.join(clips[k], "refs.json")))
        key_mat.append(k + ":" + hashlib.sha256(
            json.dumps(refs, sort_keys=True).encode()).hexdigest()[:16])
    key = hashlib.sha256("|".join(key_mat).encode()).hexdigest()[:24]
    cache_path = os.path.join(CACHE_DIR, key + ".json")
    if os.path.exists(cache_path):
        row = json.load(open(cache_path))
        if row.get("name") == a.name or True:
            print(f"[layout-eval] cached: {json.dumps(row['wer'])}")
            return 0

    eval_gguf = os.path.join(a.work, f"{a.name}-eval.gguf")
    if not os.path.exists(eval_gguf):
        print(f"[layout-eval] building eval GGUF ...", flush=True)
        r = subprocess.run([a.tool, "eval", "--pack", pack, "--gguf-in", a.gguf_ref,
                            "--gguf-out", eval_gguf], capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:], sep="\n")
            return 1

    wer = {}
    for k, v in sorted(clips.items()):
        w, n = run_wer(a.lib, eval_gguf, v)
        wer[k] = round(w, 2)
        print(f"[layout-eval] {a.name} {k}: {w:.2f}% ({n} clips)", flush=True)
    row = {"name": a.name, "rules": rules, "source": hdr["source"], "rounding": hdr["rounding"],
           "pack_bytes": os.path.getsize(pack), "wer": wer}
    json.dump(row, open(cache_path, "w"), indent=1)
    print(f"[layout-eval] {a.name}: {json.dumps(wer)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
