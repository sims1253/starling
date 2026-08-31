#!/usr/bin/env python3
"""Compare C++ audex stage dumps against the golden component tensors.

Reads the raw dumps written by the engine's probe/dump env vars
(STARLING_AUDEX_DUMP_MELSTK / STARLING_AUDEX_DUMP_ENC + STARLING_AUDEX_ONLY)
and diffs them bitwise against golden/audex_short_<stage>.pt, handling the
layout conventions (C++ feature-major vs torch time-major; f32 dumps of bf16
nodes). Local debug tooling -- not part of the test suite.

Usage: python3 scripts/audex_stage_compare.py <stage> <dump.f32> [--mel-bin]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = REPO_ROOT / "golden"


def bf16_round(x: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    return t.to(torch.bfloat16).view(torch.uint16).numpy()


def compare(name: str, got: np.ndarray, want: torch.Tensor, note: str) -> bool:
    want_np = want.contiguous().cpu()
    print(f"[{name}] cpp {got.shape} vs golden {tuple(want_np.shape)} ({note})")
    if got.size != want_np.numel():
        print(f"[{name}] SIZE MISMATCH: {got.size} vs {want_np.numel()}")
        return False
    if want_np.dtype == torch.bfloat16:
        want_u16 = want_np.view(torch.uint16).numpy().ravel()
        got_u16 = bf16_round(got.ravel()).ravel()
        eq = np.array_equal(got_u16, want_u16)
        nbad = int((got_u16 != want_u16).sum())
        print(f"[{name}] bf16 bitwise: {'EXACT' if eq else f'{nbad}/{got_u16.size} mismatched'}")
        return eq
    want_f = want_np.numpy().ravel().astype(np.float32)
    eq = np.array_equal(got.ravel(), want_f)
    if not eq:
        d = np.abs(got.ravel().astype(np.float64) - want_f.astype(np.float64))
        print(f"[{name}] f32 bitwise: FAIL max|d|={d.max():.3e} nbad={(got.ravel()!=want_f).sum()}")
    else:
        print(f"[{name}] f32 bitwise: EXACT")
    return eq


def main() -> int:
    stage, dump = sys.argv[1], sys.argv[2]
    raw = np.fromfile(dump, dtype=np.float32)
    if stage == "mel":
        # MELSTK dump: raw bf16 stream, feat-major (m, t) at m*3000+t —
        # identical memory to torch's (128, 3000) bf16.
        data = np.fromfile(dump, dtype=np.uint16)
        want = torch.load(GOLDEN / "audex_short_mel.pt")
        eq = data.size == want.numel() and np.array_equal(
            data, want.view(torch.uint16).numpy().ravel())
        print(f"[mel] bf16 bitwise: {'EXACT' if eq else 'MISMATCH'} "
              f"({data.size} vs {want.numel()})")
        return 0 if eq else 1
    if stage in ("enc", "proj"):
        # f32 dump of a bf16 node, feature-major [D, T]; golden is (T, D) bf16.
        d, t = (1280, 750) if stage == "enc" else (2048, 750)
        got = raw.reshape(d, t).T.ravel()  # -> time-major, feature-inner
        key = "enc_hidden" if stage == "enc" else "audio_embeds"
        want = torch.load(GOLDEN / f"audex_short_{key}.pt")
        ok = compare(stage, got, want, "transposed to time-major")
        return 0 if ok else 1
    if stage == "melstk_f32":
        # STARLING_AUDEX_MEL_DUMP (shared mel policy): f32 [128*3000]
        # feat-major vs golden (128, 3000) bf16.
        got = raw
        want = torch.load(GOLDEN / "audex_short_mel.pt")
        ok = compare("melf32", got, want.float(), "feat-major f32 vs bf16-rounded")
        return 0 if ok else 1
    print(f"unknown stage {stage}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
