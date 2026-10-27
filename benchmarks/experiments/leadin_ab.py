#!/usr/bin/env python3
"""Silent lead-in A/B for Parakeet-family final passes (#494).

Does a block of digital zeros in front of a take change recognition of its
first words? Two subcommands:

  prep   build the clip sets from exported FLEURS directories
         (export_fleurs.py layout: *.wav + refs.json):
           untrimmed/<lead_ms>/  the clip as exported, <lead> s of zeros first
           trimmed/<lead_ms>/    leading silence cut so the energy onset sits
                                 --margin-ms after the first sample, then the
                                 zeros. Onset = first 10 ms frame 15 dB above
                                 the clip's 10th-percentile frame.
  score  read starling-bench outputs (<tag>.<cond>.<lead_ms>.txt, one
         `--runs 1` process per directory) and report, per model and
         condition, corpus WER, first-word and first-two-word errors and
         decode time against the 0 ms arm, with a paired bootstrap CI
         (fixed seed) on the WER delta.

Usage:
  leadin_ab.py prep --src DIR [--src DIR ...] --out CLIPS
  for each model, condition and lead:
    STARLING_ENGINE=ggml STARLING_GGML_DEVICE=cpu starling-bench --model parakeet \\
      --gguf M.gguf --runs 1 CLIPS/<cond>/<lead>/*.wav > OUT/<tag>.<cond>.<lead>.txt
  leadin_ab.py score --clips CLIPS --outputs OUT [--json results.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fast_engine"))
from wer import normalize  # noqa: E402

SR = 16000
LEADS_MS = [0, 100, 250, 500]
CONDS = ["trimmed", "untrimmed"]
SEED = 494
BOOT = 2000


def read_wav(path: str) -> np.ndarray:
    with wave.open(path) as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def write_wav(path: str, x: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(x.astype(np.int16).tobytes())


def onset(x: np.ndarray) -> int:
    f = SR // 100
    n = len(x) // f
    rms = np.sqrt((x[: n * f].astype(np.float64).reshape(n, f) ** 2).mean(1) + 1e-9)
    db = 20 * np.log10(rms)
    return int(np.argmax(db > np.percentile(db, 10) + 15)) * f


def prep(a: argparse.Namespace) -> int:
    refs: dict[str, str] = {}
    meta: dict[str, dict[str, int]] = {}
    margin = a.margin_ms * SR // 1000
    for d in a.src:
        tag = os.path.basename(os.path.normpath(d))
        r = json.load(open(os.path.join(d, "refs.json")))
        for name in sorted(r):
            key = f"{tag}__{name}"
            x = read_wav(os.path.join(d, name))
            on = onset(x)
            cut = max(0, on - margin)
            refs[key] = r[name]
            meta[key] = {"onset_ms": on * 1000 // SR, "cut_ms": cut * 1000 // SR}
            for cond, y in (("untrimmed", x), ("trimmed", x[cut:])):
                for lead in LEADS_MS:
                    z = np.zeros(lead * SR // 1000, dtype=np.int16)
                    write_wav(os.path.join(a.out, cond, str(lead), key), np.concatenate([z, y]))
    json.dump(refs, open(os.path.join(a.out, "refs.json"), "w"), indent=1)
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    on = np.array([m["onset_ms"] for m in meta.values()])
    print(f"{len(refs)} clips; onset ms median {np.median(on):.0f} p10 {np.percentile(on, 10):.0f} "
          f"p90 {np.percentile(on, 90):.0f}")
    return 0


def align(ref: list[str], hyp: list[str]) -> tuple[int, list[bool]]:
    """Edit distance and, per reference word, whether it was matched exactly.

    An insertion directly before reference word i counts against word i, so a
    hallucinated word in front of the first word is a first-word error.
    """
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    ok = [False] * n
    ins_before = [False] * (n + 1)
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ok[i - 1] = ref[i - 1] == hyp[j - 1]
            i, j = i - 1, j - 1
        elif j > 0 and d[i][j] == d[i][j - 1] + 1:
            ins_before[i] = True  # hyp[j-1] sits between ref[i-1] and ref[i]
            j -= 1
        else:
            i -= 1
    ok = [o and not ins_before[k] for k, o in enumerate(ok)]
    return d[n][m], ok


_RUN = re.compile(r"^(\S+\.wav) run=0 audio=([0-9.]+)s time=([0-9.]+)ms")


def parse_bench(path: str) -> dict[str, tuple[str, float]]:
    """file key -> (transcript, decode ms) from one starling-bench output."""
    out: dict[str, tuple[str, float]] = {}
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    if not any(line == "exit=0" for line in lines):
        raise SystemExit(f"{path}: bench did not finish (no exit=0)")
    for k, line in enumerate(lines):
        mt = _RUN.match(line)
        if mt:
            text = lines[k + 1].strip() if k + 1 < len(lines) and lines[k + 1].startswith("  ") else ""
            out[os.path.basename(mt.group(1))] = (text, float(mt.group(3)))
    return out


def score(a: argparse.Namespace) -> int:
    refs = json.load(open(os.path.join(a.clips, "refs.json")))
    keys = sorted(refs)
    nref = {k: normalize(refs[k]).split() for k in keys}
    names = [os.path.basename(p) for p in glob.glob(os.path.join(a.outputs, "*.txt"))]
    tags = sorted({m.group(1) for m in (re.match(r"(.+)\.(?:trimmed|untrimmed)\.\d+\.txt$", n) for n in names) if m})
    rng = random.Random(SEED)
    boot_idx = [[rng.randrange(len(keys)) for _ in keys] for _ in range(BOOT)]
    results: dict = {"seed": SEED, "bootstrap": BOOT, "clips": len(keys), "models": {}}
    for tag in tags:
        results["models"][tag] = {}
        for cond in CONDS:
            per: dict[int, dict] = {}
            for lead in LEADS_MS:
                p = os.path.join(a.outputs, f"{tag}.{cond}.{lead}.txt")
                if not os.path.exists(p):
                    continue
                got = parse_bench(p)
                missing = [k for k in keys if k not in got]
                if missing:
                    raise SystemExit(f"{p}: {len(missing)} clips missing, e.g. {missing[0]}")
                errs, w1, w2, ms, changed = [], 0, 0, 0.0, []
                for k in keys:
                    hyp = normalize(got[k][0]).split()
                    e, ok = align(nref[k], hyp)
                    errs.append(e)
                    w1 += not ok[0]
                    w2 += not all(ok[:2])
                    ms += got[k][1]
                    changed.append(got[k][0])
                per[lead] = {"errs": errs, "w1": w1, "w2": w2, "ms": ms, "text": changed}
            if 0 not in per:
                continue
            words = [len(nref[k]) for k in keys]
            base = per[0]
            rows = {}
            for lead, r in sorted(per.items()):
                wer = 100 * sum(r["errs"]) / sum(words)
                deltas = []
                for idx in boot_idx:
                    tw = sum(words[i] for i in idx)
                    deltas.append(100 * (sum(r["errs"][i] for i in idx) - sum(base["errs"][i] for i in idx)) / tw)
                deltas.sort()
                rows[lead] = {
                    "wer": round(wer, 3),
                    "wer_delta": round(wer - 100 * sum(base["errs"]) / sum(words), 3),
                    "wer_delta_ci95": [round(deltas[int(0.025 * BOOT)], 3), round(deltas[int(0.975 * BOOT)], 3)],
                    "first_word_errors": r["w1"],
                    "first_two_word_errors": r["w2"],
                    "transcripts_changed_vs_0": sum(x != y for x, y in zip(r["text"], base["text"])),
                    "decode_ms_total": round(r["ms"]),
                    "decode_ms_vs_0_pct": round(100 * (r["ms"] / base["ms"] - 1), 1),
                }
            results["models"][tag][cond] = rows
    if a.json:
        json.dump(results, open(a.json, "w"), indent=1)
    for tag, conds in results["models"].items():
        for cond, rows in conds.items():
            print(f"\n{tag} / {cond} ({len(keys)} clips)")
            print(f"{'lead':>5} {'WER':>6} {'dWER':>7} {'95% CI':>17} {'1st':>4} {'1st2':>5} {'chg':>4} {'time':>7}")
            for lead, r in rows.items():
                ci = f"[{r['wer_delta_ci95'][0]:+.2f}, {r['wer_delta_ci95'][1]:+.2f}]"
                print(f"{lead:>5} {r['wer']:6.2f} {r['wer_delta']:+7.2f} {ci:>17} {r['first_word_errors']:4d} "
                      f"{r['first_two_word_errors']:5d} {r['transcripts_changed_vs_0']:4d} {r['decode_ms_vs_0_pct']:+6.1f}%")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--src", action="append", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--margin-ms", type=int, default=50)
    s = sub.add_parser("score")
    s.add_argument("--clips", required=True)
    s.add_argument("--outputs", required=True)
    s.add_argument("--json")
    a = ap.parse_args()
    return prep(a) if a.cmd == "prep" else score(a)


if __name__ == "__main__":
    sys.exit(main())
