#!/usr/bin/env python3
"""Corpus WER of starling-bench variants on a FLEURS clip directory.

python wer_bench.py --bench build-host/starling-bench --model qwen3 --gguf M.gguf \
    --clips /tmp/fleurs_en --out results.json --arm sg0:STARLING_FAST_SG=0 --arm sg1:STARLING_FAST_SG=1

Each arm is one starling-bench process over all clips (--runs 1, no warmup)
with STARLING_ENGINE=fast plus the arm's VAR=VALUE settings (comma separated).
Scores with benchmarks/fast_engine/wer.py normalization; prints corpus WER per
arm and how many transcripts differ from the first arm.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fast_engine"))
from wer import normalize  # noqa: E402


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def run_arm(a, env_extra):
    refs = json.load(open(os.path.join(a.clips, "refs.json")))
    names = sorted(refs)[: a.limit]
    env = dict(os.environ, STARLING_ENGINE="fast")
    env.update(env_extra)
    cmd = [a.bench, "--model", a.model, "--gguf", a.gguf, "--runs", "1"] + [os.path.join(a.clips, n) for n in names]
    t0 = time.time()
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=a.timeout).stdout
    hyp, cur = {}, None
    for line in out.splitlines():
        m = re.match(r"^(\S+\.wav) run=\d+", line)
        if m:
            cur = os.path.basename(m.group(1))
            hyp[cur] = ""
        elif line.startswith("  ") and cur is not None:
            hyp[cur] = line.strip()
            cur = None
    return names, refs, hyp, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--arm", action="append", required=True, help="name:VAR=V[,VAR=V]")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--out")
    a = ap.parse_args()
    res, first = {}, None
    for spec in a.arm:
        name, _, kv = spec.partition(":")
        env = dict(p.split("=", 1) for p in kv.split(",") if p)
        names, refs, hyp, secs = run_arm(a, env)
        errs = words = 0
        for n in names:
            r, h = normalize(refs[n]).split(), normalize(hyp.get(n, "")).split()
            errs += edit_distance(r, h)
            words += len(r)
        missing = sum(1 for n in names if n not in hyp)
        diff = sum(1 for n in names if first is not None and hyp.get(n) != first.get(n))
        res[name] = {"wer": errs / max(words, 1), "secs": secs, "missing": missing, "differ": diff, "hyp": hyp}
        print(f"{os.path.basename(a.gguf):48s} {name:6s} WER {100 * errs / max(words, 1):6.2f} %  "
              f"{secs:7.1f} s  missing {missing}  differ-from-first {diff if first is not None else '-'}", flush=True)
        if first is None:
            first = hyp
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
