#!/usr/bin/env python3
"""Corpus WER of starling-bench logs against FLEURS references.

score_fleurs.py REFS.json LOG [LOG...]
LOG is starling-bench output: "<path> run=0 ... time=Xms ..." then the
transcript on the next line, indented. Also reports transcripts that differ
between the first log and each other log, and the summed time.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fast_engine"))
from wer import normalize  # noqa: E402


def edits(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def parse(path):
    out, times = {}, {}
    lines = open(path, errors="replace").read().splitlines()
    for i, l in enumerate(lines):
        m = re.match(r"^(\S+\.wav) run=0 .*time=([\d.]+)ms", l)
        if m and i + 1 < len(lines):
            name = os.path.basename(m.group(1))
            out[name] = lines[i + 1].strip()
            times[name] = float(m.group(2))
    return out, times


refs = json.load(open(sys.argv[1]))
logs = sys.argv[2:]
parsed = [parse(p) for p in logs]
for p, (hyp, times) in zip(logs, parsed):
    e = n = 0
    for name, ref in refs.items():
        if name not in hyp:
            continue
        r, h = normalize(ref).split(), normalize(hyp[name]).split()
        e += edits(r, h)
        n += len(r)
    print(f"{os.path.basename(p)}: {len(hyp)} clips, WER {100.0 * e / max(n, 1):.2f}% "
          f"({e}/{n} words), time {sum(times.values()) / 1000:.1f} s")
base = parsed[0][0]
for p, (hyp, _) in zip(logs[1:], parsed[1:]):
    diff = [k for k in base if k in hyp and base[k] != hyp[k]]
    print(f"{os.path.basename(p)} vs {os.path.basename(logs[0])}: {len(diff)} transcripts differ")
    for k in diff[:8]:
        print(f"  {k}\n    {base[k]}\n    {hyp[k]}")
