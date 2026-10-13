#!/usr/bin/env python3
"""Medians of ab_session.sh's `ab` phase: enc+prefill (Parakeet: encoder),
wall per transcription and decode ms/token, per model, clip and arm.

python -I ab_summary.py LOGDIR
Every ab_<model>_<clip>_r<round>_sg<arm>.txt holds one starling-bench
invocation (--warmup --runs N); all timed runs of an arm are pooled.
"""
import collections
import glob
import os
import re
import statistics
import sys

d = sys.argv[1]
runs = collections.defaultdict(lambda: {"ep": [], "wall": [], "tok": []})
for f in glob.glob(f"{d}/ab_*_r*_sg*.txt"):
    m = re.match(r"ab_(.+)_(short|medium)_r\d+_sg(\d)\.txt", os.path.basename(f))
    if not m:
        continue
    key = (m.group(1), m.group(2), m.group(3))
    txt = open(f).read()
    for line in txt.splitlines():
        ep = re.search(r"(?:enc\+prefill|encoder)=([\d.]+)ms", line)
        tot = re.search(r"total=([\d.]+)ms", line)
        dec = re.search(r"decode=([\d.]+)ms \((\d+) tokens", line)
        if ep and tot:
            runs[key]["ep"].append(float(ep.group(1)))
            runs[key]["wall"].append(float(tot.group(1)))
            if dec and int(dec.group(2)) > 1:
                runs[key]["tok"].append(float(dec.group(1)) / int(dec.group(2)))
models = sorted({k[0] for k in runs})
print("| Model | clip | enc+prefill old → new | wall old → new | decode ms/tok old / new | n |")
print("| --- | --- | --- | --- | --- | --- |")
med = lambda v: statistics.median(v) if v else float("nan")
for mdl in models:
    for clip in ("short", "medium"):
        a, b = runs.get((mdl, clip, "0")), runs.get((mdl, clip, "1"))
        if not a or not b:
            continue
        ea, eb, wa, wb = med(a["ep"]), med(b["ep"]), med(a["wall"]), med(b["wall"])
        print(f"| {mdl} | {clip} | {ea:.0f} → **{eb:.0f}** ms ({100 * (eb / ea - 1):+.0f} %) | "
              f"{wa / 1000:.2f} → **{wb / 1000:.2f}** s ({100 * (wb / wa - 1):+.0f} %) | "
              f"{med(a['tok']):.1f} / {med(b['tok']):.1f} | {len(a['ep'])}+{len(b['ep'])} |")
