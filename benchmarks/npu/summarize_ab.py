#!/usr/bin/env python3
"""Summarize phone_npu_ab.sh logs: median wall / enc+prefill / decode per clip
and mode, model load, and whether the transcripts agree.
summarize_ab.py OUTDIR"""
import glob
import os
import re
import statistics as st
import sys

d = sys.argv[1]
rows = {}
for f in sorted(glob.glob(os.path.join(d, "*_*_*.txt"))):
    m = re.match(r"(.+)_(gpu|npuenc|npu)_(\d+)\.txt$", os.path.basename(f))
    if not m:
        continue
    clip, mode = m.group(1), m.group(2)
    txt = open(f, errors="replace").read()
    r = rows.setdefault((clip, mode), {"wall": [], "encpre": [], "dec": [], "load": [], "create": [], "npu": [], "tx": set()})
    # The first run of each process is the warm-up (not printed as run=); skip
    # the first timing line per file for enc+prefill too.
    walls = [float(x) for x in re.findall(r" run=\d+ audio=\S+ time=([\d.]+)ms", txt)]
    r["wall"] += walls
    tl = re.findall(r"enc\+prefill=([\d.]+)ms.*?decode=([\d.]+)ms", txt)
    for ep, dc in tl[-len(walls):] if walls else []:
        r["encpre"].append(float(ep))
        r["dec"].append(float(dc))
    r["npu"] += [float(x) for x in re.findall(r"npu enc=([\d.]+)ms", txt)[-len(walls):]] if walls else []
    r["load"] += [float(x) for x in re.findall(r"^load ([\d.]+) ms", txt, re.M)]
    r["create"] += [float(x) for x in re.findall(r"weights\+buffers ([\d.]+) ms", txt)]
    for t in re.findall(r" run=\d+ audio=.*\n(.*)", txt):
        r["tx"].add(t.strip())
med = lambda v: st.median(v) if v else float("nan")  # noqa: E731
print(f"{'clip':10s} {'mode':4s} {'wall ms':>9s} {'enc+pre':>8s} {'npu enc':>8s} {'decode':>8s} {'load ms':>8s}  runs")
for (clip, mode), r in sorted(rows.items()):
    print(f"{clip:10s} {mode:4s} {med(r['wall']):9.0f} {med(r['encpre']):8.0f} {med(r['npu']):8.0f} "
          f"{med(r['dec']):8.0f} {med(r['load']):8.0f}  {len(r['wall'])}")
for clip in sorted({c for c, _ in rows}):
    g, n = rows.get((clip, "gpu")), rows.get((clip, "npu"))
    if g and n:
        same = g["tx"] == n["tx"]
        print(f"{clip}: transcripts {'identical' if same else 'DIFFER'} "
              f"(gpu {len(g['tx'])} distinct, npu {len(n['tx'])} distinct)")
        if not same:
            for t in sorted(g["tx"] ^ n["tx"]):
                print("   ", "gpu" if t in g["tx"] else "npu", t[:300])
