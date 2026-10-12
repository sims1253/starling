"""GFLOPS of LiteRT GPU on the FC micro models (phone_session.sh micro phase).

python -I micro_summary.py LOGDIR
For each mi_fc*_M*_K*_N*_p{1,2}: the FC kernel's GPU time per launch (clshim,
steady runs; launches with the same program and shape are pooled, so the
16-FC models give the mean of 16), its GFLOPS, all kernels' time per run,
the bench's wall median, and the FC kernel's launch shape.
"""
import glob
import os
import re
import sys

d = sys.argv[1]
out = []
for log in glob.glob(f"{d}/kdump/mi_fc*/clshim.log"):
    tag = os.path.basename(os.path.dirname(log))
    M, K, N, p = map(int, re.search(r"_M(\d+)_K(\d+)_N(\d+)_p(\d)", tag).groups())
    sec = open(log).read().split("== steady runs")[-1]
    rows = re.findall(r"^\s*([\d.]+) ms\s+[\d.]+%\s+x(\d+)\s+(p\d+)\s+\S+\s+g=(\S+) l=(\S+)", sec, re.M)
    if not rows:
        continue
    total = sum(float(r[0]) for r in rows)
    agg = {}
    for r_ms, r_n, r_prog, r_g, r_l in rows:
        a = agg.setdefault((r_prog, r_g, r_l), [0.0, 0])
        a[0] += float(r_ms)
        a[1] += int(r_n)
    (prog, g, l), (ms, n) = max(agg.items(), key=lambda kv: kv[1][0])
    ms /= max(n, 1)
    src = open(f"{os.path.dirname(log)}/{prog}.cl").read()
    acc = len(re.findall(r"^\s*(?:half|float)4 r_\w+ = ", src, re.M))
    kind = ("bcast " if "sub_group_broadcast" in src else "") + ("local " if "__local" in src else "")
    wall = re.search(r"runs \d+: median ([\d.]+)", open(f"{d}/{tag}.txt").read())
    out.append((M, K, N, p, f"{tag.split('_M')[0][3:]:5s} M={M:4d} K={K:5d} N={N:5d} {'fp16' if p == 1 else 'fp32'}: main {ms:6.3f} ms = "
                f"{2 * M * K * N / ms / 1e6:5.0f} GFLOPS (x{n}) | all kernels {total:6.3f} ms | wall {wall.group(1) if wall else '?':>6} ms"
                f" | {prog} g={g} l={l} acc={acc} {kind}"))
for *_, s in sorted(out):
    print(s)
