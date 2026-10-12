"""Aggregate clshim's per-kernel table (one row per kernel object and launch
size) by program and launch size, for the 'steady runs' interval.

python -I kprof.py CLSHIM_LOG [TOP]
Prints: ms/run, share, launches/run, program, global/local size, the first
kernel-source line that declares accumulators (block shape hint).
"""
import collections
import os
import re
import sys

log = sys.argv[1]
top = int(sys.argv[2]) if len(sys.argv) > 2 else 25
txt = open(log).read()
sec = txt.split("== steady runs")[-1]
head = sec.splitlines()[0]
rows = re.findall(r"^\s*([\d.]+) ms\s+[\d.]+%\s+x(\d+)\s+(p\d+)\s+(\S+)\s+g=(\S+) l=(\S+)", sec, re.M)
agg = collections.defaultdict(lambda: [0.0, 0])
for ms, n, prog, name, g, l in rows:
    a = agg[(prog, g, l)]
    a[0] += float(ms)
    a[1] += int(n)
total = sum(v[0] for v in agg.values())
print("steady runs" + head)
print(f"{'ms/run':>8} {'share':>6} {'n':>4}  prog   global          local      accumulators")
d = os.path.dirname(log)
for (prog, g, l), (ms, n) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:top]:
    acc = ""
    try:
        src = open(f"{d}/{prog}.cl").read()
        acc = str(len(re.findall(r"^\s*(?:half|float)4 r_\w+ = ", src, re.M)))
        if "sub_group_broadcast" in src:
            acc += " bcast"
        if "__local" in src:
            acc += " local"
        acc += " half" if "half4 r_" in src else ""
    except OSError:
        pass
    print(f"{ms:8.3f} {100 * ms / total:5.1f}% {n:4d}  {prog} {g:15s} {l:10s} {acc}")
print(f"{total:8.3f} total kernel ms/run")
