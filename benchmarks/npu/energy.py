"""Energy from session4's battery samples. python -I energy.py LOGDIR"""
import glob
import re
import statistics as st
import sys

LOG = sys.argv[1]
rows = []
for line in open(f"{LOG}/pwr.txt"):
    a = line.split()
    if len(a) == 3:
        try:
            rows.append((float(a[0]), -int(a[1]) * int(a[2]) / 1e12))  # W; current_now < 0 when discharging
        except ValueError:
            pass
stamps = {}
for line in open(f"{LOG}/stamps.txt"):
    k, v = line.split()
    stamps[k] = float(v)


def window(t0, t1):
    pts = [(t, p) for t, p in rows if t0 <= t <= t1]
    e = sum((pts[i + 1][0] - pts[i][0]) * (pts[i][1] + pts[i + 1][1]) / 2 for i in range(len(pts) - 1))
    return e, pts[-1][0] - pts[0][0], len(pts)


idle = []
for k in ("idle0", "idle1"):
    e, d, n = window(stamps[k + "_start"], stamps[k + "_end"])
    idle.append(e / d)
    print(f"{k}: {e/d:.3f} W over {d:.1f}s ({n} samples)")
pi = st.mean(idle)
print(f"idle mean {pi:.3f} W")

for f in sorted(glob.glob(f"{LOG}/energy_npu_*.txt")):
    txt = open(f).read()
    m = re.search(r"loop_epoch ([\d.]+) ([\d.]+)", txt)
    runs = int(re.search(r"runs (\d+):", txt).group(1))
    e, d, n = window(float(m.group(1)), float(m.group(2)))
    print(f"{f.split('/')[-1]}: {runs} windows in {d:.1f}s, avg {e/d:.2f} W, above idle {(e-pi*d)/runs*1000:.1f} mJ/window "
          f"(total {e/runs*1000:.1f} mJ/window, {d/runs*1000:.1f} ms/window)")

for rep in (1, 2):
    try:
        ea, da, _ = window(stamps[f"gpu{rep}_n10_start"], stamps[f"gpu{rep}_n10_end"])
        eb, db, _ = window(stamps[f"gpu{rep}_n110_start"], stamps[f"gpu{rep}_n110_end"])
    except KeyError:
        continue
    tx = [float(x) for x in re.findall(r"enc\+prefill=([\d.]+)ms", open(f"{LOG}/energy_gpu_{rep}_110.txt").read())]
    tot = [float(x) for x in re.findall(r"total=([\d.]+)ms", open(f"{LOG}/energy_gpu_{rep}_110.txt").read())]
    de, dd = eb - ea, db - da
    print(f"gpu rep{rep}: marginal 100 runs: {dd:.1f}s, avg {de/dd:.2f} W, above idle {(de-pi*dd)/100:.3f} J/run "
          f"(total {de/100:.3f} J/run); median enc {st.median(tx):.0f} ms, run total {st.median(tot):.0f} ms")
