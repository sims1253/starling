#!/usr/bin/env bash
# phone_energy.sh — energy per MOSS-short transcription on the phone (#317's
# last deliverable: "report energy per transcription next to it and say how
# it was measured: the batterystats power model is an estimate, not a rail
# measurement").
#
# Method: the battery gauge's charge counter (dumpsys battery, a coulomb
# counter) is PRIMARY; the batterystats power model is reported alongside as
# an explicitly-estimated cross-check. A screen-off idle control gives the
# baseline drain rate, which is scaled to each window's duration and
# subtracted. One bench invocation per engine (--runs N, a single model load
# each — the phone's GPU health budget). The phone must be on battery.
#
# Env: RUNS (timed transcriptions per engine, default 12), MOSS_GGUF,
# OUT (directory for the raw points and bench logs, default a fresh mktemp).
set -euo pipefail
. "$(dirname "$0")/phone_common.sh"
DEV=/data/local/tmp/starling
RUNS=${RUNS:-12}
MOSS_GGUF=${MOSS_GGUF:-moss-transcribe-preview-2b-q4e8-fullimx.gguf}
V_NOM=3.87   # nominal Li-ion voltage for µAh -> µWh; the gauge integrates
             # current, so this is an approximation of energy, stated as such
OUT=${OUT:-$(mktemp -d)}
POINTS="$OUT/energy_points.txt"

command -v adb >/dev/null || { echo "adb missing" >&2; exit 1; }
adb get-state >/dev/null 2>&1 || { echo "phone not connected" >&2; exit 1; }

battery_field() {  # battery_field "status" -> value from dumpsys battery
  local out
  out=$(adb shell "dumpsys battery" | tr -d '\r')
  sed -n "s/^ *$1: *//p" <<<"$out" | sed -n 1p
}
# Charging makes the counter rise: every delta would be meaningless.
# BatteryManager status: 2 charging, 3 discharging, 4 not charging, 5 full.
on_battery() {
  local st
  st=$(battery_field status)
  [ "$st" = 3 ] || [ "$st" = 4 ] ||
    { echo "ERROR: battery status $st — unplug the phone (need 3 discharging / 4 not charging)" >&2; return 1; }
}

counter() {  # µAh; exits on a failed read
  local v
  v=$(battery_field "Charge counter")
  [[ "$v" =~ ^[0-9]+$ ]] || { echo "charge counter read failed (got: '$v')" >&2; exit 1; }
  echo "$v"
}

echo "== environment (raw points and logs: $OUT) =="
adb shell "dumpsys battery | grep -E 'status|level|Charge counter' | head -3; dumpsys thermalservice | grep -m1 Severity" || true
on_battery
kill_benches
wait_benches
trap kill_benches EXIT   # an abort must not leave a bench running on the phone

bench() {  # bench <engine>
  adb shell "cd $DEV && timeout 900 env LD_LIBRARY_PATH=. STARLING_ENGINE=$1 STARLING_GGML_THREADS=6 \
    STARLING_FAST_CACHE_DIR=$DEV ./starling-bench-cand --model moss --gguf $DEV/$MOSS_GGUF \
    --warmup --runs $RUNS $DEV/short.wav" 2>&1
}

# measure <label> <command...>: one gauge window. A failed command or a
# non-positive drain aborts with the log tail; the raw point is appended to
# $POINTS as "<label> <µAh> <seconds>".
measure() {
  local label=$1 t0 t1 c0 c1 rc=0
  shift
  screen_off
  on_battery
  t0=$(date +%s)
  c0=$(counter)
  "$@" > "$OUT/energy_$label.log" 2>&1 || rc=$?
  c1=$(counter)
  t1=$(date +%s)
  if [ "$rc" -ne 0 ]; then
    echo "ERROR: $label window failed (exit $rc):" >&2
    tail -5 "$OUT/energy_$label.log" >&2
    exit 1
  fi
  if [ $((c0 - c1)) -le 0 ]; then
    echo "ERROR: $label drain $((c0 - c1)) µAh is not positive (charging or gauge reset?)" >&2
    exit 1
  fi
  echo "$label $((c0 - c1)) $((t1 - t0))" >> "$POINTS"
  echo "$label: $((c0 - c1)) µAh over $((t1 - t0)) s"
}

: > "$POINTS"
echo "== fast engine ($RUNS transcriptions) =="
measure fast bench fast
DUR=$(awk '$1=="fast"{print $3}' "$POINTS")
echo "== idle control (${DUR} s, screen off) =="
measure idle sleep "$DUR"
echo "== ggml engine ($RUNS transcriptions) =="
wait_benches
measure ggml bench ggml

echo "== raw points (label µAh seconds) =="
cat "$POINTS"
python3 - "$RUNS" "$V_NOM" "$POINTS" <<'EOF'
import sys
runs, v, path = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
pts = {}
for line in open(path):
    label, uah, sec = line.split()
    pts[label] = (int(uah), int(sec))
missing = [k for k in ("fast", "idle", "ggml") if k not in pts]
if missing:
    sys.exit(f"missing points: {missing}")
# Idle drain is a rate: scale it to each window's duration before
# subtracting (the engines' windows differ in length).
idle_rate = pts["idle"][0] / max(pts["idle"][1], 1)   # µAh per second
print(f"idle rate {idle_rate:.2f} µAh/s")
net = {}
for label in ("fast", "ggml"):
    uah, sec = pts[label]
    net[label] = uah - idle_rate * sec
    per = net[label] * v / 1000 / runs
    flag = "" if net[label] > 0 else "  <- NOT POSITIVE: idle control unreliable"
    print(f"{label}: raw {uah} µAh over {sec} s, idle-subtracted {net[label]:.0f} µAh "
          f"= {per:.3f} mWh/transcription{flag}")
if net["fast"] > 0 and net["ggml"] > 0:
    print(f"ratio ggml/fast: {net['ggml'] / net['fast']:.2f}x")
print("method: battery charge counter (coulomb gauge), duration-scaled idle subtraction, "
      "nominal voltage; assumes constant idle drain; not a rail measurement")
EOF
echo "== batterystats power-model cross-check (an estimate, not rails) =="
adb shell "dumpsys batterystats | grep -iE 'sh=|u0a|Estimated power' | head -6" || true
echo "(the model numbers above are the vendor power profile's estimate;"
echo " the gauge deltas above are the primary measurement)"
