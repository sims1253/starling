#!/usr/bin/env bash
# energy.sh — energy per MOSS-short transcription on the phone (#317's last
# deliverable: "report energy per transcription next to it and say how it was
# measured: the batterystats power model is an estimate, not a rail
# measurement").
#
# Method: the battery gauge's charge counter (dumpsys battery, a coulomb
# counter) is PRIMARY; the batterystats power model is reported alongside as
# an explicitly-estimated cross-check. An equal-duration screen-off idle
# control subtracts the baseline drain. One bench invocation per engine
# (--runs N, a single model load each — the phone's GPU health budget).
#
# Env: RUNS (timed transcriptions per engine, default 12)
set -euo pipefail
DEV=/data/local/tmp/starling
RUNS=${RUNS:-12}
MOSS_GGUF=${MOSS_GGUF:-moss-transcribe-preview-2b-q4e8-fullimx.gguf}
V_NOM=3.87   # nominal Li-ion voltage for µAh -> µWh; the gauge integrates
             # current, so this is an approximation of energy, stated as such

command -v adb >/dev/null || { echo "adb missing" >&2; exit 1; }
adb get-state >/dev/null 2>&1 || { echo "phone not connected" >&2; exit 1; }

counter() {  # µAh (validated: a failed read aborts before any bench runs)
  local v
  v=$(adb shell "dumpsys battery | grep -m1 'Charge counter'" | grep -oE '[0-9]+' | head -1)
  [[ "$v" =~ ^[0-9]+$ ]] || { echo "charge counter read failed (got: '$v')" >&2; exit 1; }
  echo "$v"
}

screen_off() {
  if [ "$(adb shell "dumpsys power | grep -m1 mWakefulness=" | tr -d '\r')" = "  mWakefulness=Awake" ]; then
    adb shell "input keyevent KEYCODE_POWER" >/dev/null
    sleep 2
  fi
}

echo "== environment =="
adb shell "dumpsys battery | grep -E 'status|level|Charge counter' | head -3; dumpsys thermalservice | grep -m1 Severity" || true
adb shell 'for p in $(pidof starling-bench-base starling-bench-cand starling-bench); do kill -9 $p; done' >/dev/null 2>&1 || true
screen_off

bench() {  # engine runs (exported: measure() invokes it through bash -c)
  local engine=$1
  adb shell "cd $DEV && timeout 900 env LD_LIBRARY_PATH=. STARLING_ENGINE=$engine STARLING_GGML_THREADS=6 \
    STARLING_FAST_CACHE_DIR=$DEV ./starling-bench-cand --model moss --gguf $DEV/$MOSS_GGUF \
    --warmup --runs $RUNS $DEV/short.wav" 2>&1
}
export -f bench
export DEV RUNS MOSS_GGUF

measure() {  # label command
  local label=$1
  local t0=$(date +%s)
  local c0=$(counter)
  bash -c "${@:2}" > /tmp/energy_$label.log 2>&1
  local c1=$(counter)
  local t1=$(date +%s)
  local duah=$((c0 - c1))
  echo "$label $duah $((t1 - t0))" >> /tmp/energy_points.txt
  echo "$label: ${duah} µAh over $((t1 - t0)) s"
}

rm -f /tmp/energy_points.txt
echo "== fast engine ($RUNS transcriptions) =="
measure fast "bench fast"
echo "== idle control (same duration, screen off) =="
DUR=$(awk '$1=="fast"{print $3}' /tmp/energy_points.txt)
[[ "$DUR" =~ ^[0-9]+$ ]] && [ "$DUR" -gt 0 ] || { echo "idle duration invalid (DUR='$DUR') — fast measurement failed?" >&2; exit 1; }
measure idle "sleep $DUR"
echo "== ggml engine ($RUNS transcriptions) =="
measure ggml "bench ggml"

python3 - "$RUNS" "$V_NOM" <<'EOF'
import sys
runs, v = int(sys.argv[1]), float(sys.argv[2])
pts = {l.split()[0]: (int(l.split()[1]), int(l.split()[2])) for l in open("/tmp/energy_points.txt")}
fast, idle, ggml = pts["fast"][0], pts["idle"][0], pts["ggml"][0]
mwh = lambda uah: uah * v / 1000.0
print(f"gauge deltas (uAh): fast={fast} idle={idle} ggml={ggml}")
if idle:
    print(f"fast:  {mwh(fast - idle):.2f} mWh / {runs} = {(fast-idle)*v/1000/runs:.3f} mWh/transcription (idle-subtracted)")
    print(f"ggml:  {mwh(ggml - idle):.2f} mWh / {runs} = {(ggml-idle)*v/1000/runs:.3f} mWh/transcription (idle-subtracted)")
    print(f"ratio fast/ggml: {(fast-idle)/max(ggml-idle,1):.2f}x")
print("method: battery charge counter (coulomb gauge), idle-subtracted, nominal voltage")
print("assumes idle drain ~ constant; not a rail measurement")
EOF
echo "== batterystats power-model cross-check (an estimate, not rails) =="
adb shell "dumpsys batterystats | grep -iE 'sh=|u0a|Estimated power' | head -6" || true
echo "(the model numbers above are the vendor power profile's estimate;"
echo " the gauge deltas above are the primary measurement)"
