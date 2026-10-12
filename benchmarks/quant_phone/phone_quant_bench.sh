#!/usr/bin/env bash
# phone_quant_bench.sh — speed, CPU time and peak memory of quantized GGUFs
# on an adb phone (run on the host the phone is attached to).
#
# Every LABEL=kind:file spec runs as its own starling-bench process under
# `toybox time -v` (peak RSS, user/system CPU seconds), and each process does
# a warm-up plus RUNS timed transcriptions of every wav. Sustained 6-core CPU
# work throttles the phone within minutes (measured: each spec in a fixed
# order ran slower than the one before it), so every process starts only once
# the virtual skin temperature is back under SKIN_MAX, and rounds alternate
# the spec order (ABBA) so the heat a round builds falls on every spec.
# On battery, each process's energy comes from the gauge's charge counter
# (a coulomb counter; µAh x nominal 3.87 V, as phone_energy.sh) minus the
# idle drain measured screen-off at the start (IDLE_S seconds). CPU seconds of the
# whole process (load + warm-up included, the same work for every spec of a
# model) are the energy proxy for CPU-only runs; the phone's power rails are
# not readable from the shell.
#
#   benchmarks/quant_phone/phone_quant_bench.sh [-r rounds] [-n runs] [-e engine] \
#       LABEL=kind:/local/path/model.gguf ...
#
# Pushes binaries and missing GGUFs to $DEV (env: BIN_DIR with starling-bench
# and libggml*.so, default ./bin; WAVS, default the three fixtures; OUT for the
# raw logs; SKIN_MAX °C to start a process at, default the idle reading at
# start + 1; BIG_MAX °C for the big-core cluster, default 50; COOL minimum
# seconds between processes, default 20; IDLE_S,
# default 180; THREADS, default 6; PHONE_LOCK, the flock shared with other
# phone scripts, default /tmp/starling-phone.lock).
set -euo pipefail
DEV=/data/local/tmp/starling/quant
ROUNDS=2
RUNS=2
ENGINE=ggml
while getopts "r:n:e:" o; do
  case $o in r) ROUNDS=$OPTARG ;; n) RUNS=$OPTARG ;; e) ENGINE=$OPTARG ;; *) exit 2 ;; esac
done
shift $((OPTIND - 1))
[ $# -ge 1 ] || { echo "usage: $0 [-r rounds] [-n runs] [-e ggml|fast|auto] LABEL=kind:model.gguf ..." >&2; exit 2; }
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BIN_DIR=${BIN_DIR:-./bin}
WAVS=${WAVS:-"$ROOT/tests/fixtures/short.wav $ROOT/tests/fixtures/medium.wav $ROOT/tests/fixtures/long.wav"}
OUT=${OUT:-$(mktemp -d)}
COOL=${COOL:-20}
SKIN_MAX=${SKIN_MAX:-}
BIG_MAX=${BIG_MAX:-50}
IDLE_S=${IDLE_S:-180}
THREADS=${THREADS:-6}
mkdir -p "$OUT"

# One bench session on the phone at a time: concurrent work skews timing,
# heat and the charge counter. Other phone scripts on this host take the
# same lock (PHONE_LOCK); wait for it, holding it until exit.
PHONE_LOCK=${PHONE_LOCK:-/tmp/starling-phone.lock}
exec 9>"$PHONE_LOCK"
if ! flock -n 9; then
  echo "waiting for $PHONE_LOCK (another phone session) ..." >&2
  flock 9
fi

. "$ROOT/benchmarks/fast_engine/phone_common.sh"
BENCH_BINS="starling-bench-quant"
# Our own wake-lock holder: wake_hold first releases whatever holder the pid
# file names, so sharing phone_common's default path with another session's
# bench script would kill its lock (and it ours).
WAKEHOLD_DEX=$DEV/wakehold.dex
WAKEHOLD_TAG=starling-quant
trap 'kill_benches || true; wake_release || true' EXIT
kill_benches || true
wake_hold
screen_off

timeout 30 adb shell "mkdir -p $DEV" >/dev/null
timeout 120 adb push "$BIN_DIR/starling-bench" "$DEV/starling-bench-quant" >/dev/null
sos=("$BIN_DIR"/libggml*.so)
[ -e "${sos[0]}" ] || { echo "no libggml*.so in $BIN_DIR" >&2; exit 1; }
for so in "${sos[@]}"; do timeout 120 adb push "$so" "$DEV/" >/dev/null; done
dev_wavs=""
for w in $WAVS; do
  # The basenames go unquoted into the device shell command line.
  [[ $(basename "$w") =~ ^[A-Za-z0-9._-]+$ ]] || { echo "unsupported characters in '$w' (use [A-Za-z0-9._-])" >&2; exit 1; }
  timeout 60 adb push "$w" "$DEV/" >/dev/null
  dev_wavs+=" $DEV/$(basename "$w")"
done

push_model() {  # push_model LABEL local-path -> device path; skips an identical copy
  # The device name carries the label: two specs whose files share a basename
  # (e.g. two model-q4k.gguf from different directories) must not overwrite
  # each other's copy. Identity is the SHA-256, not the size: a rebuild with
  # another imatrix keeps the size and must still be pushed.
  local f=$2 dst sum dsum
  dst="$DEV/$1--$(basename "$f")"
  sum=$(sha256sum "$f" | cut -d' ' -f1)
  dsum=$(timeout 300 adb shell "sha256sum '$dst' 2>/dev/null" | tr -d '\r' | cut -d' ' -f1 || true)
  [ "$dsum" = "$sum" ] || timeout 3600 adb push "$f" "$dst" >/dev/null
  echo "$dst"
}

hal_temp() {  # hal_temp NAME: the thermal HAL's current NAME temperature in °C ("" if unreadable)
  # Only the "Current temperatures from HAL" block: the "Cached temperatures"
  # block printed before it keeps a stale value (measured: unchanged for an
  # hour of 6-core load), so a gate reading it never waits.
  timeout 30 adb shell "dumpsys thermalservice" 2>/dev/null | tr -d '\r' |
    sed -n '/^Current temperatures from HAL:/,/^[^[:space:]]/p' |
    sed -n "s/.*mValue=\([0-9.]*\), mType=[0-9-]*, mName=$1,.*/\1/p" | sed -n 1p
}
skin() { hal_temp VIRTUAL-SKIN; }
big() { hal_temp BIG; }
wait_cool() {  # up to 30 min for skin < SKIN_MAX and the big-core cluster < BIG_MAX
  local t b
  for _ in $(seq 1 180); do
    t=$(skin); b=$(big)
    [ -n "$t" ] && awk -v t="$t" -v m="$SKIN_MAX" -v b="${b:-0}" -v bm="$BIG_MAX" \
      'BEGIN { exit !(t < m && b < bm) }' && { echo "$t"; return 0; }
    sleep 10
  done
  echo "${t:-?} (big ${b:-?}, timeout)"
  return 1
}

battery_field() {  # battery_field "Charge counter" -> value ("" if unreadable)
  timeout 30 adb shell "dumpsys battery" 2>/dev/null | tr -d '\r' | sed -n "s/^ *$1: *//p" | sed -n 1p
}
on_battery() { case "$(battery_field status)" in 3|4) return 0 ;; *) return 1 ;; esac; }

labels=() kinds=() files=()
for spec in "$@"; do
  label=${spec%%=*}; rest=${spec#*=}
  kind=${rest%%:*}; file=${rest#*:}
  [ -f "$file" ] || { echo "missing model: $file" >&2; exit 1; }
  # These go unquoted into the device shell command line.
  for v in "$label" "$kind" "$(basename "$file")"; do
    [[ $v =~ ^[A-Za-z0-9._-]+$ ]] || { echo "unsupported characters in '$v' (use [A-Za-z0-9._-])" >&2; exit 1; }
  done
  labels+=("$label"); kinds+=("$kind"); files+=("$(push_model "$label" "$file")")
done
idle_rate=""   # µAh per second, screen off, nothing running
if on_battery; then
  c0=$(battery_field "Charge counter"); t0=$(date +%s.%N)
  sleep "$IDLE_S"
  c1=$(battery_field "Charge counter"); t1=$(date +%s.%N)
  if [[ $c0 =~ ^[0-9]+$ && $c1 =~ ^[0-9]+$ ]]; then
    idle_rate=$(awk -v a="$c0" -v b="$c1" -v s="$t0" -v e="$t1" 'BEGIN { printf "%.4f", (a - b) / (e - s) }')
    echo "idle drain: $idle_rate uAh/s over ${IDLE_S}s (counter $c0 -> $c1)" | tee "$OUT/idle.txt"
  else
    echo "charge counter unreadable ('$c0' -> '$c1'): no energy numbers" | tee "$OUT/idle.txt"
  fi
else
  echo "phone is not on battery: no energy numbers" | tee "$OUT/idle.txt"
fi
[ -n "$SKIN_MAX" ] || SKIN_MAX=$(awk -v t="$(skin)" 'BEGIN { printf "%.1f", (t == "" ? 37 : t) + 1.0 }')
echo "logs: $OUT  engine=$ENGINE threads=$THREADS rounds=$ROUNDS runs=$RUNS skin_max=$SKIN_MAX"

order=("${!labels[@]}")
for r in $(seq 1 "$ROUNDS"); do
  if (( r % 2 == 0 )); then
    rev=(); for (( j=${#order[@]}-1; j>=0; j-- )); do rev+=("${order[$j]}"); done; seq_r=("${rev[@]}")
  else
    seq_r=("${order[@]}")
  fi
  for i in "${seq_r[@]}"; do
    wake_held || { echo "ERROR: wake lock lost; stopping" >&2; exit 1; }
    log="$OUT/${labels[$i]}.r$r.log"
    start_t=$(wait_cool) || { echo "ERROR: the phone did not cool down ($start_t); stopping" >&2; exit 1; }
    echo "skin at start: $start_t" > "$log"
    if on_battery; then echo "charge start: $(battery_field "Charge counter") $(date +%s.%N)" >> "$log"; fi
    timeout 3600 adb shell "cd $DEV && LD_LIBRARY_PATH=. STARLING_ENGINE=$ENGINE \
      STARLING_GGML_THREADS=$THREADS STARLING_FAST_CACHE_DIR=$DEV \
      toybox time -v ./starling-bench-quant --model ${kinds[$i]} --gguf ${files[$i]} \
      --warmup --runs $RUNS $dev_wavs 2>&1" >> "$log" 2>&1 || echo "WARNING: ${labels[$i]} round $r exited non-zero" >&2
    if on_battery; then echo "charge end: $(battery_field "Charge counter") $(date +%s.%N)" >> "$log"; fi
    echo "skin at end: $(skin)" >> "$log"
    timeout 30 adb shell "dumpsys thermalservice | grep -m1 'Thermal Status'" 2>/dev/null | tr -d '\r' >> "$log" || true
    benches_gone 60 || kill_benches
    sleep "$COOL"
  done
done

python3 - "$OUT" "$ROUNDS" "$idle_rate" "${labels[@]}" <<'EOF'
import os, re, statistics, sys
out, rounds, idle, labels = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4:]
V_NOM = 3.87  # nominal cell voltage: µAh -> µWh is an approximation, as in phone_energy.sh
print(f"{'label':22s} {'load_ms':>8s} {'short_ms':>9s} {'medium_ms':>9s} {'long_ms':>9s} {'cpu_s':>7s} {'wall_s':>7s} {'rss_MB':>7s} {'mWh':>7s}  skin start->end per round")
for lab in labels:
    loads, per, cpu, wall, rss, therm, energy = [], {}, [], [], [], [], []
    for r in range(1, rounds + 1):
        p = os.path.join(out, f"{lab}.r{r}.log")
        if not os.path.exists(p):
            continue
        s = open(p, errors="replace").read()
        loads += [float(x) for x in re.findall(r"^load ([0-9.]+) ms", s, re.M)]
        for path, ms in re.findall(r"^(\S+\.wav) run=\d+ audio=[0-9.]+s time=([0-9.]+)ms", s, re.M):
            per.setdefault(os.path.basename(path), []).append(float(ms))
        u = re.search(r"User time \(s\): ([0-9.]+)", s); y = re.search(r"System time \(s\): ([0-9.]+)", s)
        w = re.search(r"Real time \(s\): ([0-9.]+)", s); m = re.search(r"Max RSS \(KiB\): ([0-9]+)", s)
        if u and y: cpu.append(float(u.group(1)) + float(y.group(1)))
        if w: wall.append(float(w.group(1)))
        if m: rss.append(int(m.group(1)) / 1024)
        cs = re.search(r"charge start: (\d+) ([0-9.]+)", s); ce = re.search(r"charge end: (\d+) ([0-9.]+)", s)
        if cs and ce and idle:
            dt = float(ce.group(2)) - float(cs.group(2))
            used = int(cs.group(1)) - int(ce.group(1)) - float(idle) * dt   # µAh above idle
            if used >= 0:  # a negative reading is a counter reset or recalibration
                energy.append(used * V_NOM / 1000.0)                         # mWh
        a = re.search(r"skin at start: (\S+)", s); b = re.search(r"skin at end: (\S*)", s)
        therm.append(f"{a.group(1) if a else '?'}->{b.group(1) if b and b.group(1) else '?'}")
    med = lambda v: f"{statistics.median(v):.0f}" if v else "FAIL"
    cols = [med(per.get(w, [])) for w in ("short.wav", "medium.wav", "long.wav")]
    mwh = f"{statistics.median(energy):.1f}" if energy else "n/a"
    print(f"{lab:22s} {med(loads):>8s} {cols[0]:>9s} {cols[1]:>9s} {cols[2]:>9s} {med(cpu):>7s} {med(wall):>7s} {med(rss):>7s} {mwh:>7s}  {' '.join(therm)}")
EOF
