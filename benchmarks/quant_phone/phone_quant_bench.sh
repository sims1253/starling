#!/usr/bin/env bash
# phone_quant_bench.sh — speed, CPU time and peak memory of quantized GGUFs
# on an adb phone (run on the host the phone is attached to).
#
# Every LABEL=kind:file spec runs as its own starling-bench process under
# `toybox time -v` (peak RSS, user/system CPU seconds), rounds interleave the
# specs so thermal drift spreads over all of them, and each process does a
# warm-up plus RUNS timed transcriptions of every wav. CPU seconds of the
# whole process (load + warm-up included, the same work for every spec of a
# model) are the energy proxy for CPU-only runs; the phone's power rails are
# not readable from the shell.
#
#   benchmarks/quant_phone/phone_quant_bench.sh [-r rounds] [-n runs] [-e engine] \
#       LABEL=kind:/local/path/model.gguf ...
#
# Pushes binaries and missing GGUFs to $DEV (env: BIN_DIR with starling-bench
# and libggml*.so, default ./bin; WAVS, default the three fixtures; OUT for the
# raw logs; COOL seconds between processes, default 20; THREADS, default 6).
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
THREADS=${THREADS:-6}
mkdir -p "$OUT"

. "$ROOT/benchmarks/fast_engine/phone_common.sh"
BENCH_BINS="starling-bench-quant"
trap 'kill_benches || true; wake_release || true' EXIT
kill_benches || true
wake_hold
screen_off

timeout 30 adb shell "mkdir -p $DEV" >/dev/null
timeout 120 adb push "$BIN_DIR/starling-bench" "$DEV/starling-bench-quant" >/dev/null
for so in "$BIN_DIR"/libggml*.so; do timeout 120 adb push "$so" "$DEV/" >/dev/null; done
dev_wavs=""
for w in $WAVS; do
  timeout 60 adb push "$w" "$DEV/" >/dev/null
  dev_wavs+=" $DEV/$(basename "$w")"
done

push_model() {  # local path -> device path; skips a same-size copy
  local f=$1 dst size dsize
  dst="$DEV/$(basename "$f")"
  size=$(stat -c %s "$f")
  dsize=$(timeout 30 adb shell "stat -c %s $dst 2>/dev/null" | tr -d '\r' || true)
  [ "$dsize" = "$size" ] || timeout 3600 adb push "$f" "$dst" >/dev/null
  echo "$dst"
}

labels=() kinds=() files=()
for spec in "$@"; do
  label=${spec%%=*}; rest=${spec#*=}
  kind=${rest%%:*}; file=${rest#*:}
  [ -f "$file" ] || { echo "missing model: $file" >&2; exit 1; }
  labels+=("$label"); kinds+=("$kind"); files+=("$(push_model "$file")")
done
echo "logs: $OUT  engine=$ENGINE threads=$THREADS rounds=$ROUNDS runs=$RUNS"

for r in $(seq 1 "$ROUNDS"); do
  for i in "${!labels[@]}"; do
    wake_held || { echo "ERROR: wake lock lost; stopping" >&2; exit 1; }
    log="$OUT/${labels[$i]}.r$r.log"
    timeout 3600 adb shell "cd $DEV && LD_LIBRARY_PATH=. STARLING_ENGINE=$ENGINE \
      STARLING_GGML_THREADS=$THREADS STARLING_FAST_CACHE_DIR=$DEV \
      toybox time -v ./starling-bench-quant --model ${kinds[$i]} --gguf ${files[$i]} \
      --warmup --runs $RUNS $dev_wavs 2>&1" > "$log" 2>&1 || echo "WARNING: ${labels[$i]} round $r exited non-zero" >&2
    timeout 30 adb shell "dumpsys thermalservice | grep -m1 'Thermal Status'" 2>/dev/null | tr -d '\r' >> "$log" || true
    benches_gone 60 || kill_benches
    sleep "$COOL"
  done
done

python3 - "$OUT" "$ROUNDS" "${labels[@]}" <<'EOF'
import os, re, statistics, sys
out, rounds, labels = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
print(f"{'label':22s} {'load_ms':>8s} {'short_ms':>9s} {'medium_ms':>9s} {'long_ms':>9s} {'cpu_s':>7s} {'wall_s':>7s} {'rss_MB':>7s}  thermal")
for lab in labels:
    loads, per, cpu, wall, rss, therm = [], {}, [], [], [], []
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
        t = re.search(r"Thermal Status: (\d+)", s)
        therm.append(t.group(1) if t else "?")
    med = lambda v: f"{statistics.median(v):.0f}" if v else "FAIL"
    cols = [med(per.get(w, [])) for w in ("short.wav", "medium.wav", "long.wav")]
    print(f"{lab:22s} {med(loads):>8s} {cols[0]:>9s} {cols[1]:>9s} {cols[2]:>9s} {med(cpu):>7s} {med(wall):>7s} {med(rss):>7s}  {','.join(therm)}")
EOF
