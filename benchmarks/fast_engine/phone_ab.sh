#!/usr/bin/env bash
# phone_ab.sh — alternating A/B of fast-engine bench binaries on an adb phone.
#
# The Pixel's GPU drifts ±2 % with temperature, so comparing two separately
# measured runs misleads; this interleaves the binaries round by round in one
# thermal window and prints each round's median per binary and model.
#
#   benchmarks/fast_engine/phone_ab.sh [-r rounds] [-n runs] NAME=path/to/starling-bench ...
#
# Needs on the phone ($DEV): the GGUFs below, medium.wav and short.wav
# (android_bench.sh pushes them), and the ggml shared libraries next to the
# binaries. Env: PK_GGUF, MOSS_GGUF, PK_WAV, MOSS_WAV, EXTRA_ENV.
set -euo pipefail
DEV=/data/local/tmp/starling
ROUNDS=3
RUNS=3
while getopts "r:n:" o; do
  case $o in r) ROUNDS=$OPTARG ;; n) RUNS=$OPTARG ;; *) exit 2 ;; esac
done
shift $((OPTIND - 1))
[ $# -ge 1 ] || { echo "usage: $0 [-r rounds] [-n runs] NAME=binary ..." >&2; exit 2; }
PK_GGUF=${PK_GGUF:-parakeet-tdt-0.6b-v3-q4_k_m-shrink16.gguf}
MOSS_GGUF=${MOSS_GGUF:-moss-transcribe-preview-2b-q4e8-fullimx.gguf}
PK_WAV=${PK_WAV:-medium.wav}
MOSS_WAV=${MOSS_WAV:-short.wav}
EXTRA_ENV=${EXTRA_ENV:-}

names=()
for spec in "$@"; do
  name=${spec%%=*}; bin=${spec#*=}
  [ -f "$bin" ] || { echo "missing binary: $bin" >&2; exit 1; }
  adb push "$bin" "$DEV/starling-bench-$name" >/dev/null
  names+=("$name")
done

median() { sort -n | awk '{a[NR]=$1} END {print (NR % 2) ? a[(NR+1)/2] : (a[NR/2] + a[NR/2+1]) / 2}'; }

run() { # name kind gguf wav -> median total ms
  adb shell "cd $DEV && LD_LIBRARY_PATH=. STARLING_ENGINE=fast STARLING_GGML_THREADS=6 \
    STARLING_FAST_CACHE_DIR=$DEV $EXTRA_ENV ./starling-bench-$1 --model $2 --gguf $DEV/$3 \
    --warmup --runs $RUNS $DEV/$4" 2>&1 |
    sed -n 's/.* time=\([0-9.]*\)ms.*/\1/p' | median
}

printf '%-6s %-10s %10s %10s\n' round binary parakeet moss
for r in $(seq 1 "$ROUNDS"); do
  for name in "${names[@]}"; do
    pk=$(run "$name" parakeet "$PK_GGUF" "$PK_WAV")
    mo=$(run "$name" moss "$MOSS_GGUF" "$MOSS_WAV")
    printf '%-6s %-10s %10s %10s\n' "$r" "$name" "$pk" "$mo"
  done
done
