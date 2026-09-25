#!/usr/bin/env bash
# measure.sh — A/B the current Android build against the base binary on the
# Pixel, one thermal window, MOSS short fixture. Prints METRIC lines:
#   moss_decode / moss_decode_base / moss_decode_delta (ms/token, median)
#   pk_medium_base / pk_medium_cand   (Parakeet medium total ms, G5 canary)
#   transcripts_match                 (1 = G1 pass: MOSS fixtures identical)
#
# Protocol matches pixel_measure.sh / phone_ab.sh: one `--warmup --runs N`
# invocation per side per round (the warmup transcription absorbs pipeline
# compilation and GPU DVFS ramp), alternating base/cand rounds. The display
# is put to sleep first: with the screen awake the compositor shares the GPU
# and decode times swing 2x (measured; screen-off reproduces the historical
# +/-2% band).
#
# Env: ROUNDS (default 3), RUNS (timed runs per invocation, default 3),
# EXTRA_ENV_CAND / EXTRA_ENV_BASE (STARLING_* for that side, default empty),
# SKIP_G1=1, G1_FULL=1 (also medium/long transcripts), DO_PK=1 (Parakeet G5
# canary — off by default; every model load costs GPU-driver health).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEV=/data/local/tmp/starling
ROUNDS=${ROUNDS:-3}
RUNS=${RUNS:-4}

screen_off() {
  if [ "$(adb shell "dumpsys power | grep -m1 mWakefulness=" | tr -d '\r')" = "  mWakefulness=Awake" ]; then
    adb shell "input keyevent KEYCODE_POWER" >/dev/null
    sleep 2
  fi
}
EXTRA_ENV_BASE=${EXTRA_ENV_BASE:-}
EXTRA_ENV_CAND=${EXTRA_ENV_CAND:-}
MOSS_GGUF=${MOSS_GGUF:-moss-transcribe-preview-2b-q4e8-fullimx.gguf}
PK_GGUF=${PK_GGUF:-parakeet-tdt-0.6b-v3-q4_k_m-shrink16.gguf}

cmake --build "$ROOT/build-android" --target starling-bench -j >/dev/null
adb push "$ROOT/build-android/starling-bench" "$DEV/starling-bench-cand" >/dev/null

BASE_COMMIT_FILE="$DEV/starling-bench-base.commit"
HEAD=$(git -C "$ROOT" rev-parse --short HEAD)
if ! adb shell "test -f $DEV/starling-bench-base" >/dev/null 2>&1 \
   || [ "$(adb shell "cat $BASE_COMMIT_FILE 2>/dev/null" | tr -d '\r')" != "$HEAD" ] \
   || [ -f "$ROOT/.auto/base_dirty" ]; then
  adb push "$ROOT/build-android/starling-bench" "$DEV/starling-bench-base" >/dev/null
  adb shell "echo $HEAD > $BASE_COMMIT_FILE"
  rm -f "$ROOT/.auto/base_dirty"
fi

# Session cleanup: a locally-timed-out adb shell leaves the remote bench
# running (its output pipe is gone; it can hang in poll forever) and every
# later pidof-wait would block on it.
adb shell 'for p in $(pidof starling-bench-base starling-bench-cand starling-bench); do kill -9 $p; done' >/dev/null 2>&1 || true

median() { sort -n | awk '{a[NR]=$1} END {print (NR % 2) ? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2}'; }

# bench <binary> <model> <gguf> <wav> <extra> -> raw output of one invocation.
# Back-to-back model loads (1.6 GB each) need the previous process gone; wait
# for it, then run with a generous timeout.
bench() {
  timeout 120 adb shell 'while pidof starling-bench starling-bench-base starling-bench-cand >/dev/null 2>&1; do sleep 1; done' >/dev/null 2>&1 || \
    adb shell 'for p in $(pidof starling-bench-base starling-bench-cand starling-bench); do kill -9 $p; done' >/dev/null 2>&1 || true
  adb shell "cd $DEV && timeout 600 env LD_LIBRARY_PATH=. STARLING_ENGINE=fast STARLING_GGML_THREADS=6 \
    STARLING_FAST_CACHE_DIR=$DEV STARLING_FAST_TIMING=1 $5 \
    ./$1 --model $2 --gguf $DEV/$3 --warmup --runs $RUNS $DEV/$4" 2>&1
}

# decode ms/token values from a bench output (one per timed run)
decode_mspt() { sed -n 's/.*decode=\([0-9.]*\)ms (\([0-9]*\) tokens.*/\1 \2/p' "$1" | awk '{print $1 / $2}'; }
total_ms() { sed -n 's/.*time=\([0-9.]*\)ms.*/\1/p' "$1"; }

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

declare -a base_vals cand_vals
screen_off
for r in $(seq 1 "$ROUNDS"); do
  screen_off
  bench starling-bench-base moss "$MOSS_GGUF" short.wav "$EXTRA_ENV_BASE" > "$TMP/b$r" ||
    { echo "base bench failed (round $r):"; tail -5 "$TMP/b$r"; exit 1; }
  bench starling-bench-cand moss "$MOSS_GGUF" short.wav "$EXTRA_ENV_CAND" > "$TMP/c$r" ||
    { echo "cand bench failed (round $r):"; tail -5 "$TMP/c$r"; exit 1; }
  base_vals+=("$(decode_mspt "$TMP/b$r" | median)")
  cand_vals+=("$(decode_mspt "$TMP/c$r" | median)")
  sleep 5   # breathe between rounds
done
base=$(printf '%s\n' "${base_vals[@]}" | median)
cand=$(printf '%s\n' "${cand_vals[@]}" | median)
echo "rounds(base): ${base_vals[*]}"
echo "rounds(cand): ${cand_vals[*]}"

PK_BASE=0 PK_CAND=0
if [ -n "${DO_PK:-}" ]; then
  bench starling-bench-base parakeet "$PK_GGUF" medium.wav "$EXTRA_ENV_BASE" > "$TMP/pkb"
  bench starling-bench-cand parakeet "$PK_GGUF" medium.wav "$EXTRA_ENV_CAND" > "$TMP/pkc"
  PK_BASE=$(total_ms "$TMP/pkb" | median)
  PK_CAND=$(total_ms "$TMP/pkc" | median)
fi

# G1 (short fixture): the A/B invocations already transcribe short.wav —
# compare their last transcript lines, no extra model loads. G1_FULL=1 adds
# medium/long (2 more loads per side; milestones only: every load on this
# phone costs GPU-driver health, ~8 loads per boot session is the budget).
MATCH=1
if [ -z "${SKIP_G1:-}" ]; then
  last_tr() { sed -n 's/^  //p' "$1" | tail -1; }
  [ "$(last_tr "$TMP/b$ROUNDS")" = "$(last_tr "$TMP/c$ROUNDS")" ] || MATCH=0
  if [ -n "${G1_FULL:-}" ]; then
    RUNS_SAVED=$RUNS; RUNS=1
    for w in medium long; do
      bench starling-bench-base moss "$MOSS_GGUF" "$w.wav" "$EXTRA_ENV_BASE" \
        | sed -n 's/^  //p' > "$TMP/ta"
      bench starling-bench-cand moss "$MOSS_GGUF" "$w.wav" "$EXTRA_ENV_CAND" \
        | sed -n 's/^  //p' > "$TMP/tb"
      cmp -s "$TMP/ta" "$TMP/tb" || MATCH=0
    done
    RUNS=$RUNS_SAVED
  fi
fi

delta=$(awk "BEGIN {printf \"%.2f\", ($cand - $base) / $base * 100}")
echo "METRIC moss_decode=$cand"
echo "METRIC moss_decode_base=$base"
echo "METRIC moss_decode_delta=$delta"
echo "METRIC pk_medium_base=$PK_BASE"
echo "METRIC pk_medium_cand=$PK_CAND"
echo "METRIC transcripts_match=$MATCH"
