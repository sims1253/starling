#!/usr/bin/env bash
# pixel_measure.sh — the phone metric of the fast-engine tuning campaign
# (RESEARCH_LOG.md): cross-build starling-bench, push it (plus any missing
# models / fixtures) to an adb phone, run Parakeet on medium.wav and MOSS on
# short.wav with STARLING_FAST_TIMING, and print METRIC lines. The headline
# `phone_ms` is the sum of the two median totals; the fixture transcripts
# are checked against pixel_baseline_transcripts.txt (exit 3 on mismatch).
#
# Env: NO_BUILD=1 (skip build/push)   RUNS=n (timed runs per model, default 3)
#      REFRESH_TUNE=1 (delete the on-device autotune cache first)
#      CAPTURE=1 (rewrite the transcript baseline instead of checking)
#      MODELS=dir (host GGUF directory, default <repo>/models)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
HERE=$ROOT/benchmarks/fast_engine
BUILD=$ROOT/build-android
DEV=/data/local/tmp/starling
RUNS=${RUNS:-3}
MODELS=${MODELS:-$ROOT/models}
PK=parakeet-tdt-0.6b-v3-q4_k_m-shrink16.gguf
MOSS=moss-transcribe-preview-2b-q4e8-fullimx.gguf
BASE=$HERE/pixel_baseline_transcripts.txt

command -v adb >/dev/null || { echo "adb missing" >&2; exit 1; }
adb get-state >/dev/null 2>&1 || { echo "phone not connected" >&2; exit 1; }
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# --- build + push ------------------------------------------------------------
if [ "${NO_BUILD:-0}" != 1 ]; then
  if [ ! -f "$BUILD/build.ninja" ]; then
    NDK=$(ls -d "${ANDROID_NDK_HOME:-/opt/android-sdk/ndk}"/* 2>/dev/null | sort -V | tail -1)
    cmake -S "$ROOT" -B "$BUILD" -G Ninja \
      -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
      -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-30 -DCMAKE_BUILD_TYPE=Release \
      -DSTARLING_FAST=ON -DGGML_NATIVE=OFF -DGGML_LLAMAFILE=OFF -DGGML_OPENMP=OFF \
      -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16+i8mm \
      -DSTARLING_GGML_TESTS=OFF -DSTARLING_QUANTIZE=OFF >/dev/null
  fi
  if ! cmake --build "$BUILD" -j --target starling-bench > "$TMP/build.log" 2>&1; then
    grep -iE "error|FAILED" "$TMP/build.log" | head -20 >&2
    echo "build failed" >&2
    exit 1
  fi
  adb shell "mkdir -p $DEV"
  adb push "$BUILD/starling-bench" "$DEV/" >/dev/null
  find "$BUILD/ggml" -name 'libggml*.so' -exec adb push {} "$DEV/" \; >/dev/null
fi

# --- inputs on the phone -----------------------------------------------------
have() { adb shell "[ -f $DEV/$1 ]" 2>/dev/null; }
for f in medium.wav short.wav; do
  if ! have "$f"; then
    [ -f "$ROOT/tests/fixtures/$f" ] || (cd "$ROOT" && uv run python tests/fixtures/make_fixtures.py)
    adb push "$ROOT/tests/fixtures/$f" "$DEV/" >/dev/null
  fi
done
for g in "$PK" "$MOSS"; do
  if ! have "$g"; then
    [ -f "$MODELS/$g" ] || { echo "missing model $g (on the phone and in $MODELS)" >&2; exit 1; }
    adb push "$MODELS/$g" "$DEV/" >/dev/null
  fi
done
[ "${REFRESH_TUNE:-0}" = 1 ] && adb shell "rm -f $DEV/starling-fast-tune-*.txt"

# --- runs --------------------------------------------------------------------
run_model() { # kind gguf wav
  adb shell "cd $DEV && LD_LIBRARY_PATH=. STARLING_ENGINE=fast STARLING_FAST_TIMING=1 \
    STARLING_GGML_THREADS=6 STARLING_FAST_CACHE_DIR=$DEV ./starling-bench \
    --model $1 --gguf $DEV/$2 --warmup --runs $RUNS $DEV/$3" > "$TMP/$1.out" 2>&1
  python3 - "$TMP/$1.out" "$1" "$TMP/$1.txt" <<'EOF'
import re, statistics as st, sys
txt, kind, tr_out = open(sys.argv[1]).read(), sys.argv[2], sys.argv[3]
tot = [float(m) for m in re.findall(r'run=\d+ audio=[\d.]+s time=([\d.]+)ms', txt)]
if not tot:
    sys.exit(f"{kind}: no timed runs in the bench output:\n{txt[-2000:]}")
print(f"METRIC {kind}_total_ms={st.median(tot):.1f}")
stages = [l for l in txt.splitlines() if l.startswith(f'[fast-{kind}]') and 'total=' in l]
if stages:
    last = stages[-1]
    for k in ('mel', 'encoder', 'enc+prefill', 'decode'):
        m = re.search(rf'{re.escape(k)}=([\d.]+)ms', last)
        if m: print(f"METRIC {kind}_{k.replace('+', '_')}_ms={float(m.group(1)):.1f}")
    m, d = re.search(r'\((\d+) tokens', last), re.search(r'decode=([\d.]+)ms', last)
    if m and d and int(m.group(1)):
        print(f"METRIC {kind}_ms_per_token={float(d.group(1)) / int(m.group(1)):.1f}")
tr = [l.strip() for l in txt.splitlines() if l.startswith('  ') and l.strip()]
open(tr_out, 'w').write("\n".join(tr) + "\n")
EOF
}
run_model parakeet "$PK" medium.wav | tee "$TMP/pk.metrics"
run_model moss "$MOSS" short.wav | tee "$TMP/moss.metrics"

# --- transcript gate ---------------------------------------------------------
cat "$TMP/parakeet.txt" "$TMP/moss.txt" > "$TMP/transcripts.txt"
if [ "${CAPTURE:-0}" = 1 ]; then
  cp "$TMP/transcripts.txt" "$BASE"
  echo "captured baseline transcripts -> $BASE" >&2
elif ! diff -u "$BASE" "$TMP/transcripts.txt" >&2; then
  echo "TRANSCRIPT MISMATCH" >&2
  exit 3
else
  echo "METRIC transcript_match=1"
fi

pk=$(sed -n 's/^METRIC parakeet_total_ms=//p' "$TMP/pk.metrics")
mo=$(sed -n 's/^METRIC moss_total_ms=//p' "$TMP/moss.metrics")
python3 -c "import sys; print(f'METRIC phone_ms={float(sys.argv[1]) + float(sys.argv[2]):.1f}')" "$pk" "$mo"
