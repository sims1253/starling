#!/usr/bin/env bash
# measure.sh — build the fast-engine Android bench, push it to the Pixel,
# run Parakeet (medium.wav) + MOSS (short.wav) with STARLING_FAST_TIMING,
# and emit METRIC lines. Median of RUNS timed runs per model.
#
# Env: NO_BUILD=1 (skip build/push), RUNS=n (default 3),
#      REFRESH_TUNE=1 (delete on-device autotune cache first),
#      CAPTURE=1 (write .auto/baseline_transcripts.txt instead of checking).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD=$ROOT/build-android
DEV=/data/local/tmp/starling
RUNS=${RUNS:-3}
PK=parakeet-tdt-0.6b-v3-q4_k_m-shrink16.gguf
MOSS=moss-transcribe-preview-2b-q4e8-fullimx.gguf

# --- fast pre-checks ---------------------------------------------------------
command -v glslc >/dev/null || { echo "glslc missing" >&2; exit 1; }
command -v adb >/dev/null || { echo "adb missing" >&2; exit 1; }
adb get-state >/dev/null 2>&1 || { echo "phone not connected" >&2; exit 1; }

# --- build + push ------------------------------------------------------------
if [ "${NO_BUILD:-0}" != 1 ]; then
  NDK=$(ls -d /opt/android-sdk/ndk/* | sort -V | tail -1)
  if [ ! -f "$BUILD/build.ninja" ]; then
    cmake -S "$ROOT" -B "$BUILD" -G Ninja \
      -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
      -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-30 -DCMAKE_BUILD_TYPE=Release \
      -DSTARLING_FAST=ON -DGGML_NATIVE=OFF -DGGML_LLAMAFILE=OFF -DGGML_OPENMP=OFF \
      -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16+i8mm \
      -DSTARLING_GGML_TESTS=OFF -DSTARLING_QUANTIZE=OFF >/dev/null
  fi
  if ! cmake --build "$BUILD" -j --target starling-bench > /tmp/m_build.log 2>&1; then
    grep -iE "error|FAILED" /tmp/m_build.log | head -20 >&2
    echo "build failed (full log: /tmp/m_build.log)" >&2
    exit 1
  fi
  adb shell "mkdir -p $DEV"
  adb push "$BUILD/starling-bench" "$DEV/" >/dev/null
  find "$BUILD/ggml" -name 'libggml*.so' -exec adb push {} "$DEV/" \; >/dev/null
fi

if [ "${REFRESH_TUNE:-0}" = 1 ]; then
  adb shell "rm -f $DEV/starling-fast-tune-*.txt"
fi

OUT=$(mktemp)
trap 'rm -f $OUT' EXIT

run_model() { # kind gguf wav
  adb shell "cd $DEV && LD_LIBRARY_PATH=. STARLING_ENGINE=fast STARLING_FAST_TIMING=1 \
    STARLING_GGML_THREADS=6 STARLING_FAST_CACHE_DIR=$DEV ./starling-bench \
    --model $1 --gguf $DEV/$2 --warmup --runs $RUNS $DEV/$3"
}

echo "== parakeet/medium ==" >&2
sleep 2
run_model parakeet "$PK" medium.wav > "$OUT" 2>&1
cat "$OUT" >&2
python3 - "$OUT" parakeet <<'EOF' > /tmp/m_pk.txt
import re, sys, statistics as st
txt = open(sys.argv[1]).read()
kind = sys.argv[2]
tot = [float(m) for m in re.findall(r'run=\d+ audio=[\d.]+s time=([\d.]+)ms', txt)]
tr  = [l.strip() for l in txt.splitlines() if l.startswith('  ') and l.strip()]
tl  = [l for l in txt.splitlines() if l.startswith(f'[fast-{kind}]') and 'total=' in l]
split = {}
if tl:
    last = tl[-1]
    for k in ('mel','encoder','enc+prefill','decode','total'):
        m = re.search(rf'{re.escape(k)}=([\d.]+)ms', last)
        if m: split[k.replace('+','_')] = float(m.group(1))
med = st.median(tot) if tot else 0
print(f"METRIC {kind}_total_ms={med:.1f}")
for k, v in split.items():
    print(f"METRIC {kind}_{k}ms={v:.1f}")
open(f"/tmp/{kind}_transcript.txt","w").write("\n".join(tr)+"\n")
EOF
cat /tmp/m_pk.txt

echo "== moss/short ==" >&2
sleep 2
run_model moss "$MOSS" short.wav > "$OUT" 2>&1
cat "$OUT" >&2
python3 - "$OUT" moss <<'EOF' > /tmp/m_moss.txt
import re, sys, statistics as st
txt = open(sys.argv[1]).read()
kind = sys.argv[2]
tot = [float(m) for m in re.findall(r'run=\d+ audio=[\d.]+s time=([\d.]+)ms', txt)]
tr  = [l.strip() for l in txt.splitlines() if l.startswith('  ') and l.strip()]
tl  = [l for l in txt.splitlines() if l.startswith(f'[fast-{kind}]') and 'total=' in l]
split = {}
if tl:
    last = tl[-1]
    for k in ('mel','enc+prefill','decode','total'):
        m = re.search(rf'{re.escape(k)}=([\d.]+)ms', last)
        if m: split[k.replace('+','_')] = float(m.group(1))
    m = re.search(r'\((\d+) tokens', last)
    if m:
        n = int(m.group(1))
        if 'decode' in split and n: split['per_tok'] = split['decode']/n
med = st.median(tot) if tot else 0
print(f"METRIC {kind}_total_ms={med:.1f}")
for k, v in split.items():
    print(f"METRIC {kind}_{k}ms={v:.1f}")
open(f"/tmp/{kind}_transcript.txt","w").write("\n".join(tr)+"\n")
EOF
cat /tmp/m_moss.txt

# --- transcript gate ---------------------------------------------------------
BASE=$ROOT/.auto/baseline_transcripts.txt
cat /tmp/parakeet_transcript.txt /tmp/moss_transcript.txt > /tmp/transcripts_all.txt
if [ "${CAPTURE:-0}" = 1 ]; then
  cp /tmp/transcripts_all.txt "$BASE"
  echo "captured baseline transcripts" >&2
else
  if ! diff -q "$BASE" /tmp/transcripts_all.txt >/dev/null 2>&1; then
    echo "TRANSCRIPT MISMATCH (gate 1 failed):" >&2
    diff "$BASE" /tmp/transcripts_all.txt >&2 || true
    cp /tmp/transcripts_all.txt /tmp/transcripts_all.failed.txt
    exit 3
  fi
  echo "METRIC transcript_match=1"
fi

# --- composite ---------------------------------------------------------------
PK_T=$(grep -oP 'parakeet_total_ms=\K[\d.]+' /tmp/m_pk.txt | head -1)
MO_T=$(grep -oP 'moss_total_ms=\K[\d.]+' /tmp/m_moss.txt | head -1)
python3 -c "print(f'METRIC phone_ms={float('$PK_T')+float('$MO_T'):.1f}')"
