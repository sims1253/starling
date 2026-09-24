#!/usr/bin/env bash
# android_bench.sh — build starling-bench for arm64 Android, push it with the
# models and fixtures to a phone over adb, and compare the ggml CPU engine
# with the Vulkan fast engine on the device.
#
#   benchmarks/fast_engine/android_bench.sh [--no-build] [--models DIR] [--wav a.wav ...]
#
# Environment: ANDROID_NDK (default: newest under $ANDROID_HOME/ndk),
# PARAKEET_GGUF / MOSS_GGUF (file names inside --models; empty = skip),
# RUNS (timed runs per file, default 3), EXTRA_ENV (e.g. "STARLING_FAST_F16=1").
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BUILD=$ROOT/build-android
MODELS=$ROOT/models
DEV_DIR=/data/local/tmp/starling
RUNS=${RUNS:-3}
PARAKEET_GGUF=${PARAKEET_GGUF-parakeet-tdt-0.6b-v3-q4_k_m-shrink16.gguf}
MOSS_GGUF=${MOSS_GGUF-moss-transcribe-preview-2b-q4e8-fullimx.gguf}
EXTRA_ENV=${EXTRA_ENV:-}
BUILD_IT=1
WAVS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --no-build) BUILD_IT=0 ;;
    --models) [ $# -ge 2 ] || { echo "--models needs a directory" >&2; exit 2; }; MODELS=$2; shift ;;
    --wav) [ $# -ge 2 ] || { echo "--wav needs a file" >&2; exit 2; }; WAVS+=("$2"); shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
  shift
done
if [ ${#WAVS[@]} -eq 0 ]; then
  WAVS=("$ROOT/tests/fixtures/short.wav" "$ROOT/tests/fixtures/medium.wav" "$ROOT/tests/fixtures/long.wav")
fi

if [ "$BUILD_IT" = 1 ]; then
  NDK=${ANDROID_NDK:-$(ls -d "${ANDROID_HOME:-/opt/android-sdk}"/ndk/* | sort -V | tail -1)}
  cmake -S "$ROOT" -B "$BUILD" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-30 -DCMAKE_BUILD_TYPE=Release \
    -DSTARLING_FAST=ON -DGGML_NATIVE=OFF -DGGML_LLAMAFILE=OFF -DGGML_OPENMP=OFF \
    -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16+i8mm \
    -DSTARLING_GGML_TESTS=OFF -DSTARLING_QUANTIZE=OFF
  cmake --build "$BUILD" -j --target starling-bench
fi

adb shell mkdir -p "$DEV_DIR"
adb push "$BUILD/starling-bench" "$DEV_DIR/" >/dev/null
find "$BUILD/ggml" -name 'libggml*.so' -exec adb push {} "$DEV_DIR/" \; >/dev/null
for w in "${WAVS[@]}"; do adb push "$w" "$DEV_DIR/" >/dev/null; done
DEV_WAVS=""
for w in "${WAVS[@]}"; do DEV_WAVS="$DEV_WAVS $DEV_DIR/$(basename "$w")"; done

run() {  # model-kind gguf engine
  local kind=$1 gguf=$2 engine=$3
  echo "== $kind / $engine =="
  # STARLING_GGML_THREADS=6: the app's default (prime + performance cores);
  # letting ggml spread over the efficiency cores makes its decoder ~3x slower.
  adb shell "cd $DEV_DIR && LD_LIBRARY_PATH=$DEV_DIR STARLING_ENGINE=$engine STARLING_FAST_TIMING=1 \
    STARLING_GGML_THREADS=\${STARLING_GGML_THREADS:-6} STARLING_PARAKEET_TIMING=1 STARLING_MOSS_TIMING=1 \
    STARLING_FAST_CACHE_DIR=$DEV_DIR $EXTRA_ENV ./starling-bench --model $kind --gguf $DEV_DIR/$gguf \
    --warmup --runs $RUNS $DEV_WAVS"
}

for pair in "parakeet:$PARAKEET_GGUF" "moss:$MOSS_GGUF"; do
  kind=${pair%%:*}; gguf=${pair#*:}
  [ -n "$gguf" ] || continue
  if ! adb shell test -f "$DEV_DIR/$gguf"; then
    echo "pushing $gguf (one time)"
    adb push "$MODELS/$gguf" "$DEV_DIR/" >/dev/null
  fi
  run "$kind" "$gguf" fast
  run "$kind" "$gguf" ggml
done
