#!/bin/bash
# Cross-build starling-bench for arm64 Android (same flags as
# benchmarks/fast_engine/android_bench.sh) into BUILD (default
# <repo>/build-android) and copy the binary + ggml libs to OUT if given.
# Run under: flock /tmp/starling-host-heavy.lock systemd-run --user --scope -q \
#   -p MemoryMax=8G -p MemorySwapMax=0 benchmarks/litert_gpu/build_android.sh [OUT]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BUILD=${BUILD:-$ROOT/build-android}
NDK=${ANDROID_NDK:-$HOME/android-sdk/ndk/28.2.13676358}
if [ ! -f "$BUILD/build.ninja" ]; then
  cmake -S "$ROOT" -B "$BUILD" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-30 -DCMAKE_BUILD_TYPE=Release \
    -DSTARLING_FAST=ON -DGGML_NATIVE=OFF -DGGML_LLAMAFILE=OFF -DGGML_OPENMP=OFF \
    -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16+i8mm \
    -DSTARLING_GGML_TESTS=OFF -DSTARLING_QUANTIZE=OFF >/dev/null
fi
cmake --build "$BUILD" -j6 --target starling-bench
if [ -n "${1:-}" ]; then
  mkdir -p "$1"
  cp "$BUILD/starling-bench" "$1/"
  find "$BUILD/ggml" -name 'libggml*.so' -exec cp {} "$1/" \;
  sha256sum "$1/starling-bench"
fi
