#!/bin/bash
# Host (Linux, RADV) build of starling-bench with the fast engine plus the
# fast-engine unit tests, into BUILD (default <repo>/build-host).
# Run under: flock /tmp/starling-host-heavy.lock systemd-run --user --scope -q \
#   -p MemoryMax=8G -p MemorySwapMax=0 benchmarks/litert_gpu/build_host.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BUILD=${BUILD:-$ROOT/build-host}
if [ ! -f "$BUILD/build.ninja" ]; then
  cmake -S "$ROOT" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DSTARLING_FAST=ON -DSTARLING_SERVE=ON -DBUILD_SHARED_LIBS=OFF \
    -DGGML_NATIVE=OFF -DGGML_LLAMAFILE=OFF >/dev/null
fi
cmake --build "$BUILD" -j6 --target starling-bench fast_weights_test greedy_termination_test
