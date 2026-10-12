#!/bin/bash
# Stop any leftover bench and release the shell wake hold (if a session died
# without its EXIT trap). Run under: flock /tmp/starling-phone.lock cleanup.sh
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-192.168.178.59:45929}
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../fast_engine/phone_common.sh"
BENCH_BINS="$BENCH_BINS litert_gpu_bench cl_alu_probe vk_alu_probe"
echo "benches: $(adb shell "pidof $BENCH_BINS" | tr -d '\r')"
echo "wake holder pid file: $(adb shell "cat $WAKEHOLD_DEX.pid 2>/dev/null" | tr -d '\r')"
kill_benches || true
wake_release || true
wake_held && echo "wake lock STILL held" || echo "wake lock released"
