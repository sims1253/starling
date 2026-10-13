#!/usr/bin/env bash
# phone_npu_fleurs.sh — FLEURS clips through Qwen3-ASR on the phone, TPU
# encoder vs GPU encoder (one starling-bench process per mode, one run per
# clip). Run under the phone lock:
#   flock /tmp/starling-phone.lock benchmarks/npu/phone_npu_fleurs.sh BUILD_DIR GGUF CLIPS_DIR OUTDIR
# CLIPS_DIR comes from benchmarks/fast_engine/export_fleurs.py; score with
# benchmarks/npu/score_fleurs.py CLIPS_DIR/refs.json OUTDIR/*.txt.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/benchmarks/fast_engine/phone_common.sh"
BUILD=$1; GGUF=$2; CLIPS=$3; OUT=$4
NPU_DIR=${NPU_DIR:-/data/local/tmp/starling-npu}
D=$NPU_DIR/bench
F=$NPU_DIR/fleurs
mkdir -p "$OUT"
state=$(wakefulness)
if [ "$state" = "mWakefulness=Awake" ]; then
  fg=$(adb shell "dumpsys activity activities | grep -m1 -E 'mResumedActivity|topResumedActivity'" | tr -d '\r')
  case "$fg" in *Launcher*|*launcher*|*SystemUI*|*keyguard*|"") ;; *) echo "PHONE IN USE: $fg"; exit 3 ;; esac
fi
trap 'kill_benches || true; wake_release || true' EXIT
wake_hold 7200 || exit 1
screen_off || exit 1
adb shell "mkdir -p $D $F"
adb push -q "$BUILD/starling-bench" $D/ >/dev/null
find "$BUILD/ggml" -name 'libggml*.so' -exec adb push -q {} $D/ \; >/dev/null
[ "$(adb shell "ls $F | grep -c wav" | tr -d '\r')" -ge 100 ] || adb push -q "$CLIPS"/*.wav $F/ >/dev/null
LIST=$(cd "$CLIPS" && ls *.wav | sed "s|^|$F/|" | tr '\n' ' ')
wait_benches || true
for mode in gpu npu; do
  env="STARLING_NPU_DIR=$NPU_DIR"; [ $mode = gpu ] && env="$env STARLING_NPU=0"
  timeout 3600 adb shell "cd $D && LD_LIBRARY_PATH=$D $env STARLING_ENGINE=fast STARLING_FAST_TIMING=1 \
    STARLING_FAST_CACHE_DIR=$D ./starling-bench --model qwen3 --gguf $GGUF --runs 1 $LIST" > "$OUT/fleurs_$mode.txt" 2>&1
  echo "$mode: $(grep -c ' run=0 ' "$OUT/fleurs_$mode.txt") clips"
  wait_benches || true
done
echo done
