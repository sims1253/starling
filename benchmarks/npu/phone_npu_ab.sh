#!/usr/bin/env bash
# phone_npu_ab.sh — Qwen3-ASR on the fast engine with the TPU encoder vs the
# GPU encoder (STARLING_NPU=0), same starling-bench binary, alternating, screen
# off with the wake hold. Run under the phone lock:
#   flock /tmp/starling-phone.lock benchmarks/npu/phone_npu_ab.sh BUILD_DIR GGUF OUTDIR [wav...]
# BUILD_DIR holds an Android starling-bench (+ ggml libs); the artifact must
# be in $NPU_DIR (default /data/local/tmp/starling-npu, see npu_compile.sh).
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/benchmarks/fast_engine/phone_common.sh"
BUILD=$1; GGUF=$2; OUT=$3; shift 3
WAVS=("$@"); [ ${#WAVS[@]} -gt 0 ] || WAVS=(short.wav medium.wav long.wav)
RUNS=${RUNS:-3}; REPS=${REPS:-2}
NPU_DIR=${NPU_DIR:-/data/local/tmp/starling-npu}
D=$NPU_DIR/bench
mkdir -p "$OUT"
state=$(wakefulness)
if [ "$state" = "mWakefulness=Awake" ]; then
  fg=$(adb shell "dumpsys activity activities | grep -m1 -E 'mResumedActivity|topResumedActivity'" | tr -d '\r')
  case "$fg" in *Launcher*|*launcher*|*SystemUI*|*keyguard*|"") ;; *) echo "PHONE IN USE: $fg"; exit 3 ;; esac
fi
trap 'kill_benches || true; wake_release || true' EXIT
wake_hold 7200 || exit 1
screen_off || exit 1
adb shell "mkdir -p $D"
adb push -q "$BUILD/starling-bench" $D/ >/dev/null
find "$BUILD/ggml" -name 'libggml*.so' -exec adb push -q {} $D/ \; >/dev/null
adb shell "dumpsys battery | grep -E '^  status|AC powered'" | tr -d '\r' | tee "$OUT/battery.txt"
wait_benches || true
for rep in $(seq 1 "$REPS"); do
  for w in "${WAVS[@]}"; do
    for mode in gpu npu; do
      env="STARLING_NPU_DIR=$NPU_DIR"; [ $mode = gpu ] && env="$env STARLING_NPU=0"
      f="$OUT/${w%.wav}_${mode}_$rep.txt"
      timeout 1200 adb shell "cd $D && LD_LIBRARY_PATH=$D $env STARLING_ENGINE=fast STARLING_FAST_TIMING=1 \
        STARLING_FAST_CACHE_DIR=$D ./starling-bench --model qwen3 --gguf $GGUF --warmup --runs $RUNS \
        /data/local/tmp/starling/$w" > "$f" 2>&1
      echo "--- $w $mode rep $rep: $(grep -oE 'load [0-9.]+ ms' "$f" | head -1) $(grep -oE 'time=[0-9.]+ms' "$f" | tr '\n' ' ')"
      grep -m1 -E "npu:" "$f" || true
      wait_benches || true
    done
  done
done
echo done
