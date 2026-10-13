#!/usr/bin/env bash
# phone_ablate.sh — split each audio-LLM's enc+prefill into encoder and LLM
# prefill on the phone's GPU path with STARLING_FAST_SKIP (results garbage;
# only the timing lines count). Run under the phone lock:
#   flock /tmp/starling-phone.lock benchmarks/npu/phone_ablate.sh BUILD_DIR OUTDIR [slug:gguf ...]
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/benchmarks/fast_engine/phone_common.sh"
BUILD=$1; OUT=$2; shift 2
S=/data/local/tmp/starling
MODELS=("$@")
[ ${#MODELS[@]} -gt 0 ] || MODELS=(qwen3:qwen3-asr-0.6b-q4e8.gguf qwen3:qwen3-asr-1.7b-q4e8-imx.gguf
  moss:moss-transcribe-preview-2b-q4e8-fullimx.gguf ark06:ark-asr-0.6b-q4k.gguf granite:granite-2b-dynq4.imx.gguf)
D=/data/local/tmp/starling-npu/bench
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
wait_benches || true
for m in "${MODELS[@]}"; do
  slug=${m%%:*}; g=${m#*:}
  for mode in full nollm noenc; do
    case $mode in
      full) skip="" ;;
      nollm) skip="STARLING_FAST_SKIP=llm_*,rope_kv" ;;
      noenc) skip="STARLING_FAST_SKIP=enc_*,win_*,shaw_bias,qf_*,adapter,projector" ;;
    esac
    f="$OUT/${g%.gguf}_$mode.txt"
    timeout 1800 adb shell "cd $D && LD_LIBRARY_PATH=$D STARLING_NPU=0 $skip STARLING_ENGINE=fast STARLING_FAST_TIMING=1 \
      STARLING_FAST_CACHE_DIR=$D ./starling-bench --model $slug --gguf $S/$g --warmup --runs 2 $S/short.wav" > "$f" 2>&1
    echo "--- $g $mode: $(grep -oE 'enc\+prefill=[0-9.]+ms|prefill=[0-9.]+ms' "$f" | tail -2 | tr '\n' ' ') $(grep -oE 'time=[0-9.]+ms' "$f" | tr '\n' ' ')"
    wait_benches || true
  done
done
echo done
