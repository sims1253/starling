#!/bin/bash
# NNAPI op-coverage probes on google-edgetpu: single-op and single-layer models,
# f32 strict / f32 relaxed (allow_fp16) / int8 / int16 activations.
set -u
export ANDROID_SERIAL=$(adb devices | awk "/^192\.168\.178\.59:[0-9]+\s+device/{print \$1; exit}")
echo "serial: $ANDROID_SERIAL"; [ -n "$ANDROID_SERIAL" ] || { echo "ERROR: phone not on adb"; exit 1; }
LOG=${1:-/tmp/npu-spike/logs/nnp}; mkdir -p "$LOG"
REPO=/home/m0hawk/Documents/starling/.claude/worktrees/agent-a250a02653568f706
source $REPO/benchmarks/fast_engine/phone_common.sh
D=/data/local/tmp/npu-spike/nn
state=$(wakefulness); echo "display: $state"
if [ "$state" = "mWakefulness=Awake" ]; then
  fg=$(adb shell "dumpsys activity activities | grep -m1 -E 'mResumedActivity|topResumedActivity'" | tr -d '\r')
  case "$fg" in *Launcher*|*launcher*|*SystemUI*|*keyguard*|"") ;; *) echo "PHONE IN USE: $fg"; exit 3 ;; esac
fi
trap 'adb shell "pkill benchmark_model; setprop debug.nn.vlog 0" >/dev/null 2>&1; wake_release || true' EXIT
wake_hold 3600 || exit 1
screen_off || exit 1
adb shell "mkdir -p $D/probes $D/cache"
adb push -q /tmp/npu-spike/out/probes/. $D/probes/ >/dev/null
adb shell "setprop debug.nn.vlog compilation,model,driver" >/dev/null 2>&1
NN="--use_nnapi=true --nnapi_accelerator_name=google-edgetpu --disable_nnapi_cpu=true --max_delegated_partitions=10000"
run() {  # run TAG MODEL FLAGS...
  local tag=$1 m=$2; shift 2
  adb logcat -c
  timeout 600 adb shell "cd $D && ./benchmark_model --graph=$D/probes/$m.tflite --use_xnnpack=false --num_threads=4 --num_runs=50 --warmup_runs=3 $*" > "$LOG/$tag.txt" 2>&1
  echo "exit $?" >> "$LOG/$tag.txt"
  adb logcat -d -v brief > "$LOG/$tag.logcat" 2>/dev/null
  printf "%-22s %s | %s | %s\n" "$tag" \
    "$(grep -oE 'Replacing [0-9]+ out of [0-9]+ node\(s\)[^,]*, yielding [0-9]+ partitions|will not be executed by the delegate|BAD_DATA[^.]*|OP_FAILED[^.]*' "$LOG/$tag.txt" | head -1)" \
    "$(grep -oE 'Ops supported = [0-9]+, not supported = [0-9]+' "$LOG/$tag.logcat" | sort | uniq -c | tr '\n' ' ')" \
    "$(grep -oE 'Init: [0-9]+|Inference \(avg\): [0-9.e+]+' "$LOG/$tag.txt" | tr '\n' ' ')"
}
for p in fc bmm softmax ln gelu conv heads layernm layer; do
  run ${p}_f32 $p $NN
  run ${p}_relaxed $p $NN --nnapi_allow_fp16=true
  run ${p}_a8w8 ${p}_a8w8 $NN
  run ${p}_a16w8 ${p}_a16w8 $NN
done
run layer_cpu layer
run layer_a8w8_cpu layer_a8w8
# compilation caching on whatever layer variant delegates
for v in layer_a8w8 layer; do
  adb shell "rm -rf $D/cache/*"
  run ${v}_cache1 $v $NN --nnapi_allow_fp16=true --delegate_serialize_dir=$D/cache --delegate_serialize_token=$v
  adb shell "ls -la $D/cache" >> "$LOG/${v}_cache1.txt"
  run ${v}_cache2 $v $NN --nnapi_allow_fp16=true --delegate_serialize_dir=$D/cache --delegate_serialize_token=$v
done
adb shell "setprop debug.nn.vlog 0"
echo done
