#!/bin/bash
# NNAPI follow-up: the stock TFLite benchmark_model (Apache-2.0, official TF nightly
# prebuilt) with the NNAPI delegate pinned to google-edgetpu, NNAPI CPU disallowed.
# No LiteRT Google Tensor dispatch/plugin libraries are used.
set -u
export ANDROID_SERIAL=$(adb devices | awk "/^192\.168\.178\.59:[0-9]+\s+device/{print \$1; exit}")
echo "serial: $ANDROID_SERIAL"; [ -n "$ANDROID_SERIAL" ] || { echo "ERROR: phone not on adb"; exit 1; }
LOG=${1:-/tmp/npu-spike/logs/nn1}; mkdir -p "$LOG"
REPO=/home/m0hawk/Documents/starling/.claude/worktrees/agent-a250a02653568f706
source $REPO/benchmarks/fast_engine/phone_common.sh
D=/data/local/tmp/npu-spike/nn
OUTH=/tmp/npu-spike/out
VARIANTS=${VARIANTS:-"nn_f32 nn_fp16wmask nn_wi8mask nn_a8w8 nn_a16w8"}
state=$(wakefulness); echo "display: $state"
if [ "$state" = "mWakefulness=Awake" ]; then
  fg=$(adb shell "dumpsys activity activities | grep -m1 -E 'mResumedActivity|topResumedActivity'" | tr -d '\r')
  case "$fg" in *Launcher*|*launcher*|*SystemUI*|*keyguard*|"") ;; *) echo "PHONE IN USE: $fg"; exit 3 ;; esac
fi
trap 'adb shell "pkill benchmark_model; setprop debug.nn.vlog 0" >/dev/null 2>&1; wake_release || true' EXIT
wake_hold 3600 || exit 1
screen_off || exit 1
adb shell "df -h /data | tail -1"
adb shell "mkdir -p $D/cache $D/out"
adb push -q /tmp/npu-spike/dl/tf-benchmark/android_aarch64_benchmark_model $D/benchmark_model >/dev/null
adb shell "chmod 755 $D/benchmark_model"
for v in $VARIANTS; do
  adb shell "[ -f $D/$v.tflite ] && [ \$(stat -c %s $D/$v.tflite) = $(stat -c %s $OUTH/qwen3_enc_w800_$v.tflite) ]" ||
    adb push -q $OUTH/qwen3_enc_w800_$v.tflite $D/$v.tflite >/dev/null
  adb push -q $OUTH/in_${v}_0.bin $OUTH/in_${v}_1.bin $D/ >/dev/null
done
adb shell "ls -la $D"
adb shell "dumpsys battery | grep -E 'status|AC powered|USB powered'" | tr -d '\r' | tee "$LOG/battery.txt"

bm() {  # bm TAG VARIANT EXTRA_FLAGS...
  local tag=$1 v=$2; shift 2
  adb logcat -c
  timeout 1500 adb shell "cd $D && ./benchmark_model --graph=$D/$v.tflite --use_xnnpack=false --num_threads=4 \
    --input_layer=serving_default_args_0,serving_default_args_1 --input_layer_shape=1,128,800:1,1,1,104 \
    --input_layer_value_files=serving_default_args_0:$D/in_${v}_0.bin,serving_default_args_1:$D/in_${v}_1.bin \
    --output_filepath=$D/out/$tag.bin $*" > "$LOG/$tag.txt" 2>&1
  echo "exit $?" >> "$LOG/$tag.txt"
  adb logcat -d -v brief 2>/dev/null | grep -iE "neuralnetworks|darwinn|edgetpu|nnapi|ExecutionBuilder|CompilationBuilder|ModelBuilder|tflite" | tail -400 > "$LOG/$tag.logcat"
  adb pull -q $D/out/$tag.bin "$LOG/" >/dev/null 2>&1
  echo "--- $tag"
  grep -E "Replacing|delegate|Init:|Inference \(avg\)|count=.*median|ERROR|rror|Failed|exit" "$LOG/$tag.txt" | grep -v "^INFO: Log parameter" | tail -8
}
NN="--use_nnapi=true --nnapi_accelerator_name=google-edgetpu --disable_nnapi_cpu=true --max_delegated_partitions=10000"
adb shell "setprop debug.nn.vlog compilation,execution,model,driver" >/dev/null 2>&1
for v in $VARIANTS; do
  bm ${v}_fp32 $v $NN --num_runs=10 --warmup_runs=1
  case $v in nn_a8w8|nn_a16w8) ;; *) bm ${v}_fp16 $v $NN --nnapi_allow_fp16=true --num_runs=10 --warmup_runs=1 --enable_op_profiling=true ;; esac
done
adb shell "setprop debug.nn.vlog 0"
# compilation caching: same process flags twice with a serialize dir + token
for v in nn_f32 nn_wi8mask; do
  adb shell "rm -rf $D/cache/*"
  bm ${v}_cache1 $v $NN --nnapi_allow_fp16=true --num_runs=5 --warmup_runs=1 --delegate_serialize_dir=$D/cache --delegate_serialize_token=$v
  adb shell "ls -la $D/cache" | tee -a "$LOG/${v}_cache1.txt"
  bm ${v}_cache2 $v $NN --nnapi_allow_fp16=true --num_runs=5 --warmup_runs=1 --delegate_serialize_dir=$D/cache --delegate_serialize_token=$v
done
echo done
