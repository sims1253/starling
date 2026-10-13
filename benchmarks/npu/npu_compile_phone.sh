#!/usr/bin/env bash
# Phone half of npu_compile.sh (run under the phone lock):
#   npu_compile_phone.sh NPU_TOOL TFLITE GGUF KEY IN OUT DIR APP WORKDIR
set -euo pipefail
NPU_TOOL=$1; TFL=$2; GGUF=$3; KEY=$4; IN=$5; OUT=$6; DIR=$7; APP=$8; T=$9
adb shell "mkdir -p $T $DIR" </dev/null
adb push -q "$NPU_TOOL" $T/starling-npu-tool </dev/null >/dev/null
adb push -q "$TFL" $T/encoder.tflite </dev/null >/dev/null
HASH=$(adb shell "$T/starling-npu-tool fingerprint $GGUF" </dev/null | tr -d '\r')
case $HASH in 0x*) ;; *) echo "fingerprint failed: $HASH" >&2; exit 1 ;; esac
NAME="qwen3-asr-enc-${HASH#0x}.stnpu"
echo "fingerprint $HASH -> $NAME"
adb shell "cd $T && ./starling-npu-tool compile encoder.tflite $DIR/$NAME --in $IN --out $OUT --key '$KEY' --hash $HASH" </dev/null
adb shell "rm -f $T/encoder.tflite; $T/starling-npu-tool info $DIR/$NAME" </dev/null
if [ -n "$APP" ]; then
  adb shell "run-as $APP mkdir -p files/npu && cat $DIR/$NAME | run-as $APP sh -c 'cat > files/npu/$NAME.tmp && mv files/npu/$NAME.tmp files/npu/$NAME' && run-as $APP ls -la files/npu" </dev/null
fi
