#!/usr/bin/env bash
# Phone half of npu_compile.sh (run under the phone lock):
#   npu_compile_phone.sh NPU_TOOL TFLITE GGUF|0xHASH KEY IN OUT DIR APPS WORKDIR
# compiles the Qwen3-ASR audio encoder graph TFLITE into an artifact.
# Refuses to compile when the phone is short of memory (MemAvailable below
# NPU_MIN_MEM_MB, default 4096) or the graph is larger than 640 MB: the
# compiler's own memory use can take the phone down.
# The model fingerprint comes from the phone-side GGUF, or is given (0x...,
# from the app's status-*.txt). APPS: comma-separated packages; a debuggable
# one gets the artifact in files/npu (run-as), any other in its external
# files dir (/sdcard/Android/data/<pkg>/files/npu). Both are on the engine's
# STARLING_NPU_DIR (NativeSupport).
set -euo pipefail
NPU_TOOL=$1; TFL=$2; SRC=$3; KEY=$4; IN=$5; OUT=$6; DIR=$7; APPS=$8; T=$9
adb shell "mkdir -p $T $DIR" </dev/null
adb push -q "$NPU_TOOL" $T/starling-npu-tool </dev/null >/dev/null
adb push -q "$TFL" $T/graph.tflite </dev/null >/dev/null
case $SRC in
  0x*) HASH=$SRC ;;
  *) HASH=$(adb shell "$T/starling-npu-tool fingerprint $SRC" </dev/null | tr -d "\r" || true) ;;
esac
case $HASH in 0x*) ;; *) echo "fingerprint failed: $HASH" >&2; exit 1 ;; esac
NAME="qwen3-asr-enc-${HASH#0x}.stnpu"
echo "fingerprint $HASH -> $NAME"
SIZE=$(stat -c %s "$TFL")
[ "$SIZE" -le $((640 << 20)) ] || { echo "graph is $((SIZE >> 20)) MB, above the 640 MB safety limit" >&2; exit 1; }
AVAIL=$(adb shell "grep MemAvailable /proc/meminfo" </dev/null | tr -dc '0-9')
[ "${AVAIL:-0}" -ge $((${NPU_MIN_MEM_MB:-4096} * 1024)) ] ||
  { echo "phone has $((${AVAIL:-0} / 1024)) MB available, below ${NPU_MIN_MEM_MB:-4096} MB: not compiling now" >&2; exit 1; }
echo "MemAvailable $((AVAIL / 1024)) MB"
adb shell "cd $T && ./starling-npu-tool compile graph.tflite $DIR/$NAME --in $IN --out $OUT --key '$KEY' --hash $HASH" </dev/null
adb shell "rm -f $T/graph.tflite; $T/starling-npu-tool info $DIR/$NAME" </dev/null
for APP in ${APPS//,/ }; do
  if adb shell "run-as $APP true" </dev/null >/dev/null 2>&1; then
    adb shell "run-as $APP mkdir -p files/npu && cat $DIR/$NAME | run-as $APP sh -c 'cat > files/npu/$NAME.tmp && mv files/npu/$NAME.tmp files/npu/$NAME'" </dev/null
    echo "installed for $APP (files/npu, run-as)"
  else
    X=/sdcard/Android/data/$APP/files/npu
    adb shell "mkdir -p $X && cp $DIR/$NAME $X/$NAME.tmp && mv $X/$NAME.tmp $X/$NAME" </dev/null
    echo "installed for $APP ($X)"
  fi
done
