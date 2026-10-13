#!/usr/bin/env bash
# Phone half of npu_compile.sh (run under the phone lock):
#   npu_compile_phone.sh NPU_TOOL TFLITE GGUF|0xHASH EXPECT_HASH KEY IN OUT DIR APPS WORKDIR
# compiles the Qwen3-ASR audio encoder graph TFLITE into an artifact.
# Refuses to compile when the phone is short of memory (MemAvailable below
# NPU_MIN_MEM_MB, default 4096) or the graph is larger than 640 MB: the
# compiler's own memory use can take the phone down.
# The model fingerprint comes from the phone-side GGUF, or is given (0x...,
# from the app's status-*.txt), and must equal EXPECT_HASH, the fingerprint of
# the checkpoint TFLITE was exported from (export_encoder.py --fingerprint).
# APPS: comma-separated packages; a debuggable
# one gets the artifact in files/npu (run-as), any other in its external
# files dir (/sdcard/Android/data/<pkg>/files/npu). Both are on the engine's
# STARLING_NPU_DIR (NativeSupport).
set -euo pipefail
NPU_TOOL=$1; TFL=$2; SRC=$3; EXPECT=$4; KEY=$5; IN=$6; OUT=$7; DIR=$8; APPS=$9; T=${10}
# A fingerprint in canonical form (0x + 16 lower-case hex digits), or "".
canon() {
  local h
  h=$(printf '%s' "$1" | tr 'A-FX' 'a-fx')
  [[ $h =~ ^0x[0-9a-f]{1,16}$ ]] || { echo ""; return; }
  h=${h#0x}
  printf '0x%s%s' "$(printf '%*s' $((16 - ${#h})) '' | tr ' ' 0)" "$h"
}
EXP=$(canon "$EXPECT")
[ -n "$EXP" ] || { echo "bad expected fingerprint '$EXPECT'" >&2; exit 2; }
adb shell "mkdir -p $T $DIR" </dev/null
adb push -q "$NPU_TOOL" $T/starling-npu-tool </dev/null >/dev/null
adb push -q "$TFL" $T/graph.tflite </dev/null >/dev/null
case $SRC in
  0x*) HASH=$SRC ;;
  *) HASH=$(adb shell "$T/starling-npu-tool fingerprint $SRC" </dev/null | tr -d "\r" || true) ;;
esac
HASH=$(canon "$HASH")
[ -n "$HASH" ] || { echo "fingerprint failed" >&2; exit 1; }
if [ "$HASH" != "$EXP" ]; then
  echo "refusing: the app's model ($HASH) is not the checkpoint the graph was exported from ($EXP)." >&2
  echo "  A fine-tune or other weights with the same shapes would get the wrong encoder." >&2
  exit 1
fi
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
