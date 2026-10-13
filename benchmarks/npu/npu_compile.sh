#!/usr/bin/env bash
# npu_compile.sh — build the Pixel TPU artifact for a Qwen3-ASR model's audio
# encoder and install it where starling-bench and the app look for it.
#
#   benchmarks/npu/npu_compile.sh HF_MODEL PHONE_GGUF [--app PKG] [--dir DIR]
#   e.g. benchmarks/npu/npu_compile.sh Qwen/Qwen3-ASR-0.6B-hf \
#            /data/local/tmp/starling/qwen3-asr-0.6b-q4e8.gguf --app dev.starling.mobile.debug
#
# 1. Host: export the encoder window graph from the HF checkpoint
#    (export_encoder.py; fp16 weights, metadata patched).
# 2. Phone, adb shell: fingerprint the GGUF the engine will load, compile the
#    graph with the phone's own on-device compiler (starling-npu-tool), and
#    write qwen3-asr-enc-<fingerprint>.stnpu. Compiling must run as the adb
#    shell user: the firmware's compile service only serves apps on Google's
#    EdgeTPU allowlist. Loading and running an artifact works from any app.
# 3. Install: --dir (default /data/local/tmp/starling-npu; starling-bench gets
#    STARLING_NPU_DIR=that) and, with --app, the debuggable app's files/npu
#    via run-as. A release (non-debuggable) app reads the same file from its
#    external files dir instead:
#      adb push ART /sdcard/Android/data/<pkg>/files/npu/
#    (NativeSupport points STARLING_NPU_DIR at both). The artifact is tied to
#    the GGUF's fingerprint, the encoder shape and the TPU firmware version;
#    anything else makes the engine ignore it and run the GPU encoder.
#
# Env: ANDROID_SERIAL; NPU_TOOL (default build-android-npu/starling-npu-tool,
# built with `cmake --build <android build> --target starling-npu-tool`);
# PY (python with litert-torch, ai-edge-quantizer, transformers, torch).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
[ $# -ge 2 ] || { sed -n 2,26p "$0"; exit 2; }
MODEL=$1; GGUF=$2; shift 2
APP=""; DIR=/data/local/tmp/starling-npu
while [ $# -gt 0 ]; do
  case $1 in
    --app) APP=$2; shift ;;
    --dir) DIR=$2; shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
  shift
done
NPU_TOOL=${NPU_TOOL:-$ROOT/build-android-npu/starling-npu-tool}
PY=${PY:-python3}
[ -x "$NPU_TOOL" ] || { echo "missing $NPU_TOOL (build target starling-npu-tool)" >&2; exit 1; }
WORK=$(mktemp -d /tmp/npu-compile.XXXXXX)
trap 'rm -rf "$WORK"' EXIT
TFL=$WORK/encoder.tflite

echo "== export $MODEL"
LOCK=/tmp/starling-host-heavy.lock
CMD=("$PY" -I "$ROOT/benchmarks/npu/export_encoder.py" "$MODEL" "$TFL")
if command -v systemd-run >/dev/null; then
  CMD=(systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 "${CMD[@]}")
fi
META=$(flock "$LOCK" "${CMD[@]}" 2>"$WORK/export.log" | tail -1) || { tail -20 "$WORK/export.log" >&2; exit 1; }
echo "$META"
field() { "$PY" -I -c "import json,sys; print(json.loads(sys.argv[1])[sys.argv[2]] if sys.argv[2]=='key' else ','.join(map(str,json.loads(sys.argv[1])[sys.argv[2]])))" "$META" "$1"; }
KEY=$(field key); IN=$(field in); OUT=$(field out)

echo "== compile on the phone"
T=/data/local/tmp/starling-npu-work
flock /tmp/starling-phone.lock "$ROOT/benchmarks/npu/npu_compile_phone.sh" "$NPU_TOOL" "$TFL" "$GGUF" "$KEY" "$IN" "$OUT" "$DIR" "$APP" "$T"
