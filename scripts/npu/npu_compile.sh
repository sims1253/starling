#!/usr/bin/env bash
# npu_compile.sh — build the Pixel TPU artifact for one GGUF and install it
# where starling-bench and the app look for it. To do this for every model
# the installed app has loaded, run npu_compile_all.sh instead.
#
#   scripts/npu/npu_compile.sh MODEL.gguf [--hash 0xHASH] [--app PKG[,PKG...]] [--dir DIR]
# MODEL.gguf: a Qwen3-ASR or Parakeet-TDT GGUF on this host, the very file the
# phone loads. Only the encoder is built (Qwen3-ASR's audio encoder window,
# Parakeet's FastConformer encoder in five pieces, export_parakeet.py): the
# graphs known to compile safely.
#
# WARNING: the phone's compiler can exhaust the phone's memory and panic its
# kernel (it rebooted twice on other graphs, see docs/fast-engine.md, Pixel
# TPU). Save your work on the phone first; npu_compile_phone.sh refuses to
# compile with less than 4 GB available or a graph larger than its window has
# been (640 MB 0.6B, 1300 MB 1.7B).
#   e.g. scripts/npu/npu_compile.sh models/qwen3-asr-0.6b-q4e8.gguf --app dev.starling.mobile
#
# 1. Host: export the encoder graph from MODEL.gguf (export_encoder.py: the
#    GGUF's own encoder weights, dequantized, stored as fp16; metadata
#    patched). The export also computes the GGUF's model hash, the digest of
#    its encoder tensors and metadata the engine computes at load, and
#    stamps it into the graph: the artifact is named
#    qwen3-asr-enc-<model hash>.stnpu, so it is only ever used beside the
#    encoder it was built from. --hash (the hash an app's status-*.txt asked
#    for) makes the export refuse a GGUF with another one.
# 2. Phone, adb shell: compile the graph with the phone's own on-device
#    compiler (starling-npu-tool). Compiling must run as the adb shell user:
#    the firmware's compile service only serves apps on Google's EdgeTPU
#    allowlist. Loading and running an artifact works from any app.
# 3. Install: --dir (default /data/local/tmp/starling-npu; starling-bench gets
#    STARLING_NPU_DIR=that) and, for each --app package, files/npu of a
#    debuggable app (run-as) or the external files dir of any other
#    (/sdcard/Android/data/<pkg>/files/npu); NativeSupport points
#    STARLING_NPU_DIR at both. An artifact is tied to the GGUF's encoder
#    tensors, the graph shape and the TPU firmware version; anything else
#    makes the engine ignore it, run the GPU, and say so (logcat tag
#    StarlingNpu, status-*.txt beside the artifacts).
#
# Env: ANDROID_SERIAL; NPU_TOOL (default build-android/starling-npu-tool,
# built with `cmake --build <android build> --target starling-npu-tool`);
# PY (python with gguf, litert-torch, ai-edge-quantizer, torch);
# NPU_MIN_MEM_MB (npu_compile_phone.sh); STARLING_PHONE_LOCK_HELD=1 when the
# caller already holds /tmp/starling-phone.lock.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
[ $# -ge 1 ] && [ "${1#-}" = "$1" ] || { sed -n 2,42p "$0"; exit 2; }
GGUF=$1; shift
HASH=""; APP=""; DIR=/data/local/tmp/starling-npu
while [ $# -gt 0 ]; do
  case $1 in
    --hash) HASH=$2; shift ;;
    --app) APP=$2; shift ;;
    --dir) DIR=$2; shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
  shift
done
[ -f "$GGUF" ] || { echo "no such file: $GGUF" >&2; exit 2; }
NPU_TOOL=${NPU_TOOL:-$ROOT/build-android/starling-npu-tool}
PY=${PY:-python3}
[ -x "$NPU_TOOL" ] || { echo "missing $NPU_TOOL (build target starling-npu-tool)" >&2; exit 1; }
WORK=$(mktemp -d /tmp/npu-compile.XXXXXX)
trap 'rm -rf "$WORK"' EXIT
LOCK=/tmp/starling-host-heavy.lock
T=/data/local/tmp/starling-npu-work
field() { "$PY" -I -c "import json,sys; v=json.loads(sys.argv[1])[sys.argv[2]]; print(v if isinstance(v,str) else ','.join(map(str,v)))" "$META" "$1"; }

same() { [ "$("$PY" -I -c 'import sys; print(int(sys.argv[1], 16) == int(sys.argv[2], 16))' "$1" "$2")" = True ]; }
PHONE_LOCK=(flock /tmp/starling-phone.lock)
[ -n "${STARLING_PHONE_LOCK_HELD:-}" ] && PHONE_LOCK=()   # the caller holds it
heavy() {   # the export, capped at 8 GB where systemd can
  if command -v systemd-run >/dev/null; then
    flock "$LOCK" systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 "$@"
  else
    flock "$LOCK" "$@"
  fi
}

if "$PY" -I "$ROOT/scripts/npu/export_parakeet.py" --model-hash "$GGUF" >/dev/null 2>&1; then
  # Parakeet-TDT: five pieces (the subsampling chunk, four 6-layer segments),
  # each its own artifact parakeet-enc-<model hash>-<piece>.stnpu.
  echo "== export $GGUF (Parakeet encoder, 5 pieces${HASH:+, expecting model hash $HASH})"
  CMD=("$PY" -I "$ROOT/scripts/npu/export_parakeet.py" "$GGUF" "$WORK/pk")
  [ -n "$HASH" ] && CMD+=(--expect-hash "$HASH")
  heavy "${CMD[@]}" 2>"$WORK/export.log" | grep '^{' >"$WORK/pieces.json" || { tail -20 "$WORK/export.log" >&2; exit 1; }
  [ "$(wc -l <"$WORK/pieces.json")" -ge 2 ] || { tail -20 "$WORK/export.log" >&2; exit 1; }
  while read -r META; do
    KEY=$(field key)
    MH=$(field model_hash)
    if [ -n "$HASH" ] && ! same "$HASH" "$MH"; then
      echo "refusing: the export's model hash $MH is not $HASH" >&2
      exit 1
    fi
    echo "== compile $(field piece) on the phone"
    PY=$PY "${PHONE_LOCK[@]}" "$ROOT/scripts/npu/npu_compile_phone.sh" "$NPU_TOOL" "$(field tflite)" "$MH" "$KEY" \
      "$DIR" "$APP" "$T" </dev/null
    rm -f "$(field tflite)"
  done <"$WORK/pieces.json"
  exit 0
fi

TFL=$WORK/enc.tflite
echo "== export $GGUF (encoder${HASH:+, expecting model hash $HASH})"
CMD=("$PY" -I "$ROOT/scripts/npu/export_encoder.py" "$GGUF" "$TFL")
[ -n "$HASH" ] && CMD+=(--expect-hash "$HASH")
META=$(heavy "${CMD[@]}" 2>"$WORK/export.log" | tail -1) || { tail -20 "$WORK/export.log" >&2; exit 1; }
echo "$META"
KEY=$(field key)
MH=$(field model_hash)
if [ -n "$HASH" ] && ! same "$HASH" "$MH"; then
  echo "refusing: the export's model hash $MH is not $HASH" >&2
  exit 1
fi
echo "== compile on the phone"
PY=$PY "${PHONE_LOCK[@]}" "$ROOT/scripts/npu/npu_compile_phone.sh" "$NPU_TOOL" "$TFL" "$MH" "$KEY" "$DIR" "$APP" "$T"
