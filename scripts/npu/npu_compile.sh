#!/usr/bin/env bash
# npu_compile.sh — build the Pixel TPU artifacts for one model and install
# them where starling-bench and the app look for them. To do this for every
# model the installed app has loaded, run npu_compile_all.sh instead.
#
#   scripts/npu/npu_compile.sh HF_MODEL (--gguf PHONE_GGUF | --hash 0xHASH --identity 0xIDENTITY)
#                                 [--app PKG[,PKG...]] [--dir DIR]
# HF_MODEL: a Qwen3-ASR checkpoint. Only its audio encoder is built: it is the
# one graph known to compile safely.
#
# WARNING: the phone's compiler can exhaust the phone's memory and panic its
# kernel (it rebooted twice on other graphs, see docs/fast-engine.md, Pixel
# TPU). Save your work on the phone first; npu_compile_phone.sh refuses to
# compile with less than 4 GB available or a graph larger than its window has been (640 MB 0.6B, 1300 MB 1.7B).
#   e.g. scripts/npu/npu_compile.sh Qwen/Qwen3-ASR-0.6B-hf \
#            --gguf /data/local/tmp/starling/qwen3-asr-0.6b-q4e8.gguf --app dev.starling.mobile
#
# 1. Host: export the encoder graph from the HF checkpoint (export_encoder.py;
#    fp16 weights, metadata patched).
# 2. Phone, adb shell: fingerprint the GGUF the engine will load (or take
#    --hash and --identity, the two fingerprints the app wrote to its
#    status-*.txt): its identity must be HF_MODEL's (a fine-tune with other
#    biases or norms is refused: the artifact would carry the base model's
#    weights; one that changed only weight matrices cannot be told apart, so
#    export the checkpoint the GGUF was converted from), and its model hash,
#    a digest of the GGUF's encoder tensors, names the artifact. Then compile
#    the graph with the phone's own on-device compiler (starling-npu-tool) and
#    write qwen3-asr-enc-<model hash>.stnpu.
#    Compiling must run as the adb shell user: the firmware's compile service
#    only serves apps on Google's EdgeTPU allowlist. Loading and running an
#    artifact works from any app.
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
# PY (python with litert-torch, ai-edge-quantizer, transformers, torch);
# NPU_MIN_MEM_MB (npu_compile_phone.sh); STARLING_PHONE_LOCK_HELD=1 when the
# caller already holds /tmp/starling-phone.lock.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
[ $# -ge 3 ] || { sed -n 2,45p "$0"; exit 2; }
MODEL=$1; shift
GGUF=""; HASH=""; ID=""; APP=""; DIR=/data/local/tmp/starling-npu
while [ $# -gt 0 ]; do
  case $1 in
    --gguf) GGUF=$2; shift ;;
    --hash) HASH=$2; shift ;;
    --identity) ID=$2; shift ;;
    --app) APP=$2; shift ;;
    --dir) DIR=$2; shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
  shift
done
if [ -n "$GGUF" ] && [ -z "$HASH$ID" ]; then SRC=$GGUF
elif [ -z "$GGUF" ] && [ -n "$HASH" ] && [ -n "$ID" ]; then SRC=$HASH,$ID
else echo "need --gguf PHONE_GGUF, or --hash 0xHASH and --identity 0xIDENTITY" >&2; exit 2
fi
NPU_TOOL=${NPU_TOOL:-$ROOT/build-android/starling-npu-tool}
PY=${PY:-python3}
[ -x "$NPU_TOOL" ] || { echo "missing $NPU_TOOL (build target starling-npu-tool)" >&2; exit 1; }
WORK=$(mktemp -d /tmp/npu-compile.XXXXXX)
trap 'rm -rf "$WORK"' EXIT
LOCK=/tmp/starling-host-heavy.lock
T=/data/local/tmp/starling-npu-work
field() { "$PY" -I -c "import json,sys; v=json.loads(sys.argv[1])[sys.argv[2]]; print(v if isinstance(v,str) else ','.join(map(str,v)))" "$META" "$1"; }
cfg() {  # cfg PYTHON-EXPR over the checkpoint's config.json (as c)
  "$PY" -I -c "import glob,json,os,sys; m=sys.argv[1]; d=m if os.path.isdir(m) else glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--'+m.replace('/','--')+'/snapshots/*/'))[0]; c=json.load(open(d+'/config.json')); print($1)" "$MODEL"
}
# Only the graph known to compile safely on the phone: the compiler can
# exhaust the phone's memory and panic its kernel, so the graph kind and
# shapes are fixed here, not tunable.
TYPE=$(cfg "c['model_type']")
[ "$TYPE" = qwen3_asr ] ||
  { echo "$MODEL ($TYPE): no TPU graphs for this model (supported: Qwen3-ASR)" >&2; exit 2; }
# The checkpoint must be the app's model: its identity (the same few F32
# tensors the GGUF's identity hashes) has to match before anything is built.
CKID=$("$PY" -I "$ROOT/scripts/npu/export_encoder.py" --fingerprint "$MODEL" | tail -1)
if [ -n "$ID" ] &&
   [ "$("$PY" -I -c 'import sys; print(int(sys.argv[1], 16) == int(sys.argv[2], 16))' "$ID" "$CKID")" != True ]; then
  echo "refusing: the app's model (identity $ID) is not $MODEL ($CKID): a fine-tune or other weights;" >&2
  echo "  its TPU encoder cannot be built from this checkpoint" >&2
  exit 1
fi
TFL=$WORK/enc.tflite
echo "== export $MODEL (encoder, identity $CKID)"
CMD=("$PY" -I "$ROOT/scripts/npu/export_encoder.py" "$MODEL" "$TFL")
if command -v systemd-run >/dev/null; then
  CMD=(systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 "${CMD[@]}")
fi
META=$(flock "$LOCK" "${CMD[@]}" 2>"$WORK/export.log" | tail -1) || { tail -20 "$WORK/export.log" >&2; exit 1; }
echo "$META"
KEY=$(field key)
[ "$(field identity)" = "$CKID" ] || { echo "export identity changed under us" >&2; exit 1; }
echo "== compile on the phone"
PHONE_LOCK=(flock /tmp/starling-phone.lock)
[ -n "${STARLING_PHONE_LOCK_HELD:-}" ] && PHONE_LOCK=()   # the caller holds it
PY=$PY "${PHONE_LOCK[@]}" "$ROOT/scripts/npu/npu_compile_phone.sh" "$NPU_TOOL" "$TFL" "$SRC" "$CKID" "$KEY" "$DIR" "$APP" "$T"
