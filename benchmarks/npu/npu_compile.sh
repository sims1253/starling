#!/usr/bin/env bash
# npu_compile.sh — build the Pixel TPU artifacts for one model and install
# them where starling-bench and the app look for them. To do this for every
# model the installed app has loaded, run npu_compile_all.sh instead.
#
#   benchmarks/npu/npu_compile.sh HF_MODEL (--gguf PHONE_GGUF | --hash 0xHASH)
#                                 [--app PKG[,PKG...]] [--dir DIR]
#                                 [--prefill 128] [--no-encoder]
# HF_MODEL: a Qwen3-ASR checkpoint: the audio encoder and, with --prefill 128
# (Qwen3-ASR-0.6B only, opt-in), the LLM prefill for prompts of up to 128
# tokens. Nothing else: only these graphs are known to compile safely.
#
# WARNING: the phone's compiler can exhaust the phone's memory and panic its
# kernel (it rebooted twice on other graphs, see docs/fast-engine.md, Pixel
# TPU). Save your work on the phone first; npu_compile_phone.sh refuses to
# compile with less than 4 GB available or graphs above 640 MB.
#   e.g. benchmarks/npu/npu_compile.sh Qwen/Qwen3-ASR-0.6B-hf \
#            --gguf /data/local/tmp/starling/qwen3-asr-0.6b-q4e8.gguf --app dev.starling.mobile
#
# 1. Host: export the graphs from the HF checkpoint (export_*.py; fp16
#    weights, metadata patched): the encoder and, with --prefill, the LLM
#    prefill chain for bucket 128 (two 14-layer segments; used only together
#    with the encoder artifact; longer prompts use the GPU prefill).
# 2. Phone, adb shell: fingerprint the GGUF the engine will load (or take
#    --hash, the fingerprint the app wrote to its status-*.txt; it then
#    applies to every piece built, so pass --no-encoder / no --prefill to
#    build one piece), compile each graph with the phone's own on-device
#    compiler (starling-npu-tool) and write <piece>-<fingerprint>*.stnpu.
#    Compiling must run as the adb shell user: the firmware's compile service
#    only serves apps on Google's EdgeTPU allowlist. Loading and running an
#    artifact works from any app.
# 3. Install: --dir (default /data/local/tmp/starling-npu; starling-bench gets
#    STARLING_NPU_DIR=that) and, for each --app package, files/npu of a
#    debuggable app (run-as) or the external files dir of any other
#    (/sdcard/Android/data/<pkg>/files/npu); NativeSupport points
#    STARLING_NPU_DIR at both. An artifact is tied to the GGUF's fingerprint,
#    the graph shape and the TPU firmware version; anything else makes the
#    engine ignore it, run the GPU, and say so (logcat tag StarlingNpu,
#    status-*.txt beside the artifacts).
#
# Env: ANDROID_SERIAL; NPU_TOOL (default build-android-npu/starling-npu-tool,
# built with `cmake --build <android build> --target starling-npu-tool`);
# PY (python with litert-torch, ai-edge-quantizer, transformers, torch);
# NPU_MIN_MEM_MB (npu_compile_phone.sh); STARLING_PHONE_LOCK_HELD=1 when the
# caller already holds /tmp/starling-phone.lock.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
[ $# -ge 3 ] || { sed -n 2,40p "$0"; exit 2; }
MODEL=$1; shift
SRC=""; APP=""; DIR=/data/local/tmp/starling-npu; PREFILL=""; ENC=1
while [ $# -gt 0 ]; do
  case $1 in
    --gguf) SRC=$2; shift ;;
    --hash) SRC=$2; shift ;;
    --app) APP=$2; shift ;;
    --dir) DIR=$2; shift ;;
    --prefill) PREFILL=$2; shift ;;
    --no-encoder) ENC=0 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
  shift
done
[ -n "$SRC" ] || { echo "need --gguf PHONE_GGUF or --hash 0xHASH" >&2; exit 2; }
NPU_TOOL=${NPU_TOOL:-$ROOT/build-android-npu/starling-npu-tool}
PY=${PY:-python3}
[ -x "$NPU_TOOL" ] || { echo "missing $NPU_TOOL (build target starling-npu-tool)" >&2; exit 1; }
WORK=$(mktemp -d /tmp/npu-compile.XXXXXX)
trap 'rm -rf "$WORK"' EXIT
LOCK=/tmp/starling-host-heavy.lock
T=/data/local/tmp/starling-npu-work
field() { "$PY" -I -c "import json,sys; print(json.loads(sys.argv[1])[sys.argv[2]] if sys.argv[2]=='key' else ','.join(map(str,json.loads(sys.argv[1])[sys.argv[2]])))" "$META" "$1"; }
build() {  # build KIND EXPORT_SCRIPT ARGS...
  local kind=$1 script=$2; shift 2
  local tfl=$WORK/$kind.tflite
  echo "== export $MODEL ($kind)"
  local cmd=("$PY" -I "$ROOT/benchmarks/npu/$script" "$MODEL" "$@" "$tfl")
  if command -v systemd-run >/dev/null; then
    cmd=(systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 "${cmd[@]}")
  fi
  META=$(flock "$LOCK" "${cmd[@]}" 2>"$WORK/export.log" | tail -1) || { tail -20 "$WORK/export.log" >&2; exit 1; }
  echo "$META"
  local key in out
  key=$(field key); in=$(field in); out=$(field out)
  echo "== compile on the phone ($kind)"
  local lock=(flock /tmp/starling-phone.lock)
  [ -n "${STARLING_PHONE_LOCK_HELD:-}" ] && lock=()   # the caller holds it
  "${lock[@]}" "$ROOT/benchmarks/npu/npu_compile_phone.sh" "$NPU_TOOL" "$tfl" "$SRC" "$key" "$in" "$out" "$DIR" "$APP" "$T" "$kind"
  rm -f "$tfl"
}
cfg() {  # cfg PYTHON-EXPR over the checkpoint's config.json (as c)
  "$PY" -I -c "import glob,json,os,sys; m=sys.argv[1]; d=m if os.path.isdir(m) else glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--'+m.replace('/','--')+'/snapshots/*/'))[0]; c=json.load(open(d+'/config.json')); print($1)" "$MODEL"
}
segments() {  # segments PREFIX SCRIPT P NLAYERS SEG: one build per SEG layers
  local pfx=$1 script=$2 p=$3 n=$4 seg=$5 l0 l1
  for ((l0 = 0; l0 < n; l0 += seg)); do
    l1=$((l0 + seg < n ? l0 + seg : n))
    build "$pfx-$p-$l0-$l1" "$script" "$p" "$l0" "$l1"
  done
}
TYPE=$(cfg "c['model_type']")
NL=0
# Only graphs known to compile safely on the phone: the compiler can exhaust
# the phone's memory and panic its kernel, so the graph kinds and shapes are
# fixed here, not tunable.
case $TYPE in
  qwen3_asr)
    if [ $ENC = 1 ]; then build enc export_encoder.py; fi
    NL=$(cfg "c['text_config']['num_hidden_layers']")
    if [ -n "$PREFILL" ]; then
      HD=$(cfg "c['text_config']['hidden_size']")
      [ "$HD" = 1024 ] && [ "$PREFILL" = 128 ] ||
        { echo "TPU prefill: only bucket 128 of Qwen3-ASR-0.6B is a known-safe graph" >&2; exit 2; }
    fi ;;
  *)
    echo "$MODEL ($TYPE): no TPU graphs for this model (supported: Qwen3-ASR)" >&2
    exit 2 ;;
esac
# Qwen3-ASR-0.6B's trunk in two 14-layer segments (a whole trunk does not compile).
for sb in ${PREFILL//,/ }; do
  segments prefill export_prefill.py "$sb" "$NL" 14
done
