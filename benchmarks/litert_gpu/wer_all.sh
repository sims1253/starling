#!/bin/bash
# FLEURS en_us (100 clips) WER, gemm.comp vs gemm_sg.comp, all seven models,
# on the host GPU with the PowerVR numerics (f32 products in both arms).
# Run under: flock /tmp/starling-host-heavy.lock systemd-run --user --scope -q \
#   -p MemoryMax=8G -p MemorySwapMax=0 benchmarks/litert_gpu/wer_all.sh OUTDIR
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:-/tmp/litert-gpu/wer}
MD=${MODELS_DIR:-/home/m0hawk/Documents/starling/models}
CLIPS=${CLIPS:-/tmp/litert-gpu/fleurs_en}
PY=${PY:-/tmp/litert-gpu/venv/bin/python}
mkdir -p "$OUT"
for e in ${MODELS:-qwen3:qwen3-asr-0.6b-q4e8.gguf parakeet:parakeet-tdt-0.6b-v3-q4_k_m-shrink16.gguf moss:moss-transcribe-preview-2b-q4e8-fullimx.gguf ark06:ark-asr-0.6b-q4k.gguf qwen3:qwen3-asr-1.7b-q4e8-imx.gguf granite:granite-2b-dynq4.imx.gguf voxtral:voxtral-mini-4b-realtime-q4k.gguf}; do
  slug=${e%%:*}; gguf=${e#*:}
  "$PY" -I "$ROOT/benchmarks/litert_gpu/wer_bench.py" --bench "$ROOT/build-host/starling-bench" \
    --model "$slug" --gguf "$MD/$gguf" --clips "$CLIPS" --out "$OUT/${gguf%.gguf}.json" \
    --arm "sg0:STARLING_FAST_SG=0,STARLING_FAST_F16=0" --arm "sg1:STARLING_FAST_SG=1,STARLING_FAST_F16=0"
done
