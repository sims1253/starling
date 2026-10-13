#!/bin/bash
# gemmx probes on the host GPU: gemm_sg.comp (g) vs gemm.comp (f / device
# default -) at encoder/prefill shapes, with the CPU-reference error.
#   benchmarks/litert_gpu/host_micro.sh [variants="g f -"] [reps=8]
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
M=""
for v in ${1:-g f -}; do
  for s in 104,896,896 104,3584,896 104,896,3584 112,6144,1024 320,1024,3072 37,64,128 600,1024,1024; do
    for k in w4 w8 f16; do M="$M;gemmx,$s,$k,${2:-8},$v"; done
  done
done
STARLING_ENGINE=fast STARLING_FAST_MICRO="${M#;}" timeout 900 "$ROOT/build-host/starling-bench" \
  --model qwen3 --gguf "${GGUF:-/home/m0hawk/Documents/starling/models/qwen3-asr-0.6b-q4e8.gguf}" \
  --runs 1 --quiet "$ROOT/tests/fixtures/2086-149220-0033.wav" 2>&1 | grep -E "fast-micro|rror"
