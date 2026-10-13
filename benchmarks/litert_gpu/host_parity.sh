#!/bin/bash
# Host (RADV) transcripts with gemm.comp (SG=0) vs gemm_sg.comp forced (SG=1).
#   benchmarks/litert_gpu/host_parity.sh "slug:gguf ..." [wav...]
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
MD=${MODELS_DIR:-/home/m0hawk/Documents/starling/models}
WAVS=("${@:2}")
[ ${#WAVS[@]} -gt 0 ] || WAVS=("$ROOT/tests/fixtures/2086-149220-0033.wav")
for e in $1; do
  slug=${e%%:*}; gguf=${e#*:}
  for w in "${WAVS[@]}"; do
    for sg in 0 1; do
      STARLING_ENGINE=fast STARLING_FAST_SG=$sg STARLING_FAST_TIMING=1 timeout 900 "$ROOT/build-host/starling-bench" \
        --model "$slug" --gguf "$MD/$gguf" --runs 1 "$w" > "/tmp/litert-gpu/hp_${gguf}_$(basename "$w")_$sg.txt" 2>&1
    done
    a=$(grep -E '^  ' "/tmp/litert-gpu/hp_${gguf}_$(basename "$w")_0.txt")
    b=$(grep -E '^  ' "/tmp/litert-gpu/hp_${gguf}_$(basename "$w")_1.txt")
    [ -n "$a" ] && [ "$a" = "$b" ] && echo "SAME $gguf $(basename "$w")" || { echo "DIFF $gguf $(basename "$w")"; echo " 0:$a"; echo " 1:$b"; }
  done
done
