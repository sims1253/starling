#!/bin/bash
set -euo pipefail
cd /home/m0hawk/Documents/starling
export PK_MODEL="${PK_MODEL:-models/parakeet-tdt-0.6b-v3-q8_0.gguf}"
PK_VK_LIB="${PK_VK_LIB:-build-ar-vk/libstarling_ggml.so}"
[[ -f "$PK_VK_LIB" ]] || { echo "ERROR missing $PK_VK_LIB"; exit 1; }
STARLING_GGML_LIB="$PK_VK_LIB" STARLING_GGML_DEVICE=Vulkan0 \
  uv run python .auto/bench_wer.py 2>/dev/null | grep -E "METRIC|VERDICT"
