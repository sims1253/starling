#!/bin/bash
set -euo pipefail
cd /home/m0hawk/Documents/starling
export PK_MODEL="${PK_MODEL:-models/parakeet-tdt-0.6b-v3-q4-fullimx.gguf}"
PK_MODEL_VK="${PK_MODEL_VK:-models/parakeet-tdt-0.6b-v3-q4-fullimx.gguf}"
PK_MODEL_CPU="${PK_MODEL_CPU:-$PK_MODEL}"
PK_VK_LIB="${PK_VK_LIB:-build-ar-vk/libstarling_ggml.so}"
PK_CPU_LIB="${PK_CPU_LIB:-build-ar-cpu/libstarling_ggml.so}"

# Pre-check: syntax-level (lib exists) — fast fail
[[ -f "$PK_VK_LIB" ]] || { echo "ERROR missing $PK_VK_LIB"; exit 1; }

echo "== Vulkan (primary) =="
PK_MODEL="$PK_MODEL_VK" STARLING_GGML_LIB="$PK_VK_LIB" STARLING_GGML_DEVICE=Vulkan0 \
  uv run python .auto/bench_speed.py 2>/dev/null | grep -E "^METRIC"

if [[ -f "$PK_CPU_LIB" ]]; then
  echo "== CPU (secondary) =="
  # 6 = physical cores on this Zen3 (SMT hurts the ggml kernels ~25%).
  PK_MODEL="$PK_MODEL_CPU" STARLING_GGML_THREADS="${PK_CPU_THREADS:-6}" \
  STARLING_GGML_LIB="$PK_CPU_LIB" STARLING_GGML_DEVICE=cpu \
    PK_REPS="${PK_REPS:-3}" \
    uv run python .auto/bench_speed.py 2>/dev/null \
    | grep -E "^METRIC (rtf_med|med_ms|peak_rss)" | sed 's/rtf_med/rtf_cpu_med/; s/^METRIC med_ms/METRIC cpu_med_ms/; s/^METRIC peak_rss_mb/METRIC cpu_peak_rss_mb/'
fi
