#!/bin/bash
# Autotune sweep for the AMD_GCN m_warptile_mmq on this iGPU.
set -uo pipefail
cd /home/m0hawk/Documents/starling
LINE=4394
BASE="{ 256, 64, 64, 32, 16, 16, 2, 2, 2, 1, 16 }"
SHAPES=("q8_0 1024 279 1024" "q8_0 4096 279 1024" "q8_0 1024 279 4096" "f16 2048 279 1024" "q8_0 2560 1 640" "q8_0 8198 1 640")
export LD_LIBRARY_PATH=build-ar-vk/third_party/ggml/src:build-ar-vk/third_party/ggml/src/ggml-vulkan

configs=(
  "base:{ 256, 64, 64, 32, 16, 16, 2, 2, 2, 1, 16 }"
  "warp64:{ 256, 64, 64, 32, 32, 32, 2, 2, 2, 1, 64 }"
  "blk512:{ 512, 128, 64, 32, 32, 32, 2, 2, 2, 1, 64 }"
  "wmiter1:{ 256, 64, 64, 32, 16, 16, 1, 2, 2, 1, 16 }"
  "tm4:{ 256, 64, 64, 32, 16, 16, 2, 4, 1, 1, 16 }"
  "bn128:{ 256, 64, 128, 32, 32, 16, 2, 2, 2, 1, 16 }"
  "blk128:{ 128, 64, 64, 32, 64, 32, 2, 2, 2, 1, 64 }"
)

for cfg in "${configs[@]}"; do
  name="${cfg%%:*}"
  wt="${cfg#*:}"
  sed -i "${LINE}s/.*/            m_warptile_mmq = m_warptile_mmq_int = ${wt};/" third_party/ggml/src/ggml-vulkan/ggml-vulkan.cpp
  if ! cmake --build build-ar-vk -j12 --target ggml-vulkan 2>&1 | grep -q "error"; then
    echo "== $name $wt =="
    for s in "${SHAPES[@]}"; do
      ./.auto/gemm_bench $s 100 2>/dev/null | grep RESULT | sed "s/RESULT /$name /"
    done
  else
    echo "== $name BUILD FAILED =="
  fi
done
# restore base for safety; caller re-patches the winner
sed -i "${LINE}s/.*/            m_warptile_mmq = m_warptile_mmq_int = ${BASE};/" third_party/ggml/src/ggml-vulkan/ggml-vulkan.cpp
