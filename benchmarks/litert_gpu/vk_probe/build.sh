#!/bin/bash
# Builds the Vulkan probes for arm64 Android (glslc SPIR-V embedded as C arrays,
# NDK libvulkan):
#   vk_alu_probe  alu.comp per MODE: f32/f16 multiply-add vs fma(), broadcast loop
#   vk_fc_probe   fc.comp (ML Drift's FC kernel in Vulkan) per F16/SL/ATEX variant
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
NDK=${NDK:-$HOME/android-sdk/ndk/28.2.13676358}
TC=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin
OUT=${OUT:-/tmp/litert-gpu/phone}
B=${BUILD:-/tmp/litert-gpu/build/vk}
mkdir -p "$B" "$OUT"
: > "$B/alu_spv.h"
for m in 0 1 2 3 4 5 6; do
  glslc --target-env=vulkan1.1 -O -DMODE=$m -mfmt=c "$HERE/alu.comp" -o "$B/mode$m.inc"
  { echo "static const uint32_t alu_mode$m[] ="; cat "$B/mode$m.inc"; echo ";"; } >> "$B/alu_spv.h"
done
"$TC/aarch64-linux-android31-clang" -O2 -I"$B" "$HERE/vk_alu_probe.c" -lvulkan -o "$OUT/vk_alu_probe"
: > "$B/fc_spv.h"
for v in 0_1_1 1_1_1 1_1_0 1_2_1 1_4_1 0_4_1; do
  IFS=_ read -r f16 sl atex <<< "$v"
  glslc --target-env=vulkan1.1 -O -DF16="$f16" -DSL="$sl" -DATEX="$atex" -mfmt=c "$HERE/fc.comp" -o "$B/fc_$v.inc"
  { echo "static const uint32_t fc_$v[] ="; cat "$B/fc_$v.inc"; echo ";"; } >> "$B/fc_spv.h"
done
"$TC/aarch64-linux-android31-clang" -O2 -I"$B" "$HERE/vk_fc_probe.c" -lvulkan -lm -o "$OUT/vk_fc_probe"
ls -la "$OUT/vk_alu_probe" "$OUT/vk_fc_probe"
