#!/bin/bash
# Builds the Vulkan probes for arm64 Android (glslc SPIR-V embedded as C arrays,
# NDK libvulkan):
#   vk_alu_probe  alu.comp per MODE: f32/f16 multiply-add vs fma(), broadcast loop
#   vk_fc_probe   fc.comp (ML Drift's FC kernel in Vulkan) per F16/SL/ATEX variant
#   vk_fc3_probe  fc3.comp (the same idea on cpp/fast's W4/W8/F16 layouts)
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
# fc3.comp: the cpp/fast prototype (engine layouts), BKIND x NB x RM x ATEX
: > "$B/fc3_spv.h"
vars=""
for bk in 0 1 2; do
  for v in ${FC3_SET:-8_1_1_0 16_1_0_0 16_1_1_0 32_1_1_0 8_1_1_1 16_1_1_1 16_1_0_1 12_1_1_1}; do
    IFS=_ read -r nb rm atex accv <<< "$v"
    id="fc3_${bk}_$v"
    glslc --target-env=vulkan1.1 -O -DBKIND="$bk" -DNB="$nb" -DRM="$rm" -DATEX="$atex" -DACCV="$accv" -mfmt=c "$HERE/fc3.comp" -o "$B/$id.inc"
    { echo "static const uint32_t $id[] ="; cat "$B/$id.inc"; echo ";"; } >> "$B/fc3_spv.h"
    kind=$([ $bk = 0 ] && echo w4 || ([ $bk = 1 ] && echo w8 || echo f16))
    lbl="$kind NB$nb RM$rm $([ $atex = 1 ] && echo tex || echo ssbo)$([ "$accv" = 1 ] && echo ' accv')"
    vars="$vars{\"$lbl\", $bk, $nb, $rm, $atex, $id, sizeof $id}, "
  done
done
echo "#define FC3_VARIANTS $vars" >> "$B/fc3_spv.h"
"$TC/aarch64-linux-android31-clang" -O2 -I"$B" "$HERE/vk_fc3_probe.c" -lvulkan -lm -o "$OUT/vk_fc3_probe"
ls -la "$OUT/vk_alu_probe" "$OUT/vk_fc_probe" "$OUT/vk_fc3_probe"
