#!/bin/bash
# Pull the phone's vendor OpenCL libraries (read-only) so clshim/gen_tramp.py can
# export exactly the symbols the real library exports.
# Run under: flock /tmp/starling-phone.lock benchmarks/litert_gpu/pull_vendor_cl.sh OUTDIR
set -eu
export ANDROID_SERIAL=${ANDROID_SERIAL:-192.168.178.59:45929}
OUT=${1:-/tmp/litert-gpu/vendor}
mkdir -p "$OUT"
for l in libOpenCL.so libOpenCL-pixel.so; do
  adb pull -q /vendor/lib64/$l "$OUT/$l"
done
adb shell 'getprop ro.build.fingerprint; ls /vendor/lib64/ | grep -iE "pvr|srv|IMG|glslc|usc|vulkan" ; ls /vendor/lib64/egl /vendor/lib64/hw | grep -iE "pvr|powervr|vulkan|egl"' > "$OUT/vendor_ls.txt" 2>&1 || true
ls -la "$OUT"
