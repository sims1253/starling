#!/bin/bash
# Cross-builds litert_gpu_bench (LiteRT 2.3.0 C API) and the clshim OpenCL
# interposer for arm64 Android, and stages them with LiteRT's libraries in
# $STAGE (default /tmp/litert-gpu/phone) for phone_session.sh to push.
#
# Inputs (all Apache-2.0; no Google Tensor SDK component):
#   LITERT_SDK   litert_cc_sdk from the LiteRT v2.3.0 GitHub release (headers;
#                build_config.h generated, as in benchmarks/npu/build_npu_bench.sh)
#   LITERT_AAR   unpacked Maven com.google.ai.edge.litert:litert:2.3.0 (libLiteRt.so)
#   GPU_AAR      unpacked Maven com.google.ai.edge.litert:litert-gpu:2.3.0
#                (libLiteRtClGlAccelerator.so, ML Drift OpenCL/OpenGL)
#   VENDOR_CL    the phone's /vendor/lib64/libOpenCL.so (pull_vendor_cl.sh), for the
#                list of entry points the interposer forwards
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
NDK=${NDK:-$HOME/android-sdk/ndk/28.2.13676358}
TC=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin
LITERT_SDK=${LITERT_SDK:-/tmp/npu-spike/dl/litert_cc_sdk-2.3.0/x/litert_cc_sdk}
LITERT_GEN=${LITERT_GEN:-/tmp/npu-spike/harness/gen}
LITERT_AAR=${LITERT_AAR:-/tmp/npu-spike/dl/maven-litert-2.3.0/x}
GPU_AAR=${GPU_AAR:-/tmp/litert-gpu/dl/litert-gpu-2.3.0/x}
VENDOR_CL=${VENDOR_CL:-/tmp/litert-gpu/vendor/libOpenCL.so}
STAGE=${STAGE:-/tmp/litert-gpu/phone}
B=${BUILD:-/tmp/litert-gpu/build}
mkdir -p "$STAGE/lib" "$STAGE/shim" "$B"

cp "$LITERT_AAR/jni/arm64-v8a/libLiteRt.so" "$STAGE/lib/"
cp "$GPU_AAR/jni/arm64-v8a/libLiteRtClGlAccelerator.so" "$STAGE/lib/"

"$TC/aarch64-linux-android31-clang++" -O2 -std=c++17 -I"$LITERT_SDK" -I"$LITERT_GEN" \
  "$HERE/litert_gpu_bench.cc" -L"$LITERT_AAR/jni/arm64-v8a" -lLiteRt -ldl \
  -Wl,-rpath,'$ORIGIN/lib' -static-libstdc++ -o "$STAGE/litert_gpu_bench"

python3 -I "$HERE/clshim/gen_tramp.py" "$VENDOR_CL" "$HERE/clshim/clshim.c" "$B/tramp.S"
"$TC/aarch64-linux-android31-clang" -O2 -fPIC -shared -Wl,-soname,libOpenCL.so \
  "$HERE/clshim/clshim.c" "$B/tramp.S" -ldl -o "$STAGE/shim/libOpenCL.so"
"$TC/aarch64-linux-android31-clang" -O2 -fPIC -shared -Wl,-soname,libOpenCL-pixel.so \
  "$HERE/clshim/pixel_fwd.c" -ldl -o "$STAGE/shim/libOpenCL-pixel.so"
"$TC/aarch64-linux-android31-clang" -O2 "$HERE/cl_alu_probe.c" -ldl -o "$STAGE/cl_alu_probe"
ls -la "$STAGE" "$STAGE/lib" "$STAGE/shim"
