#!/bin/bash
# Cross-builds the Pixel TPU client for arm64 Android: npu_tool (adb test
# binary) and libstarling_npu_probe.so (JNI probe for the test APK).
# No third-party libraries: the TPU runtime is the phone's own
# libedgetpu_litert.so, opened at run time.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
NDK=${ANDROID_NDK:-$(ls -d "${ANDROID_HOME:-$HOME/android-sdk}"/ndk/* | sort -V | tail -1)}
CXX=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang++
OUT=${OUT:-$HERE/../../build-npu}
mkdir -p "$OUT"
SRC="$HERE/tpu_runtime.cpp $HERE/npu_probe.cpp"
FLAGS="-O2 -std=c++17 -Wall -Wextra -Wno-unused-parameter -fPIC -static-libstdc++"
$CXX $FLAGS $SRC "$HERE/gguf_fingerprint.cpp" "$HERE/npu_tool.cpp" -lnativewindow -ldl -llog -o "$OUT/npu_tool"
$CXX $FLAGS -shared $SRC "$HERE/jni_probe.cpp" -lnativewindow -ldl -llog -o "$OUT/libstarling_npu_probe.so"
ls -la "$OUT/npu_tool" "$OUT/libstarling_npu_probe.so"
