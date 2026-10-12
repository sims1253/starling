#!/bin/bash
# Tensor G5 NPU spike (2026-10-12): cross-builds npu_bench (LiteRT C API) for arm64.
# Inputs (all Apache-2.0, no Google Tensor SDK): litert_cc_sdk.zip from the LiteRT v2.3.0
# GitHub release (headers; build_config.h generated with both DISABLE flags 0), and the
# Maven AARs com.google.ai.edge.litert:litert:2.3.0 (libLiteRt.so) and
# litert-npu-runtime-google-tensor:2.3.0 (JIT compiler plugin + dispatch library, which
# dlopen /vendor/lib64/libedgetpu_litert.so on the phone).
set -e
NDK=$HOME/android-sdk/ndk/28.2.13676358
CXX=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang++
INC=/tmp/npu-spike/dl/litert_cc_sdk-2.3.0/x/litert_cc_sdk
LIB=/tmp/npu-spike/dl/maven-litert-2.3.0/x/jni/arm64-v8a
mkdir -p /tmp/npu-spike/phone/lib
cp $LIB/libLiteRt.so /tmp/npu-spike/phone/lib/
cp /tmp/npu-spike/dl/maven-litert-npu-runtime-google-tensor-2.3.0/x/jni/arm64-v8a/*.so /tmp/npu-spike/phone/lib/
$CXX -O2 -std=c++17 -I$INC -I/tmp/npu-spike/harness/gen /tmp/npu-spike/harness/npu_bench.cc -L$LIB -lLiteRt -Wl,-rpath,'$ORIGIN/lib' -static-libstdc++ -o /tmp/npu-spike/phone/npu_bench
ls -la /tmp/npu-spike/phone /tmp/npu-spike/phone/lib
