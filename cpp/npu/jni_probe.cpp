// Copyright 2026 The Starling Authors. Apache-2.0.
//
// JNI entry points for the NPU probe APK (cpp/npu/android_probe): proves the
// compile / load / run path from an untrusted app process.
#include "npu_probe.hpp"
#include "tpu_runtime.hpp"

#include <android/log.h>
#include <jni.h>

#include <string>
#include <vector>

using namespace starling::npu;

namespace {
std::string str(JNIEnv* env, jstring s) {
    if (!s) return "";
    const char* c = env->GetStringUTFChars(s, nullptr);
    std::string r = c ? c : "";
    if (c) env->ReleaseStringUTFChars(s, c);
    return r;
}
jstring report(JNIEnv* env, const std::string& r, bool ok) {
    const std::string t = std::string(ok ? "OK\n" : "FAIL\n") + r;
    __android_log_print(ok ? ANDROID_LOG_INFO : ANDROID_LOG_ERROR, "StarlingNpu", "%s", t.c_str());
    return env->NewStringUTF(t.c_str());
}
}  // namespace

extern "C" JNIEXPORT jstring JNICALL Java_dev_starling_npuprobe_NpuProbe_compile(JNIEnv* env, jclass, jstring tflite,
                                                                                 jstring artifact, jstring fn,
                                                                                 jlongArray in, jlongArray out) {
    auto vec = [&](jlongArray a) {
        std::vector<uint64_t> v(env->GetArrayLength(a));
        jlong* p = env->GetLongArrayElements(a, nullptr);
        for (size_t i = 0; i < v.size(); ++i) v[i] = (uint64_t)p[i];
        env->ReleaseLongArrayElements(a, p, JNI_ABORT);
        return v;
    };
    bool ok = false;
    std::string r = probe_compile(str(env, tflite), str(env, artifact), str(env, fn), vec(in), vec(out), 0, ok);
    return report(env, r, ok);
}

extern "C" JNIEXPORT jstring JNICALL Java_dev_starling_npuprobe_NpuProbe_run(JNIEnv* env, jclass, jstring artifact,
                                                                             jstring in0, jstring in1,
                                                                             jstring out_prefix, jint runs) {
    bool ok = false;
    std::vector<std::string> ins;
    if (in0) ins.push_back(str(env, in0));
    if (in1) ins.push_back(str(env, in1));
    std::string r = probe_run(str(env, artifact), LoadMode::Mmap, ins, str(env, out_prefix), runs, ok);
    return report(env, r, ok);
}
