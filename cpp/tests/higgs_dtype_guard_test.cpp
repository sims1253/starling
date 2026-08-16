// higgs_dtype_guard_test.cpp — read_tensor_to_f32 must accept BF16/F32
// exactly and reject every other dtype loudly (empty vector + err), never a
// silent zero-fill. Synthesizes a tiny GGUF with F32/BF16/F16 tensors so the
// check runs anywhere (no model files, CPU-only, no skip).
//
// Usage: ./higgs_dtype_guard_test
#include "higgs/audio_encoder.hpp"
#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool ok, const char* what, const std::string& err = "") {
    std::printf("[%s] %s%s%s\n", ok ? "PASS" : "FAIL", what,
                (!ok && !err.empty()) ? " -- " : "", ok ? "" : err.c_str());
    if (!ok) failures++;
}

int main() {
    const char* path = "/tmp/higgs_dtype_guard_test.gguf";
    std::remove(path);

    // Synthesize a GGUF holding one F32, one BF16, and one F16 tensor. The
    // BF16 values are exactly representable so the round-trip is bit-exact.
    ggml_context* gctx = ggml_init({1024 * 1024, nullptr, false});
    gguf_context* gf = gguf_init_empty();
    const int64_t ne[1] = {8};
    const float f32_vals[8] = {1.0f, -2.5f, 3.25f, 4.0f, -5.75f, 6.0f, 0.5f, -7.0f};
    const float bf16_vals[8] = {1.0f, 2.0f, -0.5f, 3.0f, 4.0f, -8.0f, 0.25f, 16.0f};
    ggml_tensor* t_f32 = ggml_new_tensor(gctx, GGML_TYPE_F32, 1, ne);
    ggml_set_name(t_f32, "t.f32");
    std::memcpy(t_f32->data, f32_vals, sizeof f32_vals);
    ggml_tensor* t_bf16 = ggml_new_tensor(gctx, GGML_TYPE_BF16, 1, ne);
    ggml_set_name(t_bf16, "t.bf16");
    for (int i = 0; i < 8; ++i)
        ((ggml_bf16_t*) t_bf16->data)[i] = ggml_fp32_to_bf16(bf16_vals[i]);
    ggml_tensor* t_f16 = ggml_new_tensor(gctx, GGML_TYPE_F16, 1, ne);
    ggml_set_name(t_f16, "t.f16");
    for (int i = 0; i < 8; ++i)
        ((ggml_fp16_t*) t_f16->data)[i] = ggml_fp32_to_fp16(f32_vals[i]);
    gguf_add_tensor(gf, t_f32);
    gguf_add_tensor(gf, t_bf16);
    gguf_add_tensor(gf, t_f16);
    const bool wrote = gguf_write_to_file(gf, path, /*only_meta=*/false);
    gguf_free(gf);
    ggml_free(gctx);
    if (!wrote) {
        std::printf("[FAIL] synthesized GGUF written (%s)\n", path);
        return 1;
    }

    starling::ggml::ModelLoader ml;
    if (!ml.load(path)) {
        std::printf("[FAIL] ModelLoader parses synthesized GGUF -- %s\n",
                    ml.last_error().c_str());
        return 1;
    }

    using starling::ggml::higgs::read_tensor_to_f32;
    std::string err;
    {
        err.clear();
        std::vector<float> v = read_tensor_to_f32(ml, "t.f32", err);
        bool ok = err.empty() && v.size() == 8 && v[1] == -2.5f && v[7] == -7.0f;
        check(ok, "F32 tensor reads back exactly", err);
    }
    {
        err.clear();
        std::vector<float> v = read_tensor_to_f32(ml, "t.bf16", err);
        bool ok = err.empty() && v.size() == 8 && v[2] == -0.5f && v[7] == 16.0f;
        check(ok, "BF16 tensor converts exactly", err);
    }
    {
        err.clear();
        std::vector<float> v = read_tensor_to_f32(ml, "t.f16", err);
        bool ok = v.empty() && err.find("f16") != std::string::npos &&
                  err.find("t.f16") != std::string::npos;
        check(ok, "F16 tensor rejected loudly (no zero-fill)", err);
    }
    {
        err.clear();
        std::vector<float> v = read_tensor_to_f32(ml, "t.absent", err);
        bool ok = v.empty() && err.find("missing") != std::string::npos;
        check(ok, "absent tensor rejected loudly", err);
    }

    std::remove(path);
    std::printf("%s\n", failures ? "DTYPE GUARD FAILED" : "DTYPE GUARD OK");
    return failures ? 1 : 0;
}
