// npu_fingerprint_test.cpp — the Pixel TPU encoder's model hash
// (cpp/npu/encoder_contract.hpp, model_fingerprint.hpp, gguf_fingerprint.cpp).
//
//  * FNV-1a 64 and XXH64 against reference vectors (FNV's standard offset
//    basis; XXH64 values from xxhsum -H1);
//  * the encoder digest (the artifact's model hash) of a synthesized GGUF is
//    pinned — scripts/npu/export_encoder.py computes the same value from the
//    same file (tests/test_npu_fingerprint.py pins the same constants) — is
//    the same read from the file (starling-npu-tool) and from the loaded
//    model (the engine), covers every encoder/projector tensor whatever its
//    type — a change confined to one byte of a weight matrix changes it —
//    and ignores the rest.
//
// Usage: npu_fingerprint_test [MODEL.gguf ...]
//   With Qwen3-ASR GGUFs: the model hash from the file and from ModelLoader
//   (must agree), and how long the engine's digest takes.
// Exit 0 = pass, 1 = failure.
#include "ggml.h"
#include "gguf.h"
#include "npu/gguf_fingerprint.hpp"
#include "npu/model_fingerprint.hpp"
#include "runtime/model_loader.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#ifdef _WIN32
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

using namespace starling::npu;

namespace {

int g_fail = 0;

void check(bool ok, const std::string& what) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

void check_hex(uint64_t got, uint64_t want, const std::string& what) {
    char b[80];
    std::snprintf(b, sizeof b, " (0x%016llx, want 0x%016llx)", (unsigned long long)got, (unsigned long long)want);
    check(got == want, what + b);
}

// The encoder digest of write_model(path, 0), and of write_model(path, 1):
// pinned so the digest's layout cannot drift silently (artifacts are named
// by it); tests/test_npu_fingerprint.py writes the same GGUFs and pins the
// same values for export_encoder.py.
constexpr uint64_t kDigestVector = 0xae909cf3a449d7f7ull;
constexpr uint64_t kDigestTweakedVector = 0x7de98079b43d9342ull;

// A tiny Qwen3-ASR-like GGUF: five F32 encoder vectors, k = 0..4 holding
// 4 + k values (16k + i) / 4 - 1, an encoder weight matrix of each stored
// kind (Q8_0 blocks, BF16), and two tensors outside the encoder; every other
// byte b of a tensor is (b * 37 + 11 + strlen(name)) & 255. `tweak` flips one
// byte: 1 in the Q8_0 encoder matrix, 2 in the non-encoder tensor.
bool write_model(const char* path, int tweak) {
    ggml_context* ctx = ggml_init({1 << 20, nullptr, false});
    gguf_context* g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "qwen3");
    auto add = [&](const char* name, ggml_type type, std::vector<int64_t> ne) {
        ggml_tensor* t = ggml_new_tensor(ctx, type, (int)ne.size(), ne.data());
        ggml_set_name(t, name);
        uint8_t* p = static_cast<uint8_t*>(t->data);
        for (size_t i = 0; i < ggml_nbytes(t); ++i) p[i] = (uint8_t)(i * 37 + 11 + std::strlen(name));
        return t;
    };
    static const char* const kVectors[] = {"enc.conv1.bias", "enc.ln_post.weight", "enc.ln_post.bias",
                                           "proj.linear_2.bias", "enc.blk.0.attn_norm.weight"};
    for (int k = 0; k < 5; ++k) {
        ggml_tensor* t = add(kVectors[k], GGML_TYPE_F32, {4 + k});
        float* f = static_cast<float*>(t->data);
        for (int i = 0; i < 4 + k; ++i) f[i] = (16 * k + i) * 0.25f - 1.0f;
    }
    ggml_tensor* q = add("enc.blk.0.attn_q.weight", GGML_TYPE_Q8_0, {32, 2});
    add("enc.conv1.weight", GGML_TYPE_BF16, {3, 3, 1, 4});
    ggml_tensor* llm = add("llm.blk.0.attn_q.weight", GGML_TYPE_F32, {4, 2});
    add("audio.mel_filters", GGML_TYPE_F32, {2, 2});
    if (tweak == 1) static_cast<uint8_t*>(q->data)[40] ^= 1;
    if (tweak == 2) static_cast<uint8_t*>(llm->data)[3] ^= 1;
    for (ggml_tensor* t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) gguf_add_tensor(g, t);
    const bool ok = gguf_write_to_file(g, path, false);
    gguf_free(g);
    ggml_free(ctx);
    return ok;
}

// The engine's path: ModelLoader's mapped tensors.
bool loaded_model_hash(const char* path, uint64_t& hash, double& digest_ms, std::string& err) {
    starling::ggml::ModelLoader ml;
    if (!ml.load(path)) {
        err = ml.last_error();
        return false;
    }
    const TensorLookup tensor = [&ml](const std::string& n) -> const ggml_tensor* { return ml.tensor(n.c_str()); };
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = qwen3_model_hash(ml.tensor_names(), tensor, hash, err);
    digest_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return ok;
}

void test_vectors() {
    check_hex(fnv1a_update(kFnvBasis, "", 0), 0xcbf29ce484222325ull, "FNV-1a 64 of \"\" is the standard basis");
    check_hex(fnv1a_update(kFnvBasis, "a", 1), 0xaf63dc4c8601ec8cull, "FNV-1a 64 of \"a\"");
    check_hex(fnv1a_update(kFnvBasis, "foobar", 6), 0x85944171f73967e8ull, "FNV-1a 64 of \"foobar\"");
    check_hex(xxh64("", 0), 0xef46db3751d8e999ull, "XXH64 of \"\"");
    check_hex(xxh64("abc", 3), 0x44bc2cf5ad770999ull, "XXH64 of \"abc\"");
    std::vector<uint8_t> v37(37), v1032;
    for (int i = 0; i < 37; ++i) v37[i] = (uint8_t)((i * 7 + 3) & 255);
    for (int r = 0; r < 4; ++r)
        for (int i = 0; i < 256; ++i) v1032.push_back((uint8_t)i);
    for (char c : std::string("starling")) v1032.push_back((uint8_t)c);
    check_hex(xxh64(v37.data(), v37.size()), 0xe32ef63802f5a3fdull, "XXH64 of 37 bytes (tail paths)");
    check_hex(xxh64(v1032.data(), v1032.size()), 0x89d35540496386b9ull, "XXH64 of 1032 bytes (stripes)");
}

void test_synthesized() {
    char base[256], tw1[256], tw2[256];
    std::snprintf(base, sizeof base, "/tmp/npu_fingerprint_test_%d.gguf", (int)getpid());
    std::snprintf(tw1, sizeof tw1, "/tmp/npu_fingerprint_test_%d_w.gguf", (int)getpid());
    std::snprintf(tw2, sizeof tw2, "/tmp/npu_fingerprint_test_%d_l.gguf", (int)getpid());
    if (!write_model(base, 0) || !write_model(tw1, 1) || !write_model(tw2, 2)) {
        check(false, "synthesized GGUFs written");
        return;
    }
    uint64_t f0 = 0, f1 = 0, f2 = 0, l0 = 0;
    std::string err;
    double ms = 0;
    const bool ok = gguf_model_hash(base, f0, err) && gguf_model_hash(tw1, f1, err) && gguf_model_hash(tw2, f2, err);
    check(ok, "gguf_model_hash reads the synthesized GGUFs " + err);
    const bool ok_loaded = loaded_model_hash(base, l0, ms, err);
    check(ok_loaded, "the loaded model's hash " + err);
    if (ok) {
        check_hex(f0, kDigestVector, "encoder digest (pinned, = export_encoder.py's)");
        check_hex(l0, f0, "digest from the loaded model = from the file");
        check_hex(f1, kDigestTweakedVector, "one changed byte of an encoder weight matrix: another digest (pinned)");
        check(f1 != f0, "... which differs from the original's");
        check_hex(f2, f0, "a changed non-encoder tensor leaves the digest alone");
    }
    uint64_t none = 0;
    check(!gguf_model_hash("/nonexistent/model.gguf", none, err), "a missing file is an error");
    std::remove(base);
    std::remove(tw1);
    std::remove(tw2);
}

}  // namespace

int main(int argc, char** argv) {
    test_vectors();
    test_synthesized();
    for (int i = 1; i < argc; ++i) {
        uint64_t file = 0, loaded = 0;
        std::string err;
        double digest_ms = 0;
        // The engine's digest first: on a cold page cache it pays the reads.
        const bool ok_loaded = loaded_model_hash(argv[i], loaded, digest_ms, err);
        check(ok_loaded, std::string("loaded model hash of ") + argv[i] + " " + err);
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok_file = gguf_model_hash(argv[i], file, err);
        const double file_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        check(ok_file, "file model hash " + err);
        check_hex(loaded, file, "  model hash: loaded = file");
        std::printf("%s: model_hash 0x%016llx; engine digest %.1f ms, tool (file) %.1f ms\n", argv[i],
                    (unsigned long long)file, digest_ms, file_ms);
    }
    if (g_fail) {
        std::printf("npu_fingerprint_test: %d check(s) FAILED\n", g_fail);
        return 1;
    }
    std::printf("npu_fingerprint_test: ok\n");
    return 0;
}
