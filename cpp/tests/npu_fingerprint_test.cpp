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
//    and the encoder metadata the graph is built from — another LayerNorm
//    epsilon changes it — and ignores the rest.
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
constexpr uint64_t kDigestVector = 0x9b4f28654f4e15b6ull;
constexpr uint64_t kDigestTweakedVector = 0x66351613e1266017ull;
constexpr uint64_t kDigestEpsVector = 0xf237d4837eb57312ull;

// A tiny Qwen3-ASR-like GGUF: five F32 encoder vectors, k = 0..4 holding
// 4 + k values (16k + i) / 4 - 1, an encoder weight matrix of each stored
// kind (Q8_0 blocks, BF16), and two tensors outside the encoder; every other
// byte b of a tensor is (b * 37 + 11 + strlen(name)) & 255. Encoder metadata
// of each kind (u32, f32, string; a bool, which no side reads) and one key
// outside the encoder. `tweak` changes one thing: 1 a byte of the Q8_0
// encoder matrix, 2 a byte of the non-encoder tensor, 3 the LayerNorm
// epsilon, 4 the non-encoder key; 5 adds an array under an encoder key.
bool write_model(const char* path, int tweak) {
    ggml_context* ctx = ggml_init({1 << 20, nullptr, false});
    gguf_context* g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "qwen3");
    gguf_set_val_u32(g, "qwen3.enc.heads", 14);
    gguf_set_val_f32(g, "qwen3.enc.layer_norm_eps", tweak == 3 ? 1e-6f : 1e-5f);
    gguf_set_val_u32(g, "qwen3.proj.output_dim", 1024);
    gguf_set_val_str(g, "qwen3.enc.activation", "gelu");
    gguf_set_val_bool(g, "qwen3.enc.flag", true);
    gguf_set_val_u32(g, "qwen3.llm.hidden_size", tweak == 4 ? 2048 : 1024);
    if (tweak == 5) {
        const int32_t w[2] = {1, 2};
        gguf_set_arr_data(g, "qwen3.enc.windows", GGUF_TYPE_INT32, w, 2);
    }
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

// The encoder digest of write_parakeet_model(path, 0): pinned like
// kDigestVector (tests/test_npu_fingerprint.py pins the same value for
// scripts/npu/export_parakeet.py).
constexpr uint64_t kParakeetDigestVector = 0x29faef0bc7c1f073ull;

// A tiny Parakeet-TDT-like GGUF: encoder tensors (a Q8_0 matrix, F32
// vectors, the joint encoder projection's bias) and tensors outside the
// encoder (the prediction network, the joint output); encoder metadata (u32,
// string) and parakeet.preprocessor.n_mels, plus keys outside the hash.
// `tweak`: 1 a byte of the encoder matrix, 2 a byte of a decoder tensor,
// 3 n_mels, 4 the hop length (outside the hash).
bool write_parakeet_model(const char* path, int tweak) {
    ggml_context* ctx = ggml_init({1 << 20, nullptr, false});
    gguf_context* g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "parakeet_tdt");
    gguf_set_val_u32(g, "parakeet.encoder.d_model", 1024);
    gguf_set_val_u32(g, "parakeet.encoder.n_layers", 24);
    gguf_set_val_str(g, "parakeet.encoder.conv_norm_type", "batch_norm");
    gguf_set_val_u32(g, "parakeet.preprocessor.n_mels", tweak == 3 ? 80 : 128);
    gguf_set_val_u32(g, "parakeet.preprocessor.hop_length", tweak == 4 ? 320 : 160);
    gguf_set_val_u32(g, "parakeet.decoder.pred_hidden", 640);
    auto add = [&](const char* name, ggml_type type, std::vector<int64_t> ne) {
        ggml_tensor* t = ggml_new_tensor(ctx, type, (int)ne.size(), ne.data());
        ggml_set_name(t, name);
        uint8_t* p = static_cast<uint8_t*>(t->data);
        for (size_t i = 0; i < ggml_nbytes(t); ++i) p[i] = (uint8_t)(i * 37 + 11 + std::strlen(name));
        return t;
    };
    ggml_tensor* q = add("encoder.layers.0.self_attn.linear_q.weight", GGML_TYPE_Q8_0, {32, 2});
    add("encoder.pre_encode.out.bias", GGML_TYPE_F32, {4});
    add("encoder.layers.0.norm_out.weight", GGML_TYPE_F32, {4});
    add("joint.enc.bias", GGML_TYPE_F32, {3});
    ggml_tensor* dec = add("decoder.prediction.embed.weight", GGML_TYPE_F32, {4, 2});
    add("joint.joint_net.2.bias", GGML_TYPE_F32, {5});
    if (tweak == 1) static_cast<uint8_t*>(q->data)[40] ^= 1;
    if (tweak == 2) static_cast<uint8_t*>(dec->data)[3] ^= 1;
    for (ggml_tensor* t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) gguf_add_tensor(g, t);
    const bool ok = gguf_write_to_file(g, path, false);
    gguf_free(g);
    ggml_free(ctx);
    return ok;
}

// The engine's path: ModelLoader's mapped tensors and its metadata (the
// Parakeet engine's digest for a parakeet_tdt GGUF, else Qwen3-ASR's).
bool loaded_model_hash(const char* path, uint64_t& hash, double& digest_ms, std::string& err) {
    starling::ggml::ModelLoader ml;
    if (!ml.load(path)) {
        err = ml.last_error();
        return false;
    }
    const TensorLookup tensor = [&ml](const std::string& n) -> const ggml_tensor* { return ml.tensor(n.c_str()); };
    const auto arch = ml.kv().find("general.architecture");
    const bool parakeet = arch != ml.kv().end() && arch->second.s == "parakeet_tdt";
    std::vector<EncoderMeta> meta;
    if (!(parakeet ? parakeet_encoder_meta(ml.kv(), meta, err) : qwen3_encoder_meta(ml.kv(), meta, err))) return false;
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = parakeet ? parakeet_model_hash(ml.tensor_names(), tensor, meta, hash, err)
                             : qwen3_model_hash(ml.tensor_names(), tensor, meta, hash, err);
    digest_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return ok;
}

void test_parakeet() {
    char path[5][256];
    for (int t = 0; t < 5; ++t) {
        std::snprintf(path[t], sizeof path[t], "/tmp/npu_fingerprint_test_pk_%d_%d.gguf", (int)getpid(), t);
        if (!write_parakeet_model(path[t], t)) {
            check(false, "synthesized Parakeet GGUFs written");
            return;
        }
    }
    uint64_t f[5] = {}, l[5] = {};
    std::string err;
    double ms = 0;
    bool ok = true;
    for (int t = 0; t < 5; ++t)
        ok = ok && gguf_model_hash(path[t], f[t], err) && loaded_model_hash(path[t], l[t], ms, err);
    check(ok, "Parakeet: hashes of the synthesized GGUFs " + err);
    if (ok) {
        check_hex(f[0], kParakeetDigestVector, "Parakeet encoder digest (pinned, = export_parakeet.py's)");
        for (int t = 0; t < 5; ++t) check_hex(l[t], f[t], "Parakeet: digest from the loaded model = from the file");
        check(f[1] != f[0], "Parakeet: one changed byte of an encoder matrix: another digest");
        check_hex(f[2], f[0], "Parakeet: a changed decoder tensor leaves the digest alone");
        check(f[3] != f[0], "Parakeet: another n_mels: another digest");
        check_hex(f[4], f[0], "Parakeet: a key outside the encoder leaves the digest alone");
    }
    // The artifact contract the engine and export_parakeet.py share.
    const std::vector<std::string> pieces = parakeet_enc_pieces(24);
    check(pieces.size() == 5 && pieces[0] == "sub" && pieces[1] == "l0-6" && pieces[4] == "l18-24",
          "Parakeet: pieces sub, l0-6 .. l18-24");
    std::vector<uint64_t> in, out;
    parakeet_enc_io("sub", 128, 1024, 24, 640, in, out);
    check(in == std::vector<uint64_t>{110080, 428, 212, 104} && out == std::vector<uint64_t>{106496},
          "Parakeet: sub I/O sizes");
    parakeet_enc_io("l6-12", 128, 1024, 24, 640, in, out);
    check(in == std::vector<uint64_t>{1703936, 1664, 1664} && out == std::vector<uint64_t>{1703936},
          "Parakeet: segment I/O sizes");
    parakeet_enc_io("l18-24", 128, 1024, 24, 640, in, out);
    check(out == std::vector<uint64_t>{1064960}, "Parakeet: the last segment outputs the joint projection");
    check(parakeet_enc_key("l0-6", 128, 1024, 24, 8, 4096, 9, 256, 640) ==
              "parakeet-enc/1 piece=l0-6 mels=128 chunk=26 rows=416 d=1024 L=24 H=8 ff=4096 ck=9 sc=256 jh=640",
          "Parakeet: contract key (= export_parakeet.py piece_key)");
    check(parakeet_enc_artifact_name(0x0f2812c9375238b0ull, "sub") == "parakeet-enc-0f2812c9375238b0-sub.stnpu",
          "Parakeet: artifact name");
    for (auto& p : path) std::remove(p);
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
    char path[6][256];
    for (int t = 0; t < 6; ++t) {
        std::snprintf(path[t], sizeof path[t], "/tmp/npu_fingerprint_test_%d_%d.gguf", (int)getpid(), t);
        if (!write_model(path[t], t)) {
            check(false, "synthesized GGUFs written");
            return;
        }
    }
    uint64_t f[5] = {}, l[5] = {};
    std::string err;
    double ms = 0;
    bool ok = true, ok_loaded = true;
    for (int t = 0; t < 5; ++t) {
        ok = ok && gguf_model_hash(path[t], f[t], err);
        ok_loaded = ok_loaded && loaded_model_hash(path[t], l[t], ms, err);
    }
    check(ok, "gguf_model_hash reads the synthesized GGUFs " + err);
    check(ok_loaded, "the loaded models' hashes " + err);
    if (ok && ok_loaded) {
        check_hex(f[0], kDigestVector, "encoder digest (pinned, = export_encoder.py's)");
        for (int t = 0; t < 5; ++t) check_hex(l[t], f[t], "digest from the loaded model = from the file");
        check_hex(f[1], kDigestTweakedVector, "one changed byte of an encoder weight matrix: another digest (pinned)");
        check(f[1] != f[0], "... which differs from the original's");
        check_hex(f[2], f[0], "a changed non-encoder tensor leaves the digest alone");
        check_hex(f[3], kDigestEpsVector, "another LayerNorm epsilon: another digest (pinned)");
        check(f[3] != f[0], "... which differs from the original's");
        check_hex(f[4], f[0], "a changed non-encoder metadata key leaves the digest alone");
    }
    uint64_t none = 0;
    std::string arr_err;
    bool refused = !gguf_model_hash(path[5], none, arr_err) && arr_err.find("array") != std::string::npos;
    check(refused, "an array under an encoder key is an error (file): " + arr_err);
    double unused = 0;
    arr_err.clear();
    refused = !loaded_model_hash(path[5], none, unused, arr_err) && arr_err.find("array") != std::string::npos;
    check(refused, "an array under an encoder key is an error (loaded): " + arr_err);
    check(!gguf_model_hash("/nonexistent/model.gguf", none, err), "a missing file is an error");
    for (auto& p : path) std::remove(p);
}

}  // namespace

int main(int argc, char** argv) {
    test_vectors();
    test_synthesized();
    test_parakeet();
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
