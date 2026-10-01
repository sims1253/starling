// granite_kv_factors_test.cpp — the issue #59 low-rank K/V factor file
// (granite/kv_factors.hpp, STARLING_GRANITE_KVFACT / STARLING_GRANITE_KVINR).
//
//   1. Loader units over synthesized factor files: the layer-major payload
//      (each layer's K pair, then its V pair — the order every exporter
//      writes) lands in the right layer for v1 and v3 files; ranks above
//      head_dim, truncated payloads, trailing bytes, bad magic and dims
//      mismatches are rejected; f2tk exists iff in-r was requested at load.
//
//   2. End-to-end through starling_ggml_granite_decode on the tiny zero-
//      weight granite GGUF (CPU-only, no downloads): a factor file changes
//      nothing on the zero model, an over-rank file fails the load instead
//      of aborting in graph build, and setting STARLING_GRANITE_KVINR AFTER
//      the load (previously a near-null f2tk dereference) is ignored.
//
// Usage: ./granite_kv_factors_test
#include "granite/kv_factors.hpp"
#include "tiny_granite_fixture.hpp"
#include "trace_test_support.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

extern "C" {
void* starling_ggml_granite_load(const char* gguf_path, const char** err_out);
void starling_ggml_granite_free(void* handle);
char* starling_ggml_granite_decode(void* handle, const float* pcm, int64_t n,
                                   const char** err_out);
}

namespace {

using starling::ggml::granite::KVFactors;

int failures = 0;

void check(bool ok, const std::string& what, const std::string& detail = "") {
    std::printf("[%s] %s%s%s\n", ok ? "PASS" : "FAIL", what.c_str(),
                (!ok && !detail.empty()) ? " -- " : "", ok ? "" : detail.c_str());
    if (!ok) ++failures;
}

std::filesystem::path tmp(const char* name) {
    return std::filesystem::temp_directory_path() / name;
}

// Little-endian writer for synthesized factor files.
struct FactorFile {
    std::vector<unsigned char> bytes;
    void magic(const char* m) { bytes.insert(bytes.end(), m, m + 8); }
    void u32(uint32_t v) {
        for (int i = 0; i < 4; ++i) bytes.push_back((unsigned char) (v >> (8 * i)));
    }
    // `n` floats all equal to `v` (a payload block tagged by its value).
    void block(size_t n, float v) {
        const unsigned char* p = reinterpret_cast<const unsigned char*>(&v);
        for (size_t i = 0; i < n; ++i) bytes.insert(bytes.end(), p, p + sizeof(float));
    }
    bool write(const std::filesystem::path& path) const {
        FILE* f = std::fopen(path.string().c_str(), "wb");
        if (!f) return false;
        const bool ok = std::fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size();
        return std::fclose(f) == 0 && ok;
    }
};

// Tag for (layer, half, f1/f2) so a misassigned block is detectable.
float tag(int layer, int half, int which) {
    return (float) (100 * (layer + 1) + 10 * half + which);
}

bool all_eq(const std::vector<float>& v, size_t n, float x) {
    if (v.size() != n) return false;
    for (float e : v)
        if (e != x) return false;
    return true;
}

bool load_file(const FactorFile& ff, const char* name, KVFactors& kvf, std::string& err,
               int layers, int hidden, int heads, bool in_r = false) {
    const auto path = tmp(name);
    if (!ff.write(path)) { err = "cannot write test file"; return false; }
    const bool ok = kvf.load(path.string().c_str(), layers, hidden, heads, in_r, err);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    return ok;
}

void unit_checks() {
    // 2 layers, hidden 4, 2 heads -> head_dim 2.
    constexpr int L = 2, HID = 4, NH = 2;
    {
        // v1 K+V rank 1: payload K0 V0 K1 V1 (exporter order).
        FactorFile ff;
        ff.magic("STLGKVF1");
        for (uint32_t v : {1u, (uint32_t) L, (uint32_t) HID, (uint32_t) NH, 1u, 1u}) ff.u32(v);
        const size_t n = (size_t) NH * 1 * HID;
        for (int l = 0; l < L; ++l)
            for (int half = 0; half < 2; ++half) {
                ff.block(n, tag(l, half, 1));
                ff.block(n, tag(l, half, 2));
            }
        KVFactors kvf;
        std::string err;
        const bool ok = load_file(ff, "kvf_v1_kv.bin", kvf, err, L, HID, NH);
        check(ok, "v1 K+V file loads", err);
        bool placed = ok;
        for (int l = 0; ok && l < L; ++l)
            placed = placed && all_eq(kvf.f1k[l], n, tag(l, 0, 1)) &&
                     all_eq(kvf.f2k[l], n, tag(l, 0, 2)) &&
                     all_eq(kvf.f1v[l], n, tag(l, 1, 1)) &&
                     all_eq(kvf.f2v[l], n, tag(l, 1, 2));
        check(placed, "v1 K+V: every layer's K and V land in that layer");
        check(ok && !kvf.in_r && kvf.f2tk.empty(), "v1: no in-r request -> no f2tk");
    }
    {
        // v3 per-layer ranks: layer 0 K r2 + V r1, layer 1 K r0 + V r2.
        FactorFile ff;
        ff.magic("STLGKVF3");
        for (uint32_t v : {3u, (uint32_t) L, (uint32_t) HID, (uint32_t) NH}) ff.u32(v);
        const int rk[L] = {2, 0}, rv[L] = {1, 2};
        for (int r : rk) ff.u32((uint32_t) r);
        for (int r : rv) ff.u32((uint32_t) r);
        for (int l = 0; l < L; ++l) {
            if (rk[l]) { ff.block((size_t) NH * rk[l] * HID, tag(l, 0, 1));
                         ff.block((size_t) NH * rk[l] * HID, tag(l, 0, 2)); }
            if (rv[l]) { ff.block((size_t) NH * rv[l] * HID, tag(l, 1, 1));
                         ff.block((size_t) NH * rv[l] * HID, tag(l, 1, 2)); }
        }
        KVFactors kvf;
        std::string err;
        const bool ok = load_file(ff, "kvf_v3_kv.bin", kvf, err, L, HID, NH, true);
        check(ok, "v3 per-layer K+V file loads", err);
        check(ok && kvf.rank_layer_k(0) == 2 && kvf.rank_layer_k(1) == 0 &&
                  kvf.rank_layer_v(0) == 1 && kvf.rank_layer_v(1) == 2 &&
                  kvf.rank_k == 2 && kvf.rank_v == 2,
              "v3: per-layer rank tables");
        check(ok && all_eq(kvf.f1k[0], (size_t) NH * 2 * HID, tag(0, 0, 1)) &&
                  kvf.f1k[1].empty() &&
                  all_eq(kvf.f2v[0], (size_t) NH * 1 * HID, tag(0, 1, 2)) &&
                  all_eq(kvf.f1v[1], (size_t) NH * 2 * HID, tag(1, 1, 1)),
              "v3 K+V: payloads land in their layers");
        check(ok && kvf.in_r && kvf.f2tk.size() == (size_t) L && !kvf.f2tk[0].empty() &&
                  kvf.f2tk[1].empty(),
              "v3: in-r requested at load -> f2tk for compressed K layers");
    }
    auto expect_reject = [&](const FactorFile& ff, const char* what, const char* needle) {
        KVFactors kvf;
        std::string err;
        const bool ok = load_file(ff, "kvf_bad.bin", kvf, err, L, HID, NH, true);
        check(!ok && err.find(needle) != std::string::npos, what, ok ? "loaded" : err);
        check(!kvf.enabled() && !kvf.in_r && kvf.f1k.empty(),
              std::string(what) + " leaves the factors disabled");
    };
    {
        FactorFile ff;  // v1 rank 3 > head_dim 2 (no payload needed: rejected first)
        ff.magic("STLGKVF1");
        for (uint32_t v : {1u, (uint32_t) L, (uint32_t) HID, (uint32_t) NH, 3u, 0u}) ff.u32(v);
        expect_reject(ff, "v1 rank > head_dim rejected", "exceeds head_dim");
    }
    {
        FactorFile ff;  // v3 rank that would wrap negative as int
        ff.magic("STLGKVF3");
        for (uint32_t v : {3u, (uint32_t) L, (uint32_t) HID, (uint32_t) NH}) ff.u32(v);
        for (uint32_t v : {0u, 0u, 0u, 0xFFFFFFFFu}) ff.u32(v);
        expect_reject(ff, "v3 wrapped rank rejected", "exceeds head_dim");
    }
    {
        FactorFile ff;  // truncated V payload
        ff.magic("STLGKVF1");
        for (uint32_t v : {1u, (uint32_t) L, (uint32_t) HID, (uint32_t) NH, 1u, 1u}) ff.u32(v);
        ff.block((size_t) NH * HID * 3, 1.0f);
        expect_reject(ff, "truncated payload rejected", "truncated V payload in layer 0");
    }
    {
        FactorFile ff;  // trailing bytes
        ff.magic("STLGKVF1");
        for (uint32_t v : {1u, (uint32_t) L, (uint32_t) HID, (uint32_t) NH, 1u, 0u}) ff.u32(v);
        ff.block((size_t) NH * HID * 2 * L + 1, 1.0f);
        expect_reject(ff, "trailing bytes rejected", "trailing bytes");
    }
    {
        FactorFile ff;
        ff.magic("STLGKVF2");
        expect_reject(ff, "bad magic rejected", "bad magic");
    }
    {
        FactorFile ff;
        ff.magic("STLGKVF1");
        for (uint32_t v : {1u, (uint32_t) L + 1, (uint32_t) HID, (uint32_t) NH, 1u, 0u})
            ff.u32(v);
        expect_reject(ff, "dims mismatch rejected", "model dims mismatch");
    }
}

#ifndef _WIN32
// Tiny granite encoder: 1 layer, hidden 1024, 8 heads (head_dim 128).
constexpr int kLayers = 1, kHidden = 1024, kHeads = 8;

bool write_tiny_factors(const std::filesystem::path& path, uint32_t rk, uint32_t rv) {
    FactorFile ff;
    ff.magic("STLGKVF1");
    for (uint32_t v : {1u, (uint32_t) kLayers, (uint32_t) kHidden, (uint32_t) kHeads, rk, rv})
        ff.u32(v);
    if (rk <= 128 && rv <= 128)
        for (uint32_t r : {rk, rv})
            if (r) ff.block((size_t) 2 * kHeads * r * kHidden, 0.01f);
    return ff.write(path);
}

std::string decode_once(const char* gguf, const char* label, bool kvinr_after_load) {
    const char* err = nullptr;
    void* handle = starling_ggml_granite_load(gguf, &err);
    check(handle != nullptr, std::string("e2e: ") + label + " loads", err ? err : "");
    if (!handle) return "";
    if (kvinr_after_load) SETENV("STARLING_GRANITE_KVINR", "1");
    std::vector<float> pcm(8000, 0.0f);
    err = nullptr;
    char* text = starling_ggml_granite_decode(handle, pcm.data(), (int64_t) pcm.size(), &err);
    check(text != nullptr, std::string("e2e: ") + label + " decodes", err ? err : "");
    std::string out = text ? text : "";
    std::free(text);
    starling_ggml_granite_free(handle);
    UNSETENV("STARLING_GRANITE_KVINR");
    return out;
}

void e2e_checks() {
    TinyGraniteFixture fixture(tmp("granite_kv_factors_test.gguf"));
    check(fixture.wrote(), "e2e: synthesized tiny granite GGUF written");
    if (!fixture.wrote()) return;
    const std::string gguf = fixture.path.string();
    const auto kv_path = tmp("granite_kv_factors_test_kv.bin");
    const auto over_path = tmp("granite_kv_factors_test_over.bin");

    // The in-r switch is read once at load; flipping it afterwards must not
    // send the encoder down the in-r branch without f2tk. This must be the
    // process's FIRST encoder run: the old per-process latch fired there.
    UNSETENV("STARLING_GRANITE_KVINR");
    check(write_tiny_factors(kv_path, 16, 8), "e2e: K+V factor file written");
    SETENV("STARLING_GRANITE_KVFACT", kv_path.string().c_str());
    const std::string after = decode_once(gguf.c_str(), "KVINR set after load", true);
    const std::string factored = decode_once(gguf.c_str(), "K+V factors", false);
    SETENV("STARLING_GRANITE_KVINR", "1");
    const std::string in_r = decode_once(gguf.c_str(), "KVINR set before load", false);

    UNSETENV("STARLING_GRANITE_KVFACT");
    UNSETENV("STARLING_GRANITE_KVINR");
    const std::string base = decode_once(gguf.c_str(), "no factors", false);
    check(!base.empty() && factored == base,
          "e2e: factors leave the zero model's transcript unchanged");
    check(after == base, "e2e: KVINR set after load is ignored (no crash)");
    check(in_r == base, "e2e: in-r path decodes the zero model unchanged");

    check(write_tiny_factors(over_path, 129, 0), "e2e: over-rank factor file written");
    SETENV("STARLING_GRANITE_KVFACT", over_path.string().c_str());
    const char* err = nullptr;
    void* handle = starling_ggml_granite_load(gguf.c_str(), &err);
    check(handle == nullptr && err && std::string(err).find("exceeds head_dim") !=
                                          std::string::npos,
          "e2e: rank 129 on head_dim 128 fails the load", err ? err : "loaded");
    if (handle) starling_ggml_granite_free(handle);

    UNSETENV("STARLING_GRANITE_KVFACT");
    UNSETENV("STARLING_GRANITE_KVINR");
    std::error_code ignored;
    std::filesystem::remove(kv_path, ignored);
    std::filesystem::remove(over_path, ignored);
}
#endif

} // namespace

int main() {
    unit_checks();
#ifdef _WIN32
    std::printf("[SKIP] e2e factor-path checks (POSIX only)\n");
#else
    SETENV("STARLING_GGML_DEVICE", "cpu");
    e2e_checks();
#endif
    std::printf("%s\n", failures ? "GRANITE KV FACTORS FAILED" : "GRANITE KV FACTORS OK");
    return failures ? 1 : 0;
}
