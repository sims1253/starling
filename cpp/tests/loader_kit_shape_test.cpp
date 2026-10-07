// loader_kit_shape_test.cpp — direct unit tests for the shared
// lib::shape_eq helper (cpp/lib/loader_kit.hpp) that every model loader's
// exact-shape gate now goes through: exact-dim matching, the trailing-1
// trim (ggml_n_dims() strips trailing 1s from ne[], so want {N,1} must
// match a tensor stored 1-dim), the GGML_MAX_DIMS caller-bug guard, and
// the missing-tensor / mismatch error strings (the wording is load-failure
// diagnostics surface, so it is pinned here against silent rewording).
// Synthesizes a tiny GGUF so the check runs anywhere (no model files,
// CPU-only, no skip).
//
// Usage: ./loader_kit_shape_test
#include "ggml.h"
#include "gguf.h"
#include "lib/loader_kit.hpp"
#include "runtime/model_loader.hpp"
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

static int failures = 0;

static void check(bool ok, const char* what, const std::string& err = "") {
    std::printf("[%s] %s%s%s\n", ok ? "PASS" : "FAIL", what,
                (!ok && !err.empty()) ? " -- " : "", ok ? "" : err.c_str());
    if (!ok) failures++;
}

static bool contains(const std::string& hay, const char* needle) {
    return hay.find(needle) != std::string::npos;
}

int main() {
    // Unique path per process so parallel test runs cannot collide.
    char path[256];
    std::snprintf(path, sizeof path, "/tmp/loader_kit_shape_test_%d.gguf", (int) getpid());
    std::remove(path);

    // Tensors: ne=[8] (1-dim), ne=[6,1] (2-dim whose trailing dim is 1, so
    // ggml_n_dims() reports 1 after a GGUF round-trip), and ne=[3,4].
    ggml_context* gctx = ggml_init({1024 * 1024, nullptr, false});
    gguf_context* gf = gguf_init_empty();
    {
        const int64_t ne1[1] = {8};
        ggml_tensor* a = ggml_new_tensor(gctx, GGML_TYPE_F32, 1, ne1);
        ggml_set_name(a, "t.one_dim");
        const int64_t ne2[2] = {6, 1};
        ggml_tensor* b = ggml_new_tensor(gctx, GGML_TYPE_F32, 2, ne2);
        ggml_set_name(b, "t.trailing_one");
        const int64_t ne3[2] = {3, 4};
        ggml_tensor* c = ggml_new_tensor(gctx, GGML_TYPE_F32, 2, ne3);
        ggml_set_name(c, "t.two_dim");
        gguf_add_tensor(gf, a);
        gguf_add_tensor(gf, b);
        gguf_add_tensor(gf, c);
    }
    const bool wrote = gguf_write_to_file(gf, path, /*only_meta=*/false);
    gguf_free(gf);
    ggml_free(gctx);
    if (!wrote) {
        std::printf("[FAIL] synthesized GGUF written (%s)\n", path);
        return 1;
    }

    starling::ggml::ModelLoader m;
    if (!m.load(path)) {
        std::printf("[FAIL] ModelLoader parses synthesized GGUF -- %s\n",
                    m.last_error().c_str());
        return 1;
    }

    using starling::ggml::lib::shape_eq;
    std::string err;

    // --- exact matches -----------------------------------------------------
    err.clear();
    check(shape_eq(m, "TEST", "t.one_dim", {8}, err), "exact 1-dim match", err);
    err.clear();
    check(shape_eq(m, "TEST", "t.two_dim", {3, 4}, err), "exact 2-dim match", err);

    // --- trailing-1 trimming ------------------------------------------------
    // The stored ne=[6,1] reads back 1-dim; a want ending in 1 is trimmed the
    // same way, so both spellings match.
    err.clear();
    check(shape_eq(m, "TEST", "t.trailing_one", {6}, err),
          "trailing-1 tensor matches trimmed want {6}", err);
    err.clear();
    check(shape_eq(m, "TEST", "t.trailing_one", {6, 1}, err),
          "trailing-1 tensor matches want {6,1} (trailing 1 trimmed)", err);
    // A trailing 1 that is NOT redundant must still fail: ne=[8] has no
    // second dim, want {8,2} trims to nothing extra... {8,2} has no trailing
    // 1, so the count check rejects it.
    err.clear();
    check(!shape_eq(m, "TEST", "t.one_dim", {8, 2}, err),
          "want {8,2} still rejected against ne=[8]", err);

    // --- GGML_MAX_DIMS caller-bug guard --------------------------------------
    // Derived from the macro so a build with a different GGML_MAX_DIMS still
    // exercises the guard instead of failing on a hardcoded "max 4".
    {
        std::vector<int64_t> too_many((size_t) GGML_MAX_DIMS + 1, 1);
        too_many[0] = 3;
        too_many[1] = 4;
        const std::string want_msg = "shape_eq: wanted " +
            std::to_string(too_many.size()) + " dims (max " +
            std::to_string(GGML_MAX_DIMS) + ")";
        err.clear();
        check(!shape_eq(m, "TEST", "t.two_dim", too_many, err) &&
                  contains(err, want_msg.c_str()),
              "want beyond GGML_MAX_DIMS fails with dedicated error", err);
    }

    // --- missing tensor -----------------------------------------------------
    err.clear();
    check(!shape_eq(m, "TEST", "t.absent", {8}, err) &&
              contains(err, "TEST GGUF missing required tensor: t.absent"),
          "missing tensor error wording pinned", err);

    // --- mismatch error: dims + trimmed want print ---------------------------
    err.clear();
    check(!shape_eq(m, "TEST", "t.one_dim", {9}, err) &&
              contains(err, "TEST GGUF tensor t.one_dim has ne=[8], expected [9]"),
          "mismatch error wording pinned (ne + want)", err);
    err.clear();
    check(!shape_eq(m, "TEST", "t.one_dim", {9, 1}, err) &&
              contains(err, "expected [9]") &&
              !contains(err, "expected [9,1]"),
          "mismatch error prints the TRIMMED want (no trailing 1)", err);
    // Multi-dim value mismatch: pins the got/want printing loop (which
    // indexes the want list) against silent rewording regressions.
    err.clear();
    check(!shape_eq(m, "TEST", "t.two_dim", {3, 5}, err) &&
              contains(err, "TEST GGUF tensor t.two_dim has ne=[3,4], expected [3,5]"),
          "multi-dim mismatch error wording pinned (ne + want)", err);

    std::remove(path);
    std::printf("%s\n", failures ? "LOADER KIT SHAPE TEST FAILED" : "LOADER KIT SHAPE TEST OK");
    return failures ? 1 : 0;
}
