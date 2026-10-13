// npu_fingerprint_test.cpp — the Pixel TPU encoder's model fingerprint
// (cpp/npu/encoder_contract.hpp, gguf_fingerprint.cpp).
//
//  * FNV-1a 64 against reference vectors (the standard offset basis);
//  * the fingerprint of a synthesized GGUF equals the value
//    scripts/npu/export_encoder.py computes for the same tensor values
//    (tests/test_npu_fingerprint.py pins the same constant), so the compile
//    scripts' comparison of the two can match.
//
// Usage: npu_fingerprint_test     Exit 0 = pass, 1 = failure.
#include "ggml.h"
#include "gguf.h"
#include "npu/encoder_contract.hpp"
#include "npu/gguf_fingerprint.hpp"

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

// Shared with tests/test_npu_fingerprint.py: identity tensor k (in
// kQwen3EncFingerprintTensors order) holds 4 + k f32 values (16k + i) / 4 - 1.
constexpr uint64_t kIdentityVector = 0xad085128de48605dull;

bool write_model(const char* path) {
    ggml_context* ctx = ggml_init({1 << 20, nullptr, false});
    gguf_context* g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "qwen3");
    for (int k = 0; k < 5; ++k) {
        const int64_t ne = 4 + k;
        ggml_tensor* t = ggml_new_tensor(ctx, GGML_TYPE_F32, 1, &ne);
        ggml_set_name(t, kQwen3EncFingerprintTensors[k]);
        float* f = static_cast<float*>(t->data);
        for (int i = 0; i < 4 + k; ++i) f[i] = (16 * k + i) * 0.25f - 1.0f;
        gguf_add_tensor(g, t);
    }
    const bool ok = gguf_write_to_file(g, path, false);
    gguf_free(g);
    ggml_free(ctx);
    return ok;
}

}  // namespace

int main() {
    check_hex(fnv1a_update(kFnvBasis, "", 0), 0xcbf29ce484222325ull, "FNV-1a 64 of \"\" is the standard basis");
    check_hex(fnv1a_update(kFnvBasis, "a", 1), 0xaf63dc4c8601ec8cull, "FNV-1a 64 of \"a\"");
    check_hex(fnv1a_update(kFnvBasis, "foobar", 6), 0x85944171f73967e8ull, "FNV-1a 64 of \"foobar\"");
    char path[256];
    std::snprintf(path, sizeof path, "/tmp/npu_fingerprint_test_%d.gguf", (int)getpid());
    uint64_t h = 0;
    std::string err;
    const bool ok = write_model(path) && gguf_fingerprint(path, h, err);
    check(ok, "fingerprint of the synthesized GGUF " + err);
    if (ok) check_hex(h, kIdentityVector, "= export_encoder.py's value for the same tensors");
    std::remove(path);
    if (g_fail) {
        std::printf("npu_fingerprint_test: %d check(s) FAILED\n", g_fail);
        return 1;
    }
    std::printf("npu_fingerprint_test: ok\n");
    return 0;
}
