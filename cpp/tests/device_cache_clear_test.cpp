// device_cache_clear_test.cpp — unit test for DeviceCache::zero()'s on-device
// clear (S01: K/V cleared via ggml_backend_tensor_memset instead of host
// zero uploads). Verifies, on the CPU backend:
//
//   (1) init()'s initial clear: fresh K/V read back as all-zero bytes;
//   (2) after poison + zero(): every layer's K and V read back byte-exact
//       zero over the tensor's full extent;
//   (3) the RoPE cos/sin tables sharing the same backend allocation are
//       untouched by zero() (a whole-allocation clear would destroy them);
//   (4) per-cache isolation: zeroing one DeviceCache never touches another
//       instance's tensors (each request owns its cache);
//   (5) clear-then-prefill ordering: a graph compute that reads a K tensor
//       as a leaf sees the poison before zero() and exact 0.0 after — the
//       memset is complete and visible before any subsequent graph runs.
//
// CPU-only. Exit 0 = pass, 1 = failure.

#include "lib/device_cache.hpp"
#include "runtime/backend.hpp"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using starling::ggml::lib::DeviceCache;

namespace {

int g_failures = 0;

void check(bool cond, const std::string& what) {
    if (!cond) {
        std::printf("FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

constexpr int kLayers = 3, kD = 8, kKV = 2, kMaxCache = 16;
constexpr size_t kElems = (size_t) kD * kMaxCache * kKV;
constexpr size_t kNbytes = kElems * sizeof(ggml_bf16_t);

std::vector<uint8_t> readback(const ggml_tensor* t) {
    std::vector<uint8_t> bytes(ggml_nbytes(t));
    ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
    return bytes;
}

bool all_zero(const std::vector<uint8_t>& bytes) {
    for (uint8_t b : bytes)
        if (b != 0) return false;
    return true;
}

void poison(ggml_tensor* t) {
    const std::vector<uint8_t> p(kNbytes, 0xA5);
    ggml_backend_tensor_set(t, p.data(), 0, p.size());
}

// Sum of a KV tensor's contents as a subsequent graph (a prefill) would see
// them: the cache tensor is a leaf with device storage already allocated
// (the same weight-style leaf clone_weight produces), converted bf16 -> f32
// and reduced. Exercises the exact read path prefill uses for the cache.
bool graph_sum(starling::ggml::Backend& backend, ggml_tensor* t, float& out) {
    std::vector<float> res;
    const bool ok = backend.compute([&](ggml_context* c) -> ggml_tensor* {
        int64_t ne[3] = {t->ne[0], t->ne[1], t->ne[2]};
        ggml_tensor* f = ggml_new_tensor(c, GGML_TYPE_F32, 3, ne);
        return ggml_sum(c, ggml_cpy(c, t, f));
    }, res);
    if (!ok || res.size() != 1) return false;
    out = res[0];
    return true;
}

} // namespace

int main() {
    starling::ggml::Backend backend(2);
    std::printf("device_cache_clear_test: backend=%s\n", backend.device_name());

    std::string err;
    DeviceCache dc;
    check(dc.init(kLayers, kD, kKV, kMaxCache, 10000.0f, backend.handle(), err),
          "DeviceCache::init succeeds: " + err);

    // (1) Constructor initial clear: init() itself leaves K/V zeroed.
    for (int i = 0; i < kLayers; ++i) {
        check(all_zero(readback(dc.k[i])),
              "init leaves K[" + std::to_string(i) + "] zero");
        check(all_zero(readback(dc.v[i])),
              "init leaves V[" + std::to_string(i) + "] zero");
    }

    // Snapshot the RoPE tables (shared allocation) for the (3) check. The
    // table is naturally nonzero (cos(0) = 1 in row 0), so a later all-zero
    // readback would be detectable.
    const std::vector<uint8_t> rope_cos_before = readback(dc.rope_cos);
    const std::vector<uint8_t> rope_sin_before = readback(dc.rope_sin);
    check(rope_cos_before.size() == (size_t) kD * kMaxCache * sizeof(ggml_bf16_t),
          "rope_cos snapshot has the expected byte count");
    bool rope_nonzero = false;
    for (uint8_t b : rope_cos_before)
        if (b != 0) { rope_nonzero = true; break; }
    check(rope_nonzero, "rope_cos snapshot is nontrivial (nonzero table)");

    // (5a) A graph reading K[0] sees live cache memory: poison first.
    poison(dc.k[0]);
    poison(dc.v[0]);
    for (int i = 1; i < kLayers; ++i) { poison(dc.k[i]); poison(dc.v[i]); }
    float before = -1.0f;
    check(graph_sum(backend, dc.k[0], before) && before != 0.0f,
          "graph sees poisoned (nonzero) K before zero()");

    // (4) Per-cache isolation: zeroing a second cache must not touch dc.
    DeviceCache other;
    check(other.init(kLayers, kD, kKV, kMaxCache, 10000.0f, backend.handle(), err),
          "second DeviceCache::init succeeds: " + err);
    poison(other.k[0]);
    other.zero();
    check(!all_zero(readback(dc.k[0])),
          "zero() of another cache leaves this cache's K untouched");
    check(all_zero(readback(other.k[0])),
          "zero() clears its own cache's K");

    // (2) Clear after poison: every layer, byte-exact zero, full extent.
    dc.zero();
    for (int i = 0; i < kLayers; ++i) {
        const std::vector<uint8_t> kb = readback(dc.k[i]);
        const std::vector<uint8_t> vb = readback(dc.v[i]);
        check(kb.size() == kNbytes, "K[" + std::to_string(i) + "] byte count");
        check(vb.size() == kNbytes, "V[" + std::to_string(i) + "] byte count");
        check(all_zero(kb), "K[" + std::to_string(i) + "] zeroed after zero()");
        check(all_zero(vb), "V[" + std::to_string(i) + "] zeroed after zero()");
    }

    // (3) RoPE tables in the same allocation untouched.
    check(readback(dc.rope_cos) == rope_cos_before,
          "rope_cos untouched by zero()");
    check(readback(dc.rope_sin) == rope_sin_before,
          "rope_sin untouched by zero()");

    // (5b) Clear-then-prefill ordering: the next graph compute sees exact 0.
    float after = -1.0f;
    check(graph_sum(backend, dc.k[0], after) && after == 0.0f,
          "graph sees exact 0.0 K after zero() (clear precedes next compute)");
    float vsum = -1.0f;
    check(graph_sum(backend, dc.v[0], vsum) && vsum == 0.0f,
          "graph sees exact 0.0 V after zero()");

    if (g_failures) {
        std::printf("device_cache_clear_test: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("device_cache_clear_test: all checks passed\n");
    return 0;
}
