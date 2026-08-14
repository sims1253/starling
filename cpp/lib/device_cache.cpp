// device_cache.cpp — device-resident KV cache (see device_cache.hpp).
#include "device_cache.hpp"
#include "runtime/graph.hpp"
#include <cmath>
#include <memory>

namespace starling::ggml::lib {

bool DeviceCache::init(int n_layers_, int D_, int KV_, int max_cache_,
                       float rope_theta, ggml_backend_t backend, std::string& e) {
    n_layers = n_layers_;
    D = D_;
    KV = KV_;
    max_cache = max_cache_;
    max_pos = max_cache_;  // decode positions stay < max_cache

    const size_t n_tensors = 2 * (size_t) n_layers + 2;
    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (n_tensors + 8),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx = ggml_init(params);
    if (!ctx) { e = "DeviceCache: ggml_init failed"; return false; }

    int64_t kv_ne[3] = {D, max_cache, KV};
    k.resize(n_layers);
    v.resize(n_layers);
    for (int i = 0; i < n_layers; ++i) {
        k[i] = ggml_new_tensor(ctx, GGML_TYPE_BF16, 3, kv_ne);
        v[i] = ggml_new_tensor(ctx, GGML_TYPE_BF16, 3, kv_ne);
    }
    int64_t rope_ne[2] = {D, max_pos};
    rope_cos = ggml_new_tensor(ctx, GGML_TYPE_BF16, 2, rope_ne);
    rope_sin = ggml_new_tensor(ctx, GGML_TYPE_BF16, 2, rope_ne);

    buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { e = "DeviceCache: backend alloc failed"; return false; }

    // Precompute the RoPE cos/sin tables with the f32 std::pow-based formula
    // (duplicated halves, rounded to bf16).
    std::vector<ggml_bf16_t> cos_t((size_t) D * max_pos), sin_t((size_t) D * max_pos);
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < D / 2; ++i) {
            float inv = 1.0f / std::pow(rope_theta, (2.0f * i) / D);
            float a = (float) p * inv;
            ggml_bf16_t c = ggml_fp32_to_bf16(std::cos(a));
            ggml_bf16_t s = ggml_fp32_to_bf16(std::sin(a));
            cos_t[(size_t) p * D + i] = cos_t[(size_t) p * D + i + D / 2] = c;
            sin_t[(size_t) p * D + i] = sin_t[(size_t) p * D + i + D / 2] = s;
        }
    }
    ggml_backend_tensor_set(rope_cos, cos_t.data(), 0, cos_t.size() * sizeof(ggml_bf16_t));
    ggml_backend_tensor_set(rope_sin, sin_t.data(), 0, sin_t.size() * sizeof(ggml_bf16_t));

    zero();
    return true;
}

void DeviceCache::zero() {
    std::vector<ggml_bf16_t> z((size_t) D * max_cache * KV, ggml_bf16_t{0});
    for (int i = 0; i < n_layers; ++i) {
        ggml_backend_tensor_set(k[i], z.data(), 0, z.size() * sizeof(ggml_bf16_t));
        ggml_backend_tensor_set(v[i], z.data(), 0, z.size() * sizeof(ggml_bf16_t));
    }
}

} // namespace starling::ggml::lib
