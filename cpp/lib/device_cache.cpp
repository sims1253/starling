// device_cache.cpp — device-resident KV cache (see device_cache.hpp).
#include "device_cache.hpp"
#include "runtime/graph.hpp"
#include <cmath>
#include <memory>

// ggml-backend-impl.h (under third_party/ggml/src/, on the core target's
// include path) declares the buffer iface so zero() can probe for
// memset_tensor support and keep the host-upload fallback for buffers that
// lack it (none of the backends Starling builds: CPU, CUDA/HIP, Metal,
// Vulkan — all register one).
#include "ggml-backend-impl.h"

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
    // [S01] Clear on-device instead of uploading host BF16 zeros. A bf16 +0.0
    // is the all-zero byte pattern, so ggml_backend_tensor_memset(t, 0, ...)
    // yields byte-identical contents to the old host-vector upload while
    // skipping the staging allocation and the whole H2D payload (default MOSS
    // geometry: 224 MiB across 56 tensor-sets -> 56 device-side fills, 0 host
    // bytes). Ordering semantics are unchanged: every backend that implements
    // memset_tensor completes it synchronously with the same visibility the
    // tensor_set path had (CPU: host memset; CUDA/HIP: cudaMemsetAsync +
    // cudaStreamSynchronize on the stream tensor_set uses; Vulkan: fillBuffer
    // submit + fence wait). Each K/V tensor is cleared individually — the
    // buffer also owns the RoPE tables, so a whole-buffer clear is never safe.
    // Buffers without memset support fall back to the original upload.
    const size_t n_elems = (size_t) D * max_cache * KV;
    const size_t nbytes  = n_elems * sizeof(ggml_bf16_t);
    std::vector<ggml_bf16_t> z;  // fallback staging, built only if needed
    for (int i = 0; i < n_layers; ++i) {
        for (ggml_tensor* t : {k[i], v[i]}) {
            ggml_backend_buffer_t tb = t->view_src ? t->view_src->buffer : t->buffer;
            if (tb != nullptr && tb->iface.memset_tensor != nullptr) {
                ggml_backend_tensor_memset(t, 0, 0, nbytes);
            } else {
                if (z.empty()) z.assign(n_elems, ggml_bf16_t{0});
                ggml_backend_tensor_set(t, z.data(), 0, nbytes);
            }
        }
    }
}

} // namespace starling::ggml::lib
