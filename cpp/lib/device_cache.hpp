// device_cache.hpp — device-resident KV cache + RoPE tables. One cache per
// process; graphs reference the k/v tensors and RoPE tables as fixed leaves;
// zeroed at the start of each utterance and freed by the decode-cache
// clearer before backend teardown. Per-model get_device_cache() wrappers
// bind this to their LlmConfig.
#pragma once

#include "runtime/backend.hpp"
#include "runtime/graph.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include <string>
#include <vector>

namespace starling::ggml::lib {

struct DeviceCache {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor*> k, v;    // [n_layers], each [D, max_cache, KV] bf16
    ggml_tensor* rope_cos = nullptr;  // [D, max_pos] bf16
    ggml_tensor* rope_sin = nullptr;  // [D, max_pos] bf16
    int max_cache = 0, max_pos = 0;
    int n_layers = 0, D = 0, KV = 0;

    bool init(int n_layers_, int D_, int KV_, int max_cache_, float rope_theta,
              ggml_backend_t backend, std::string& e);
    void zero();
    ~DeviceCache() {
        if (shutting_down()) return;  // driver gone -> leak (fine at exit)
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
    }
};

} // namespace starling::ggml::lib
