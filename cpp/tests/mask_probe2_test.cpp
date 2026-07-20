// mask_probe2_test.cpp — replicate the MOSS LLM layer's mask usage pattern:
// one bf16 mask [K,S] input -> 16 f32 casts -> 16 softmaxes (per-head scores),
// outputs concatenated. Checks each head's probs for mask violations.
#include "runtime/backend.hpp"
#include "runtime/graph.hpp"
#include "ggml.h"
#include <cstdio>
#include <vector>
#include <cmath>

using namespace starling::ggml;

int main() {
    const int K = 107, S = 107, H = 16;
    std::vector<float> sc((size_t)K * S);
    for (size_t i = 0; i < sc.size(); ++i) sc[i] = std::sin((double)i * 0.37) * 12.0f;
    std::vector<ggml_bf16_t> mk((size_t)K * S);
    const ggml_bf16_t neg = ggml_fp32_to_bf16(-3.3895313892515355e38f);
    const ggml_bf16_t zero = ggml_fp32_to_bf16(0.0f);
    for (int qi = 0; qi < S; ++qi)
        for (int j = 0; j < K; ++j)
            mk[(size_t)qi * K + j] = (j <= qi) ? zero : neg;

    (void)global_backend();
    std::vector<float> out;
    std::vector<float> cap_mask;
    bool ok = run_graph([&](ggml_context* c) {
        int64_t ne[2] = {K, S};
        auto* m = graph_input_tensor(c, GGML_TYPE_BF16, 2, ne, mk.data(), mk.size() * sizeof(mk[0]));
        capture_graph_output(ggml_cast(c, m, GGML_TYPE_F32), &cap_mask);
        ggml_tensor* joined = nullptr;
        for (int h = 0; h < H; ++h) {
            auto* s = graph_input_tensor(c, GGML_TYPE_F32, 2, ne, sc.data(), sc.size() * sizeof(float));
            auto* mf = ggml_cast(c, m, GGML_TYPE_F32);
            auto* pr = ggml_soft_max_ext(c, s, mf, 1.0f, 0.0f);
            joined = joined ? ggml_concat(c, joined, pr, 0) : pr;
        }
        return joined;
    }, out);
    if (!ok) { std::printf("graph FAILED\n"); return 1; }

    // check each head's probs: row qi zero at j>qi
    // ggml joined [K*H, S]: element for (head h, query qi, key j) at (h*K + j) + (K*H)*qi
    int bad = 0;
    for (int h = 0; h < H; ++h) {
        for (int qi = 0; qi < S; ++qi)
            for (int j = qi + 1; j < K; ++j)
                if (std::abs(out[(size_t)(h*K + j) + (size_t)(K*H)*qi]) > 1e-6) { ++bad; if (bad < 5) std::printf("head %d row %d col %d: %.6f\n", h, qi, j, out[(size_t)(h*K + j) + (size_t)(K*H)*qi]); break; }
    }
    std::printf("bad cells: %d\n", bad);
    // mask capture check
    int mbad = 0;
    for (int qi = 0; qi < S; ++qi)
        for (int j = 0; j < K; ++j) {
            float exp_v = (j <= qi) ? 0.0f : -3.3895313892515355e38f;
            if (cap_mask[(size_t)qi*K+j] != exp_v) ++mbad;
        }
    std::printf("mask capture wrong cells: %d / %d\n", mbad, K*S);
    shutdown_backend();
    return (bad || mbad) ? 2 : 0;
}
