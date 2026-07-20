// mask_probe.cpp — minimal reproduction for the CUDA soft_max_ext mask issue.
// Graph: scores [K,S] (f32) + mask [K,S] (bf16->f32 cast) -> soft_max_ext -> out.
// Checks that masked positions get ~0 probability, on the active backend.
#include "runtime/backend.hpp"
#include "runtime/graph.hpp"
#include "ggml.h"
#include <cstdio>
#include <vector>
#include <cmath>

using namespace starling::ggml;

int main() {
    const int K = 107, S = 107;
    // scores: deterministic pseudo-random in [-12, 12]
    std::vector<float> sc((size_t)K * S);
    for (size_t i = 0; i < sc.size(); ++i) sc[i] = std::sin((double)i * 0.37) * 12.0f;
    // causal mask row qi: j<=qi -> 0 else BF16_MIN
    std::vector<ggml_bf16_t> mk((size_t)K * S);
    const ggml_bf16_t neg = ggml_fp32_to_bf16(-3.3895313892515355e38f);
    const ggml_bf16_t zero = ggml_fp32_to_bf16(0.0f);
    for (int qi = 0; qi < S; ++qi)
        for (int j = 0; j < K; ++j)
            mk[(size_t)qi * K + j] = (j <= qi) ? zero : neg;

    std::vector<float> out;
    (void)global_backend();  // run_graph uses g_backend; create it first
    bool ok = run_graph([&](ggml_context* c) {
        int64_t sne[2] = {K, S};
        auto* s = graph_input_tensor(c, GGML_TYPE_F32, 2, sne, sc.data(), sc.size() * sizeof(float));
        auto* m = graph_input_tensor(c, GGML_TYPE_BF16, 2, sne, mk.data(), mk.size() * sizeof(mk[0]));
        auto* mf = ggml_cast(c, m, GGML_TYPE_F32);
        return ggml_soft_max_ext(c, s, mf, 1.0f, 0.0f);
    }, out);
    if (!ok) { std::printf("graph FAILED\n"); return 1; }

    // out [K,S]: row qi must have ~0 mass at j>qi
    double max_leak = 0; int bad_rows = 0;
    for (int qi = 0; qi < S; ++qi) {
        double row_sum = 0, leak = 0;
        for (int j = 0; j < K; ++j) {
            float p = out[(size_t)qi * K + j];
            row_sum += p;
            if (j > qi) { leak += p; if (std::abs(p) > max_leak) max_leak = std::abs(p); }
        }
        if (leak > 1e-6) ++bad_rows;
        if (qi < 3) std::printf("row %d: sum=%.6f leak=%.9g p0=%.4f\n", qi, row_sum, leak, out[(size_t)qi*K]);
    }
    std::printf("bad_rows=%d max_leak=%.9g\n", bad_rows, max_leak);
    shutdown_backend();
    return bad_rows ? 2 : 0;
}
