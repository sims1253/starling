// host_conv_test.cpp — lib::host_conv1d_k3 against the scalar reference loop
// it replaced in ark/voxtral (sequential f64 accumulation, explicit zero
// padding). The contract is BIT equality on every output, for symmetric
// (ark) and left-only (voxtral) padding, stride 1 and 2, OC/OL sizes that
// are not multiples of the tile, and any thread count. No model needed.
//
// Usage: host_conv_test     Exit 0 = pass, 1 = mismatch.
#include "lib/host_conv.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using starling::ggml::lib::host_conv1d_k3;

namespace {

std::vector<float> reference(const std::vector<float>& wf, const std::vector<float>& bf,
                             const std::vector<ggml_bf16_t>& in, int64_t IC, int64_t L,
                             int64_t pad_l, int64_t pad_r, int64_t stride) {
    const int64_t OC = (int64_t) bf.size(), K = 3, Lp = L + pad_l + pad_r;
    const int64_t OL = (Lp - K) / stride + 1;
    std::vector<float> x((size_t) IC * Lp, 0.0f);
    for (int64_t c = 0; c < IC; ++c)
        for (int64_t t = 0; t < L; ++t)
            x[(size_t) c * Lp + t + pad_l] = ggml_bf16_to_fp32(in[(size_t) c * L + t]);
    std::vector<float> y((size_t) OC * OL);
    for (int64_t oc = 0; oc < OC; ++oc)
        for (int64_t t = 0; t < OL; ++t) {
            double acc = (double) bf[(size_t) oc];
            for (int64_t c = 0; c < IC; ++c)
                for (int64_t k = 0; k < K; ++k)
                    acc += (double) wf[((size_t) oc * IC + c) * K + k] *
                           (double) x[(size_t) c * Lp + t * stride + k];
            y[(size_t) oc * OL + t] = (float) acc;
        }
    return y;
}

int run_case(std::mt19937& rng, int64_t OC, int64_t IC, int64_t L, int64_t pad_l,
             int64_t pad_r, int64_t stride, int threads, bool f32_weights) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto round_bf16 = [](float v) { return ggml_bf16_to_fp32(ggml_fp32_to_bf16(v)); };
    std::vector<float> w((size_t) OC * IC * 3), b((size_t) OC);
    for (auto& v : w) v = f32_weights ? nd(rng) * 0.05f : round_bf16(nd(rng) * 0.05f);
    for (auto& v : b) v = round_bf16(nd(rng) * 0.1f);
    std::vector<ggml_bf16_t> in((size_t) IC * L);
    for (auto& v : in) v = ggml_fp32_to_bf16(nd(rng) * 3.0f);
    const std::vector<float> ref = reference(w, b, in, IC, L, pad_l, pad_r, stride);
    std::vector<float> got;
    const int64_t OL = host_conv1d_k3(w.data(), b.data(), OC, IC, in.data(), L, pad_l,
                                      pad_r, stride, got, threads);
    if (got.size() != ref.size() || (size_t) (OC * OL) != ref.size()) {
        std::printf("FAIL size OC=%lld IC=%lld L=%lld s=%lld: %zu vs %zu\n",
                    (long long) OC, (long long) IC, (long long) L, (long long) stride,
                    got.size(), ref.size());
        return 1;
    }
    if (std::memcmp(got.data(), ref.data(), ref.size() * sizeof(float)) != 0) {
        size_t bad = 0;
        for (size_t i = 0; i < ref.size(); ++i)
            if (std::memcmp(&got[i], &ref[i], sizeof(float)) != 0) ++bad;
        std::printf("FAIL bits OC=%lld IC=%lld L=%lld pad=(%lld,%lld) s=%lld thr=%d: "
                    "%zu/%zu differ\n", (long long) OC, (long long) IC, (long long) L,
                    (long long) pad_l, (long long) pad_r, (long long) stride, threads,
                    bad, ref.size());
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    std::mt19937 rng(1234);
    int fails = 0;
    // ark: symmetric pad 1; voxtral: left pad 2 (s1) / 1 (s2).
    const int64_t pads[3][3] = {{1, 1, 1}, {2, 0, 1}, {1, 0, 2}};  // {pad_l, pad_r, stride}
    for (const auto& p : pads)
        for (int64_t OC : {4, 37, 130})
            for (int64_t IC : {1, 65, 128})
                for (int64_t L : {2, 3, 63, 150})
                    for (int threads : {1, 3})
                        fails += run_case(rng, OC, IC, L, p[0], p[1], p[2], threads,
                                          /*f32_weights=*/OC == 37);
    // Stride 2 with symmetric padding (ark conv2).
    fails += run_case(rng, 96, 200, 301, 1, 1, 2, 4, false);
    if (fails) {
        std::printf("host_conv_test: %d case(s) FAILED\n", fails);
        return 1;
    }
    std::printf("host_conv_test: PASS\n");
    return 0;
}
