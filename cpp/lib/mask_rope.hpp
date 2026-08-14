// mask_rope.hpp — shared causal attention masks + host RoPE tables.
//
// build_causal_mask is extracted verbatim from the four identical copies in
// moss/ark/higgs/hojo llm.cpp. The host f32 RoPE builder (duplicated-halves
// layout, std::pow frequencies) is hojo's decode-time table builder; the
// moss/ark/higgs decoders use the device-resident tables in DeviceCache
// instead (same formula).
#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

namespace starling::ggml::lib {

// Causal additive mask [K, S] f32: 0 where allowed, -3.3895313892515355e38
// beyond (row qi covers keys j <= past+qi).
inline std::vector<float> build_causal_mask(int64_t S, int64_t past) {
    const int64_t K = past + S;
    std::vector<float> mask((size_t) K * S);
    const float neg = -3.3895313892515355e38f;
    for (int64_t qi = 0; qi < S; ++qi)
        for (int64_t j = 0; j < K; ++j)
            mask[(size_t) qi * K + j] = (j <= past + qi) ? 0.0f : neg;
    return mask;
}

// Precompute RoPE cos/sin for positions [0, max_pos) as f32 host tables
// (duplicated halves). Returned as two [D, max_pos] f32 tables.
struct RopeTables {
    std::vector<float> cos, sin;  // [D, max_pos]
    int D = 0, max_pos = 0;
};

inline RopeTables build_rope_tables(int head_dim, float rope_theta, int max_pos) {
    RopeTables r;
    r.D = head_dim;
    r.max_pos = max_pos;
    r.cos.assign((size_t) r.D * max_pos, 0.0f);
    r.sin.assign((size_t) r.D * max_pos, 0.0f);
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < r.D / 2; ++i) {
            float inv = 1.0f / std::pow((float) rope_theta, (2.0f * i) / r.D);
            float a = (float) p * inv;
            float c = std::cos(a), s = std::sin(a);
            r.cos[(size_t) p * r.D + i] = c;
            r.cos[(size_t) p * r.D + i + r.D / 2] = c;
            r.sin[(size_t) p * r.D + i] = s;
            r.sin[(size_t) p * r.D + i + r.D / 2] = s;
        }
    }
    return r;
}

} // namespace starling::ggml::lib
