#pragma once

#include "loader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace starling::ggml::granite {

constexpr double kChunkSampleRate = 16000.0;

inline int64_t effective_chunk_samples(const Config& cfg) {
    const double token_limited =
        std::max(0.1, ((double)(int)cfg.max_new_tokens - 32.0) / 5.0);
    return (int64_t)std::llround(
        std::min(cfg.chunk_seconds, token_limited) * kChunkSampleRate);
}

// prompt_len is the padded chunk's prompt length. Omit it for the C API's
// single-piece path, where the ordinary decoder validates cache capacity.
inline int32_t decode_budget(const Config& cfg, double duration_s,
                             int64_t prompt_len = -1) {
    int64_t estimated = (int64_t)std::ceil(duration_s * 5.0) + 32;
    if (estimated < 1) estimated = 1;
    const int64_t cap = cfg.max_new_tokens > 0 ? cfg.max_new_tokens : 1;
    int64_t budget = std::min(cap, estimated);
    if (prompt_len >= 0) {
        const int64_t headroom = (int64_t)cfg.llm.max_cache - prompt_len - 1;
        budget = std::min(budget, std::max<int64_t>(1, headroom));
    }
    return (int32_t)budget;
}

} // namespace starling::ggml::granite
