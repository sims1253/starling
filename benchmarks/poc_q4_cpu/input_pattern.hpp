#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace independent_q4k {

inline void fill_input(float* output, size_t count, const std::string& pattern) {
    if (pattern != "cos" && pattern != "random" && pattern != "zero")
        throw std::invalid_argument("input pattern must be cos, random, or zero");
    uint32_t rng = 0x7a3b1289u;
    for (size_t i = 0; i < count; ++i) {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        if (pattern == "cos") output[i] = std::cos(float(i % 137) * 0.041f) * 0.14f;
        else if (pattern == "random")
            output[i] = (float(rng & 0xffffu) / 32767.5f - 1.0f) * 0.2f;
        else output[i] = 0.0f;
    }
}

} // namespace independent_q4k
