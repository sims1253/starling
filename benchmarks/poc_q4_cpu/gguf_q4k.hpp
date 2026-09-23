#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace independent_q4k {

struct TensorBytes {
    int64_t cols = 0;
    int64_t rows = 0;
    std::vector<uint8_t> bytes;
};

// Reads one 2-D Q4_K tensor from GGUF v3 without ggml. Other tensor data and
// metadata values are skipped. Throws on unsupported/corrupt input.
TensorBytes read_gguf_q4k(const std::string& path, const std::string& name);

} // namespace independent_q4k
