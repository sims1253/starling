#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// Standalone Parakeet Q4_K x Q8_K CPU matmul. This library does not include,
// link to, or call ggml. The benchmark supplies GGUF tensor bytes and links
// ggml separately for the reference result.
namespace independent_q4k {

constexpr int block_width = 256;
constexpr int weight_block_bytes = 144;

struct ActivationBlock {
    float scale;
    int8_t values[block_width];
    int16_t sums[block_width / 32];
};
static_assert(sizeof(ActivationBlock) == 276, "unexpected Q4 activation layout");

struct PackedBlock {
    float scale;
    float min_scale;
    uint8_t group_scale[8];
    uint8_t group_min[8];
    uint8_t values[block_width];
};

struct RawMetadata {
    float scale;
    float min_scale;
    uint8_t group_scale[8];
    uint8_t group_min[8];
};

class Matrix {
public:
    Matrix(const void* weight_bytes, size_t size, int64_t cols, int64_t rows,
           bool prepack = true);
    void run(const float* input, int batch, float* output, int threads,
             bool raw = false);
    size_t packed_bytes() const { return blocks_.size() * sizeof(PackedBlock); }
    size_t metadata_bytes() const { return metadata_.size() * sizeof(RawMetadata); }

private:
    int64_t cols_;
    int64_t rows_;
    int64_t blocks_per_row_;
    const uint8_t* raw_;
    std::vector<RawMetadata> metadata_;
    std::vector<PackedBlock> blocks_;
    std::vector<ActivationBlock> activations_;
};

} // namespace independent_q4k
