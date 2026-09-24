// cpu_kernels.hpp — CPU GEMV kernels for the sequential decoders.
//
// The transducer/LLM greedy loops are strictly sequential with tiny
// per-step work (a few matrix-vector products), where a GPU round trip per
// step costs more than the math. These kernels multiply int8 weights (CpuQ8:
// f32 scale per 32) by an int8-quantized activation vector with the
// dot-product instructions of the target (ARMv8.2 SDOT, x86 AVX2), four rows
// at a time so each activation load feeds four accumulators.

#pragma once

#include "weights.hpp"

#include <cstdint>
#include <vector>

namespace starling::fast::cpu {

// Activation quantized per 32 (absmax / 127, f32 scale).
struct QVec {
    std::vector<int8_t> q;
    std::vector<float> s;
};
void quantize(const float* x, uint32_t K, QVec& out);

// y[n] = sum_k W[n][k] * x[k] (+ bias[n] when non-null) for rows [r0, r1).
void gemv(const CpuQ8& W, const QVec& x, const float* bias, float* y,
          uint32_t r0 = 0, uint32_t r1 = UINT32_MAX);

// Name of the compiled kernel family ("neon-dotprod", "avx2", "scalar").
const char* isa_name();

} // namespace starling::fast::cpu
