// pos_enc.cpp — FastConformer relative positional encoding table (host-side).
//
// Starling-authored port of NeMo's RelPositionalEncoding.sinusoidal computations.
// Numerics are load-bearing for byte-exactness vs the encoder golden: every
// detail mirrors the parakeet.cpp reference exactly (double-internal div_term,
// positions running +(T-1)..-(T-1), even dims = sin, odd dims = cos, cast f32).

#include "pos_enc.hpp"

#include <cassert>
#include <cmath>

namespace starling::ggml::parakeet {

// NeMo multi_head_attention.py INF_VAL.
static constexpr double kInfVal = 10000.0;

void rel_pos_encoding(int T, int d_model, std::vector<float>& out) {
    assert(T > 0 && d_model > 0 && (d_model % 2) == 0);
    const int P    = 2 * T - 1;       // relative positions
    const int half = d_model / 2;     // (sin, cos) pairs

    // div_term[i] = exp(2i * -(log(INF_VAL)/d_model)) for i in [0, half).
    // Compute in double for parity with NeMo / parakeet.cpp.
    std::vector<double> div_term(half);
    const double factor = -(std::log(kInfVal) / (double)d_model);
    for (int i = 0; i < half; ++i) {
        div_term[i] = std::exp((double)(2 * i) * factor);
    }

    out.assign((size_t)P * d_model, 0.0f);
    // Positions run from +(T-1) down to -(T-1) inclusive.
    for (int p = 0; p < P; ++p) {
        const double pos = (double)((T - 1) - p);  // (T-1), (T-2), ..., -(T-1)
        float* row = out.data() + (size_t)p * d_model;
        for (int i = 0; i < half; ++i) {
            const double arg = pos * div_term[i];
            row[2 * i]     = (float)std::sin(arg);   // even dims
            row[2 * i + 1] = (float)std::cos(arg);   // odd dims
        }
    }
}

} // namespace starling::ggml::parakeet
