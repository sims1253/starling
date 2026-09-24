// pk_mel.hpp — Parakeet log-mel frontend for the fast engine.
//
// Bit-identical to parakeet::MelFrontend (the byte-exact NeMo reference:
// double preemphasis, the radix-2 FFT with its twiddle recurrence, double
// filterbank accumulation, per-feature CMVN) but restructured for speed:
//   * the twiddle recurrence of every FFT stage is replayed once into a table
//     (same values, no per-frame recomputation), bit reversal is a table,
//     and no per-frame allocations happen;
//   * the triangular filterbank is applied over each filter's nonzero bins
//     only (adding exact zeros never changes a double sum);
//   * output is time-major [T][n_mels], the layout the encoder reads.

#pragma once

#include <cstdint>
#include <vector>

namespace starling::ggml::parakeet { struct MelConstants; }

namespace starling::fast {

class PkMel {
public:
    explicit PkMel(const ggml::parakeet::MelConstants& c);
    // pcm: mono f32 at 16 kHz. Writes time-major feats [T][n_mels].
    void compute(const float* pcm, size_t n, std::vector<float>& feats, int& T) const;
    uint32_t n_mels() const { return n_mels_; }

private:
    uint32_t n_fft_, hop_, n_mels_, n_bins_;
    double preemph_, guard_, mag_power_;
    bool per_feature_;
    std::vector<float> window_;
    std::vector<uint32_t> bitrev_;
    std::vector<double> tw_re_, tw_im_;         // per-stage twiddles, concatenated
    std::vector<uint32_t> fb_lo_, fb_hi_;       // nonzero bin range per filter
    std::vector<float> fb_;                     // [n_mels][n_bins]
};

} // namespace starling::fast
