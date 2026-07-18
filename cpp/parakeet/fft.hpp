// fft.hpp — iterative radix-2 Cooley-Tukey FFT, the byte-exact reference.
//
// Mirrors parakeet.cpp/src/fft.cpp exactly (bit-reversal + butterfly, twiddle
// angle -2*pi/len, no normalization), operating in-place on a complex double
// buffer. This is the byte-exactness anchor for the mel frontend: matching it
// bit-for-bit reproduces NeMo's torch.stft numerics on CPU.

#pragma once

#include <complex>
#include <cstddef>
#include <vector>

namespace starling::ggml::parakeet {

// In-place radix-2 FFT of `a` (length n, must be a power of 2). Forward
// transform (sign = -1 in the twiddle), no normalization — matches torch.stft's
// unnormalized rFFT for the 0..n/2 bins.
void rfft(std::vector<std::complex<double>>& a, size_t n);

} // namespace starling::ggml::parakeet
