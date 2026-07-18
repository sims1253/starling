// fft.cpp — iterative radix-2 Cooley-Tukey FFT.
//
// Byte-exact port of parakeet.cpp/src/fft.cpp (the byte-identical CPU reference).
// Operates on separate real/imaginary double arrays with the exact same
// bit-reversal permutation, butterfly arithmetic, and manual twiddle advance
// (cur_wr/cur_wi) as the reference. The mel frontend's byte-exactness vs the
// golden depends on matching this FFT's double-precision rounding bit-for-bit;
// the std::complex<double> variant diverges by ~1e-2 after CMVN.

#include "fft.hpp"

#include <cassert>
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <vector>

namespace starling::ggml::parakeet {

// In-place radix-2 FFT of `a` (length n, must be a power of 2). Forward
// transform (sign = -1 in the twiddle), no normalization — matches torch.stft's
// unnormalized rFFT for the 0..n/2 bins. Mirrors parakeet.cpp/src/fft.cpp's
// fft_inplace exactly (same bit-reversal + butterfly + twiddle advance).
void rfft(std::vector<std::complex<double>>& a, size_t n) {
    assert(n > 0 && (n & (n - 1)) == 0 && "rfft: n must be a power of 2");

    // Split into separate real/imag double arrays (parakeet.cpp's representation).
    std::vector<double> re(n), im(n);
    for (size_t i = 0; i < n; ++i) {
        re[i] = a[i].real();
        im[i] = a[i].imag();
    }
    const int nn = static_cast<int>(n);

    // Bit-reversal permutation (parakeet.cpp variant).
    for (int i = 1, j = 0; i < nn; ++i) {
        int bit = nn >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }

    // Butterfly stages with manual twiddle advance (parakeet.cpp variant).
    for (int len = 2; len <= nn; len <<= 1) {
        double ang = -2.0 * M_PI / len;
        double wr = std::cos(ang);
        double wi = std::sin(ang);
        for (int i = 0; i < nn; i += len) {
            double cur_wr = 1.0, cur_wi = 0.0;
            for (int k = 0; k < len / 2; ++k) {
                int u = i + k;
                int v = i + k + len / 2;
                double tr = cur_wr * re[v] - cur_wi * im[v];
                double ti = cur_wr * im[v] + cur_wi * re[v];
                re[v] = re[u] - tr;
                im[v] = im[u] - ti;
                re[u] = re[u] + tr;
                im[u] = im[u] + ti;
                // advance twiddle factor
                double new_wr = cur_wr * wr - cur_wi * wi;
                double new_wi = cur_wr * wi + cur_wi * wr;
                cur_wr = new_wr;
                cur_wi = new_wi;
            }
        }
    }

    // Write back into the complex buffer.
    for (size_t i = 0; i < n; ++i) a[i] = std::complex<double>(re[i], im[i]);
    // No normalization (matches torch.stft's unnormalized forward FFT).
}

} // namespace starling::ggml::parakeet
