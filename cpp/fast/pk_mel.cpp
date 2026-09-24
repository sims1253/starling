// pk_mel.cpp — see pk_mel.hpp.

#include "pk_mel.hpp"

#include "parakeet/mel.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace starling::fast {

PkMel::PkMel(const ggml::parakeet::MelConstants& c)
    : n_fft_(c.n_fft), hop_(c.hop_length), n_mels_(c.n_mels), n_bins_(c.n_bins),
      preemph_(c.preemph), guard_(c.log_zero_guard), mag_power_(c.mag_power),
      per_feature_(c.normalize == "per_feature"), window_(c.window), fb_(c.filterbank) {
    const int nn = (int)n_fft_;
    // Bit reversal exactly as the reference computes it.
    bitrev_.resize(n_fft_);
    for (int i = 0; i < nn; ++i) bitrev_[i] = (uint32_t)i;
    for (int i = 1, j = 0; i < nn; ++i) {
        int bit = nn >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(bitrev_[i], bitrev_[j]);
    }
    // Twiddle recurrence of each stage, replayed once. The reference restarts
    // it at every block, so every block of a stage sees the same sequence.
    for (int len = 2; len <= nn; len <<= 1) {
        const double ang = -2.0 * M_PI / len;
        const double wr = std::cos(ang), wi = std::sin(ang);
        double cr = 1.0, ci = 0.0;
        for (int k = 0; k < len / 2; ++k) {
            tw_re_.push_back(cr);
            tw_im_.push_back(ci);
            const double nr = cr * wr - ci * wi;
            const double ni = cr * wi + ci * wr;
            cr = nr;
            ci = ni;
        }
    }
    fb_lo_.assign(n_mels_, 0);
    fb_hi_.assign(n_mels_, 0);
    for (uint32_t m = 0; m < n_mels_; ++m) {
        const float* row = &fb_[(size_t)m * n_bins_];
        uint32_t lo = n_bins_, hi = 0;
        for (uint32_t b = 0; b < n_bins_; ++b)
            if (row[b] != 0.0f) { lo = std::min(lo, b); hi = b + 1; }
        fb_lo_[m] = lo < hi ? lo : 0;
        fb_hi_[m] = hi;
    }
}

void PkMel::compute(const float* pcm, size_t S, std::vector<float>& feats, int& out_T) const {
    const size_t pad = n_fft_ / 2;
    const size_t T = 1 + S / hop_;              // = 1 + (S + n_fft - n_fft) / hop
    out_T = (int)T;
    // Preemphasis (double) into a zero-padded buffer.
    std::vector<double> padded(S + n_fft_, 0.0);
    if (S > 0) {
        padded[pad] = (double)pcm[0];
        for (size_t t = 1; t < S; ++t)
            padded[pad + t] = (double)pcm[t] - preemph_ * (double)pcm[t - 1];
    }
    feats.assign(T * n_mels_, 0.0f);
    std::vector<double> re(n_fft_), im(n_fft_), power(n_bins_);
    const int nn = (int)n_fft_;
    for (size_t t = 0; t < T; ++t) {
        const double* src = padded.data() + t * hop_;
        for (int i = 0; i < nn; ++i) {
            re[bitrev_[i]] = (double)(float)(src[i] * (double)window_[i]);
            im[bitrev_[i]] = 0.0;
        }
        size_t tw = 0;
        for (int len = 2; len <= nn; len <<= 1) {
            const int half = len / 2;
            for (int i = 0; i < nn; i += len) {
                for (int k = 0; k < half; ++k) {
                    const double cr = tw_re_[tw + k], ci = tw_im_[tw + k];
                    const int u = i + k, v = u + half;
                    const double tr = cr * re[v] - ci * im[v];
                    const double ti = cr * im[v] + ci * re[v];
                    re[v] = re[u] - tr;
                    im[v] = im[u] - ti;
                    re[u] = re[u] + tr;
                    im[u] = im[u] + ti;
                }
            }
            tw += half;
        }
        for (uint32_t b = 0; b < n_bins_; ++b) {
            const double r = (double)(float)re[b], m = (double)(float)im[b];
            const double mag = std::sqrt(r * r + m * m);
            power[b] = mag_power_ == 2.0 ? mag * mag : std::pow(mag, mag_power_);
        }
        float* out = &feats[t * n_mels_];
        for (uint32_t mm = 0; mm < n_mels_; ++mm) {
            const float* row = &fb_[(size_t)mm * n_bins_];
            double acc = 0.0;
            for (uint32_t b = fb_lo_[mm]; b < fb_hi_[mm]; ++b) acc += (double)row[b] * power[b];
            out[mm] = (float)std::log(acc + guard_);
        }
    }
    if (!per_feature_) return;
    const size_t valid = std::min(S / hop_, T);
    if (valid < 1) return;
    const double ddof = valid >= 2 ? (double)(valid - 1) : 1.0;
    std::vector<double> mean(n_mels_, 0.0), var(n_mels_, 0.0);
    for (size_t t = 0; t < valid; ++t)
        for (uint32_t m = 0; m < n_mels_; ++m) mean[m] += feats[t * n_mels_ + m];
    for (uint32_t m = 0; m < n_mels_; ++m) mean[m] /= (double)valid;
    for (size_t t = 0; t < valid; ++t)
        for (uint32_t m = 0; m < n_mels_; ++m) {
            const double d = feats[t * n_mels_ + m] - mean[m];
            var[m] += d * d;
        }
    for (uint32_t m = 0; m < n_mels_; ++m) var[m] = std::sqrt(var[m] / ddof) + (double)1e-5f;   // the reference's float eps
    for (size_t t = 0; t < T; ++t)
        for (uint32_t m = 0; m < n_mels_; ++m) {
            float& v = feats[t * n_mels_ + m];
            v = t < valid ? (float)(((double)v - mean[m]) / var[m]) : 0.0f;
        }
}

} // namespace starling::fast
