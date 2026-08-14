// whisper_mel.cpp — shared Whisper log-mel frontend (see whisper_mel.hpp).
// Body assembled verbatim from the four mel.cpp copies; every per-model
// branch is driven by MelPolicy.
#include "whisper_mel.hpp"
#include "pocketfft_hdronly.h"
#include "threads.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace starling::ggml::lib {
namespace {

size_t reflect_index(int64_t i, size_t n) {
    if (n <= 1) return 0;
    const int64_t period = 2 * (int64_t) n - 2;
    i %= period;
    if (i < 0) i += period;
    return (size_t)(i < (int64_t) n ? i : period - i);
}

} // namespace

bool compute_log_mel(const MelPolicy& p, const ModelLoader& ml, const float* pcm,
                     size_t S, MelOutput& out, std::string& err) {
    // hojo truncates unconditionally BEFORE the checks; the others check the
    // input first and cap afterwards with a >0 guard.
    if (p.cap_n_samples && p.cap_before_checks) {
        if (S > (size_t) p.n_samples) S = (size_t) p.n_samples;
    }
    const size_t N = p.n_fft, H = p.hop_length, M = p.n_mels, B = N / 2 + 1;
    if (!pcm && S) { err = "null PCM input"; return false; }
    if (S < 2) {
        err = std::string(p.label) + " reflect padding requires at least 2 PCM samples";
        return false;
    }
    if (p.cap_n_samples && !p.cap_before_checks) {
        const size_t n_samples_cap = p.n_samples > 0 ? (size_t) p.n_samples : S;
        if (S > n_samples_cap) S = n_samples_cap;
    }
    auto* wt = ml.tensor("audio.mel_window");
    auto* ft = ml.tensor("audio.mel_filters");
    if (!wt || !ft || wt->type != GGML_TYPE_F32 || ft->type != GGML_TYPE_F32) {
        err = std::string(p.label) + " mel constants missing or not F32";
        return false;
    }
    if ((size_t) ggml_nelements(wt) != N || (size_t) ggml_nelements(ft) != M * B) {
        err = std::string(p.label) + " mel constant shape mismatch";
        return false;
    }
    std::vector<float> window_host, bank_host;
    const float* window = (const float*) wt->data;
    const float* bank = (const float*) ft->data;
    // Weight realization repoints loader tensors to device tensors; mel stays
    // on host, so read constants back when a prior call realized the model.
    if (wt->buffer) {
        window_host.resize(N);
        ggml_backend_tensor_get(wt, window_host.data(), 0, N * sizeof(float));
        window = window_host.data();
    }
    if (ft->buffer) {
        bank_host.resize(M * B);
        ggml_backend_tensor_get(ft, bank_host.data(), 0, M * B * sizeof(float));
        bank = bank_host.data();
    }
    const size_t fullT = S / H + 1;
    size_t T;
    switch (p.t_rule) {
        case MelPolicy::T_FULLT_MINUS_1:  T = fullT - 1; break;
        case MelPolicy::T_CEIL_S_OVER_H:  T = (S + H - 1) / H; break;
        default:                          T = S / H; break;
    }
    std::vector<float> logmel(M * fullT);
    std::vector<double> powers(B * fullT);
    std::vector<double> mel64(M * fullT);
    const size_t nthr = mel_thread_count();
    // Transpose the filterbank to bank_t[m*B+b] for cache-friendly dot products.
    std::vector<float> bank_t(M * B);
    for (size_t m = 0; m < M; ++m)
        for (size_t b = 0; b < B; ++b) bank_t[m * B + b] = bank[b * M + m];
    // Loop 1: per frame reflect-pad + window + r2c FFT + power.
    parallel_for(nthr, fullT, [&](size_t /*tid*/, size_t lo, size_t hi) {
        std::vector<double> frame(N);
        std::vector<std::complex<double>> z(B);
        for (size_t t = lo; t < hi; ++t) {
            const int64_t start = (int64_t)(t * H) - (int64_t)(N / 2);
            for (size_t i = 0; i < N; ++i)
                frame[i] = (double) pcm[reflect_index(start + (int64_t) i, S)] *
                           (double) window[i];
            pocketfft::r2c({N}, {sizeof(double)}, {sizeof(std::complex<double>)}, 0, true,
                           frame.data(), z.data(), 1.0);
            for (size_t b = 0; b < B; ++b) {
                // NumPy stores the RFFT as complex64, computes abs(complex64) in
                // float32, then squares that float32 for power=2. Preserve both
                // rounding boundaries rather than widening magnitude/power to f64.
                const float re = (float) z[b].real(), im = (float) z[b].imag();
                const float mag = std::hypot(re, im);
                const float power = mag * mag;
                powers[t * B + b] = (double) power;
            }
        }
    });
    // Loop 2: mel filterbank.
    parallel_for(nthr, M * fullT, [&](size_t /*tid*/, size_t lo, size_t hi) {
        for (size_t idx = lo; idx < hi; ++idx) {
            const size_t m = idx / fullT, t = idx % fullT;
            double a = 0;
            const float* fb = &bank_t[m * B];
            const double* pw = &powers[t * B];
            for (size_t b = 0; b < B; ++b) a += (double) fb[b] * pw[b];
            mel64[m * fullT + t] = a;
        }
    });
    // Loop 3a: log10 + per-chunk max for a deterministic global-max reduction.
    // MAX_ALL_FRAMES: flat split over all M*fullT entries (moss/ark).
    // MAX_KEPT_FRAMES: per-m split over the KEPT frames [0, T) only, matching
    // the eager Whisper extractor (which drops the trailing frame via [:, :-1]
    // BEFORE the global max-clamp); including the dropped frame shifts the
    // clamp threshold and perturbs every kept frame (higgs/hojo).
    std::vector<float> chunk_max(nthr, -std::numeric_limits<float>::infinity());
    if (p.max_scope == MelPolicy::MAX_ALL_FRAMES) {
        parallel_for(nthr, M * fullT, [&](size_t tid, size_t lo, size_t hi) {
            float cm = -std::numeric_limits<float>::infinity();
            for (size_t idx = lo; idx < hi; ++idx) {
                float v = (float) std::log10(std::max(mel64[idx], p.mel_floor));
                logmel[idx] = v;
                cm = std::max(cm, v);
            }
            chunk_max[tid] = cm;
        });
    } else {
        parallel_for(nthr, M, [&](size_t tid, size_t m_lo, size_t m_hi) {
            float cm = -std::numeric_limits<float>::infinity();
            for (size_t m = m_lo; m < m_hi; ++m) {
                for (size_t t = 0; t < T; ++t) {
                    const size_t idx = m * fullT + t;
                    float v = (float) std::log10(std::max(mel64[idx], p.mel_floor));
                    logmel[idx] = v;
                    cm = std::max(cm, v);
                }
            }
            chunk_max[tid] = cm;
        });
    }
    float mx = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < nthr; ++i) mx = std::max(mx, chunk_max[i]);
    out.n_mels = M;
    out.n_frames = T;
    out.f32.resize(M * T);
    if (p.emit_bf16) out.bf16.resize(M * T);
    // Loop 3b: clamp + Whisper normalize + (optional) bf16. norm_in_double
    // keeps hojo's double-constant arithmetic (clamp threshold and the
    // (v+off)/div pass in double, rounded once at the store).
    if (p.norm_in_double) {
        const double off = p.normalization_offset, div = p.normalization_divisor;
        parallel_for(nthr, M * T, [&](size_t /*tid*/, size_t lo, size_t hi) {
            for (size_t idx = lo; idx < hi; ++idx) {
                const size_t m = idx / T, t = idx % T;
                float v = std::max(logmel[m * fullT + t], (float)((double) mx - p.dynamic_range));
                v = (float)((v + off) / div);
                out.f32[idx] = v;
            }
        });
    } else {
        const float off = (float) p.normalization_offset, div = (float) p.normalization_divisor;
        const float dr = (float) p.dynamic_range;
        parallel_for(nthr, M * T, [&](size_t /*tid*/, size_t lo, size_t hi) {
            for (size_t idx = lo; idx < hi; ++idx) {
                const size_t m = idx / T, t = idx % T;
                float v = std::max(logmel[m * fullT + t], mx - dr);
                v = (v + off) / div;
                out.f32[idx] = v;
                if (p.emit_bf16) out.bf16[idx] = ggml_fp32_to_bf16(v);
            }
        });
    }
    if (p.dump_env) {
        if (const char* env = std::getenv(p.dump_env)) {
            if (FILE* f = std::fopen(env, "wb")) {
                std::fwrite(out.f32.data(), sizeof(float), out.f32.size(), f);
                std::fclose(f);
            }
        }
    }
    return true;
}

} // namespace starling::ggml::lib
