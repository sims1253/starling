// mel.cpp — Whisper log-mel frontend for higgs-audio-v3-stt.
//
// Byte-exact replica of the eager WhisperFeatureExtractor (n_fft=400, hop=160,
// 128 bins, reflect pad, Hann window, power=2, log10, then
// max-clamp(dynamic_range=8) and normalize (x+4)/4). Threaded like ark/mel.cpp;
// the per-frame / per-(m,t) loop nests keep the same reduction order so the
// output is bit-identical to the serial path. pocketfft is reentrant. This is a
// verbatim copy of ark/mel.cpp (the two frontends are byte-exact except the
// namespace + error-string prefixes); the Higgs and ARK mel frontends wrap the
// same Whisper extractor with the same constants.
#include "mel.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <thread>
#include <vector>
#include "pocketfft_hdronly.h"
#include "ggml-backend.h"

namespace starling::ggml::higgs {
namespace {
// Vendored from moss/mel.cpp (the two frontends are byte-exact except n_fft +
// normalization constants, which come from config).
size_t reflect_index(int64_t i, size_t n) {
    if (n <= 1) return 0;
    const int64_t period = 2 * (int64_t) n - 2;
    i %= period;
    if (i < 0) i += period;
    return (size_t)(i < (int64_t) n ? i : period - i);
}

size_t mel_thread_count() {
    if (const char* p = std::getenv("STARLING_MEL_THREADS")) {
        char* end = nullptr;
        long v = std::strtol(p, &end, 10);
        if (end != p && v >= 1) return static_cast<size_t>(v);
    }
    unsigned hc = std::thread::hardware_concurrency();
    if (hc == 0) hc = 1;
    if (hc > 16) hc = 16;
    return static_cast<size_t>(hc);
}

template <typename Body>
void mel_parallel(size_t nthr, size_t total, Body&& body) {
    if (total == 0) return;
    if (nthr <= 1) { body((size_t)0, (size_t)0, total); return; }
    if (nthr > total) nthr = total;
    std::vector<std::thread> ths;
    ths.reserve(nthr);
    const size_t chunk = (total + nthr - 1) / nthr;
    for (size_t i = 0; i < nthr; ++i) {
        const size_t lo = i * chunk;
        if (lo >= total) break;
        const size_t hi = std::min(lo + chunk, total);
        ths.emplace_back([&, i, lo, hi]() { body(i, lo, hi); });
    }
    for (auto& t : ths) t.join();
}
} // namespace

bool compute_log_mel(const Config& cfg, const ModelLoader& ml, const float* pcm,
                     size_t S, MelFeatures& out, std::string& err) {
    const auto& c = cfg.frontend;
    const size_t N = c.n_fft, H = c.hop_length, M = c.n_mels, B = N / 2 + 1;
    if (!pcm && S) { err = "null PCM input"; return false; }
    if (S < 2) { err = "Higgs reflect padding requires at least 2 PCM samples"; return false; }
    // Whisper truncates the raw waveform to n_samples (chunk_length * sample_rate
    // = 30s) before the spectrogram, capping the mel at n_samples/hop frames
    // (3000 for the default config). This bounds the encoder's T_enc (<=1500) so
    // the baked absolute positional embedding table (1500 rows) always covers it;
    // without the cap long audio overruns the table view and aborts. HF applies
    // the same cap.
    const size_t n_samples_cap = c.n_samples > 0 ? (size_t) c.n_samples : S;
    if (S > n_samples_cap) S = n_samples_cap;
    auto* wt = ml.tensor("audio.mel_window");
    auto* ft = ml.tensor("audio.mel_filters");
    if (!wt || !ft || wt->type != GGML_TYPE_F32 || ft->type != GGML_TYPE_F32) {
        err = "Higgs mel constants missing or not F32";
        return false;
    }
    if ((size_t) ggml_nelements(wt) != N || (size_t) ggml_nelements(ft) != M * B) {
        err = "Higgs mel constant shape mismatch";
        return false;
    }
    std::vector<float> window_host, bank_host;
    const float* window = (const float*) wt->data;
    const float* bank = (const float*) ft->data;
    // Weight realization repoints loader tensors to device tensors; mel stays on
    // host, so read constants back when a prior call realized the model.
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
    const size_t fullT = S / H + 1, T = fullT - 1;
    std::vector<float> logmel(M * fullT);
    std::vector<double> powers(B * fullT);
    std::vector<double> mel64(M * fullT);
    const size_t nthr = mel_thread_count();
    // Transpose the filterbank to bank_t[m*B+b] for cache-friendly dot products.
    std::vector<float> bank_t(M * B);
    for (size_t m = 0; m < M; ++m)
        for (size_t b = 0; b < B; ++b) bank_t[m * B + b] = bank[b * M + m];
    // Loop 1: per frame reflect-pad + window + r2c FFT + power.
    mel_parallel(nthr, fullT, [&](size_t /*tid*/, size_t lo, size_t hi) {
        std::vector<double> frame(N);
        std::vector<std::complex<double>> z(B);
        for (size_t t = lo; t < hi; ++t) {
            const int64_t start = (int64_t)(t * H) - (int64_t)(N / 2);
            for (size_t i = 0; i < N; ++i)
                frame[i] = (double) pcm[reflect_index(start + (int64_t) i, S)] * (double) window[i];
            pocketfft::r2c({N}, {sizeof(double)}, {sizeof(std::complex<double>)}, 0, true,
                           frame.data(), z.data(), 1.0);
            for (size_t b = 0; b < B; ++b) {
                const float re = (float) z[b].real(), im = (float) z[b].imag();
                const float mag = std::hypot(re, im);
                const float power = mag * mag;
                powers[t * B + b] = (double) power;
            }
        }
    });
    // Loop 2: mel filterbank.
    mel_parallel(nthr, M * fullT, [&](size_t /*tid*/, size_t lo, size_t hi) {
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
    std::vector<float> chunk_max(nthr, -std::numeric_limits<float>::infinity());
    mel_parallel(nthr, M * fullT, [&](size_t tid, size_t lo, size_t hi) {
        float cm = -std::numeric_limits<float>::infinity();
        for (size_t idx = lo; idx < hi; ++idx) {
            float v = (float) std::log10(std::max(mel64[idx], (double) c.mel_floor));
            logmel[idx] = v;
            cm = std::max(cm, v);
        }
        chunk_max[tid] = cm;
    });
    float mx = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < nthr; ++i) mx = std::max(mx, chunk_max[i]);
    out.n_mels = M;
    out.n_frames = T;
    out.f32.resize(M * T);
    out.data.resize(M * T);
    // Loop 3b: clamp + Whisper normalize (x+4)/4 + bf16.
    mel_parallel(nthr, M * T, [&](size_t /*tid*/, size_t lo, size_t hi) {
        for (size_t idx = lo; idx < hi; ++idx) {
            const size_t m = idx / T, t = idx % T;
            float v = std::max(logmel[m * fullT + t], mx - c.dynamic_range);
            v = (v + c.normalization_offset) / c.normalization_divisor;
            const size_t i = m * T + t;
            out.f32[i] = v;
            out.data[i] = ggml_fp32_to_bf16(v);
        }
    });
    if (const char* p = std::getenv("STARLING_HIGGS_MEL_DUMP")) {
        if (FILE* f = std::fopen(p, "wb")) {
            std::fwrite(out.f32.data(), sizeof(float), out.f32.size(), f);
            std::fclose(f);
        }
    }
    return true;
}
} // namespace starling::ggml::higgs
