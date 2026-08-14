// mel.cpp — Whisper log-mel frontend for Hojo-ASR-V1.
//
// Byte-exact replica of the eager WhisperFeatureExtractor (n_fft=400, hop=160,
// 128 bins, reflect pad, Hann window, power=2, log10, then
// max-clamp(dynamic_range=8) and normalize (x+4)/4) that hojo_asr's dataset
// wraps (feat_extractor = WhisperFeatureExtractor.from_pretrained(whisper_path,
// chunk_length=40)). chunk_length only affects the attention-mask framing, not
// the mel computation, so the frontend is byte-identical to higgs/ark/moss.
// This is a verbatim copy of higgs/mel.cpp (renamed namespace).
#include "mel.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <thread>
#include <vector>
#include "lib/pocketfft_hdronly.h"
#include "lib/threads.hpp"
#include "ggml-backend.h"

namespace starling::ggml::hojo {
namespace {
size_t reflect_index(int64_t i, size_t n) {
    if (n <= 1) return 0;
    const int64_t period = 2 * (int64_t) n - 2;
    i %= period;
    if (i < 0) i += period;
    return (size_t)(i < (int64_t) n ? i : period - i);
}

} // namespace

bool compute_log_mel(const Config& cfg, const ModelLoader& ml, const float* pcm,
                     size_t S, MelFeatures& out, std::string& err) {
    const auto& c = cfg.frontend;
    // WhisperFeatureExtractor (truncation=True) truncates audio longer than
    // n_samples (= chunk_length*sr = 40s for hojo) to n_samples BEFORE the STFT.
    // Audio shorter than n_samples is used as-is (hojo calls with padding=False,
    // so no padding). Match this here: cap the sample count at n_samples.
    if (S > (size_t) c.n_samples) S = (size_t) c.n_samples;
    const size_t N = c.n_fft, H = c.hop_length, M = c.n_mels, B = N / 2 + 1;
    if (!pcm && S) { err = "null PCM input"; return false; }
    if (S < 2) { err = "Hojo reflect padding requires at least 2 PCM samples"; return false; }
    auto* wt = ml.tensor("audio.mel_window");
    auto* ft = ml.tensor("audio.mel_filters");
    if (!wt || !ft || wt->type != GGML_TYPE_F32 || ft->type != GGML_TYPE_F32) {
        err = "Hojo mel constants missing or not F32";
        return false;
    }
    if ((size_t) ggml_nelements(wt) != N || (size_t) ggml_nelements(ft) != M * B) {
        err = "Hojo mel constant shape mismatch";
        return false;
    }
    std::vector<float> window_host, bank_host;
    const float* window = (const float*) wt->data;
    const float* bank = (const float*) ft->data;
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
    // HF Whisper STFT computes fullT = S/H + 1 frames then DROPS the last
    // (`log_spec[:, :-1]`, magnitudes[..., :-1]) -> T = S // H valid frames.
    // Hojo uses padding=False (arbitrary S), so the last-frame truncation is
    // load-bearing (higgs/ark use padding=True with S a hop multiple, where
    // ceil(S/H) == S//H and the difference is invisible). fullT FFT frames are
    // still computed (the dropped frame's power feeds nothing), but the valid
    // count is S//H.
    const size_t fullT = S / H + 1, T = S / H;
    std::vector<float> logmel(M * fullT);
    std::vector<double> powers(B * fullT);
    std::vector<double> mel64(M * fullT);
    const size_t nthr = lib::mel_thread_count();
    std::vector<float> bank_t(M * B);
    for (size_t m = 0; m < M; ++m)
        for (size_t b = 0; b < B; ++b) bank_t[m * B + b] = bank[b * M + m];
    lib::parallel_for(nthr, fullT, [&](size_t /*tid*/, size_t lo, size_t hi) {
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
    lib::parallel_for(nthr, M * fullT, [&](size_t /*tid*/, size_t lo, size_t hi) {
        for (size_t idx = lo; idx < hi; ++idx) {
            const size_t m = idx / fullT, t = idx % fullT;
            double a = 0;
            const float* fb = &bank_t[m * B];
            const double* pw = &powers[t * B];
            for (size_t b = 0; b < B; ++b) a += (double) fb[b] * pw[b];
            mel64[m * fullT + t] = a;
        }
    });
    std::vector<float> chunk_max(nthr, -std::numeric_limits<float>::infinity());
    lib::parallel_for(nthr, M, [&](size_t tid, size_t m_lo, size_t m_hi) {
        float cm = -std::numeric_limits<float>::infinity();
        for (size_t m = m_lo; m < m_hi; ++m) {
            for (size_t t = 0; t < T; ++t) {
                const size_t idx = m * fullT + t;
                float v = (float) std::log10(std::max(mel64[idx], (double) c.mel_floor));
                logmel[idx] = v;
                cm = std::max(cm, v);
            }
        }
        chunk_max[tid] = cm;
    });
    float mx = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < nthr; ++i) mx = std::max(mx, chunk_max[i]);
    out.n_mels = M;
    out.n_frames = T;
    out.data.resize(M * T);
    lib::parallel_for(nthr, M * T, [&](size_t /*tid*/, size_t lo, size_t hi) {
        for (size_t idx = lo; idx < hi; ++idx) {
            const size_t m = idx / T, t = idx % T;
            float v = std::max(logmel[m * fullT + t], (float)(mx - c.dynamic_range));
            v = (float)((v + c.normalization_offset) / c.normalization_divisor);
            out.data[m * T + t] = v;
        }
    });
    if (const char* p = std::getenv("STARLING_HOJO_MEL_DUMP")) {
        if (FILE* f = std::fopen(p, "wb")) {
            std::fwrite(out.data.data(), sizeof(float), out.data.size(), f);
            std::fclose(f);
        }
    }
    return true;
}
} // namespace starling::ggml::hojo
