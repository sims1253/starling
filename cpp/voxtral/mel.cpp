// mel.cpp — Voxtral log-mel frontend (see mel.hpp).
//
// Thin adapter over the shared frontend (lib/whisper_mel.hpp) plus the
// offline pad and the mel-constant synthesis. Voxtral policy bits: fullT-1
// valid frames (the extractor's stft[..., :-1] drops the trailing TIME
// frame), NO n_samples truncation (the offline pad owns length), global max
// over ALL computed frames with the FIXED max 1.5 (see below), f32
// arithmetic, bf16 + f32 output.
//
// Fixed max: voxtral's stock extractor clamps against the FIXED constant
// global_log_mel_max=1.5, not a per-utterance max (streaming-safe: a
// per-utterance max would leak future audio). lib's fixed_log_mel_max
// carries it; the clamp + (x+4)/4 math is otherwise the shared path.
//
// Mel constants: the real GGUF carries no audio.mel_* tensors (its metadata
// holds only the frontend dims), so they are synthesized into the loader at
// first use via add_owned_tensor (a no-op when present):
//   * window: periodic Hann, np.hanning(401)[:-1] (transformers
//     window_function(400, "hann")).
//   * bank: slaney mel_filter_bank (num_frequency_bins=201, 128 filters,
//     0..8000 Hz @16kHz, norm="slaney", mel_scale="slaney"), computed in f64
//     and rounded once to f32 — the same values the transformers helper
//     returns (verified bin-for-bin by the encoder test's mel check).
#include "mel.hpp"
#include "loader.hpp"
#include "lib/threads.hpp"
#include "lib/whisper_mel.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace starling::ggml::voxtral {
namespace {

// Slaney hertz<->mel (transformers audio_utils, mel_scale="slaney").
double hertz_to_mel_slaney(double f) {
    constexpr double min_log_hertz = 1000.0, min_log_mel = 15.0;
    constexpr double logstep = 27.0 / 6.4;  // 27/log(6.4) via std::log below
    if (f < min_log_hertz) return 3.0 * f / 200.0;
    return min_log_mel + std::log(f / min_log_hertz) * (27.0 / std::log(6.4));
}
double mel_to_hertz_slaney(double m) {
    constexpr double min_log_hertz = 1000.0, min_log_mel = 15.0;
    if (m < min_log_mel) return 200.0 * m / 3.0;
    return min_log_hertz * std::exp((std::log(6.4) / 27.0) * (m - min_log_mel));
}

const lib::EngineMelPolicy kMelPolicy = {
    lib::MelPolicy::T_FULLT_MINUS_1,
    /*cap_n_samples=*/false,
    /*cap_before_checks=*/false,
    lib::MelPolicy::MAX_ALL_FRAMES,
    /*norm_in_double=*/false,
    /*emit_bf16=*/true,
    "STARLING_VOXTRAL_MEL_DUMP",
    "VOXTRAL",
};

// Synthesize audio.mel_window [N] + audio.mel_filters [M*B] (GGUF layout:
// filter-major [B, M], element (b,m) at b*M+m) into the loader when absent.
void ensure_mel_constants(const Config& cfg, ModelLoader& ml) {
    const uint32_t N = cfg.frontend.n_fft, M = cfg.frontend.n_mels;
    const uint32_t B = N / 2 + 1;
    if (!ml.tensor("audio.mel_window")) {
        // Periodic Hann: np.hanning(N+1)[:-1] = 0.5-0.5*cos(2*pi*i/N)
        // (transformers window_function(N, "hann")).
        std::vector<float> win(N);
        const size_t nthr = lib::mel_thread_count();
        lib::parallel_for(nthr, N, [&](size_t, size_t lo, size_t hi) {
            for (size_t i = lo; i < hi; ++i)
                win[i] = (float)(0.5 - 0.5 * std::cos(2.0 * M_PI * (double) i /
                                                      (double) N));
        });
        ml.add_owned_tensor("audio.mel_window", win, N, 1);
    }
    if (!ml.tensor("audio.mel_filters")) {
        // Slaney bank in f64, one round to f32 (matches mel_filter_bank).
        const double sr = 16000.0, fmin = 0.0, fmax = 8000.0;
        std::vector<double> mels(M + 2);
        const double mel_lo = hertz_to_mel_slaney(fmin);
        const double mel_hi = hertz_to_mel_slaney(fmax);
        for (uint32_t i = 0; i < M + 2; ++i)
            mels[i] = mel_lo + (mel_hi - mel_lo) * (double) i / (double)(M + 1);
        std::vector<double> fcent(M + 2);
        for (uint32_t i = 0; i < M + 2; ++i) fcent[i] = mel_to_hertz_slaney(mels[i]);
        // fft_freqs: linspace(0, sr/2, B). Triangular bank [B, M]:
        // down=-slopes[:,:-2]/diff[:-1], up=slopes[:,2:]/diff[1:],
        // bank=max(0,min(down,up)), then slaney enorm 2/(f[i+2]-f[i]).
        std::vector<float> bank((size_t) B * M);
        const size_t nthr = lib::mel_thread_count();
        lib::parallel_for(nthr, B, [&](size_t, size_t blo, size_t bhi) {
            for (size_t b = blo; b < bhi; ++b) {
                const double fb = (sr / 2.0) * (double) b / (double)(B - 1);
                for (uint32_t m = 0; m < M; ++m) {
                    const double lo = fcent[m], ce = fcent[m + 1], hi = fcent[m + 2];
                    double v = 0.0;
                    if (fb >= lo && fb <= ce && ce > lo)
                        v = (fb - lo) / (ce - lo);
                    else if (fb >= ce && fb <= hi && hi > ce)
                        v = (hi - fb) / (hi - ce);
                    v *= 2.0 / (fcent[m + 2] - fcent[m]);
                    bank[b * M + m] = (float) v;
                }
            }
        });
        ml.add_owned_tensor("audio.mel_filters", bank, M, B);
    }
}

} // namespace

bool compute_log_mel(const Config& cfg, const ModelLoader& ml, const float* pcm,
                     size_t S, MelFeatures& out, std::string& err) {
    // Check the audio-derived cap before padding or allocating mel/attention.
    // Quotient/remainder avoids overflowing on a foreign PCM length.
    const size_t cap = S / 1280 + (S % 1280 != 0) + 49;
    if (cap > cfg.llm.max_cache) {
        err = "VOXTRAL total length cap " + std::to_string(cap) +
              " exceeds max_cache_len " + std::to_string(cfg.llm.max_cache);
        return false;
    }
    // T_enc = 4 * cap: a float32 [T_enc,T_enc] mask reaches
    // the encoder's 1 GiB budget at cap == 4096.
    if (cap > 4096) {
        err = "VOXTRAL audio too long for the encoder mask budget; use shorter audio";
        return false;
    }
    ensure_mel_constants(cfg, const_cast<ModelLoader&>(ml));
    // Offline pad: ceil to whole 1280-sample audio tokens, plus the 32 left
    // + 17 right streaming-pad tokens of zeros.
    const size_t P = (size_t) offline_padded_samples((int64_t) S);
    std::vector<float> padded(P, 0.0f);
    if (pcm && S) std::copy(pcm, pcm + S, padded.begin());
    lib::MelPolicy pol = lib::make_mel_policy(cfg.frontend, kMelPolicy);
    // The FIXED global log-mel max (streaming-safe normalization): the STOCK
    // extractor clamps against this constant, not a per-utterance max.
    pol.fixed_log_mel_max = cfg.frontend.log_mel_max;
    lib::MelOutput mo;
    if (!lib::compute_log_mel(pol, ml, padded.data(), P, mo, err)) return false;
    out.n_mels = (int64_t) mo.n_mels;
    out.n_frames = (int64_t) mo.n_frames;
    out.f32 = std::move(mo.f32);
    out.data = std::move(mo.bf16);
    return true;
}

} // namespace starling::ggml::voxtral
