// mel.cpp — whisper-style log-mel frontend for Qwen3-ASR-1.7B.
//
// Thin adapter over the shared frontend (lib/whisper_mel.hpp) plus the two
// qwen3-specific steps around it: clips under min_length samples are
// zero-padded BEFORE the mel (matching Qwen3ASRFeatureExtractor, which does
// not touch the attention mask there), and after the normalize the mel time
// axis is right-padded with ZEROS to a multiple of 2*n_window = 100 frames
// (the value 0.0, NOT the silence-mel — those padded frames feed the conv
// stack and their values leak into valid outputs through the 3-wide conv
// kernels). Policy: torch.stft center=True gives fullT = S/H + 1 frames of
// which the LAST is dropped (T_FULLT_MINUS_1, the moss/ark rule — the
// extractor computes the mel over stft[..., :-1]), no n_samples cap, global
// max over ALL computed frames, f32 arithmetic, bf16 + f32 out. Output is
// transposed to time-major so the encoder's chunk reshape is contiguous.
#include "mel.hpp"
#include "lib/whisper_mel.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

namespace starling::ggml::qwen3 {
namespace {
const lib::EngineMelPolicy kMelPolicy = {
    lib::MelPolicy::T_FULLT_MINUS_1,
    /*cap_n_samples=*/false,
    /*cap_before_checks=*/false,
    lib::MelPolicy::MAX_ALL_FRAMES,
    /*norm_in_double=*/false,
    /*emit_bf16=*/true,
    "STARLING_QWEN3_MEL_DUMP",
    "QWEN3",
};
} // namespace

bool compute_log_mel(const Config& cfg, const ModelLoader& ml, const float* pcm,
                     size_t S, MelFeatures& out, std::string& err) {
    // min_length zero-pad (single-clip padding=True: no other raw pad).
    std::vector<float> padded;
    const size_t min_len = cfg.frontend.min_length;
    if (S < min_len) {
        padded.assign(min_len, 0.0f);
        if (S > 0) std::memcpy(padded.data(), pcm, S * sizeof(float));
        pcm = padded.data();
        S = min_len;
    }
    lib::MelOutput mo;
    if (!lib::compute_log_mel(lib::make_mel_policy(cfg.frontend, kMelPolicy),
                              ml, pcm, S, mo, err))
        return false;
    // mo is feat-major [128, T] (element (m, t) at m * T + t) — exactly the
    // [T, 128] ggml layout the encoder's chunk reshape wants (time
    // innermost). Zero-pad each mel row's time axis out to the chunk
    // multiple (the padded frames carry mel value 0.0).
    const int64_t M = (int64_t) mo.n_mels;
    const int64_t T = (int64_t) mo.n_frames;
    const int64_t chunk = 2 * (int64_t) cfg.frontend.n_window;
    const int64_t T_pad = (T + chunk - 1) / chunk * chunk;
    out.data.assign((size_t) M * T_pad, ggml_bf16_t{0});
    out.f32.assign((size_t) M * T_pad, 0.0f);
    for (int64_t m = 0; m < M; ++m) {
        std::memcpy(out.data.data() + (size_t) m * T_pad,
                    mo.bf16.data() + (size_t) m * T, (size_t) T * sizeof(ggml_bf16_t));
        for (int64_t t = 0; t < T; ++t)
            out.f32[(size_t) m * T_pad + t] = mo.f32[(size_t) m * T + t];
    }
    out.n_mels = M;
    out.n_frames = T_pad;
    out.valid_frames = T;
    return true;
}
} // namespace starling::ggml::qwen3
