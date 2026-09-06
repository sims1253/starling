// mel.hpp — Voxtral log-mel frontend: offline pad + shared Whisper frontend.
//
// The offline streaming-pad (prompt.hpp's rule: ceil to whole 1280-sample
// audio tokens plus the 32 left + 17 right pad tokens) owns the length, so
// no n_samples truncation. The shared frontend runs its stock STFT semantics
// (reflect pad, center=True -> fullT frames, then the T_FULLT_MINUS_1 rule
// drops the trailing TIME frame, reproducing the extractor's stft[..., :-1]);
// the only voxtral policy bit is the FIXED global log-mel max 1.5 replacing
// the computed per-utterance max (streaming-safe normalization).
#pragma once

#include "config.hpp"
#include "runtime/model_loader.hpp"
#include "ggml.h"
#include <cstddef>
#include <string>
#include <vector>

namespace starling::ggml::voxtral {

struct MelFeatures {
    std::vector<ggml_bf16_t> data;  // bf16 copy, feat-major [n_mels, n_frames]
    std::vector<float> f32;         // f32 copy, same layout
    int64_t n_mels = 0, n_frames = 0;
};

// Offline-pad `pcm[0,S)` per offline_padded_samples, run the shared frontend
// with the fixed-max policy, and return the mel. The mel constants
// (audio.mel_window / audio.mel_filters) are synthesized into the loader when
// absent (see mel.cpp); the real GGUF carries none.
bool compute_log_mel(const Config& cfg, const ModelLoader& ml, const float* pcm,
                     size_t S, MelFeatures& out, std::string& err);

} // namespace starling::ggml::voxtral
