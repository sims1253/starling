// mel.hpp — Whisper log-mel frontend for Hojo-ASR-V1.
//
// Byte-exact replica of the WhisperFeatureExtractor (n_fft=400, hop=160,
// 128 bins, reflect pad, Hann window, power=2, log10, then
// max-clamp(dynamic_range=8) and normalize (x+4)/4) that hojo_asr's dataset
// wraps. This is a verbatim copy of higgs/mel.cpp (the two frontends are
// byte-exact except the namespace); Hojo and Higgs share the same Whisper
// extractor + constants. Output is feat-major [n_mels, n_frames] f32.
#pragma once
#include "config.hpp"
#include "runtime/model_loader.hpp"
#include "ggml.h"
#include <cstddef>
#include <string>
#include <vector>

namespace starling::ggml::hojo {
struct MelFeatures {
    std::vector<float> data;   // feat-major [n_mels, n_frames] f32
    int64_t n_mels = 0, n_frames = 0;
};
// Whisper log-mel. Mirrors higgs/ark/moss (identical frontend).
bool compute_log_mel(const Config&, const ModelLoader&, const float* pcm, size_t n,
                     MelFeatures&, std::string& err);
} // namespace starling::ggml::hojo
