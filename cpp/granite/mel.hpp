#pragma once
#include "config.hpp"
#include "runtime/model_loader.hpp"
#include "ggml.h"
#include <cstddef>
#include <string>
#include <vector>

namespace starling::ggml::granite {
struct MelFeatures {
    // Stacked frames: element (f, t) at t*160 + f (frame-major rows), bf16 +
    // f32. n_mels is 2x the frontend's 80 (the pair stack).
    std::vector<ggml_bf16_t> data;
    std::vector<float> f32;
    int64_t n_mels = 160, n_frames = 0;
};
// torchaudio log-mel (n_fft=512, hop=160, 80 bins) + the granite odd-frame
// drop and consecutive-pair stack (80 -> 160 dims). Mirrors
// GraniteSpeechFeatureExtractor._extract_mel_spectrograms.
bool compute_log_mel(const Config&, const ModelLoader&, const float* pcm, size_t n,
                     MelFeatures&, std::string& err);
} // namespace starling::ggml::granite
