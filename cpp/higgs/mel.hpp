#pragma once
#include "config.hpp"
#include "runtime/model_loader.hpp"
#include "ggml.h"
#include <cstddef>
#include <string>
#include <vector>

namespace starling::ggml::higgs {
struct MelFeatures {
    std::vector<ggml_bf16_t> data;
    std::vector<float> f32;
    int64_t n_mels = 0, n_frames = 0;
};
// Whisper log-mel (n_fft=400, hop=160, 128 bins), feat-major
// [n_mels, n_frames] bf16 (and f32); implemented via lib/whisper_mel.hpp with
// the policy set in mel.cpp. Constants come from
// FrontendConfig.
bool compute_log_mel(const Config&, const ModelLoader&, const float* pcm, size_t n,
                     MelFeatures&, std::string& err);
} // namespace starling::ggml::higgs
