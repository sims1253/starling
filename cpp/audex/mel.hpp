// mel.hpp — whisper-style log-mel frontend for Nemotron-Labs-Audex-2B.
#pragma once

#include "config.hpp"
#include "runtime/model_loader.hpp"
#include "ggml.h"

#include <cstddef>
#include <string>
#include <vector>

namespace starling::ggml::audex {

struct MelFeatures {
    std::vector<ggml_bf16_t> data;  // FEAT-major [n_mels * n_frames], time innermost
    int64_t n_mels = 0, n_frames = 0, valid_frames = 0;
};

// Every clip is zero-padded (or truncated) to frontend.n_samples = 480000
// samples first (the extractor's padding="max_length"), so the output is
// ALWAYS exactly 3000 frames — the fixed encoder input shape.
bool compute_log_mel(const Config& cfg, const ModelLoader& ml, const float* pcm,
                     size_t S, MelFeatures& out, std::string& err);

} // namespace starling::ggml::audex
