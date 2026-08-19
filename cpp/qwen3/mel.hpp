#pragma once
#include "config.hpp"
#include "runtime/model_loader.hpp"
#include "ggml.h"
#include <cstddef>
#include <string>
#include <vector>

namespace starling::ggml::qwen3 {
struct MelFeatures {
    // Chunk-padded mel, FEAT-major: element (m, t) at m*n_frames + t (time
    // innermost — the [T_pad, 128] ggml layout), bf16. n_frames is the padded
    // frame count (multiple of 100); valid_frames is the real frame count T
    // before the zero pad.
    std::vector<ggml_bf16_t> data;
    int64_t n_mels = 128, n_frames = 0, valid_frames = 0;
};
// Whisper-style log-mel (n_fft=400, hop=160, 128 bins, drop-last-frame rule)
// + the qwen3 pre/post steps: zero-pad the waveform to min_length samples,
// then right-pad the mel axis with ZEROS to a multiple of 2*n_window frames.
// Mirrors Qwen3ASRFeatureExtractor.__call__ for a single clip (padding=True:
// no batch-longest raw pad beyond min_length).
bool compute_log_mel(const Config&, const ModelLoader&, const float* pcm, size_t n,
                     MelFeatures&, std::string& err);
} // namespace starling::ggml::qwen3
