#pragma once
#include "config.hpp"
#include "runtime/model_loader.hpp"
#include "ggml.h"
#include <cstddef>
#include <string>
#include <vector>

namespace starling::ggml::moss {
struct MelFeatures { std::vector<ggml_bf16_t> data; std::vector<float> f32; int64_t n_mels=0,n_frames=0; };
bool compute_log_mel(const Config&, const ModelLoader&, const float* pcm, size_t n, MelFeatures&, std::string& err);
} // namespace starling::ggml::moss
