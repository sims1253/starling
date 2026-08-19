// encoder.hpp — Nemotron-Labs-Audex-2B audio encoder + projector binding.
#pragma once

#include "loader.hpp"
#include "mel.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace starling::ggml::audex {

struct AudioEmbeds {
    std::vector<float> data;  // [width * n_tokens], feature-major (LLM hidden)
    int64_t n_tokens = 0, width = 0;
};

// Fixed-shape fused path: [3000, 128] bf16 mel -> conv frontend -> +pos -> 32
// full-attention layers -> avg-pool (750) -> ln_post -> projector -> f32
// [2048, 750] audio embeddings. GPU: one captured ReplayGraph (LRU-bounded);
// CPU / debug / stage probes: the one-shot build.
bool encode_audio_and_project(const AudexModel& model, const MelFeatures& mel,
                              AudioEmbeds& out, std::string& err);

// Number of captured encoder graphs (diagnostic + bounded-LRU test hook).
size_t encoder_replay_cache_size();

} // namespace starling::ggml::audex
