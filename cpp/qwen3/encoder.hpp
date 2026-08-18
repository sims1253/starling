#pragma once
#include "loader.hpp"
#include "mel.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace starling::ggml::qwen3 {

// The fused encoder + projector output: [output_dim, L] f32 (the Qwen3
// decoder's audio embeddings), column per packed audio token.
struct AudioEmbeds {
    std::vector<float> data;
    int64_t n_tokens = 0;
    int64_t width = 0;
};

// Fused windowed-attention conv encoder + MLP projector. On GPU this is ONE
// captured ReplayGraph keyed on (padded mel length, packed length) — both
// determine the whole windowing structure (the cache is LRU-bounded); on CPU
// / debug it is the one-shot build. STARLING_QWEN3_DUMP_ENC=<file>
// additionally dumps the encoder's last hidden state (f32 [hidden, L]) for
// divergence localization.
bool encode_audio_and_project(const Qwen3Model& model, const MelFeatures& mel,
                              AudioEmbeds& out, std::string& err);

// Current number of cached fused encoder graphs (diagnostic). Zero on CPU /
// before first GPU encode.
size_t encoder_replay_cache_size();

} // namespace starling::ggml::qwen3
