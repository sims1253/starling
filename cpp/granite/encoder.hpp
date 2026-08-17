#pragma once
#include "loader.hpp"
#include "mel.hpp"
#include "projector.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace starling::ggml::granite {

// The fused encoder + projector output: [output_dim, N] f32 (the Granite
// decoder's audio embeddings), column per audio token.
struct AudioEmbeds {
    std::vector<float> data;
    int64_t n_tokens = 0;
    int64_t width = 0;
};

// Fused CTC-conformer encoder + BLIP2 Q-Former projector. On GPU this is ONE
// captured ReplayGraph keyed on the stacked-mel length (block-local attention
// => one graph per T; the cache is LRU-bounded); on CPU / debug it is the
// one-shot build. STARLING_GRANITE_DUMP_ENC=<file> additionally dumps the
// encoder's last hidden state (f32 [hidden, T]) for divergence localization.
bool encode_audio_and_project(const GraniteModel& model, const MelFeatures& mel,
                              AudioEmbeds& out, std::string& err);

// Current number of cached fused encoder graphs (diagnostic). Zero on CPU /
// before first GPU encode.
size_t encoder_replay_cache_size();

} // namespace starling::ggml::granite
