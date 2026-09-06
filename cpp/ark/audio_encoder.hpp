#pragma once
#include "loader.hpp"
#include "mel.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace starling::ggml::ark {
// The adapter output: [adapter_output, N] f32 (the Qwen2.5 audio embeddings).
struct AudioEncoding {
    // run_graph converts the BF16 graph output to f32 on readback; values are
    // exactly BF16-representable.
    std::vector<float> data;
    int64_t n_tokens = 0;
    int64_t width = 0;
};

// Fused Whisper encoder + LayerNorm + MLP adapter. On GPU this is ONE captured
// ReplayGraph keyed on the mel length (global attention -> one graph per T_enc,
// LRU-bounded); on CPU / debug it is the one-shot encode + adapt pair.
bool encode_audio_and_adapt(const ArkModel& model, const MelFeatures& mel,
                            AudioEncoding& out, std::string& err);

// Current number of cached fused encoder graphs (diagnostic). Zero on CPU /
// before first GPU encode.
size_t encoder_replay_cache_size(const ArkModel& model);
} // namespace starling::ggml::ark
