#pragma once
#include "loader.hpp"
#include "mel.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace starling::ggml::higgs {
// The projector output: [output, N] f32 (the Qwen3 audio embeddings, width=2048).
struct AudioEncoding {
    // run_graph converts the BF16 graph output to f32 on readback; values are
    // exactly BF16-representable.
    std::vector<float> data;
    int64_t n_tokens = 0;
    int64_t width = 0;
};

// Fused Whisper encoder + ln_post + avg_pool + MLP projector. On GPU this is ONE
// captured ReplayGraph keyed on the mel length (global attention -> one graph per
// mel_T, LRU-bounded); on CPU / debug it is the one-shot encode + project pair.
bool encode_audio_and_project(const HiggsModel& model, const MelFeatures& mel,
                              AudioEncoding& out, std::string& err);

// Read a weight tensor as f32 (BF16/F32 only; any other dtype fails loudly:
// empty return + err set — never a silent zero-fill).
std::vector<float> read_tensor_to_f32(const ModelLoader& ml, const char* name,
                                      std::string& err);

// Current number of cached fused encoder graphs (diagnostic). Zero on CPU /
// before first GPU encode.
size_t encoder_replay_cache_size(const HiggsModel& model);
} // namespace starling::ggml::higgs
