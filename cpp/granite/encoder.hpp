#pragma once
#include "loader.hpp"
#include "mel.hpp"
#include "projector.hpp"
#include <cstdint>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;

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

// Offline/native parity path for the optional BPE CTC head. Returns tokenizer
// IDs (CTC label 0 is blank; all other labels are mapped to label - 1).
// This does not change greedy transcription or its graph cache. A one-shot
// encoder graph returns the pre-feedback mid CTC logits and final hidden
// state, then projects the importance-weighted 4-frame pools.
bool extract_ctc_draft(const GraniteModel& model, const MelFeatures& mel,
                       std::vector<int32_t>& token_ids, std::string& err);

// Opt-in shared-encoder path for speculative transcription. One explicit
// graph output contains the CTC intermediate/final bundle and projected
// audio embeddings. The optional head then produces draft IDs. This avoids
// rerunning the 16-layer encoder; the ordinary greedy graph stays unchanged.
bool encode_audio_project_and_extract_ctc(const GraniteModel& model,
                                          const MelFeatures& mel,
                                          AudioEmbeds& audio,
                                          std::vector<int32_t>& token_ids,
                                          std::string& err,
                                          double* stage_ms = nullptr);

// Row-wise argmax with torch's first-index behavior on exact ties. `iota`
// holds vocab-index as descending f32 values (vocab <= 2^24). `iota` and `one`
// back graph inputs by address, so both must outlive graph execution.
ggml_tensor* ctc_argmax_first(ggml_context* c, ggml_tensor* logits,
                              const std::vector<float>& iota, const float& one);

// Current number of cached fused encoder graphs (diagnostic). Zero on CPU /
// before first GPU encode.
size_t encoder_replay_cache_size(const GraniteModel& model);

} // namespace starling::ggml::granite
