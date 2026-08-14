#pragma once
#include "conformer.hpp"
#include <cstdint>
#include <string>
#include <vector>
namespace starling::ggml::hojo {
struct InputsEmbeds {
    std::vector<float> data;   // [hidden, n_tokens] bf16-as-f32 (cast at boundary)
    int64_t n_tokens = 0, width = 0;
};
// Build inputs_embeds = cat([embed_tokens(bos_id), ln_speech(bottleneck)], dim=1).
// The bos token (151644 <|im_start|>) embedding is prepended to the speech
// embeddings. NO text prompt, NO audio placeholder. Then ln_speech is applied
// to the bottleneck output. Returns the f32 embeds [hidden, n_tokens].
bool build_inputs_embeds(const HojoModel& m, const BottleneckOutput& bn,
                         InputsEmbeds& out, std::string& err);
} // namespace starling::ggml::hojo
