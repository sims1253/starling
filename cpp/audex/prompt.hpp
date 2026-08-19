// prompt.hpp — audex prompt construction + audio-embedding scatter injection.
#pragma once

#include "config.hpp"
#include "encoder.hpp"
#include "loader.hpp"
#include "lib/qwen_decode.hpp"

#include <cstdint>
#include <vector>

namespace starling::ggml::audex {

struct Prompt {
    std::vector<int32_t> ids;
    std::vector<uint8_t> audio_mask;  // 1 where an audio embedding goes
};

using InputsEmbeds = lib::InputsEmbeds;

// The ChatML layout baked into the GGUF: prefix + <so_embedding> x 750 +
// suffix (the audio-slot count is FIXED per 30 s clip, unlike qwen3's
// sample-count formula).
Prompt build_transcribe_prompt(const Config& c);

bool build_inputs_embeds(const AudexModel& m, const Prompt& p, const AudioEmbeds& a,
                         InputsEmbeds& out, std::string& err);

} // namespace starling::ggml::audex
