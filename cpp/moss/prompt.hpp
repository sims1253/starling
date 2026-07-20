#pragma once
#include "adapter.hpp"
#include <cstdint>
#include <string>
#include <vector>
namespace starling::ggml::moss {
struct Prompt { std::vector<int32_t> ids; std::vector<uint8_t> audio_mask; };
struct InputsEmbeds { std::vector<float> data; int64_t n_tokens=0, width=0; };
Prompt build_transcribe_prompt(const Config&, int64_t mel_frames);
bool build_inputs_embeds(const MossModel&, const Prompt&, const AudioEncoding& audio,
                         InputsEmbeds&, std::string& err);
}
