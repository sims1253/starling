#pragma once
#include "adapter.hpp"
#include "lib/qwen_decode.hpp"
#include <cstdint>
#include <string>
#include <vector>
namespace starling::ggml::moss {
struct Prompt { std::vector<int32_t> ids; std::vector<uint8_t> audio_mask; };
using InputsEmbeds = lib::InputsEmbeds;
Prompt build_transcribe_prompt(const Config&, int64_t mel_frames);
bool build_inputs_embeds(const MossModel&, const Prompt&, const AudioEncoding& audio,
                         InputsEmbeds&, std::string& err);
}
