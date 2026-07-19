#pragma once
#include "audio_encoder.hpp"
namespace starling::ggml::moss {
bool apply_adapter(const MossModel& model, const AudioEncoding& input,
                   AudioEncoding& out, std::string& err);
} // namespace starling::ggml::moss
