// embed_scatter.hpp — prompt embedding lookup + audio-feature scatter.
// (moss is not covered: its variant is the compact exact-count form with no
// zero-pad path.)
#pragma once

#include "runtime/backend.hpp"
#include "runtime/graph.hpp"
#include "ggml.h"
#include <cstdint>
#include <string>
#include <vector>

namespace starling::ggml::lib {

// Token-embedding lookup over `ids`, then overwrite each audio slot (mask
// != 0) with the corresponding audio feature row. When the audio path emits
// fewer features than there are slots (long audio, mel capped), the overflow
// slots are ZEROED — matching the HF scatter (zero-init then overwrite),
// which writes a literal zero, NOT the embedded placeholder id the lookup
// left there. Returns the f32 embeds [ids.size() * hidden].
bool embed_and_scatter_audio(const ModelLoader& ml, int64_t hidden,
                             const std::vector<int32_t>& ids,
                             const std::vector<uint8_t>& mask,
                             const float* audio_data, size_t audio_len,
                             int64_t audio_width, int64_t audio_tokens,
                             std::vector<float>& out, const char* label,
                             std::string& err);

} // namespace starling::ggml::lib
