#pragma once
#include "loader.hpp"
#include "mel.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace starling::ggml::moss {
struct AudioEncoding {
    // run_graph converts the BF16 graph output to f32 on readback. Values are
    // nevertheless exactly representable BF16 values.
    std::vector<float> data;
    int64_t n_tokens = 0;
    int64_t width = 0;
};

// The processor/module deepstack length function from spec section 2.3.
int64_t audio_token_length(int64_t mel_frames);

bool encode_audio(const MossModel& model, const MelFeatures& mel,
                  AudioEncoding& out, std::string& err);
} // namespace starling::ggml::moss
