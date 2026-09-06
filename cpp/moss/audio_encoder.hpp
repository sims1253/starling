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

// Fused encode_audio + apply_adapter. On GPU this is ONE captured graph keyed
// on the mel shape (C, tail); on CPU / under STARLING_MOSS_DEBUG it is the
// one-shot encode_audio + apply_adapter pair. Output is the adapter embedding
// ([adapter_output, A] f32). C API surface is unchanged.
bool encode_audio_and_adapt(const MossModel& model, const MelFeatures& mel,
                            AudioEncoding& out, std::string& err);

// Current number of cached fused encoder+adapter graphs (diagnostic + the Wave H
// bounded-LRU regression-test hook). Zero on CPU / before first GPU encode.
size_t encoder_replay_cache_size(const MossModel& model);
} // namespace starling::ggml::moss
