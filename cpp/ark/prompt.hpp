#pragma once
#include "audio_encoder.hpp"
#include <cstdint>
#include <string>
#include <vector>
namespace starling::ggml::ark {
// The chat-templated prompt: prefix + N <|audio|> placeholders + suffix, where
// N is derived from the uncapped mel-frame count. audio_mask marks the slots
// that get clobbered by the adapter audio embeddings.
struct Prompt {
    std::vector<int32_t> ids;
    std::vector<uint8_t> audio_mask;
};
struct InputsEmbeds {
    std::vector<float> data;
    int64_t n_tokens = 0, width = 0;
};
// Build the transcribe prompt for `mel_frames` (the uncapped frame count). The
// prefix/suffix + N are baked in the GGUF (empirically captured from the HF
// processor: <|user|><|begin_of_audio|> + <|audio|>*N + <|end_of_audio|> +
// instruction + <|assistant|>).
Prompt build_transcribe_prompt(const Config& c, int64_t mel_frames);
// Look up embed_tokens and scatter the adapter audio features into the audio
// slots. Handles the long-audio zero-pad / truncate alignment exactly like
// modeling_arkasr._inject_audio_embeddings.
bool build_inputs_embeds(const ArkModel& m, const Prompt& p, const AudioEncoding& audio,
                         InputsEmbeds& out, std::string& err);
} // namespace starling::ggml::ark
