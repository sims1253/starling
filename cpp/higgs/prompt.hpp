#pragma once
#include "audio_encoder.hpp"
#include <cstdint>
#include <string>
#include <vector>
namespace starling::ggml::higgs {
// The ChatML prompt: prefix + N <|AUDIO|> placeholders + suffix, where N is
// derived from the uncapped mel-frame count. audio_mask marks the slots that get
// clobbered by the projector audio embeddings (the <|AUDIO|> positions).
struct Prompt {
    std::vector<int32_t> ids;
    std::vector<uint8_t> audio_mask;
};
struct InputsEmbeds {
    std::vector<float> data;
    int64_t n_tokens = 0, width = 0;
};
// Build the transcribe prompt for `mel_frames` (the uncapped frame count). The
// prompt is the ChatML layout from src/starling/higgs/pipeline.py:
//   <|im_start|>user\n + instruction + " " + <|audio_bos|> + (<|AUDIO|> * N) +
//   <|audio_eos|> + \n + <|im_end|> + \n + <|im_start|> + assistant\n
// The prefix/suffix token-id arrays are baked in the GGUF (pre-tokenized by the
// converter, since the C++ side has a decoder-only tokenizer); N = audio_token_count.
Prompt build_transcribe_prompt(const Config& c, int64_t mel_frames);
// Look up embed_tokens and scatter the projector audio features into the audio
// slots. Handles the long-audio zero-pad / truncate alignment exactly like
// merge_input_ids_with_audio_features (the single <|AUDIO|> placeholder expands
// to audio_features_length tokens; overflow/underflow slots zero-fill).
bool build_inputs_embeds(const HiggsModel& m, const Prompt& p, const AudioEncoding& audio,
                         InputsEmbeds& out, std::string& err);
} // namespace starling::ggml::higgs
