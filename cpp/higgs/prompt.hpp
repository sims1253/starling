#pragma once
#include "audio_encoder.hpp"
#include <cstdint>
#include <string>
#include <vector>
namespace starling::ggml::higgs {
// The ChatML prompt: head + one <|audio_bos|>..(<|AUDIO|>*N_k)..|audio_eos|>
// segment PER audio chunk + tail. audio_mask marks the slots that get clobbered
// by the projector audio embeddings (the <|AUDIO|> positions). chunk_tokens[k]
// is the number of audio tokens chunk k contributes (from its projector output).
struct Prompt {
    std::vector<int32_t> ids;
    std::vector<uint8_t> audio_mask;
};
struct InputsEmbeds {
    std::vector<float> data;
    int64_t n_tokens = 0, width = 0;
};
// Build the transcribe prompt with one audio segment per chunk. The prompt is
// the ChatML layout from src/starling/higgs/pipeline.py + the collator's
// per-chunk <|audio_bos|>...<|audio_eos|> expansion:
//   <|im_start|>user\n + instruction +
//     (<|audio_bos|> + (<|AUDIO|> * N_k) + <|audio_eos|>)*num_chunks +
//   <|im_end|>\n<|im_start|>assistant\n
// The head/tail text is pre-tokenized in the GGUF (higgs.prompt_prefix /
// higgs.prompt_suffix); build_transcribe_prompt splits the audio boundary tokens
// out of those baked arrays so it re-emits them once per chunk.
Prompt build_transcribe_prompt(const Config& c, const std::vector<int64_t>& chunk_tokens);
// Look up embed_tokens and scatter the projector audio features into the audio
// slots. Handles the long-audio zero-pad / truncate alignment exactly like
// merge_input_ids_with_audio_features (the single <|AUDIO|> placeholder expands
// to audio_features_length tokens; overflow/underflow slots zero-fill).
bool build_inputs_embeds(const HiggsModel& m, const Prompt& p, const AudioEncoding& audio,
                         InputsEmbeds& out, std::string& err);
} // namespace starling::ggml::higgs
