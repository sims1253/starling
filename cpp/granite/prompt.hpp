#pragma once
#include "encoder.hpp"
#include "lib/qwen_decode.hpp"
#include <cstdint>
#include <string>
#include <vector>
namespace starling::ggml::granite {
// The chat-templated prompt: prefix + N <|audio|> placeholders + suffix, where
// N comes from the raw sample count (mirroring
// GraniteSpeechFeatureExtractor._get_num_audio_features). audio_mask marks the
// slots that get clobbered by the projector's audio embeddings.
struct Prompt {
    std::vector<int32_t> ids;
    std::vector<uint8_t> audio_mask;
};
using InputsEmbeds = lib::InputsEmbeds;
// Build the transcribe prompt for `n_samples` PCM samples (16 kHz). The
// prefix/suffix token arrays are baked in the GGUF (captured from the HF
// processor under the reference tokenizer).
Prompt build_transcribe_prompt(const Config& c, int64_t n_samples);
// Look up embed_tokens and scatter the projector's audio embeddings into the
// audio slots — the byte-exact replica of get_merged_audio_embeddings (the
// Granite embedding multiplier is NOT applied here; the shared decode stack
// applies it to the whole merged tensor at prefill, exactly as stock).
bool build_inputs_embeds(const GraniteModel& m, const Prompt& p, const AudioEmbeds& audio,
                         InputsEmbeds& out, std::string& err);
} // namespace starling::ggml::granite
