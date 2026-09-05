// prompt.hpp — Voxtral offline prompt + PCM pad rule (Phase 1).
//
// The prompt is the baked prefix alone: mistral-common's offline
// encode_streaming_tokens emits [BOS] + 38 streaming-pad (32 left-pad + 6
// delay), verified against the stock processor on all three fixtures (P == 39,
// every id after BOS is 32). Audio injection is ADDITIVE (inputs_embeds +=
// audio_embeds at every position of the current forward slice), so there are
// no placeholder slots to mask. Phase 2 consumes these helpers in the mel ->
// encode -> prefill path.
#pragma once

#include "loader.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace starling::ggml::voxtral {

// Prompt token ids for a request: the baked prefix verbatim.
inline std::vector<int32_t> build_transcribe_prompt(const Config& c) {
    return c.prompt_prefix;
}

// Stock total-length bound: ceil(mel_T / 8) (from _prepare_generation_config:
// num_audio_tokens = ceil(mel_len / audio_length_per_tok), used as the default
// max_length and hard clamp). Offline the exact conv-chain token count is
// mel_T/8, one under this bound.
inline int64_t stock_max_length(int64_t mel_T) {
    if (mel_T <= 0) return 1;
    return (mel_T + 8 - 1) / 8;
}

// Total-sequence-length cap mirroring the stock logic: the stock bound, with
// an optional user max_new_tokens budget clamped under it.
inline int64_t generation_cap(int64_t prompt_len, int64_t mel_T,
                              int64_t max_new_tokens = -1) {
    const int64_t bound = stock_max_length(mel_T);
    if (max_new_tokens < 0) return bound;
    const int64_t want = prompt_len + (max_new_tokens > 0 ? max_new_tokens : 0);
    return want < bound ? want : bound;
}

} // namespace starling::ggml::voxtral
