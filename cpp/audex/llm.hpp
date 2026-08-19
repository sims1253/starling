// llm.hpp — audex text-decoder binding over the shared qwen_decode stack.
#pragma once

#include "loader.hpp"
#include "prompt.hpp"
#include "lib/qwen_decode.hpp"

#include <cstddef>
#include <cstdint>

namespace starling::ggml::audex {

// Re-exports of the shared decode-stack state types (see lib/qwen_decode.hpp).
using lib::LayerKvCache;
using lib::LlmState;
using lib::PrefillResult;
using lib::GenerateResult;

struct GenerateOptions {
    int32_t max_new_tokens = 200;
    int32_t max_cache_len = 4096;
    int32_t eos_token_id = 11;  // <|im_end|> — the serving path's stop
};

bool llm_prefill(const AudexModel& m, const InputsEmbeds& i, int32_t max_cache_len,
                 PrefillResult& o, std::string& e);
bool greedy_generate(const AudexModel& m, const InputsEmbeds& i,
                     const GenerateOptions& op, GenerateResult& o, std::string& e);
size_t prefill_replay_cache_size();

} // namespace starling::ggml::audex
