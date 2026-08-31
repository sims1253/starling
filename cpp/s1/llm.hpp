// llm.hpp — S1-mini text decoder on the shared Qwen-trunk decode stack.
#pragma once

#include "config.hpp"
#include "loader.hpp"
#include "lib/qwen_decode.hpp"

#include <string>

namespace starling::ggml::s1 {

struct GenerateOptions {
    int32_t max_new_tokens = 0;   // caller computes the 1.3*T + 32 budget
    int32_t max_cache_len = 4096;
    int32_t eos_token_id = 151645;
    int32_t eos2_token_id = 151643;
};

bool llm_prefill(const S1Model& m, const lib::InputsEmbeds& i, int32_t maxc,
                 lib::PrefillResult& o, std::string& e);
bool greedy_generate(const S1Model& m, const lib::InputsEmbeds& i,
                     const GenerateOptions& op, lib::GenerateResult& o,
                     std::string& e);
size_t prefill_replay_cache_size();

} // namespace starling::ggml::s1
