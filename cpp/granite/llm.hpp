#pragma once
#include "prompt.hpp"
#include "lib/qwen_decode.hpp"
#include <cstdint>
#include <string>
#include <vector>
namespace starling::ggml::granite {
using LayerKvCache = lib::LayerKvCache;
using LlmState = lib::LlmState;
using PrefillResult = lib::PrefillResult;
using GenerateResult = lib::GenerateResult;
// Defaults are model-specific (max_cache_len 640 for granite-speech).
struct GenerateOptions {
    int32_t max_new_tokens = 200, max_cache_len = 640, eos_token_id = 100257;
};
bool llm_prefill(const GraniteModel&, const InputsEmbeds&, int32_t max_cache_len,
                 PrefillResult&, std::string&);
bool greedy_generate(const GraniteModel&, const InputsEmbeds&, const GenerateOptions&,
                     GenerateResult&, std::string&);
// Current number of cached per-S prefill graphs (diagnostic + the
// bounded-LRU regression-test hook). Zero on CPU / before first GPU prefill.
size_t prefill_replay_cache_size();
} // namespace starling::ggml::granite
