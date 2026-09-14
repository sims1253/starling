#pragma once
#include "prompt.hpp"
#include <cstdint>
#include <string>
#include <vector>
namespace starling::ggml::higgs {
struct GenerateOptions {
    int32_t max_new_tokens = 200;
    int32_t max_cache_len = 4096;
    // Primary EOS (151643 <|endoftext|>). ASR also stops on <|im_end|> (151645) —
    // see greedy_generate; eos_token_id is the canonical stop, im_end_id is the
    // secondary stop from config.im_end_id.
    int32_t eos_token_id = 151643;
    int32_t im_end_id = 151645;
};
struct GenerateResult {
    std::vector<int32_t> ids;
    bool hit_eos = false;
    std::vector<float> prefill_logits;
};
// Greedy-decode the merged inputs_embeds through the Qwen3 trunk. Prefill runs
// as a captured per-S ReplayGraph (GPU) / one-shot graph (CPU); decode runs as a
// K-step captured multistep graph (GPU) or one-step graphs (CPU). Stops on EITHER
// eos_token_id or im_end_id (matches EOS_TOKEN_IDS in config.py).
bool greedy_generate(const HiggsModel& m, const InputsEmbeds& inputs,
                     const GenerateOptions& op, GenerateResult& out, std::string& err);
// Current number of cached per-S prefill graphs (diagnostic). Zero on CPU /
// before first GPU prefill.
size_t prefill_replay_cache_size(const HiggsModel& model);
} // namespace starling::ggml::higgs
