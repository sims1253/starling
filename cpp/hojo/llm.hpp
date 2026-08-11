#pragma once
#include "prompt.hpp"
#include <cstdint>
#include <string>
#include <vector>
namespace starling::ggml::hojo {
struct GenerateOptions {
    int32_t max_new_tokens = 200;
    int32_t max_cache_len = 4096;
    int32_t eos_token_id = 151645;
    int32_t pad_token_id = 151645;
    uint32_t num_beams = 4;
    double repetition_penalty = 2.0;
    double length_penalty = 1.0;
    uint32_t min_length = 1;
};
struct GenerateResult {
    std::vector<int32_t> ids;   // winning beam's token ids (incl. eos if stopped)
    bool hit_eos = false;
    std::vector<float> prefill_logits;
};
// Beam-4 decode the inputs_embeds through the Qwen3-4B trunk (qk_norm, SEPARATE
// lm_head). Prefill runs as a one-shot graph; decode runs a per-step graph for
// each active beam with KV-cache reordering (correctness first). Matches HF's
// beam search (num_beams, repetition_penalty, length_penalty, do_sample=False).
bool beam_generate(const HojoModel& m, const InputsEmbeds& inputs,
                   const GenerateOptions& op, GenerateResult& out, std::string& err);
// Greedy fallback (used if num_beams == 1 or for diagnostics). Same Qwen3 trunk.
bool greedy_generate(const HojoModel& m, const InputsEmbeds& inputs,
                     const GenerateOptions& op, GenerateResult& out, std::string& err);
} // namespace starling::ggml::hojo
