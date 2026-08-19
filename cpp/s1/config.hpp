// config.hpp — S1-mini architecture constants + prompt contract.
//
// S1-mini (superwhisper/s1-mini) is a pure text-to-text normalizer: a
// decoder-only Qwen3-0.6B (28 layers, hidden 1024, GQA 16Q/8KV, head_dim 128,
// tied embeddings, per-head q/k norm) with NO audio front-end. Defaults mirror
// the baked GGUF metadata (scripts/convert_s1_gguf.py); the loader overrides
// from `s1.*` KV keys.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace starling::ggml::s1 {

struct LlmConfig {
    uint32_t n_layers = 28, hidden = 1024, n_heads = 16, n_kv_heads = 8;
    uint32_t head_dim = 128, intermediate = 3072, vocab = 151936;
    uint32_t max_position_embeddings = 40960, max_cache = 4096;
    float rope_theta = 1000000.0f, rms_norm_eps = 1e-6f;
    bool tied_embeddings = true, has_qk_norm = true;
};

struct Config {
    LlmConfig llm;

    // Stop ids from generation_config.json: stop on <|im_end|> (151645) OR
    // <|endoftext|> (151643, also the pad token).
    int32_t eos_token_id = 151645;
    int32_t eos2_token_id = 151643;
    int32_t pad_token_id = 151643;

    // Trained input/output contract (model card): prompts up to ~1000 tokens;
    // greedy budget 1.3 * input + 32.
    int32_t max_input_tokens = 1000;
    float max_new_tokens_input_factor = 1.3f;
    float max_new_tokens_fixed = 32.0f;

    // Control-space values the engine validates against (values outside the
    // trained sets make the model hallucinate — reject them).
    std::vector<std::string> styling_values = {"casual", "semi-casual", "semi-formal", "formal"};
    std::vector<std::string> structure_values = {"prose", "lists"};
    std::vector<std::string> context_values = {"general", "email"};

    // Chat-template layout baked in the GGUF: prefix ends after "user\n",
    // suffix starts at the <|im_end|> after the transcript (includes the
    // enable_thinking=False assistant prefix "<think>\n\n</think>\n\n").
    std::vector<int32_t> prompt_prefix, prompt_suffix;
};

} // namespace starling::ggml::s1
