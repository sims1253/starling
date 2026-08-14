// qwen_decode.hpp — the shared Qwen-trunk text decoder stack (whole-model
// prefill/decode graphs, device-resident KV cache, captured K-step greedy
// generation) plus the state types the model bundles re-export.
//
// A model binds the stack through a QwenDecodeCtx: a QwenDecodeSpec (the only
// places the two trunks differ — projection family, env-var prefix, message
// label, probe filename prefix), the ModelLoader holding the realized weights,
// and the config dims the graphs are shaped by. The process-global caches
// (device KV, per-S prefill graphs, per-K K-step graphs) are singletons keyed
// by spec: one set per model, first requester sizes them.
#pragma once

#include "runtime/model_loader.hpp"
#include "ggml.h"

#include <cstdint>
#include <string>
#include <vector>

namespace starling::ggml::lib {

// The moss/ark decode-stack differences. qkv_bias selects the projection
// family: true = Qwen2.5 trunk (biased q/k/v addressed by BASE name "attn.q"
// etc., no q_norm/k_norm); false = Qwen3 trunk (bias-free over full weight
// names "attn.q.weight", per-head q_norm/k_norm after the reshape).
struct QwenDecodeSpec {
    bool qkv_bias;
    const char* env;           // env-var prefix, e.g. "STARLING_MOSS"
    const char* label;         // error/log label, e.g. "MOSS"
    const char* stage_prefix;  // L0-probe dump filename prefix
};

// The config fields the decode graphs are shaped by.
struct QwenLlmDims {
    uint32_t n_layers = 0, hidden = 0, n_heads = 0, n_kv_heads = 0;
    uint32_t head_dim = 0, max_cache = 0;
    float rope_theta = 0.0f, rms_norm_eps = 0.0f;
};

// One model's binding of the decode stack.
struct QwenDecodeCtx {
    const QwenDecodeSpec& spec;
    const ModelLoader& loader;
    QwenLlmDims dims;
};

// Host-side KV mirror (probe path only).
struct LayerKvCache { std::vector<ggml_bf16_t> k, v; };
struct LlmState { std::vector<LayerKvCache> layers; int64_t length = 0; };
// inputs_embeds, [hidden, n_tokens] f32 token-major (column per token).
struct InputsEmbeds { std::vector<float> data; int64_t n_tokens = 0, width = 0; };
struct PrefillResult { std::vector<float> logits; int32_t first_token = -1; LlmState state; };
struct GenerateResult {
    std::vector<int32_t> ids;
    bool hit_eos = false;
    std::vector<float> prefill_logits;
};
// greedy_generate controls. Model bundles keep their own GenerateOptions
// (default max_cache_len differs per model) and convert.
struct GenerateParams { int32_t max_new_tokens; int32_t max_cache_len; int32_t eos_token_id; };

bool llm_prefill(const QwenDecodeCtx& m, const InputsEmbeds& i, int32_t max_cache_len,
                 PrefillResult& o, std::string& e);
bool greedy_generate(const QwenDecodeCtx& m, const InputsEmbeds& i, const GenerateParams& op,
                     GenerateResult& o, std::string& e);
// Number of captured per-S prefill graphs for this spec (diagnostic + the
// bounded-LRU regression-test hook). Zero on CPU / before first GPU prefill.
size_t prefill_replay_cache_size(const QwenDecodeSpec& spec);

} // namespace starling::ggml::lib
