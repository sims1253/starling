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

// The moss/ark/granite decode-stack differences. qkv_bias selects the
// projection family: true = Qwen2.5 trunk (biased q/k/v addressed by BASE name
// "attn.q" etc., no q_norm/k_norm); false = bias-free over full weight names
// "attn.q.weight" (Qwen3 trunk with per-head q_norm/k_norm, or granite with
// them off via qk_norm). The granite-family fields below default so the
// moss/ark graphs stay byte-identical: qk_norm is consulted only in the
// bias-free branch, the multipliers use skip-when-default (the op sequence is
// untouched), and tied_lm_head keeps llm.embed.weight as the head.

// MLP activation family: the stock silu-gated gate/up/down trunk, or
// Nemotron's plain squared-ReLU up/down MLP (no gate tensor at all).
enum class QwenMlpAct { kSiluGated, kRelu2Plain };

struct QwenDecodeSpec {
    bool qkv_bias;
    const char* env;           // env-var prefix, e.g. "STARLING_MOSS"
    const char* label;         // error/log label, e.g. "MOSS"
    const char* stage_prefix;  // L0-probe dump filename prefix
    bool qk_norm = true;       // per-head q_norm/k_norm (moss; granite: off)
    bool tied_lm_head = true;  // lm_head == llm.embed (granite is untied)
    float attention_scale = 0.0f;       // 0 -> 1/sqrt(head_dim) (granite: 0.0078125)
    float embedding_multiplier = 1.0f;  // hidden = embeds * m at prefill AND decode
    float residual_multiplier = 1.0f;   // residual + m*y, attn and mlp (granite: 0.22)
    float logits_scaling = 1.0f;        // logits / s after lm_head (granite: 8.0)
    // torch reads the lm_head output stored as bf16 and its argmax keeps the
    // FIRST index on exact ties; the raw f32 logits (host pick) and ggml's
    // CUDA argmax (warp-order ties) both disagree on such ties. When set, the
    // greedy picks round the logits to bf16 first; the host keeps the first
    // index on the exact ties that creates, and the K-step graph masks the
    // rounded logits by equality with their max and weights the masked
    // columns by a descending column iota, making the lowest tied column a
    // unique argmax (order-independent) (qwen3; off keeps moss/ark/granite
    // byte-identical).
    bool argmax_low_ties = false;
    // Nemotron's MLP is up -> relu(x)^2 -> down (F.relu(x).pow(2): relu is
    // exact, one bf16 round after the square), with no gate projection. The
    // default keeps the historical silu-gated sequence byte-identical.
    QwenMlpAct mlp_activation = QwenMlpAct::kSiluGated;
    // Nemotron normalizes with F.rms_norm: normalize AND affine in f32, ONE
    // bf16 round at the end. The default is the Llama-style two-round
    // discipline (round after the rsqrt, round again after the weight mul)
    // the stack was built on (moss/ark/granite/qwen3 byte-identity).
    bool rms_norm_single_round = false;
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
