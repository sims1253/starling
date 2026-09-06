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
    // Generation suppression: sorted token ids banned from greedy picks (the
    // model card's bad_words_ids constraint: special and codec ids the
    // reference never emits). Appended last so existing positional field
    // initializers (audex sets mlp_activation) stay valid. Points into the
    // owning model's config storage, which outlives inference; nullptr/0
    // disables suppression: no masking branch runs and the graphs keep their
    // exact historical op sequence. The spec ADDRESS keys the process-global
    // decode caches, so a patched spec must live in the model, not a temp.
    const int32_t* banned_ids = nullptr;
    size_t n_banned = 0;
    // Voxtral's AdaRMSNorm (MLP branch only): h = h * (1 + fc2(gelu(fc0(t_cond))))
    // per layer, recomputed in-graph from the baked llm.t_cond leaf every
    // forward (t_cond is fixed per utterance, so this matches the stock
    // per-step recompute with no host round-trip). Off keeps every other
    // engine's graph byte-identical: no ada branch runs. The suffixes name
    // the per-layer ada weights under the layer prefix ("llm.blk.<i>." +
    // suffix).
    bool ada_rms_norm = false;
    const char* ada_fc0_suffix = nullptr;
    const char* ada_fc2_suffix = nullptr;
    // Voxtral's additive audio injection: the decode step adds a per-step
    // [hidden] audio row to the looked-up token embedding (llm_decode_step's
    // audio_row); prefill embeds get their rows host-side. Off (and a null
    // row) keeps every other engine's decode graph byte-identical.
    bool decode_add = false;
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
struct GenerateParams {
    int32_t max_new_tokens;
    int32_t max_cache_len;
    int32_t eos_token_id;
    // Optional secondary stop id (s1's generation_config stops on BOTH
    // <|im_end|> and <|endoftext|>). -1 (the default, also when a model
    // bundle aggregate-initializes only three fields) disables it, keeping
    // the moss/ark/granite/qwen3 stop behavior byte-identical.
    int32_t eos2_token_id = -1;
};

bool llm_prefill(const QwenDecodeCtx& m, const InputsEmbeds& i, int32_t max_cache_len,
                 PrefillResult& o, std::string& e);
// One decode step from the previous token id: embed lookup (+ the per-step
// audio row when spec.decode_add and audio_row != nullptr), one layer stack
// pass over the device KV, lm_head logits out. `state.length` is the write
// position (set by llm_prefill, advanced per step). Voxtral's offline loop
// drives prefill + this directly (its per-step audio rows cannot ride the
// shared greedy_generate).
bool llm_decode_step(const QwenDecodeCtx& m, int32_t prev_token,
                     const float* audio_row, LlmState& state,
                     std::vector<float>& logits, std::string& e);
// Greedy pick over host logits under the spec's tie/suppression policy
// (bf16-round + first-on-ties when argmax_low_ties; banned ids skipped).
int32_t spec_argmax(const QwenDecodeSpec& s, const std::vector<float>& x);
bool greedy_generate(const QwenDecodeCtx& m, const InputsEmbeds& i, const GenerateParams& op,
                     GenerateResult& o, std::string& e);
// Number of captured per-S prefill graphs for this spec (diagnostic + the
// bounded-LRU regression-test hook). Zero on CPU / before first GPU prefill.
size_t prefill_replay_cache_size(const QwenDecodeSpec& spec);

} // namespace starling::ggml::lib
