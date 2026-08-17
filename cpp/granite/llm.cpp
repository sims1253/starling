// llm.cpp — the Granite-4.0-1b text decoder on the Starling ggml runtime.
//
// The decode stack (whole-model prefill/decode graphs, device-resident KV,
// K-step greedy) is the shared lib/qwen_decode stack; this file binds it to
// GraniteModel through the spec below. Granite is a THIRD trunk variant:
// bias-free projections like the Qwen3 trunk (moss) but WITHOUT the per-head
// q_norm/k_norm, an UNTIED lm_head, and the Granite numerics — the attention
// scale is attention_multiplier (0.0078125) rather than 1/sqrt(D), residuals
// scale by residual_multiplier (0.22), the merged inputs_embeds scale by
// embedding_multiplier (12.0) at prefill AND the embed lookup at decode, and
// logits divide by logits_scaling (8.0).
//
// Correctness contract: byte-exact bf16 vs the Transformers golden path on
// CUDA. CPU bf16 GEMMs are not bit-identical to cuBLAS and are a fallback
// only.

#include "llm.hpp"

#include "lib/qwen_decode.hpp"

#include <string>

namespace starling::ggml::granite {
namespace {

// Model-identity constants (the trunk-shape equivalents): granite-speech-4.1
// carries exactly these multipliers in its HF config. Env surface:
// STARLING_GRANITE_{L0_STAGE,STAGE_DIR,DUMP_LAYERS,PERHEAD,FULLCAP,KSTEP,
// NOKSTEP,TIMING,DUMP_LOGITS,DUMP_IDS}.
const lib::QwenDecodeSpec kSpec = {
    /*qkv_bias=*/false,
    /*env=*/"STARLING_GRANITE",
    /*label=*/"GRANITE",
    /*stage_prefix=*/"granite_stage_",
    /*qk_norm=*/false,
    /*tied_lm_head=*/false,
    /*attention_scale=*/0.0078125f,
    /*embedding_multiplier=*/12.0f,
    /*residual_multiplier=*/0.22f,
    /*logits_scaling=*/8.0f,
};

lib::QwenDecodeCtx decode_ctx(const GraniteModel& m) {
    const auto& lc = m.config.llm;
    return lib::QwenDecodeCtx{kSpec, m.loader,
                              {lc.n_layers, lc.hidden, lc.n_heads, lc.n_kv_heads,
                               lc.head_dim, lc.max_cache, lc.rope_theta,
                               lc.rms_norm_eps}};
}

} // namespace

bool llm_prefill(const GraniteModel& m, const InputsEmbeds& i, int32_t maxc,
                 PrefillResult& o, std::string& e) {
    return lib::llm_prefill(decode_ctx(m), i, maxc, o, e);
}

bool greedy_generate(const GraniteModel& m, const InputsEmbeds& i,
                     const GenerateOptions& op, GenerateResult& o, std::string& e) {
    const lib::GenerateParams p{op.max_new_tokens, op.max_cache_len, op.eos_token_id};
    return lib::greedy_generate(decode_ctx(m), i, p, o, e);
}

size_t prefill_replay_cache_size() {
    return lib::prefill_replay_cache_size(kSpec);
}

} // namespace starling::ggml::granite
