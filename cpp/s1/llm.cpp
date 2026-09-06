// llm.cpp — the S1-mini text decoder on the Starling ggml runtime.
//
// S1-mini's trunk IS the stock Qwen3 variant the shared lib/qwen_decode
// stack was built for (bias-free projections over full weight names,
// per-head q_norm/k_norm, TIED lm_head == llm.embed.weight, no multipliers)
// — identical to the qwen3-ASR spec modulo identity strings and the dims.
//
// Correctness contract: byte-exact bf16 vs the Transformers golden path on
// CUDA. CPU bf16 GEMMs are not bit-identical to cuBLAS and are a fallback
// only.

#include "llm.hpp"

#include <string>

namespace starling::ggml::s1 {
namespace {

// Model-identity constants. Env surface (per spec):
// STARLING_S1_{L0_STAGE,STAGE_DIR,DUMP_LAYERS,PERHEAD,FULLCAP,KSTEP,
// NOKSTEP,TIMING,DUMP_LOGITS,DUMP_IDS}.
const lib::QwenDecodeSpec kSpec = {
    /*qkv_bias=*/false,
    /*env=*/"STARLING_S1",
    /*label=*/"S1",
    /*stage_prefix=*/"s1_stage_",
    /*qk_norm=*/true,
    /*tied_lm_head=*/true,
    /*attention_scale=*/0.0f,
    /*embedding_multiplier=*/1.0f,
    /*residual_multiplier=*/1.0f,
    /*logits_scaling=*/1.0f,
    /*argmax_low_ties=*/true,
};

lib::QwenDecodeCtx decode_ctx(const S1Model& m) {
    const auto& lc = m.config.llm;
    return lib::QwenDecodeCtx{kSpec, m.loader,
                              {lc.n_layers, lc.hidden, lc.n_heads, lc.n_kv_heads,
                               lc.head_dim, lc.max_cache, lc.rope_theta,
                               lc.rms_norm_eps}};
}

} // namespace

bool llm_prefill(const S1Model& m, const lib::InputsEmbeds& i, int32_t maxc,
                 lib::PrefillResult& o, std::string& e) {
    return lib::llm_prefill(decode_ctx(m), i, maxc, o, e);
}

bool greedy_generate(const S1Model& m, const lib::InputsEmbeds& i,
                     const GenerateOptions& op, lib::GenerateResult& o,
                     std::string& e) {
    const lib::GenerateParams p{op.max_new_tokens, op.max_cache_len,
                                op.eos_token_id, op.eos2_token_id};
    return lib::greedy_generate(decode_ctx(m), i, p, o, e);
}

size_t prefill_replay_cache_size(const S1Model& model) {
    return lib::prefill_replay_cache_size(model.loader);
}

} // namespace starling::ggml::s1
