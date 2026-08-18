// llm.cpp — the Qwen3 text decoder on the Starling ggml runtime.
//
// The decode stack (whole-model prefill/decode graphs, device-resident KV,
// K-step greedy) is the shared lib/qwen_decode stack; this file binds it to
// Qwen3Model through the spec below. Qwen3-ASR's trunk is the STOCK variant
// the stack was built for: bias-free projections over full weight names,
// per-head q_norm/k_norm, a TIED lm_head (== llm.embed.weight) and no
// multipliers — identical to the moss spec modulo identity strings.
//
// Correctness contract: byte-exact bf16 vs the Transformers golden path on
// CUDA. CPU bf16 GEMMs are not bit-identical to cuBLAS and are a fallback
// only.

#include "llm.hpp"

#include "lib/qwen_decode.hpp"

#include <string>

namespace starling::ggml::qwen3 {
namespace {

// Model-identity constants. Env surface (per spec):
// STARLING_QWEN3_{L0_STAGE,STAGE_DIR,DUMP_LAYERS,PERHEAD,FULLCAP,KSTEP,
// NOKSTEP,TIMING,DUMP_LOGITS,DUMP_IDS}.
const lib::QwenDecodeSpec kSpec = {
    /*qkv_bias=*/false,
    /*env=*/"STARLING_QWEN3",
    /*label=*/"QWEN3",
    /*stage_prefix=*/"qwen3_stage_",
    /*qk_norm=*/true,
    /*tied_lm_head=*/true,
    /*attention_scale=*/0.0f,
    /*embedding_multiplier=*/1.0f,
    /*residual_multiplier=*/1.0f,
    /*logits_scaling=*/1.0f,
};

lib::QwenDecodeCtx decode_ctx(const Qwen3Model& m) {
    const auto& lc = m.config.llm;
    return lib::QwenDecodeCtx{kSpec, m.loader,
                              {lc.n_layers, lc.hidden, lc.n_heads, lc.n_kv_heads,
                               lc.head_dim, lc.max_cache, lc.rope_theta,
                               lc.rms_norm_eps}};
}

} // namespace

bool llm_prefill(const Qwen3Model& m, const InputsEmbeds& i, int32_t maxc,
                 PrefillResult& o, std::string& e) {
    return lib::llm_prefill(decode_ctx(m), i, maxc, o, e);
}

bool greedy_generate(const Qwen3Model& m, const InputsEmbeds& i,
                     const GenerateOptions& op, GenerateResult& o, std::string& e) {
    const lib::GenerateParams p{op.max_new_tokens, op.max_cache_len, op.eos_token_id};
    return lib::greedy_generate(decode_ctx(m), i, p, o, e);
}

size_t prefill_replay_cache_size() {
    return lib::prefill_replay_cache_size(kSpec);
}

} // namespace starling::ggml::qwen3
