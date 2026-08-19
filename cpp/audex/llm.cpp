// llm.cpp — the Nemotron-Labs-Audex-2B text decoder on the Starling ggml
// runtime.
//
// The decode stack (whole-model prefill/decode graphs, device-resident KV,
// K-step greedy) is the shared lib/qwen_decode stack; this file binds it to
// AudexModel through the spec below. Audex's trunk is a Nemotron-Dense 2B:
// bias-free projections over full weight names with NO q/k norm, an UNTIED
// lm_head, stock attention/residual/logits numerics, torch's first-index
// bf16-tie argmax, a plain squared-ReLU up/down MLP (no gate tensor), and
// F.rms_norm's single-round RMSNorm — the last two via the audex-added
// skip-when-default spec fields.
//
// Correctness contract: byte-exact bf16 vs the Transformers golden path on
// CUDA. CPU bf16 GEMMs are not bit-identical to cuBLAS and are a fallback
// only.

#include "llm.hpp"

#include "lib/qwen_decode.hpp"

#include <string>

namespace starling::ggml::audex {
namespace {

// Model-identity constants. Env surface (per spec):
// STARLING_AUDEX_{L0_STAGE,STAGE_DIR,DUMP_LAYERS,PERHEAD,FULLCAP,KSTEP,
// NOKSTEP,TIMING,DUMP_LOGITS,DUMP_IDS}.
const lib::QwenDecodeSpec kSpec = {
    /*qkv_bias=*/false,
    /*env=*/"STARLING_AUDEX",
    /*label=*/"AUDEX",
    /*stage_prefix=*/"audex_stage_",
    /*qk_norm=*/false,
    /*tied_lm_head=*/false,
    /*attention_scale=*/0.0f,
    /*embedding_multiplier=*/1.0f,
    /*residual_multiplier=*/1.0f,
    /*logits_scaling=*/1.0f,
    /*argmax_low_ties=*/true,
    /*mlp_activation=*/lib::QwenMlpAct::kRelu2Plain,
    /*rms_norm_single_round=*/true,
};

lib::QwenDecodeCtx decode_ctx(const AudexModel& m) {
    const auto& lc = m.config.llm;
    return lib::QwenDecodeCtx{kSpec, m.loader,
                              {lc.n_layers, lc.hidden, lc.n_heads, lc.n_kv_heads,
                               lc.head_dim, lc.max_cache, lc.rope_theta,
                               lc.rms_norm_eps}};
}

} // namespace

bool llm_prefill(const AudexModel& m, const InputsEmbeds& i, int32_t maxc,
                 PrefillResult& o, std::string& e) {
    return lib::llm_prefill(decode_ctx(m), i, maxc, o, e);
}

bool greedy_generate(const AudexModel& m, const InputsEmbeds& i,
                     const GenerateOptions& op, GenerateResult& o, std::string& e) {
    const lib::GenerateParams p{op.max_new_tokens, op.max_cache_len, op.eos_token_id};
    return lib::greedy_generate(decode_ctx(m), i, p, o, e);
}

size_t prefill_replay_cache_size() {
    return lib::prefill_replay_cache_size(kSpec);
}

} // namespace starling::ggml::audex
