// llm.cpp — MOSS (Qwen3-style) text decoder on the Starling ggml runtime.
//
// The decode stack (whole-model prefill/decode graphs, device-resident KV,
// K-step greedy) is the shared lib/qwen_decode stack; this file binds it to
// MossModel through the spec below.
//
// Correctness contract: byte-exact bf16 vs the Transformers golden path on
// CUDA (see docs/ggml-moss-spec.md). CPU bf16 GEMMs are not bit-identical to
// cuBLAS and are a fallback only.

#include "llm.hpp"

#include "lib/qwen_decode.hpp"

#include <string>

namespace starling::ggml::moss {
namespace {

// Qwen3 trunk: bias-free q/k/v over full weight names + per-head q_norm /
// k_norm. Env surface: STARLING_MOSS_{L0_STAGE,STAGE_DIR,DUMP_LAYERS,PERHEAD,
// FULLCAP,KSTEP,NOKSTEP,TIMING,DUMP_LOGITS,DUMP_IDS}. L0-probe stage files are
// named moss_stage_<stage>.f32.
const lib::QwenDecodeSpec kSpec = {
    /*qkv_bias=*/false,
    /*env=*/"STARLING_MOSS",
    /*label=*/"MOSS",
    /*stage_prefix=*/"moss_stage_",
};

lib::QwenDecodeCtx decode_ctx(const MossModel& m) {
    const auto& lc = m.config.llm;
    return lib::QwenDecodeCtx{kSpec, m.loader,
                              {lc.n_layers, lc.hidden, lc.n_heads, lc.n_kv_heads,
                               lc.head_dim, lc.max_cache, lc.rope_theta,
                               lc.rms_norm_eps}};
}

} // namespace

bool llm_prefill(const MossModel& m, const InputsEmbeds& i, int32_t maxc,
                 PrefillResult& o, std::string& e) {
    return lib::llm_prefill(decode_ctx(m), i, maxc, o, e);
}

bool greedy_generate(const MossModel& m, const InputsEmbeds& i,
                     const GenerateOptions& op, GenerateResult& o, std::string& e) {
    const lib::GenerateParams p{op.max_new_tokens, op.max_cache_len, op.eos_token_id};
    return lib::greedy_generate(decode_ctx(m), i, p, o, e);
}

size_t prefill_replay_cache_size() {
    return lib::prefill_replay_cache_size(kSpec);
}

} // namespace starling::ggml::moss
