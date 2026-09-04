// llm.cpp — Higgs Qwen3-1.7B text decoder on the Starling ggml runtime.
//
// The trunk IS the shared lib/qwen_decode stack (moss = the tied-lm_head
// Qwen3-with-qk_norm sibling); higgs previously carried a frozen ~600-line
// port of it
// whose CPU decode diverged (greedy decode fell into a 4-token repetition
// loop: the port drifted from the maintained stack). This file is now the
// thin spec binding, mirroring cpp/moss/llm.cpp.
//
// Higgs differs from moss only in: (a) SEPARATE lm_head (llm.lm_head.weight,
// NOT tied to embed), (b) ASR stops on EITHER <|endoftext|> (151643) OR
// <|im_end|> (151645) — config.EOS_TOKEN_IDS, mapped to the spec's
// eos2_token_id, (c) dims (28 layers, d2048, GQA 16/8, head_dim 128,
// vocab 151936 — config-driven, no code change).
//
// Correctness contract: byte-exact bf16 vs the Transformers golden path on
// CUDA.
#include "llm.hpp"

#include "lib/qwen_decode.hpp"

#include <string>

namespace starling::ggml::higgs {
namespace {

// Qwen3 trunk: bias-free q/k/v over full weight names + per-head q_norm /
// k_norm. Env surface: STARLING_HIGGS_{L0_STAGE,STAGE_DIR,DUMP_LAYERS,PERHEAD,
// FULLCAP,KSTEP,NOKSTEP,TIMING,DUMP_LOGITS,DUMP_IDS}. L0-probe stage files are
// named higgs_stage_<stage>.f32.
const lib::QwenDecodeSpec kSpec = {
    /*qkv_bias=*/false,
    /*env=*/"STARLING_HIGGS",
    /*label=*/"HIGGS",
    /*stage_prefix=*/"higgs_stage_",
    /*qk_norm=*/true,
    /*tied_lm_head=*/false,
};

lib::QwenDecodeCtx decode_ctx(const HiggsModel& m) {
    const auto& lc = m.config.llm;
    return lib::QwenDecodeCtx{kSpec, m.loader,
                              {lc.n_layers, lc.hidden, lc.n_heads, lc.n_kv_heads,
                               lc.head_dim, lc.max_cache, lc.rope_theta,
                               lc.rms_norm_eps}};
}

} // namespace

bool greedy_generate(const HiggsModel& m, const InputsEmbeds& inputs,
                     const GenerateOptions& op, GenerateResult& out, std::string& err) {
    // Secondary stop: <|im_end|> rides the spec's eos2_token_id. The higgs
    // Inputs/Generate structs are structurally identical to the lib:: ones
    // but distinct types; convert (one copy of the merged embeds).
    const lib::GenerateParams p{op.max_new_tokens, op.max_cache_len,
                                op.eos_token_id, op.im_end_id};
    lib::InputsEmbeds li{inputs.data, inputs.n_tokens, inputs.width};
    lib::GenerateResult lo;
    if (!lib::greedy_generate(decode_ctx(m), li, p, lo, err)) return false;
    // Leading-EOS (near-silence input): the shared stack stops on a
    // prefill-argmax stop token for eos2 engines (qwen_decode.cpp's
    // prefill_stop gate) — the deleted port's behavior, at its old cost
    // (no wasted max_new_tokens decode).
    out.ids = std::move(lo.ids);
    out.hit_eos = lo.hit_eos;
    out.prefill_logits = std::move(lo.prefill_logits);
    return true;
}

size_t prefill_replay_cache_size() {
    return lib::prefill_replay_cache_size(kSpec);
}

} // namespace starling::ggml::higgs
