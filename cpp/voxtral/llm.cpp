// llm.cpp — Voxtral text decoder binding + offline greedy loop.
//
// The decode stack (whole-model prefill/decode graphs, device-resident KV) is
// the shared lib/qwen_decode stack; this file binds it to VoxtralModel
// through the spec below and drives the additive-injection loop the shared
// greedy_generate cannot serve (per-step audio rows).
//
// Correctness contract: byte-exact bf16 vs the Transformers golden path on
// CUDA. CPU bf16 GEMMs are not bit-identical to cuBLAS and are a fallback
// only.

#include "llm.hpp"

#include "lib/graph_helpers.hpp"
#include "lib/qwen_decode.hpp"

#include <string>

namespace starling::ggml::voxtral {
namespace {

// Bias-free full-name projections, no q_norm/k_norm, tied lm_head, stock
// scales, first-on-ties bf16 argmax (the reference argmaxes bf16-stored
// logits; torch keeps the first index on ties), SwiGLU, two-round RMSNorm
// (all stack defaults), plus AdaRMSNorm (ada.fc0/fc2 under each layer) and
// the decode-step audio-row add. Env surface: STARLING_VOXTRAL_{L0_STAGE,
// STAGE_DIR,DUMP_LAYERS,PERHEAD,FULLCAP,TIMING,DUMP_LOGITS,DUMP_IDS}.
const lib::QwenDecodeSpec kSpec = {
    /*qkv_bias=*/false,
    /*env=*/"STARLING_VOXTRAL",
    /*label=*/"VOXTRAL",
    /*stage_prefix=*/"voxtral_stage_",
    /*qk_norm=*/false,
    /*tied_lm_head=*/true,
    /*attention_scale=*/0.0f,
    /*embedding_multiplier=*/1.0f,
    /*residual_multiplier=*/1.0f,
    /*logits_scaling=*/1.0f,
    /*argmax_low_ties=*/true,
    /*mlp_activation=*/lib::QwenMlpAct::kSiluGated,
    /*rms_norm_single_round=*/false,
    /*banned_ids=*/nullptr,
    /*n_banned=*/0,
    /*ada_rms_norm=*/true,
    /*ada_fc0_suffix=*/"ada.fc0.weight",
    /*ada_fc2_suffix=*/"ada.fc2.weight",
    /*decode_add=*/true,
};

// Synthesize the llm.ada_ones leaf (1-D [hidden] f32 ones) the ada branch
// adds to each layer's modulation (see apply_ada). A no-op when present;
// owned until loader destruction, so captured prefill graphs keep a stable
// source. 1-D keeps it at 12 KiB for the real model, inside the compat
// context's 1 MiB budget. The const_cast follows the mel-constants precedent
// (mel.cpp): the loader is logically mutated once (synthesis), then read-only.
void ensure_ada_ones(const VoxtralModel& m) {
    if (m.loader.tensor("llm.ada_ones")) return;
    auto& ml = const_cast<ModelLoader&>(m.loader);
    const int64_t H = m.config.llm.hidden;
    ml.add_owned_tensor("llm.ada_ones", std::vector<float>((size_t)H, 1.0f),
                        H, 1);
}

lib::QwenDecodeCtx decode_ctx(const VoxtralModel& m) {
    const auto& lc = m.config.llm;
    // Materialize the model's spec once. Stored on the model because the
    // spec address keys the process-global decode caches.
    if (!m.decode_spec_ready) {
        m.decode_spec = kSpec;
        m.decode_spec_ready = true;
    }
    ensure_ada_ones(m);
    return lib::QwenDecodeCtx{m.decode_spec, m.loader,
                              {lc.n_layers, lc.hidden, lc.n_heads, lc.n_kv_heads,
                               lc.head_dim, lc.max_cache, lc.rope_theta,
                               lc.rms_norm_eps}};
}

} // namespace

bool build_inputs_embeds(const VoxtralModel& m, const std::vector<int32_t>& ids,
                         const AudioEncoding& audio, InputsEmbeds& out,
                         std::string& err) {
    const int64_t H = m.config.llm.hidden;
    if (audio.width != H) {
        err = "VOXTRAL audio width != llm hidden";
        return false;
    }
    if (audio.n_tokens < (int64_t) ids.size()) {
        err = "VOXTRAL audio rows fewer than prompt tokens";
        return false;
    }
    // Token rows, host-side: the embedding table is bf16; read_f32 casts up
    // exactly, and adding the (bf16-valued) audio rows in f32 before the
    // prefill's single tobf round reproduces the stock bf16 add bit-exactly
    // (an f32 sum of bf16 values is exact; one round follows).
    std::vector<float> table = lib::read_f32(m.loader, "llm.embed.weight");
    if (table.size() < (size_t) m.config.llm.vocab * H) {
        err = "VOXTRAL llm.embed.weight unreadable";
        return false;
    }
    out.data.assign((size_t) ids.size() * H, 0.0f);
    for (size_t i = 0; i < ids.size(); ++i) {
        const int32_t id = ids[i];
        if (id < 0 || (int64_t) id >= m.config.llm.vocab) {
            err = "VOXTRAL prompt id out of range";
            return false;
        }
        const float* erow = table.data() + (size_t) id * H;
        const float* arow = audio.data.data() + i * H;
        float* drow = out.data.data() + i * H;
        for (int64_t h = 0; h < H; ++h) drow[h] = erow[h] + arow[h];
    }
    out.n_tokens = (int64_t) ids.size();
    out.width = H;
    return true;
}

bool greedy_generate(const VoxtralModel& m, const InputsEmbeds& prefill,
                     const AudioEncoding& audio, int64_t mel_T,
                     const GenerateOptions& op, GenerateResult& out,
                     std::string& e) {
    const int64_t P = prefill.n_tokens;
    const int64_t cap = generation_cap(P, mel_T);
    // Mirror the pipeline's _check_cache_fit: the cap must fit the static KV.
    if (cap > op.max_cache_len) {
        e = "VOXTRAL total length cap " + std::to_string(cap) +
            " exceeds max_cache_len " + std::to_string(op.max_cache_len);
        return false;
    }
    if (prefill.width != (int64_t) m.config.llm.hidden ||
        audio.width != (int64_t) m.config.llm.hidden) {
        e = "VOXTRAL prefill/audio width != llm hidden";
        return false;
    }
    // Offline the cap equals the audio-row count, so every consumed row
    // (prompt rows 0..P-1, decode step t row P+t-1 with P+t-1 < cap) exists.
    if (audio.n_tokens < cap) {
        e = "VOXTRAL audio rows fewer than the total length cap";
        return false;
    }
    const lib::QwenDecodeCtx ctx = decode_ctx(m);
    lib::PrefillResult p;
    if (!lib::llm_prefill(ctx, prefill, op.max_cache_len, p, e)) return false;
    out.prefill_logits = p.logits;
    const int64_t H = m.config.llm.hidden;
    int32_t prev = lib::spec_argmax(ctx.spec, p.logits);
    out.ids.push_back(prev);
    while (P + (int64_t) out.ids.size() < cap && prev != op.eos_token_id) {
        const int64_t row = P + (int64_t) out.ids.size() - 1;
        const float* arow = audio.data.data() + row * H;
        std::vector<float> dl;
        if (!lib::llm_decode_step(ctx, prev, arow, p.state, dl, e)) return false;
        prev = lib::spec_argmax(ctx.spec, dl);
        out.ids.push_back(prev);
    }
    out.hit_eos = (prev == op.eos_token_id);
    return true;
}

size_t prefill_replay_cache_size() {
    return lib::prefill_replay_cache_size(kSpec);
}

} // namespace starling::ggml::voxtral
