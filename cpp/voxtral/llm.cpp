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
#include "runtime/backend.hpp"
#include "runtime/graph.hpp"

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
    /*f32_acts=*/false,
    /*kstep_bucket=*/false,
    /*banned_ids=*/nullptr,
    /*n_banned=*/0,
    /*ada_rms_norm=*/true,
    /*ada_fc0_suffix=*/"ada.fc0.weight",
    /*ada_fc2_suffix=*/"ada.fc2.weight",
    /*decode_add=*/true,
};

// Embedding lookup of `ids` in a 2-D weight ([width, rows]) as an in-graph
// get_rows (every backend dequantizes every block type it stores), read back
// as host f32 rows: an embedding lookup without staging the whole table.
// Going through a graph also tells the CPU repacker (runtime/cpu_repack.hpp)
// that the table is read by GET_ROWS, so a quantized tied embedding/lm_head
// is never repacked for its matmul. Returns {} on failure.
std::vector<float> lookup_rows_f32(const ModelLoader& ml, const char* name,
                                          const std::vector<int32_t>& ids) {
    std::vector<float> out;
    if (ids.empty() || !ml.tensor(name)) return out;
    std::vector<int32_t> idx(ids);  // the graph input borrows this host buffer
    const bool ok = run_graph([&](ggml_context* c) -> ggml_tensor* {
        int64_t ne[1] = {(int64_t) idx.size()};
        ggml_tensor* rows = graph_input_tensor(c, GGML_TYPE_I32, 1, ne, idx.data(),
                                               idx.size() * sizeof(int32_t));
        return ggml_get_rows(c, lib::weight(c, ml, name), rows);
    }, out);
    if (!ok) out.clear();
    return out;
}
lib::QwenDecodeCtx decode_ctx(const VoxtralModel& m) {
    const auto& lc = m.config.llm;
    // Materialize the model's spec once (decode state is keyed per loader
    // post-#71; the model simply owns the spec alongside its loader).
    if (!m.decode_spec_ready) {
        m.decode_spec = kSpec;
        m.decode_spec_ready = true;
    }
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
    // Token rows, only the prompt's rows, through an in-graph get_rows (the
    // whole 131k-row table read to host f32 was ~1.5 GiB per call, and a
    // host read hides the lookup from the CPU repacker, which would then
    // repack a quantized tied head for its matmul). A bf16 table casts up
    // exactly, and adding the (bf16-valued) audio rows in f32 before the
    // prefill's single tobf round reproduces the stock bf16 add bit-exactly
    // (an f32 sum of bf16 values is exact; one round follows). A quantized
    // table dequantizes its rows.
    for (const int32_t id : ids) {
        if (id < 0 || (int64_t) id >= m.config.llm.vocab) {
            err = "VOXTRAL prompt id out of range";
            return false;
        }
    }
    const std::vector<float> rows = lookup_rows_f32(m.loader, "llm.embed.weight", ids);
    if (rows.size() != ids.size() * (size_t) H) {
        err = "VOXTRAL llm.embed.weight unreadable";
        return false;
    }
    out.data.assign((size_t) ids.size() * H, 0.0f);
    for (size_t i = 0; i < ids.size(); ++i) {
        const float* erow = rows.data() + i * H;
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
    out = GenerateResult{};
    const int64_t cap = generation_cap(P, mel_T, op.max_new_tokens);
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
    if (cap <= P) return true;
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
    // S03 termination contract, mapped from this loop's own stop condition:
    // last token was the EOS -> complete; the audio-row cap ended the loop ->
    // budget-exhausted (truncated, not a completed decode).
    out.stop_reason = prev == op.eos_token_id ? lib::GenStopReason::kEos
                                              : lib::GenStopReason::kBudgetExhausted;
    return true;
}

size_t prefill_replay_cache_size(const VoxtralModel& model) {
    return lib::prefill_replay_cache_size(model.loader);
}

} // namespace starling::ggml::voxtral
