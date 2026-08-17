#include "loader.hpp"
#include <cstdio>
#include <vector>

#include "lib/loader_kit.hpp"
namespace starling::ggml::granite {
namespace {
// GGUF-metadata helpers shared across the engines (lib/loader_kit.hpp).
using lib::f32;
using lib::f64;
using lib::str;
} // namespace

int64_t audio_token_count(int64_t n_samples, const Config& c) {
    const uint32_t hop = c.frontend.hop_length > 0 ? c.frontend.hop_length : 1;
    const int64_t mel = n_samples / hop + 1;
    const int64_t enc = mel / 2;
    const int64_t nblocks = (enc + c.projector.window_size - 1) / c.projector.window_size;
    const int64_t per_block = c.projector.window_size / c.projector.downsample_rate;
    const int64_t n = nblocks * per_block;
    return n > 0 ? n : per_block;
}

bool GraniteModel::load(const char* path, std::string& err) {
    if (!loader.load(path)) {
        err = loader.last_error();
        return false;
    }
    auto& c = config;
    const auto& m = loader;
#define U(field, key) do { if (!lib::u32(m, "granite." key, field, field, err)) return false; } while (0)
#define F(field, key) field = f32(m, "granite." key, field)
    // Frontend (torchaudio MelSpectrogram: 80 mels, n_fft 512, win 400, hop 160).
    U(c.frontend.sample_rate, "frontend.sample_rate");
    U(c.frontend.n_fft, "frontend.n_fft");
    U(c.frontend.win_length, "frontend.win_length");
    U(c.frontend.hop_length, "frontend.hop_length");
    U(c.frontend.n_mels, "frontend.n_mels");
    U(c.frontend.power, "frontend.power");
    U(c.frontend.chunk_length, "frontend.chunk_length");
    F(c.frontend.mel_floor, "frontend.mel_floor");
    F(c.frontend.normalization_offset, "frontend.normalization_offset");
    F(c.frontend.normalization_divisor, "frontend.normalization_divisor");
    F(c.frontend.dynamic_range, "frontend.dynamic_range");
    // Encoder (CTC conformer).
    U(c.encoder.input_dim, "enc.input_dim");
    U(c.encoder.hidden, "enc.hidden");
    U(c.encoder.n_layers, "enc.layers");
    U(c.encoder.n_heads, "enc.heads");
    U(c.encoder.head_dim, "enc.head_dim");
    U(c.encoder.ffn_dim, "enc.ffn_dim");
    U(c.encoder.conv_kernel, "enc.conv_kernel");
    U(c.encoder.context_size, "enc.context_size");
    U(c.encoder.max_pos_emb, "enc.max_pos_emb");
    U(c.encoder.output_dim, "enc.output_dim");
    U(c.encoder.mid_layer, "enc.mid_layer");
    F(c.encoder.layer_norm_eps, "enc.layer_norm_eps");
    // Projector (BLIP2 Q-Former).
    U(c.projector.window_size, "proj.window_size");
    U(c.projector.downsample_rate, "proj.downsample_rate");
    U(c.projector.num_queries, "proj.num_queries");
    U(c.projector.hidden, "proj.hidden");
    U(c.projector.qformer_layers, "proj.qformer_layers");
    U(c.projector.qformer_heads, "proj.qformer_heads");
    U(c.projector.qformer_intermediate, "proj.qformer_intermediate");
    U(c.projector.output_dim, "proj.output_dim");
    F(c.projector.layer_norm_eps, "proj.layer_norm_eps");
    // LLM (Granite-4.0-1b trunk + numerics).
    U(c.llm.hidden, "llm.hidden_size");
    U(c.llm.n_layers, "llm.num_layers");
    U(c.llm.n_heads, "llm.num_heads");
    U(c.llm.n_kv_heads, "llm.num_kv_heads");
    U(c.llm.head_dim, "llm.head_dim");
    U(c.llm.intermediate, "llm.intermediate_size");
    U(c.llm.vocab, "llm.vocab_size");
    U(c.llm.max_position_embeddings, "llm.max_position_embeddings");
    U(c.llm.max_cache, "llm.max_cache_len");
    F(c.llm.rope_theta, "llm.rope_theta");
    F(c.llm.rms_norm_eps, "llm.rms_norm_eps");
    F(c.llm.attention_multiplier, "llm.attention_multiplier");
    F(c.llm.embedding_multiplier, "llm.embedding_multiplier");
    F(c.llm.residual_multiplier, "llm.residual_multiplier");
    F(c.llm.logits_scaling, "llm.logits_scaling");
    {
        int64_t tied = 0, qn = 0;
        if (m.kv_int("granite.llm.tied_embeddings", tied)) c.llm.tied_embeddings = tied != 0;
        if (m.kv_int("granite.llm.has_qk_norm", qn)) c.llm.has_qk_norm = qn != 0;
    }
    // Token ids + generation + chunk policy.
    {
        uint32_t t;
#define T(field, key) do { if (!lib::u32(m, "granite." key, (uint32_t) field, t, err)) return false; field = (int32_t) t; } while (0)
        T(c.audio_token_id, "audio_token_id");
        T(c.pad_token_id, "pad_token_id");
        T(c.bos_token_id, "bos_token_id");
        T(c.eos_token_id, "eos_token_id");
#undef T
    }
    U(c.max_new_tokens, "max_new_tokens");
    c.chunk_seconds = f64(m, "granite.chunk_seconds", 30.0);
#undef U
#undef F
    std::vector<int64_t> a;
    if (m.kv_arr_int("granite.prompt_prefix", a))
        for (auto v : a) c.prompt_prefix.push_back((int32_t) v);
    a.clear();
    if (m.kv_arr_int("granite.prompt_suffix", a))
        for (auto v : a) c.prompt_suffix.push_back((int32_t) v);

    // --- Validate untrusted GGUF metadata (mirror ark/loader.cpp). ---
    if (!lib::check_gguf_header(m, "granite", "GRANITE", {"bf16_exact"}, err))
        return false;
#define POS(v, name) do { if (!(v)) { err = "GRANITE GGUF " name " must be positive"; return false; } } while (0)
    POS(c.encoder.n_layers, "enc.layers");
    POS(c.encoder.hidden, "enc.hidden");
    POS(c.encoder.n_heads, "enc.heads");
    POS(c.encoder.head_dim, "enc.head_dim");
    POS(c.encoder.ffn_dim, "enc.ffn_dim");
    POS(c.encoder.conv_kernel, "enc.conv_kernel");
    POS(c.encoder.context_size, "enc.context_size");
    POS(c.encoder.output_dim, "enc.output_dim");
    POS(c.projector.window_size, "proj.window_size");
    POS(c.projector.num_queries, "proj.num_queries");
    POS(c.projector.qformer_layers, "proj.qformer_layers");
    POS(c.projector.qformer_heads, "proj.qformer_heads");
    POS(c.projector.output_dim, "proj.output_dim");
    POS(c.llm.hidden, "llm.hidden_size");
    POS(c.llm.n_heads, "llm.num_heads");
    POS(c.llm.n_kv_heads, "llm.num_kv_heads");
    POS(c.llm.head_dim, "llm.head_dim");
    POS(c.llm.intermediate, "llm.intermediate_size");
    POS(c.llm.vocab, "llm.vocab_size");
    POS(c.llm.max_cache, "llm.max_cache_len");
    POS(c.max_new_tokens, "max_new_tokens");
#undef POS
    if (c.encoder.input_dim != 2 * c.frontend.n_mels) {
        err = "GRANITE GGUF enc.input_dim must equal 2 * frontend.n_mels (pair stack)";
        return false;
    }
    if (c.encoder.hidden != c.encoder.n_heads * c.encoder.head_dim) {
        err = "GRANITE GGUF enc.hidden != enc.heads * enc.head_dim";
        return false;
    }
    if (c.encoder.conv_inner != c.encoder.hidden * 2) {
        err = "GRANITE GGUF conv expansion must be 2 (enc.conv_inner)";
        return false;
    }
    if (c.projector.num_queries != c.projector.window_size / c.projector.downsample_rate) {
        err = "GRANITE GGUF proj.num_queries != window/downsample";
        return false;
    }
    if (c.llm.hidden != c.llm.n_heads * c.llm.head_dim) {
        err = "GRANITE GGUF llm.hidden_size != llm.num_heads * llm.head_dim";
        return false;
    }
    if (c.llm.n_heads % c.llm.n_kv_heads != 0) {
        err = "GRANITE GGUF llm.num_heads must be a multiple of llm.num_kv_heads";
        return false;
    }
    if (c.frontend.n_fft != 512 || c.frontend.win_length != 400 ||
        c.frontend.n_mels != 80 || c.frontend.hop_length != 160) {
        err = "unsupported GRANITE frontend metadata (requires n_fft=512, win=400, hop=160, n_mels=80)";
        return false;
    }
    if (c.llm.tied_embeddings) {
        err = "unsupported GRANITE GGUF: granite-speech carries an untied lm_head";
        return false;
    }
    if (c.prompt_prefix.empty() || c.prompt_suffix.empty()) {
        err = "GRANITE GGUF missing prompt prefix/suffix arrays";
        return false;
    }

    // Require every expected tensor so a structural change fails loudly.
    for (const char* n : {"audio.mel_filters", "audio.mel_window", "enc.input_linear.weight",
                          "enc.input_linear.bias", "enc.out.weight", "enc.out.bias",
                          "enc.out_mid.weight", "enc.out_mid.bias", "proj.query",
                          "proj.qformer_ln.weight", "proj.qformer_ln.bias",
                          "proj.out.weight", "proj.out.bias", "llm.embed.weight",
                          "llm.lm_head.weight", "llm.final_norm.weight"})
        if (!lib::require(m, n, "GRANITE", err)) return false;
    // Encoder layers: 32 tensors each (ff halves, Shaw attn incl. the baked
    // rel-pos bias, conv module with BatchNorm stats, post_norm).
    for (uint32_t i = 0; i < c.encoder.n_layers; ++i) {
        char n[128];
        for (const char* tail : {"ff1_norm.weight", "ff1_norm.bias", "ff1_up.weight",
                                 "ff1_up.bias", "ff1_down.weight", "ff1_down.bias",
                                 "ff2_norm.weight", "ff2_norm.bias", "ff2_up.weight",
                                 "ff2_up.bias", "ff2_down.weight", "ff2_down.bias",
                                 "attn_norm.weight", "attn_norm.bias", "attn_q.weight",
                                 "attn_kv.weight", "attn_o.weight", "attn_o.bias",
                                 "rel_pos_bias",
                                 "conv_norm.weight", "conv_norm.bias", "conv_up.weight",
                                 "conv_up.bias", "conv_depth.weight", "bn_weight",
                                 "bn_bias", "bn_mean", "bn_var", "conv_down.weight",
                                 "conv_down.bias", "post_norm.weight", "post_norm.bias"}) {
            std::snprintf(n, sizeof n, "enc.blk.%u.%s", i, tail);
            if (!lib::require(m, n, "GRANITE", err)) return false;
        }
    }
    // Projector qformer layers: self-attn, cross-attn, FF (query path) = 26 each.
    for (uint32_t i = 0; i < c.projector.qformer_layers; ++i) {
        char n[128];
        for (const char* tail : {"self_q.weight", "self_q.bias", "self_k.weight",
                                 "self_k.bias", "self_v.weight", "self_v.bias",
                                 "self_out.weight", "self_out.bias",
                                 "self_ln.weight", "self_ln.bias",
                                 "cross_q.weight", "cross_q.bias", "cross_k.weight",
                                 "cross_k.bias", "cross_v.weight", "cross_v.bias",
                                 "cross_out.weight", "cross_out.bias",
                                 "cross_ln.weight", "cross_ln.bias",
                                 "ff_up.weight", "ff_up.bias", "ff_down.weight",
                                 "ff_down.bias", "ff_ln.weight", "ff_ln.bias"}) {
            std::snprintf(n, sizeof n, "proj.blk.%u.%s", i, tail);
            if (!lib::require(m, n, "GRANITE", err)) return false;
        }
    }
    // LLM layers: bias-free trunk, no q/k norm = 9 each.
    for (uint32_t i = 0; i < c.llm.n_layers; ++i) {
        char n[128];
        for (const char* tail : {"attn_norm.weight", "attn.q.weight", "attn.k.weight",
                                 "attn.v.weight", "attn.o.weight", "ffn_norm.weight",
                                 "ffn.gate.weight", "ffn.up.weight", "ffn.down.weight"}) {
            std::snprintf(n, sizeof n, "llm.blk.%u.%s", i, tail);
            if (!lib::require(m, n, "GRANITE", err)) return false;
        }
    }
    return true;
}
} // namespace starling::ggml::granite
