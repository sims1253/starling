#include "loader.hpp"
#include <cstdio>
#include <vector>

#include "lib/loader_kit.hpp"
namespace starling::ggml::qwen3 {
namespace {
// GGUF-metadata helpers shared across the engines (lib/loader_kit.hpp).
using lib::f32;
using lib::f64;
using lib::str;
} // namespace

int64_t audio_token_count(int64_t n_samples, const Config& c) {
    const uint32_t hop = c.frontend.hop_length > 0 ? c.frontend.hop_length : 1;
    const int64_t min_len = c.frontend.min_length;
    int64_t S = n_samples;
    if (S < min_len) S = min_len;
    const int64_t T = S / hop;
    const int64_t chunk = 2 * (int64_t) c.frontend.n_window;
    int64_t r = T % chunk;
    int64_t full = T / chunk;
    // c3: triple ceil-halving ((x-1)/2+1), zero stays zero.
    for (int i = 0; i < 3 && r > 0; ++i) r = (r - 1) / 2 + 1;
    const int64_t per_full = c.encoder.max_pos_emb;  // 13 post-CNN rows
    return full * per_full + r;
}

bool Qwen3Model::load(const char* path, std::string& err) {
    if (!loader.load(path)) {
        err = loader.last_error();
        return false;
    }
    auto& c = config;
    const auto& m = loader;
#define U(field, key) do { if (!lib::u32(m, "qwen3." key, field, field, err)) return false; } while (0)
#define F(field, key) field = f32(m, "qwen3." key, field)
    // Frontend (torch.stft mel: 128 bins, n_fft 400, hop 160).
    U(c.frontend.sample_rate, "frontend.sample_rate");
    U(c.frontend.n_fft, "frontend.n_fft");
    U(c.frontend.win_length, "frontend.win_length");
    U(c.frontend.hop_length, "frontend.hop_length");
    U(c.frontend.n_mels, "frontend.n_mels");
    U(c.frontend.power, "frontend.power");
    U(c.frontend.chunk_length, "frontend.chunk_length");
    U(c.frontend.min_length, "frontend.min_length");
    U(c.frontend.n_window, "frontend.n_window");
    F(c.frontend.mel_floor, "frontend.mel_floor");
    F(c.frontend.normalization_offset, "frontend.normalization_offset");
    F(c.frontend.normalization_divisor, "frontend.normalization_divisor");
    F(c.frontend.dynamic_range, "frontend.dynamic_range");
    // Encoder (windowed-attention conv stack).
    U(c.encoder.n_mel, "enc.n_mel");
    U(c.encoder.hidden, "enc.hidden");
    U(c.encoder.n_layers, "enc.layers");
    U(c.encoder.n_heads, "enc.heads");
    U(c.encoder.head_dim, "enc.head_dim");
    U(c.encoder.ffn_dim, "enc.ffn_dim");
    U(c.encoder.downsample_hidden, "enc.downsample_hidden");
    U(c.encoder.n_window, "enc.n_window");
    U(c.encoder.n_window_infer, "enc.n_window_infer");
    U(c.encoder.max_pos_emb, "enc.max_pos_emb");
    U(c.encoder.output_dim, "enc.output_dim");
    F(c.encoder.layer_norm_eps, "enc.layer_norm_eps");
    // Projector (2-layer MLP).
    U(c.projector.hidden, "proj.hidden");
    U(c.projector.output_dim, "proj.output_dim");
    // LLM (Qwen3 trunk, stock numerics).
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
    {
        int64_t tied = 0, qn = 0;
        if (m.kv_int("qwen3.llm.tied_embeddings", tied)) c.llm.tied_embeddings = tied != 0;
        if (m.kv_int("qwen3.llm.has_qk_norm", qn)) c.llm.has_qk_norm = qn != 0;
    }
    // Token ids + generation + chunk policy.
    {
        uint32_t t;
#define T(field, key) do { if (!lib::u32(m, "qwen3." key, (uint32_t) field, t, err)) return false; field = (int32_t) t; } while (0)
        T(c.audio_token_id, "audio_token_id");
        T(c.pad_token_id, "pad_token_id");
        T(c.eos_token_id, "eos_token_id");
#undef T
    }
    U(c.max_new_tokens, "max_new_tokens");
    c.chunk_seconds = f64(m, "qwen3.chunk_seconds", 30.0);
#undef U
#undef F
    std::vector<int64_t> a;
    if (m.kv_arr_int("qwen3.prompt_prefix", a))
        for (auto v : a) c.prompt_prefix.push_back((int32_t) v);
    a.clear();
    if (m.kv_arr_int("qwen3.prompt_suffix", a))
        for (auto v : a) c.prompt_suffix.push_back((int32_t) v);

    // --- Validate untrusted GGUF metadata (mirror granite/loader.cpp). ---
    if (!lib::check_gguf_header(m, "qwen3", "QWEN3", {"bf16_exact"}, err))
        return false;
#define POS(v, name) do { if (!(v)) { err = "QWEN3 GGUF " name " must be positive"; return false; } } while (0)
    POS(c.encoder.n_layers, "enc.layers");
    POS(c.encoder.hidden, "enc.hidden");
    POS(c.encoder.n_heads, "enc.heads");
    POS(c.encoder.head_dim, "enc.head_dim");
    POS(c.encoder.ffn_dim, "enc.ffn_dim");
    POS(c.encoder.downsample_hidden, "enc.downsample_hidden");
    POS(c.encoder.n_window, "enc.n_window");
    POS(c.encoder.n_window_infer, "enc.n_window_infer");
    POS(c.encoder.max_pos_emb, "enc.max_pos_emb");
    POS(c.encoder.output_dim, "enc.output_dim");
    POS(c.projector.hidden, "proj.hidden");
    POS(c.projector.output_dim, "proj.output_dim");
    POS(c.llm.hidden, "llm.hidden_size");
    POS(c.llm.n_layers, "llm.num_layers");
    POS(c.llm.n_heads, "llm.num_heads");
    POS(c.llm.n_kv_heads, "llm.num_kv_heads");
    POS(c.llm.head_dim, "llm.head_dim");
    POS(c.llm.intermediate, "llm.intermediate_size");
    POS(c.llm.vocab, "llm.vocab_size");
    POS(c.llm.max_cache, "llm.max_cache_len");
    POS(c.max_new_tokens, "max_new_tokens");
#undef POS
    if (c.encoder.n_mel != c.frontend.n_mels) {
        err = "QWEN3 GGUF enc.n_mel must equal frontend.n_mels";
        return false;
    }
    if (c.encoder.hidden != c.encoder.n_heads * c.encoder.head_dim) {
        err = "QWEN3 GGUF enc.hidden != enc.heads * enc.head_dim";
        return false;
    }
    // conv_out in_features: downsample_hidden * (mel 128 -> 64 -> 32 -> 16).
    if (c.encoder.downsample_hidden * 16 != 7680) {
        err = "QWEN3 GGUF conv stack must reduce 128 mel bins to 16";
        return false;
    }
    // Attention windows must be chunk-aligned: n_window_infer a multiple of
    // 2*n_window (get_audio_cu_seqlens' n_window_ratio).
    if (c.encoder.n_window_infer % (2 * c.encoder.n_window) != 0) {
        err = "QWEN3 GGUF enc.n_window_infer must be a multiple of 2*enc.n_window";
        return false;
    }
    if (c.llm.hidden != c.llm.n_heads * c.llm.head_dim) {
        err = "QWEN3 GGUF llm.hidden_size != llm.num_heads * llm.head_dim";
        return false;
    }
    if (c.llm.n_heads % c.llm.n_kv_heads != 0) {
        err = "QWEN3 GGUF llm.num_heads must be a multiple of llm.num_kv_heads";
        return false;
    }
    if (c.frontend.n_fft != 400 || c.frontend.win_length != 400 ||
        c.frontend.n_mels != 128 || c.frontend.hop_length != 160 ||
        c.frontend.n_window != 50) {
        err = "unsupported QWEN3 frontend metadata (requires n_fft=400, win=400, hop=160, n_mels=128, n_window=50)";
        return false;
    }
    if (c.encoder.n_window_infer != 800) {
        err = "unsupported QWEN3 GGUF: enc.n_window_infer must be 800";
        return false;
    }
    if (!c.llm.tied_embeddings) {
        err = "unsupported QWEN3 GGUF: Qwen3-ASR ties lm_head to the embedding table";
        return false;
    }
    if (!c.llm.has_qk_norm) {
        err = "unsupported QWEN3 GGUF: the Qwen3 trunk carries per-head q/k norm";
        return false;
    }
    if (c.prompt_prefix.empty() || c.prompt_suffix.empty()) {
        err = "QWEN3 GGUF missing prompt prefix/suffix arrays";
        return false;
    }

    // Require every expected tensor so a structural change fails loudly.
    for (const char* n : {"audio.mel_filters", "audio.mel_window",
                          "enc.conv1.weight", "enc.conv1.bias",
                          "enc.conv2.weight", "enc.conv2.bias",
                          "enc.conv3.weight", "enc.conv3.bias",
                          "enc.out.weight", "enc.pos_embed",
                          "enc.ln_post.weight", "enc.ln_post.bias",
                          "proj.linear_1.weight", "proj.linear_1.bias",
                          "proj.linear_2.weight", "proj.linear_2.bias",
                          "llm.embed.weight", "llm.final_norm.weight"})
        if (!lib::require(m, n, "QWEN3", err)) return false;
    // Encoder layers: 16 tensors each (biased MHA + two biased LayerNorms +
    // biased FFN).
    for (uint32_t i = 0; i < c.encoder.n_layers; ++i) {
        char n[128];
        for (const char* tail : {"attn_norm.weight", "attn_norm.bias",
                                 "attn_q.weight", "attn_q.bias",
                                 "attn_k.weight", "attn_k.bias",
                                 "attn_v.weight", "attn_v.bias",
                                 "attn_o.weight", "attn_o.bias",
                                 "ffn_norm.weight", "ffn_norm.bias",
                                 "ff_up.weight", "ff_up.bias",
                                 "ff_down.weight", "ff_down.bias"}) {
            std::snprintf(n, sizeof n, "enc.blk.%u.%s", i, tail);
            if (!lib::require(m, n, "QWEN3", err)) return false;
        }
    }
    // LLM layers: bias-free Qwen3 trunk with q/k norm = 11 each.
    for (uint32_t i = 0; i < c.llm.n_layers; ++i) {
        char n[128];
        for (const char* tail : {"attn_norm.weight", "attn.q.weight", "attn.k.weight",
                                 "attn.v.weight", "attn.o.weight",
                                 "attn.q_norm.weight", "attn.k_norm.weight",
                                 "ffn_norm.weight", "ffn.gate.weight",
                                 "ffn.up.weight", "ffn.down.weight"}) {
            std::snprintf(n, sizeof n, "llm.blk.%u.%s", i, tail);
            if (!lib::require(m, n, "QWEN3", err)) return false;
        }
    }
    return true;
}
} // namespace starling::ggml::qwen3
