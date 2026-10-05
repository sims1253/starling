#include "loader.hpp"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ggml.h"
#include "lib/loader_kit.hpp"
namespace starling::ggml::qwen3 {
namespace {
// GGUF-metadata helpers shared across the engines (lib/loader_kit.hpp).
using lib::f32;
using lib::f64;
using lib::str;

// Appends `tail` to the layer prefix already written into `n` and runs the
// exact-shape check on the combined name — the ONE shared implementation
// behind the encoder/LLM layer loops. Fails loudly on snprintf truncation
// instead of probing a wrong tensor name.
bool layer_shape(const ModelLoader& m, char* n, size_t cap, size_t pre_len,
                 const char* tail, std::initializer_list<int64_t> want,
                 std::string& err) {
    const int written = std::snprintf(n + pre_len, cap - pre_len, "%s", tail);
    if (written < 0 || (size_t) written >= cap - pre_len) {
        err = std::string("QWEN3 GGUF tensor name truncated: ") + n;
        return false;
    }
    return lib::shape_eq(m, "QWEN3", n, want, err);
}
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

// Exact tensor-shape checks (shared lib/loader_kit.hpp shape_eq, QWEN3
// label) as plain lambdas rather than macros: compiler diagnostics and
// debuggers point at the real call site, and there is no #define/#undef
// lifecycle to keep in order. GGUF ne[] is the reversed checkpoint shape:
// a torch Linear weight [OC, IC] reads back ne0 = IC (the mul_mat
// contraction dim), ne1 = OC.

bool Qwen3Model::load(const char* path, std::string& err) {
    if (!loader.load(path)) {
        err = loader.last_error();
        return false;
    }
    auto& c = config;
    const auto& m = loader;
    // Top-level tensors take a literal name.
    const auto shape = [&](const char* name, std::initializer_list<int64_t> want) {
        return lib::shape_eq(m, "QWEN3", name, want, err);
    };
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
    if (!lib::check_gguf_header(m, "qwen3", "QWEN3", {"bf16_exact", "quantized"}, err))
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
    // Attention windows must be chunk-aligned: n_window_infer a multiple of
    // 2*n_window (get_audio_cu_seqlens' n_window_ratio).
    if (c.encoder.n_window_infer % (2 * c.encoder.n_window) != 0) {
        err = "QWEN3 GGUF enc.n_window_infer must be a multiple of 2*enc.n_window";
        return false;
    }
    // NOTE: llm.hidden_size and the attention width heads*head_dim are
    // INDEPENDENT fields. The 1.7B happens to equate them (2048 == 16*128);
    // the 0.6B runs a 1024-wide trunk under a 2048-wide attention (16 heads x
    // 128), which the shared decode stack has supported since voxtral's
    // 4096-wide attention over a 3072 trunk. The dimensions are validated
    // against the actual tensor shapes below instead of equated.
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

    // Require every expected tensor with its EXACT shape so a structural
    // change or a metadata/tensor mismatch fails loudly at load. All ne[] are
    // ggml order (reversed checkpoint shape: ne0 = Linear in-features).
    // Frontend constants ([n_mels, 1+n_fft/2] filterbank, [win] window).
    if (!shape("audio.mel_filters", {c.encoder.n_mel, (int64_t) 1 + c.frontend.n_fft / 2}))
        return false;
    if (!shape("audio.mel_window", {(int64_t) c.frontend.win_length})) return false;
    // Conv stack: torch [480, ic, 3, 3] -> ne [3, 3, ic, 480]; three stride-2
    // k3 convs reduce the 128-bin mel axis to 16 (checked via enc.out below).
    if (!shape("enc.conv1.weight", {3, 3, 1, c.encoder.downsample_hidden})) return false;
    if (!shape("enc.conv1.bias", {c.encoder.downsample_hidden})) return false;
    if (!shape("enc.conv2.weight",
               {3, 3, c.encoder.downsample_hidden, c.encoder.downsample_hidden}))
        return false;
    if (!shape("enc.conv2.bias", {c.encoder.downsample_hidden})) return false;
    if (!shape("enc.conv3.weight",
               {3, 3, c.encoder.downsample_hidden, c.encoder.downsample_hidden}))
        return false;
    if (!shape("enc.conv3.bias", {c.encoder.downsample_hidden})) return false;
    // conv_out: bias-free Linear(downsample_hidden * n_mel/8 -> hidden). The
    // mel axis is downsampled by the conv stack's fixed factor 8 (three
    // stride-2 convs, encoder.cpp's ec.n_mel / 8); guard divisibility so a
    // future frontend with a non-multiple n_mel fails loudly here instead of
    // passing a truncated-division shape that mismatches the runtime graph.
    if (c.encoder.n_mel % 8 != 0) {
        err = "QWEN3 GGUF enc.n_mel must be divisible by the conv downsample factor (8)";
        return false;
    }
    if (!shape("enc.out.weight",
               {(int64_t) c.encoder.downsample_hidden * (c.encoder.n_mel / 8),
                c.encoder.hidden}))
        return false;
    if (!shape("enc.pos_embed", {c.encoder.hidden, c.encoder.max_pos_emb})) return false;
    if (!shape("enc.ln_post.weight", {c.encoder.hidden})) return false;
    if (!shape("enc.ln_post.bias", {c.encoder.hidden})) return false;
    // Projector: Linear(hidden -> hidden) + GELU + Linear(hidden -> output).
    if (!shape("proj.linear_1.weight", {c.projector.hidden, c.projector.hidden}))
        return false;
    if (!shape("proj.linear_1.bias", {c.projector.hidden})) return false;
    if (!shape("proj.linear_2.weight", {c.projector.hidden, c.projector.output_dim}))
        return false;
    if (!shape("proj.linear_2.bias", {c.projector.output_dim})) return false;
    if (!shape("llm.embed.weight", {c.llm.hidden, c.llm.vocab})) return false;
    if (!shape("llm.final_norm.weight", {c.llm.hidden})) return false;
    // Encoder layers: 16 tensors each (biased MHA + two biased LayerNorms +
    // biased FFN); MHA projects hidden -> hidden (kv heads == heads).
    for (uint32_t i = 0; i < c.encoder.n_layers; ++i) {
        char n[128];
        std::snprintf(n, sizeof n, "enc.blk.%u.", i);
        const size_t pre_len = std::strlen(n);
        // Append the tensor tail to the layer prefix in the SAME stack
        // buffer — no per-tensor heap string (load-time churn the original
        // require-loop avoided by snprintf-ing straight into n).
        const auto enc_shape = [&](const char* tail, std::initializer_list<int64_t> want) {
            return layer_shape(m, n, sizeof n, pre_len, tail, want, err);
        };
        if (!enc_shape("attn_norm.weight", {c.encoder.hidden})) return false;
        if (!enc_shape("attn_norm.bias", {c.encoder.hidden})) return false;
        if (!enc_shape("attn_q.weight", {c.encoder.hidden, c.encoder.hidden}))
            return false;
        if (!enc_shape("attn_q.bias", {c.encoder.hidden})) return false;
        if (!enc_shape("attn_k.weight", {c.encoder.hidden, c.encoder.hidden}))
            return false;
        if (!enc_shape("attn_k.bias", {c.encoder.hidden})) return false;
        if (!enc_shape("attn_v.weight", {c.encoder.hidden, c.encoder.hidden}))
            return false;
        if (!enc_shape("attn_v.bias", {c.encoder.hidden})) return false;
        if (!enc_shape("attn_o.weight", {c.encoder.hidden, c.encoder.hidden}))
            return false;
        if (!enc_shape("attn_o.bias", {c.encoder.hidden})) return false;
        if (!enc_shape("ffn_norm.weight", {c.encoder.hidden})) return false;
        if (!enc_shape("ffn_norm.bias", {c.encoder.hidden})) return false;
        if (!enc_shape("ff_up.weight", {c.encoder.hidden, c.encoder.ffn_dim}))
            return false;
        if (!enc_shape("ff_up.bias", {c.encoder.ffn_dim})) return false;
        if (!enc_shape("ff_down.weight", {c.encoder.ffn_dim, c.encoder.hidden}))
            return false;
        if (!enc_shape("ff_down.bias", {c.encoder.hidden})) return false;
    }
    // LLM layers: bias-free Qwen3 trunk with per-head q/k norm (11 each).
    // q/k/v project hidden -> {QW, KVW, KVW}; o projects the concatenated
    // heads QW back to hidden — the width relation the 0.6B exercises.
    const int64_t QW = (int64_t) c.llm.n_heads * c.llm.head_dim;
    const int64_t KVW = (int64_t) c.llm.n_kv_heads * c.llm.head_dim;
    for (uint32_t i = 0; i < c.llm.n_layers; ++i) {
        char n[128];
        std::snprintf(n, sizeof n, "llm.blk.%u.", i);
        const size_t pre_len = std::strlen(n);
        // Same stack-buffer reuse + truncation guard as the encoder loop.
        const auto llm_shape = [&](const char* tail, std::initializer_list<int64_t> want) {
            return layer_shape(m, n, sizeof n, pre_len, tail, want, err);
        };
        if (!llm_shape("attn_norm.weight", {c.llm.hidden})) return false;
        if (!llm_shape("attn.q.weight", {c.llm.hidden, QW})) return false;
        if (!llm_shape("attn.k.weight", {c.llm.hidden, KVW})) return false;
        if (!llm_shape("attn.v.weight", {c.llm.hidden, KVW})) return false;
        if (!llm_shape("attn.o.weight", {QW, c.llm.hidden})) return false;
        if (!llm_shape("attn.q_norm.weight", {c.llm.head_dim})) return false;
        if (!llm_shape("attn.k_norm.weight", {c.llm.head_dim})) return false;
        if (!llm_shape("ffn_norm.weight", {c.llm.hidden})) return false;
        if (!llm_shape("ffn.gate.weight", {c.llm.hidden, c.llm.intermediate}))
            return false;
        if (!llm_shape("ffn.up.weight", {c.llm.hidden, c.llm.intermediate}))
            return false;
        if (!llm_shape("ffn.down.weight", {c.llm.intermediate, c.llm.hidden}))
            return false;
    }
    return true;
}
} // namespace starling::ggml::qwen3
