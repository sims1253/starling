#include "loader.hpp"
#include <cstdio>
#include <vector>

#include "lib/loader_kit.hpp"
namespace starling::ggml::audex {
namespace {
// GGUF-metadata helpers shared across the engines (lib/loader_kit.hpp).
using lib::f32;
using lib::f64;
using lib::str;
} // namespace

bool AudexModel::load(const char* path, std::string& err) {
    if (!loader.load(path)) {
        err = loader.last_error();
        return false;
    }
    auto& c = config;
    const auto& m = loader;
#define U(field, key) do { if (!lib::u32(m, "audex." key, field, field, err)) return false; } while (0)
#define F(field, key) field = f32(m, "audex." key, field)
    // Frontend (WhisperFeatureExtractor: 128 bins, n_fft 400, hop 160,
    // fixed 30 s clips padded to n_samples).
    U(c.frontend.sample_rate, "frontend.sample_rate");
    U(c.frontend.n_fft, "frontend.n_fft");
    U(c.frontend.win_length, "frontend.win_length");
    U(c.frontend.hop_length, "frontend.hop_length");
    U(c.frontend.n_mels, "frontend.n_mels");
    U(c.frontend.power, "frontend.power");
    U(c.frontend.chunk_length, "frontend.chunk_length");
    U(c.frontend.n_samples, "frontend.n_samples");
    F(c.frontend.mel_floor, "frontend.mel_floor");
    F(c.frontend.normalization_offset, "frontend.normalization_offset");
    F(c.frontend.normalization_divisor, "frontend.normalization_divisor");
    F(c.frontend.dynamic_range, "frontend.dynamic_range");
    // Encoder (full-attention whisper-shaped stack).
    U(c.encoder.n_mel, "enc.n_mel");
    U(c.encoder.hidden, "enc.hidden");
    U(c.encoder.n_layers, "enc.layers");
    U(c.encoder.n_heads, "enc.heads");
    U(c.encoder.head_dim, "enc.head_dim");
    U(c.encoder.ffn_dim, "enc.ffn_dim");
    U(c.encoder.max_pos_emb, "enc.max_pos_emb");
    U(c.encoder.out_frames, "enc.out_frames");
    F(c.encoder.layer_norm_eps, "enc.layer_norm_eps");
    // Projector (RMSNorm -> fc1 -> relu2 -> fc2, bias-free).
    U(c.projector.hidden, "proj.hidden");
    U(c.projector.intermediate, "proj.intermediate");
    U(c.projector.output_dim, "proj.output_dim");
    F(c.projector.norm_eps, "proj.norm_eps");
    // LLM (Nemotron-Dense trunk: untied lm_head, no q/k norm, relu2 MLP).
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
        if (m.kv_int("audex.llm.tied_embeddings", tied)) c.llm.tied_embeddings = tied != 0;
        if (m.kv_int("audex.llm.has_qk_norm", qn)) c.llm.has_qk_norm = qn != 0;
    }
    // Token ids + generation + chunk policy.
    {
        uint32_t t;
#define T(field, key) do { if (!lib::u32(m, "audex." key, (uint32_t) field, t, err)) return false; field = (int32_t) t; } while (0)
        T(c.audio_token_id, "audio_token_id");
        T(c.pad_token_id, "pad_token_id");
        T(c.eos_token_id, "eos_token_id");
        T(c.sound_start_token_id, "sound_start_token_id");
        T(c.sound_end_token_id, "sound_end_token_id");
#undef T
    }
    U(c.sound_embedding_size, "sound_embedding_size");
    U(c.max_new_tokens, "max_new_tokens");
    c.chunk_seconds = f64(m, "audex.chunk_seconds", 30.0);
#undef U
#undef F
    std::vector<int64_t> a;
    if (m.kv_arr_int("audex.prompt_prefix", a))
        for (auto v : a) c.prompt_prefix.push_back((int32_t) v);
    a.clear();
    if (m.kv_arr_int("audex.prompt_suffix", a))
        for (auto v : a) c.prompt_suffix.push_back((int32_t) v);

    // --- Validate untrusted GGUF metadata (mirror qwen3/loader.cpp). ---
    if (!lib::check_gguf_header(m, "audex", "AUDEX", {"bf16_exact", "quantized"}, err))
        return false;
#define POS(v, name) do { if (!(v)) { err = "AUDEX GGUF " name " must be positive"; return false; } } while (0)
    POS(c.encoder.n_layers, "enc.layers");
    POS(c.encoder.hidden, "enc.hidden");
    POS(c.encoder.n_heads, "enc.heads");
    POS(c.encoder.head_dim, "enc.head_dim");
    POS(c.encoder.ffn_dim, "enc.ffn_dim");
    POS(c.encoder.max_pos_emb, "enc.max_pos_emb");
    POS(c.encoder.out_frames, "enc.out_frames");
    POS(c.projector.hidden, "proj.hidden");
    POS(c.projector.intermediate, "proj.intermediate");
    POS(c.projector.output_dim, "proj.output_dim");
    POS(c.llm.hidden, "llm.hidden_size");
    POS(c.llm.n_layers, "llm.num_layers");
    POS(c.llm.n_heads, "llm.num_heads");
    POS(c.llm.n_kv_heads, "llm.num_kv_heads");
    POS(c.llm.head_dim, "llm.head_dim");
    POS(c.llm.intermediate, "llm.intermediate_size");
    POS(c.llm.vocab, "llm.vocab_size");
    POS(c.llm.max_cache, "llm.max_cache_len");
    POS(c.sound_embedding_size, "sound_embedding_size");
    POS(c.max_new_tokens, "max_new_tokens");
    POS(c.frontend.n_samples, "frontend.n_samples");
    POS(c.frontend.hop_length, "frontend.hop_length");
#undef POS
    if (c.encoder.n_mel != c.frontend.n_mels) {
        err = "AUDEX GGUF enc.n_mel must equal frontend.n_mels";
        return false;
    }
    if (c.encoder.hidden != c.encoder.n_heads * c.encoder.head_dim) {
        err = "AUDEX GGUF enc.hidden != enc.heads * enc.head_dim";
        return false;
    }
    if (c.llm.hidden != c.llm.n_heads * c.llm.head_dim) {
        err = "AUDEX GGUF llm.hidden_size != llm.num_heads * llm.head_dim";
        return false;
    }
    if (c.llm.n_heads % c.llm.n_kv_heads != 0) {
        err = "AUDEX GGUF llm.num_heads must be a multiple of llm.num_kv_heads";
        return false;
    }
    if (c.frontend.n_fft != 400 || c.frontend.win_length != 400 ||
        c.frontend.n_mels != 128 || c.frontend.hop_length != 160) {
        err = "unsupported AUDEX frontend metadata (requires n_fft=400, win=400, hop=160, n_mels=128)";
        return false;
    }
    // The encoder graph is fixed-shape: the whisper frontend must produce
    // exactly max_pos_emb*2 = 3000 frames per clip (expected_seq_length in
    // Qwen2AudioEncoder.forward raises otherwise) and the avg-pooler then
    // halves them to the baked 750 <so_embedding> slots.
    if (c.frontend.n_samples / c.frontend.hop_length != 2 * (int64_t) c.encoder.max_pos_emb ||
        c.frontend.n_samples % c.frontend.hop_length != 0) {
        err = "AUDEX GGUF frontend.n_samples must be 2*enc.max_pos_emb*frontend.hop_length";
        return false;
    }
    if (c.encoder.max_pos_emb % 2 != 0 || c.encoder.out_frames != c.encoder.max_pos_emb / 2) {
        err = "AUDEX GGUF enc.out_frames must be half of enc.max_pos_emb";
        return false;
    }
    if (c.llm.tied_embeddings) {
        err = "unsupported AUDEX GGUF: the Nemotron-Dense trunk carries an untied lm_head";
        return false;
    }
    if (c.llm.has_qk_norm) {
        err = "unsupported AUDEX GGUF: the Nemotron-Dense trunk has no q/k norm";
        return false;
    }
    if (str(m, "audex.llm.hidden_act", "relu2") != "relu2") {
        err = "unsupported AUDEX GGUF: llm.hidden_act must be relu2";
        return false;
    }
    if (str(m, "audex.proj.activation", "relu2") != "relu2") {
        err = "unsupported AUDEX GGUF: proj.activation must be relu2";
        return false;
    }
    if (c.prompt_prefix.empty() || c.prompt_suffix.empty()) {
        err = "AUDEX GGUF missing prompt prefix/suffix arrays";
        return false;
    }
    // The prompt layout plus the fixed audio slots must fit the KV cache
    // with decode headroom (mirrors the Python LLMMega max_new_tokens guard).
    if ((int64_t) c.prompt_prefix.size() + c.sound_embedding_size +
            (int64_t) c.prompt_suffix.size() + 1 >=
        (int64_t) c.llm.max_cache) {
        err = "AUDEX GGUF prompt + audio slots do not fit llm.max_cache_len";
        return false;
    }

    // Require every expected tensor so a structural change fails loudly.
    for (const char* n : {"audio.mel_filters", "audio.mel_window",
                          "enc.conv1.weight", "enc.conv1.bias",
                          "enc.conv2.weight", "enc.conv2.bias",
                          "enc.pos_embed",
                          "enc.ln_post.weight", "enc.ln_post.bias",
                          "proj.norm.weight", "proj.fc1.weight", "proj.fc2.weight",
                          "llm.embed.weight", "llm.final_norm.weight",
                          "llm.lm_head.weight"})
        if (!lib::require(m, n, "AUDEX", err)) return false;
    // Encoder layers: 15 tensors each (biased q/v/out + bias-free k, two
    // biased LayerNorms, biased FFN).
    for (uint32_t i = 0; i < c.encoder.n_layers; ++i) {
        char n[128];
        for (const char* tail : {"attn_norm.weight", "attn_norm.bias",
                                 "attn_q.weight", "attn_q.bias",
                                 "attn_k.weight",
                                 "attn_v.weight", "attn_v.bias",
                                 "attn_o.weight", "attn_o.bias",
                                 "ffn_norm.weight", "ffn_norm.bias",
                                 "ff_up.weight", "ff_up.bias",
                                 "ff_down.weight", "ff_down.bias"}) {
            std::snprintf(n, sizeof n, "enc.blk.%u.%s", i, tail);
            if (!lib::require(m, n, "AUDEX", err)) return false;
        }
    }
    // LLM layers: bias-free Nemotron trunk = 8 each (no gate, no q/k norm).
    for (uint32_t i = 0; i < c.llm.n_layers; ++i) {
        char n[128];
        for (const char* tail : {"attn_norm.weight", "attn.q.weight", "attn.k.weight",
                                 "attn.v.weight", "attn.o.weight",
                                 "ffn_norm.weight",
                                 "ffn.up.weight", "ffn.down.weight"}) {
            std::snprintf(n, sizeof n, "llm.blk.%u.%s", i, tail);
            if (!lib::require(m, n, "AUDEX", err)) return false;
        }
    }
    return true;
}
} // namespace starling::ggml::audex
