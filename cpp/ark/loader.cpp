#include "loader.hpp"
#include <cstdio>
#include <vector>

namespace starling::ggml::ark {
namespace {
uint32_t u32(const ModelLoader& m, const char* k, uint32_t d) {
    int64_t v;
    return m.kv_int(k, v) ? (uint32_t) v : d;
}
float f32(const ModelLoader& m, const char* k, float d) {
    double v;
    return m.kv_float(k, v) ? (float) v : d;
}
std::string str(const ModelLoader& m, const char* k, const char* d) {
    std::string v;
    return m.kv_str(k, v) ? v : d;
}
bool require(const ModelLoader& m, const std::string& n, std::string& err) {
    if (m.tensor(n.c_str())) return true;
    err = "ARK GGUF missing required tensor: " + n;
    return false;
}
} // namespace

int64_t audio_token_count(int64_t mel_frames, uint32_t merge_factor) {
    if (mel_frames <= 0) return 1;
    const uint32_t mf = merge_factor > 0 ? merge_factor : 1;
    int64_t downsampled = (mel_frames + 1) / 2;
    int64_t merged = downsampled / mf;
    return merged > 0 ? merged : 1;
}

bool ArkModel::load(const char* path, std::string& err) {
    if (!loader.load(path)) {
        err = loader.last_error();
        return false;
    }
    auto& c = config;
    const auto& m = loader;
#define U(field, key) field = u32(m, "ark." key, field)
#define F(field, key) field = f32(m, "ark." key, field)
    // Frontend.
    U(c.frontend.sample_rate, "frontend.sample_rate");
    U(c.frontend.n_fft, "frontend.n_fft");
    U(c.frontend.win_length, "frontend.win_length");
    U(c.frontend.hop_length, "frontend.hop_length");
    U(c.frontend.n_mels, "frontend.n_mels");
    U(c.frontend.power, "frontend.power");
    U(c.frontend.nb_max_frames, "frontend.nb_max_frames");
    U(c.frontend.n_samples, "frontend.n_samples");
    U(c.frontend.chunk_length, "frontend.chunk_length");
    F(c.frontend.mel_floor, "frontend.mel_floor");
    F(c.frontend.normalization_offset, "frontend.normalization_offset");
    F(c.frontend.normalization_divisor, "frontend.normalization_divisor");
    F(c.frontend.dynamic_range, "frontend.dynamic_range");
    c.frontend.pad_mode = str(m, "ark.frontend.pad_mode", "reflect");
    c.frontend.mel_scale = str(m, "ark.frontend.mel_scale", "slaney");
    c.frontend.mel_norm = str(m, "ark.frontend.mel_norm", "slaney");
    c.frontend.log = str(m, "ark.frontend.log", "log");
    c.frontend.output_dtype = str(m, "ark.frontend.output_dtype", "bf16");
    // Encoder.
    U(c.encoder.num_mel_bins, "enc.num_mel_bins");
    U(c.encoder.n_layers, "enc.encoder_layers");
    U(c.encoder.d_model, "enc.d_model");
    U(c.encoder.n_heads, "enc.encoder_attention_heads");
    U(c.encoder.head_dim, "enc.head_dim");
    U(c.encoder.ffn_dim, "enc.encoder_ffn_dim");
    U(c.encoder.max_source_positions, "enc.max_source_positions");
    U(c.encoder.conv_kernel, "enc.conv_kernel");
    U(c.encoder.merge_factor, "enc.merge_factor");
    U(c.encoder.use_rope, "enc.use_rope");
    U(c.encoder.rope_dim, "enc.rope_dim");
    F(c.encoder.rope_base, "enc.rope_base");
    F(c.encoder.layer_norm_eps, "enc.layer_norm_eps");
    // Adapter.
    U(c.adapter.input_size, "adapter.input_size");
    U(c.adapter.hidden_size, "adapter.hidden_size");
    U(c.adapter.output_size, "adapter.output_size");
    U(c.adapter.merge_factor, "adapter.merge_factor");
    c.adapter.act = str(m, "ark.adapter.act", "gelu");
    // LLM.
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
    c.llm.rope_scaling = str(m, "ark.llm.rope_scaling", "none");
    {
        int64_t tied = 1;
        if (m.kv_int("ark.llm.tied_embeddings", tied)) c.llm.tied_embeddings = tied != 0;
        int64_t qn = 0;
        if (m.kv_int("ark.llm.has_qk_norm", qn)) c.llm.has_qk_norm = qn != 0;
    }
    // Token ids + generation.
    c.audio_token_id = (int32_t) u32(m, "ark.audio_token_id", c.audio_token_id);
    c.begin_audio_id = (int32_t) u32(m, "ark.begin_audio_id", c.begin_audio_id);
    c.end_audio_id = (int32_t) u32(m, "ark.end_audio_id", c.end_audio_id);
    c.user_id = (int32_t) u32(m, "ark.user_id", c.user_id);
    c.assistant_id = (int32_t) u32(m, "ark.assistant_id", c.assistant_id);
    c.pad_token_id = (int32_t) u32(m, "ark.pad_token_id", c.pad_token_id);
    c.bos_token_id = (int32_t) u32(m, "ark.bos_token_id", c.bos_token_id);
    c.eos_token_id = (int32_t) u32(m, "ark.eos_token_id", c.eos_token_id);
    U(c.max_new_tokens, "max_new_tokens");
    c.default_instruction = str(m, "ark.default_instruction", c.default_instruction.c_str());
#undef U
#undef F
    std::vector<int64_t> a;
    if (m.kv_arr_int("ark.prompt_prefix", a))
        for (auto v : a) c.prompt_prefix.push_back((int32_t) v);
    a.clear();
    if (m.kv_arr_int("ark.prompt_suffix", a))
        for (auto v : a) c.prompt_suffix.push_back((int32_t) v);

    // --- Validate untrusted GGUF metadata (mirror moss/loader.cpp). The
    // consumers divide by these (encoder head grouping; llm head grouping
    // h / (n_heads/n_kv_heads)), so a zero or inconsistent value is div-by-zero.
    if (std::string arch; m.kv_str("general.architecture", arch) && arch != "ark") {
        err = "unsupported ARK GGUF architecture: " + arch;
        return false;
    }
    if (std::string prof; m.kv_str("starling.numeric_profile", prof) &&
        prof != "bf16_exact" && prof != "f16" && prof != "q8_0") {
        err = "unsupported ARK numeric profile: " + prof;
        return false;
    }
    if (int64_t fv; m.kv_int("starling.format_version", fv) && fv != 1) {
        err = "unsupported Starling GGUF format version: " + std::to_string(fv);
        return false;
    }
    if (m.tensor_names().empty()) {
        err = "ARK GGUF contains no tensors";
        return false;
    }
#define POS(v, name) do { if (!(v)) { err = "ARK GGUF " name " must be positive"; return false; } } while (0)
    POS(c.encoder.n_layers, "enc.encoder_layers");
    POS(c.encoder.d_model, "enc.d_model");
    POS(c.encoder.n_heads, "enc.encoder_attention_heads");
    POS(c.encoder.head_dim, "enc.head_dim");
    POS(c.encoder.ffn_dim, "enc.encoder_ffn_dim");
    POS(c.encoder.merge_factor, "enc.merge_factor");
    POS(c.adapter.input_size, "adapter.input_size");
    POS(c.adapter.output_size, "adapter.output_size");
    POS(c.llm.hidden, "llm.hidden_size");
    POS(c.llm.n_heads, "llm.num_heads");
    POS(c.llm.n_kv_heads, "llm.num_kv_heads");
    POS(c.llm.head_dim, "llm.head_dim");
    POS(c.llm.intermediate, "llm.intermediate_size");
    POS(c.llm.vocab, "llm.vocab_size");
#undef POS
    if (c.encoder.d_model != c.encoder.n_heads * c.encoder.head_dim) {
        err = "ARK GGUF enc.d_model != enc.encoder_attention_heads * enc.head_dim";
        return false;
    }
    if (c.llm.hidden != c.llm.n_heads * c.llm.head_dim) {
        err = "ARK GGUF llm.hidden_size != llm.num_heads * llm.head_dim";
        return false;
    }
    if (c.llm.n_heads % c.llm.n_kv_heads != 0) {
        err = "ARK GGUF llm.num_heads must be a multiple of llm.num_kv_heads";
        return false;
    }
    if (c.frontend.n_fft != 400 || c.frontend.win_length != 400 ||
        c.frontend.n_mels != 128 || c.frontend.hop_length != 160) {
        err = "unsupported ARK frontend metadata (requires n_fft/win=400, hop=160, n_mels=128)";
        return false;
    }

    // Require every expected tensor so a structural change fails loudly.
    for (const char* n : {"audio.mel_filters", "audio.mel_window", "enc.rope_cos",
                          "enc.rope_sin", "enc.conv1.weight", "enc.conv1.bias",
                          "enc.conv2.weight", "enc.conv2.bias", "enc.ln_post.weight",
                          "enc.ln_post.bias", "adapter.fc0.weight", "adapter.fc0.bias",
                          "adapter.fc2.weight", "adapter.fc2.bias", "llm.embed.weight"})
        if (!require(m, n, err)) return false;
    // Encoder layers: attn_norm(w+b), attn.q(w+b), attn.k(w), attn.v(w+b),
    // attn.o(w+b), ffn_norm(w+b), ffn.fc1(w+b), ffn.fc2(w+b) = 15 each.
    for (uint32_t i = 0; i < c.encoder.n_layers; ++i) {
        char n[128];
        for (const char* tail : {"attn_norm.weight", "attn_norm.bias", "attn.q.weight",
                                 "attn.q.bias", "attn.k.weight", "attn.v.weight",
                                 "attn.v.bias", "attn.o.weight", "attn.o.bias",
                                 "ffn_norm.weight", "ffn_norm.bias", "ffn.fc1.weight",
                                 "ffn.fc1.bias", "ffn.fc2.weight", "ffn.fc2.bias"}) {
            std::snprintf(n, sizeof n, "enc.blk.%u.%s", i, tail);
            if (!require(m, n, err)) return false;
        }
    }
    // LLM layers: attn_norm(w), attn.q(w+b), attn.k(w+b), attn.v(w+b), attn.o(w),
    // ffn_norm(w), ffn.gate(w), ffn.up(w), ffn.down(w) = 12 each (no q_norm/k_norm).
    for (uint32_t i = 0; i < c.llm.n_layers; ++i) {
        char n[128];
        for (const char* tail : {"attn_norm.weight", "attn.q.weight", "attn.q.bias",
                                 "attn.k.weight", "attn.k.bias", "attn.v.weight",
                                 "attn.v.bias", "attn.o.weight", "ffn_norm.weight",
                                 "ffn.gate.weight", "ffn.up.weight", "ffn.down.weight"}) {
            std::snprintf(n, sizeof n, "llm.blk.%u.%s", i, tail);
            if (!require(m, n, err)) return false;
        }
    }
    return require(m, "llm.final_norm.weight", err);
}
} // namespace starling::ggml::ark
