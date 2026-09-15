#include "loader.hpp"
#include <cstdio>
#include <string>
#include <vector>

#include "ggml.h"
#include "lib/loader_kit.hpp"
namespace starling::ggml::voxtral {
namespace {
// GGUF-metadata helpers shared across the engines (lib/loader_kit.hpp).
using lib::f32;
using lib::str;

// Tensor-shape equation: `name` must have exactly the ggml dims in `want`.
// (GGUF stores row-major [out, in]; ggml exposes ne innermost-first, so a
// Linear(out <- in) weight reads ne0=in, ne1=out, and a Conv1d(OC, IC, K)
// weight reads ne0=K, ne1=IC, ne2=OC.) Presence is established by the
// require() pass; a missing tensor here is still a loud error.
bool shape_eq(const ModelLoader& m, const char* name,
              std::initializer_list<int64_t> want, std::string& err) {
    ggml_tensor* t = m.tensor(name);
    if (!t) {
        err = std::string("VOXTRAL GGUF missing required tensor: ") + name;
        return false;
    }
    bool ok = (int) want.size() == ggml_n_dims(t);
    int i = 0;
    for (int64_t w : want) {
        if (ok && t->ne[i] != w) ok = false;
        ++i;
    }
    if (!ok) {
        std::string got;
        for (int j = 0; j < ggml_n_dims(t); ++j) {
            if (j) got += ",";
            got += std::to_string(t->ne[j]);
        }
        std::string want_s;
        for (int64_t w : want) {
            if (!want_s.empty()) want_s += ",";
            want_s += std::to_string(w);
        }
        err = std::string("VOXTRAL GGUF shape mismatch on ") + name +
              " (ggml dims [" + got + "], want [" + want_s + "])";
        return false;
    }
    return true;
}
} // namespace

int64_t offline_padded_samples(int64_t n_samples) {
    if (n_samples < 0) return 0;
    const int64_t unit = 1280;  // raw samples per audio token
    const int64_t body = (n_samples + unit - 1) / unit * unit;
    return body + (int64_t)(32 + 17) * unit;
}

int64_t mel_frames(int64_t n_samples) {
    // Padded length is always a multiple of 1280 = 8 hops.
    return offline_padded_samples(n_samples) / 160;
}

int64_t audio_token_count(int64_t mel_T) {
    if (mel_T <= 0) return 0;
    // conv1 (k3 s1, left-pad 2) preserves length: (L+2-3)/1+1 = L.
    // conv2 (k3 s2, left-pad 1) halves: (L+1-3)/2+1 = L/2 (offline L even).
    // The projector groups by 4; offline lengths divide with no remainder.
    return (mel_T / 2) / 4;
}

bool VoxtralModel::load(const char* path, std::string& err) {
    if (!loader.load(path)) {
        err = loader.last_error();
        return false;
    }
    auto& c = config;
    const auto& m = loader;
#define U(field, key) do { if (!lib::u32(m, "voxtral." key, field, field, err)) return false; } while (0)
#define F(field, key) field = f32(m, "voxtral." key, field)
    // Frontend.
    U(c.frontend.sample_rate, "frontend.sample_rate");
    U(c.frontend.n_fft, "frontend.n_fft");
    U(c.frontend.win_length, "frontend.win_length");
    U(c.frontend.hop_length, "frontend.hop_length");
    U(c.frontend.n_mels, "frontend.n_mels");
    U(c.frontend.center, "frontend.center");
    U(c.frontend.unit_samples, "frontend.unit_samples");
    U(c.frontend.left_pad_tokens, "frontend.left_pad_tokens");
    U(c.frontend.right_pad_tokens, "frontend.right_pad_tokens");
    F(c.frontend.mel_floor, "frontend.mel_floor");
    F(c.frontend.log_mel_max, "frontend.log_mel_max");
    F(c.frontend.normalization_offset, "frontend.normalization_offset");
    F(c.frontend.normalization_divisor, "frontend.normalization_divisor");
    F(c.frontend.dynamic_range, "frontend.dynamic_range");
    c.frontend.mel_scale = str(m, "voxtral.frontend.mel_scale", "slaney");
    c.frontend.log = str(m, "voxtral.frontend.log", "log10");
    c.frontend.output_dtype = str(m, "voxtral.frontend.output_dtype", "bf16");
    // Encoder.
    U(c.encoder.num_mel_bins, "enc.num_mel_bins");
    U(c.encoder.n_layers, "enc.encoder_layers");
    U(c.encoder.d_model, "enc.d_model");
    U(c.encoder.n_heads, "enc.encoder_attention_heads");
    U(c.encoder.head_dim, "enc.head_dim");
    U(c.encoder.ffn_dim, "enc.encoder_ffn_dim");
    U(c.encoder.sliding_window, "enc.sliding_window");
    U(c.encoder.conv_kernel, "enc.conv_kernel");
    U(c.encoder.conv_left_pad1, "enc.conv_left_pad1");
    U(c.encoder.conv_left_pad2, "enc.conv_left_pad2");
    U(c.encoder.conv_stride2, "enc.conv_stride2");
    F(c.encoder.rope_theta, "enc.rope_theta");
    F(c.encoder.rms_norm_eps, "enc.rms_norm_eps");
    // Projector.
    U(c.projector.input_size, "proj.input_size");
    U(c.projector.output_size, "proj.output_size");
    U(c.projector.downsample, "proj.downsample");
    U(c.projector.mel_per_token, "proj.mel_per_token");
    c.projector.act = str(m, "voxtral.proj.act", "gelu");
    // LLM.
    U(c.llm.hidden, "llm.hidden_size");
    U(c.llm.n_layers, "llm.num_layers");
    U(c.llm.n_heads, "llm.num_heads");
    U(c.llm.n_kv_heads, "llm.num_kv_heads");
    U(c.llm.head_dim, "llm.head_dim");
    U(c.llm.intermediate, "llm.intermediate_size");
    U(c.llm.vocab, "llm.vocab_size");
    U(c.llm.sliding_window, "llm.sliding_window");
    U(c.llm.tied, "llm.tied");
    U(c.llm.num_delay_tokens, "llm.num_delay_tokens");
    U(c.llm.time_embedding_dim, "llm.time_embedding_dim");
    U(c.llm.ada_bottleneck, "llm.ada_bottleneck");
    U(c.llm.max_cache, "llm.max_cache_len");
    F(c.llm.rope_theta, "llm.rope_theta");
    F(c.llm.rms_norm_eps, "llm.rms_norm_eps");
    F(c.llm.time_embedding_theta, "llm.time_embedding_theta");
    // Token ids + generation.
    {
        uint32_t t;
#define T(field, key) do { if (!lib::u32(m, "voxtral." key, (uint32_t) field, t, err)) return false; field = (int32_t) t; } while (0)
        T(c.bos_token_id, "bos_token_id");
        T(c.eos_token_id, "eos_token_id");
        T(c.pad_token_id, "pad_token_id");
        T(c.streaming_pad_id, "streaming_pad_id");
#undef T
    }
    U(c.left_pad_tokens, "left_pad_tokens");
    U(c.right_pad_tokens, "right_pad_tokens");
    U(c.max_new_tokens, "max_new_tokens");
#undef U
#undef F
    std::vector<int64_t> a;
    if (m.kv_arr_int("voxtral.prompt_prefix", a))
        for (auto v : a) c.prompt_prefix.push_back((int32_t) v);

    // --- Validate untrusted GGUF metadata (mirror ark/loader.cpp). The
    // consumers divide by these (encoder head grouping; llm head grouping
    // h / (n_heads/n_kv_heads)), so a zero or inconsistent value is div-by-zero.
    if (!lib::check_gguf_header(m, "voxtral", "VOXTRAL", {"bf16_exact", "f16"}, err))
        return false;
#define POS(v, name) do { if (!(v)) { err = "VOXTRAL GGUF " name " must be positive"; return false; } } while (0)
    POS(c.encoder.n_layers, "enc.encoder_layers");
    POS(c.encoder.d_model, "enc.d_model");
    POS(c.encoder.n_heads, "enc.encoder_attention_heads");
    POS(c.encoder.head_dim, "enc.head_dim");
    POS(c.encoder.ffn_dim, "enc.encoder_ffn_dim");
    POS(c.encoder.sliding_window, "enc.sliding_window");
    POS(c.projector.input_size, "proj.input_size");
    POS(c.projector.output_size, "proj.output_size");
    POS(c.projector.downsample, "proj.downsample");
    POS(c.projector.mel_per_token, "proj.mel_per_token");
    POS(c.llm.hidden, "llm.hidden_size");
    POS(c.llm.n_layers, "llm.num_layers");
    POS(c.llm.n_heads, "llm.num_heads");
    POS(c.llm.n_kv_heads, "llm.num_kv_heads");
    POS(c.llm.head_dim, "llm.head_dim");
    POS(c.llm.intermediate, "llm.intermediate_size");
    POS(c.llm.vocab, "llm.vocab_size");
    POS(c.llm.num_delay_tokens, "llm.num_delay_tokens");
    POS(c.llm.time_embedding_dim, "llm.time_embedding_dim");
    POS(c.llm.ada_bottleneck, "llm.ada_bottleneck");
#undef POS
    // Attention projects to heads*head_dim (the encoder is full MHA: the
    // attribute_map folds num_key_value_heads into num_attention_heads, and
    // the checkpoint's q/k/v rows confirm it). The hidden width is NOT the
    // attention width; every relation below is an equation between metadata
    // fields and the actual tensor shapes, so any consistent dims validate.
    const int64_t AW = (int64_t) c.encoder.n_heads * c.encoder.head_dim;
    const int64_t QW = (int64_t) c.llm.n_heads * c.llm.head_dim;
    const int64_t KVW = (int64_t) c.llm.n_kv_heads * c.llm.head_dim;
    if (c.llm.n_heads % c.llm.n_kv_heads != 0) {
        err = "VOXTRAL GGUF llm.num_heads must be a multiple of llm.num_kv_heads";
        return false;
    }
    // The projector groups downsample encoder frames: input == d_model * ds.
    if ((int64_t) c.projector.input_size !=
        (int64_t) c.encoder.d_model * c.projector.downsample) {
        err = "VOXTRAL GGUF proj.input_size != enc.d_model * proj.downsample";
        return false;
    }
    if (c.frontend.n_fft != 400 || c.frontend.win_length != 400 ||
        c.frontend.n_mels != 128 || c.frontend.hop_length != 160) {
        err = "unsupported VOXTRAL frontend metadata (requires n_fft/win=400, hop=160, n_mels=128)";
        return false;
    }

    // Require every expected tensor so a structural change fails loudly, then
    // check each weight's shape against the metadata relations above.
    const int64_t Dm = c.encoder.d_model, Fm = c.encoder.ffn_dim;
    const int64_t H = c.llm.hidden, I = c.llm.intermediate;
    const int64_t V = c.llm.vocab, Ad = c.llm.ada_bottleneck;
    const int64_t Pm = c.encoder.num_mel_bins, Po = c.projector.output_size;
#define SHAPE(name, ...) do { if (!shape_eq(m, name, {__VA_ARGS__}, err)) return false; } while (0)
    SHAPE("enc.conv1.weight", 3, Pm, Dm);
    SHAPE("enc.conv1.bias", Dm);
    SHAPE("enc.conv2.weight", 3, Dm, Dm);
    SHAPE("enc.conv2.bias", Dm);
    SHAPE("enc.final_norm.weight", Dm);
    SHAPE("proj.fc0.weight", (int64_t) c.projector.input_size, Po);
    SHAPE("proj.fc2.weight", Po, Po);
    SHAPE("llm.embed.weight", H, V);
    SHAPE("llm.final_norm.weight", H);
    SHAPE("llm.t_cond", (int64_t) c.llm.time_embedding_dim);
    // Encoder layers: attn_norm(w), attn.q(w+b), attn.k(w),
    // attn.v(w+b), attn.o(w+b), ffn_norm(w), ffn.gate(w), ffn.up(w),
    // ffn.down(w+b) = 13 each. The q/k/v rows are the metadata attention
    // width AW (full MHA: kv heads == q heads), o cols are AW, gate/up rows
    // are the ffn width Fm, down cols are Fm.
    for (uint32_t i = 0; i < c.encoder.n_layers; ++i) {
        char n[128];
        std::snprintf(n, sizeof n, "enc.blk.%u.", i);
        const std::string pre = n;
#define LSHAPE(tail, ...) do { \
            if (!shape_eq(m, (pre + tail).c_str(), {__VA_ARGS__}, err)) return false; \
        } while (0)
        LSHAPE("attn_norm.weight", Dm);
        LSHAPE("attn.q.weight", Dm, AW);
        LSHAPE("attn.q.bias", AW);
        LSHAPE("attn.k.weight", Dm, AW);
        LSHAPE("attn.v.weight", Dm, AW);
        LSHAPE("attn.v.bias", AW);
        LSHAPE("attn.o.weight", AW, Dm);
        LSHAPE("attn.o.bias", Dm);
        LSHAPE("ffn_norm.weight", Dm);
        LSHAPE("ffn.gate.weight", Dm, Fm);
        LSHAPE("ffn.up.weight", Dm, Fm);
        LSHAPE("ffn.down.weight", Fm, Dm);
        LSHAPE("ffn.down.bias", Dm);
#undef LSHAPE
    }
    // LLM layers: attn_norm(w), attn.{q,k,v,o}(w), ffn_norm(w),
    // ffn.{gate,up,down}(w), ada.fc0(w), ada.fc2(w) = 11 each, no biases.
    for (uint32_t i = 0; i < c.llm.n_layers; ++i) {
        char n[128];
        std::snprintf(n, sizeof n, "llm.blk.%u.", i);
        const std::string pre = n;
#define LSHAPE(tail, ...) do { \
            if (!shape_eq(m, (pre + tail).c_str(), {__VA_ARGS__}, err)) return false; \
        } while (0)
        LSHAPE("attn_norm.weight", H);
        LSHAPE("attn.q.weight", H, QW);
        LSHAPE("attn.k.weight", H, KVW);
        LSHAPE("attn.v.weight", H, KVW);
        LSHAPE("attn.o.weight", QW, H);
        LSHAPE("ffn_norm.weight", H);
        LSHAPE("ffn.gate.weight", H, I);
        LSHAPE("ffn.up.weight", H, I);
        LSHAPE("ffn.down.weight", I, H);
        LSHAPE("ada.fc0.weight", H, Ad);
        LSHAPE("ada.fc2.weight", Ad, H);
#undef LSHAPE
    }
#undef SHAPE
    // Create graph constants BEFORE the first realize_weights() call. Adding
    // a leaf afterwards leaves a host pointer in an otherwise device graph.
    loader.add_owned_tensor("llm.ada_ones",
                            std::vector<float>((size_t) H, 1.0f), H, 1);
    return true;
}
} // namespace starling::ggml::voxtral
