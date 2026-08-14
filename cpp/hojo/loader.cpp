#include "loader.hpp"
#include "lib/loader_kit.hpp"
#include <cstdio>
#include <string>
#include <vector>

namespace starling::ggml::hojo {
namespace {
// GGUF-metadata helpers shared across the engines (lib/loader_kit.hpp).
using lib::f64;
using lib::str;
} // namespace

// Tower conv output length for one mel window: three stride-2 conv2d, each
// ((L-1)//2+1). Matches _get_feat_extract_output_lengths over the conv path.
int64_t tower_conv_output_length(int64_t mel_frames) {
    int64_t l = mel_frames;
    for (int i = 0; i < 3; ++i) l = (l - 1) / 2 + 1;
    return l;
}

bool HojoModel::load(const char* path, std::string& err) {
    if (!loader.load(path)) {
        err = loader.last_error();
        return false;
    }
    auto& c = config;
    const auto& m = loader;
#define U(field, key) do { if (!lib::u32(m, "hojo." key, field, field, err)) return false; } while (0)
#define D(field, key) field = f64(m, "hojo." key, field)
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
    D(c.frontend.mel_floor, "frontend.mel_floor");
    D(c.frontend.normalization_offset, "frontend.normalization_offset");
    D(c.frontend.normalization_divisor, "frontend.normalization_divisor");
    D(c.frontend.dynamic_range, "frontend.dynamic_range");
    c.frontend.pad_mode = str(m, "hojo.frontend.pad_mode", "reflect");
    c.frontend.mel_scale = str(m, "hojo.frontend.mel_scale", "slaney");
    c.frontend.mel_norm = str(m, "hojo.frontend.mel_norm", "slaney");
    c.frontend.log = str(m, "hojo.frontend.log", "log");
    // Validate the base frontend params BEFORE the chunk override below: the
    // override derives nb_max_frames = n_samples / hop_length, so a zero or
    // invalid hop_length (untrusted GGUF) would divide by zero here.
    if (c.frontend.n_fft != 400 || c.frontend.win_length != 400 ||
        c.frontend.n_mels != 128 || c.frontend.hop_length != 160) {
        err = "unsupported Hojo frontend metadata (requires n_fft/win=400, hop=160, n_mels=128)";
        return false;
    }
    // hojo_asr_model.HOJO_ASR loads the Whisper extractor with chunk_length=40
    // (overriding the base whisper-large-v3 preprocessor_config.json's
    // chunk_length=30). The GGUF stores the BASE Whisper config, so override
    // here to match the reference: audio is truncated to 40s (640000 samples)
    // before the STFT, yielding at most 4000 mel frames.
    c.frontend.chunk_length = 40;
    c.frontend.n_samples = c.frontend.chunk_length * c.frontend.sample_rate;
    c.frontend.nb_max_frames = c.frontend.n_samples / c.frontend.hop_length;
    // Tower.
    U(c.tower.num_mel_bins, "tower.num_mel_bins");
    U(c.tower.d_model, "tower.d_model");
    U(c.tower.encoder_layers, "tower.encoder_layers");
    U(c.tower.encoder_attention_heads, "tower.encoder_attention_heads");
    U(c.tower.head_dim, "tower.head_dim");
    U(c.tower.encoder_ffn_dim, "tower.encoder_ffn_dim");
    U(c.tower.downsample_hidden_size, "tower.downsample_hidden_size");
    U(c.tower.output_dim, "tower.output_dim");
    U(c.tower.max_source_positions, "tower.max_source_positions");
    U(c.tower.conv_kernel, "tower.conv_kernel");
    D(c.tower.layer_norm_eps, "tower.layer_norm_eps");
    U(c.tower.n_window, "tower.n_window");
    U(c.tower.n_window_infer, "tower.n_window_infer");
    U(c.tower.conv_chunksize, "tower.conv_chunksize");
    c.tower.activation_function = str(m, "hojo.tower.activation_function", "gelu");
    // Bottleneck.
    U(c.bottleneck.input_size, "bottleneck.input_size");
    U(c.bottleneck.output_size, "bottleneck.output_size");
    U(c.bottleneck.linear_units, "bottleneck.linear_units");
    U(c.bottleneck.num_blocks, "bottleneck.num_blocks");
    U(c.bottleneck.attention_heads, "bottleneck.attention_heads");
    U(c.bottleneck.cnn_module_kernel, "bottleneck.cnn_module_kernel");
    U(c.bottleneck.max_len, "bottleneck.max_len");
    D(c.bottleneck.norm_eps, "bottleneck.norm_eps");
    c.bottleneck.input_layer = str(m, "hojo.bottleneck.input_layer", "linear");
    c.bottleneck.pos_enc_layer_type = str(m, "hojo.bottleneck.pos_enc_layer_type", "rel_pos");
    c.bottleneck.selfattention_layer_type = str(m, "hojo.bottleneck.selfattention_layer_type", "rel_selfattn");
    c.bottleneck.activation_type = str(m, "hojo.bottleneck.activation_type", "swish");
    c.bottleneck.cnn_module_norm = str(m, "hojo.bottleneck.cnn_module_norm", "batch_norm");
    {
        int64_t mac = 1, nb = 1;
        if (m.kv_int("hojo.bottleneck.macaron_style", mac)) c.bottleneck.macaron_style = mac != 0;
        if (m.kv_int("hojo.bottleneck.normalize_before", nb)) c.bottleneck.normalize_before = nb != 0;
    }
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
    D(c.llm.rope_theta, "llm.rope_theta");
    D(c.llm.rms_norm_eps, "llm.rms_norm_eps");
    c.llm.rope_scaling = str(m, "hojo.llm.rope_scaling", "none");
    {
        int64_t tied = 0, qn = 1;
        if (m.kv_int("hojo.llm.tied_embeddings", tied)) c.llm.tied_embeddings = tied != 0;
        if (m.kv_int("hojo.llm.has_qk_norm", qn)) c.llm.has_qk_norm = qn != 0;
    }
    // Decode (beam-4). max_new_tokens falls back to the top-level key for
    // backward compatibility with the converter's two KV locations.
    U(c.decode.num_beams, "decode.num_beams");
    U(c.decode.min_length, "decode.min_length");
    {
        int64_t mnt = 0;
        if (m.kv_int("hojo.decode.max_new_tokens", mnt) ||
            m.kv_int("hojo.max_new_tokens", mnt)) {
            uint32_t t;
            if (!lib::u32(m, "hojo.decode.max_new_tokens", 0, t, err)) return false;
            c.decode.max_new_tokens = t;
        }
    }
    D(c.decode.repetition_penalty, "decode.repetition_penalty");
    D(c.decode.length_penalty, "decode.length_penalty");
    D(c.decode.temperature, "decode.temperature");
    D(c.decode.top_p, "decode.top_p");
    {
        int64_t ds = 0;
        if (m.kv_int("hojo.decode.do_sample", ds)) c.decode.do_sample = ds != 0;
    }
    // Token ids.
    {
        uint32_t t;
#define T(field, key) do { if (!lib::u32(m, "hojo." key, (uint32_t) field, t, err)) return false; field = (int32_t) t; } while (0)
        T(c.bos_token_id, "bos_token_id");
        T(c.eos_token_id, "eos_token_id");
        T(c.pad_token_id, "pad_token_id");
#undef T
    }
#undef U
#undef D

    // --- Validate untrusted GGUF metadata (lib/loader_kit.hpp). ---
    if (!lib::check_gguf_header(m, "hojo", "Hojo",
                                {"mixed_f32_bf16_exact", "bf16_exact", "f16"}, err))
        return false;
#define POS(v, name) do { if (!(v)) { err = "Hojo GGUF " name " must be positive"; return false; } } while (0)
    POS(c.tower.d_model, "tower.d_model");
    POS(c.tower.encoder_layers, "tower.encoder_layers");
    POS(c.tower.encoder_attention_heads, "tower.encoder_attention_heads");
    POS(c.tower.head_dim, "tower.head_dim");
    POS(c.tower.encoder_ffn_dim, "tower.encoder_ffn_dim");
    POS(c.tower.output_dim, "tower.output_dim");
    POS(c.bottleneck.output_size, "bottleneck.output_size");
    POS(c.bottleneck.num_blocks, "bottleneck.num_blocks");
    POS(c.bottleneck.attention_heads, "bottleneck.attention_heads");
    POS(c.llm.hidden, "llm.hidden_size");
    POS(c.llm.n_heads, "llm.num_heads");
    POS(c.llm.n_kv_heads, "llm.num_kv_heads");
    POS(c.llm.head_dim, "llm.head_dim");
    POS(c.llm.intermediate, "llm.intermediate_size");
    POS(c.llm.vocab, "llm.vocab_size");
#undef POS
    if (c.tower.d_model != c.tower.encoder_attention_heads * c.tower.head_dim) {
        err = "Hojo GGUF tower.d_model != tower.encoder_attention_heads * tower.head_dim";
        return false;
    }
    if (c.llm.hidden != c.llm.n_heads * c.llm.head_dim) {
        // Qwen3-4B: head_dim=128 is INDEPENDENT of hidden/num_heads (80).
        // hidden=2560, num_heads*head_dim=4096 (the q/k/v proj output dim). This
        // is expected, NOT an error — only warn-level. Tower still must satisfy
        // d_model == heads*head_dim (checked above).
    }
    if (c.llm.n_heads % c.llm.n_kv_heads != 0) {
        err = "Hojo GGUF llm.num_heads must be a multiple of llm.num_kv_heads";
        return false;
    }
    // The Conformer MHA reshapes the bottleneck output into H heads of
    // output_size/H each, so output_size must be divisible by attention_heads.
    if (c.bottleneck.attention_heads == 0 ||
        c.bottleneck.output_size % c.bottleneck.attention_heads != 0) {
        err = "Hojo GGUF bottleneck.output_size must be divisible by bottleneck.attention_heads";
        return false;
    }

    // --- Require every expected tensor so a structural change fails loudly. ---
    for (const char* n : {"audio.mel_filters", "audio.mel_window",
                          "audio.conv2d1.weight", "audio.conv2d1.bias",
                          "audio.conv2d2.weight", "audio.conv2d2.bias",
                          "audio.conv2d3.weight", "audio.conv2d3.bias",
                          "audio.conv_out.weight",
                          "audio.ln_post.weight", "audio.ln_post.bias",
                          "audio.proj1.weight", "audio.proj1.bias",
                          "audio.proj2.weight", "audio.proj2.bias",
                          "bottleneck.embed.out.0.weight", "bottleneck.embed.out.0.bias",
                          "bottleneck.embed.out.1.weight", "bottleneck.embed.out.1.bias",
                          "bottleneck.pos_enc.pe",
                          "bottleneck.after_norm.weight", "bottleneck.after_norm.bias",
                          "ln_speech.weight", "ln_speech.bias",
                          "llm.embed.weight", "llm.lm_head.weight", "llm.final_norm.weight"})
        if (!lib::require(m, n, "Hojo", err)) return false;
    // Tower layers: attn_norm(w+b), attn.q/k/v/o (w+b), ffn_norm(w+b), ffn.fc1/fc2
    // (w+b) = 16 each.
    for (uint32_t i = 0; i < c.tower.encoder_layers; ++i) {
        char n[128];
        for (const char* tail : {"attn_norm.weight", "attn_norm.bias",
                                 "attn.q.weight", "attn.q.bias",
                                 "attn.k.weight", "attn.k.bias",
                                 "attn.v.weight", "attn.v.bias",
                                 "attn.o.weight", "attn.o.bias",
                                 "ffn_norm.weight", "ffn_norm.bias",
                                 "ffn.fc1.weight", "ffn.fc1.bias",
                                 "ffn.fc2.weight", "ffn.fc2.bias"}) {
            std::snprintf(n, sizeof n, "audio.blk.%u.%s", i, tail);
            if (!lib::require(m, n, "Hojo", err)) return false;
        }
    }
    // Bottleneck layers: norm_mha/ff/ff_macaron/conv/final (w+b) = 5*2, mha
    // (linear_q/k/v/out w+b, linear_pos w, pos_bias_u/v) = 4*2+1+2 = 11,
    // ffn + ffn_macaron (w_1/w_2 w+b) = 2*4 = 8, conv (pointwise1/2 w+b,
    // depthwise w+b, norm weight/bias/running_mean/running_var/num_batches_tracked)
    // = 2*2+2+5 = 11. Total 40 per block.
    for (uint32_t i = 0; i < c.bottleneck.num_blocks; ++i) {
        char n[160];
        for (const char* tail : {"norm_mha.weight", "norm_mha.bias",
                                 "norm_ff.weight", "norm_ff.bias",
                                 "norm_ff_macaron.weight", "norm_ff_macaron.bias",
                                 "norm_conv.weight", "norm_conv.bias",
                                 "norm_final.weight", "norm_final.bias",
                                 "mha.linear_q.weight", "mha.linear_q.bias",
                                 "mha.linear_k.weight", "mha.linear_k.bias",
                                 "mha.linear_v.weight", "mha.linear_v.bias",
                                 "mha.linear_out.weight", "mha.linear_out.bias",
                                 "mha.linear_pos.weight",
                                 "mha.pos_bias_u", "mha.pos_bias_v",
                                 "ffn.w_1.weight", "ffn.w_1.bias",
                                 "ffn.w_2.weight", "ffn.w_2.bias",
                                 "ffn_macaron.w_1.weight", "ffn_macaron.w_1.bias",
                                 "ffn_macaron.w_2.weight", "ffn_macaron.w_2.bias",
                                 "conv.pointwise_conv1.weight", "conv.pointwise_conv1.bias",
                                 "conv.pointwise_conv2.weight", "conv.pointwise_conv2.bias",
                                 "conv.depthwise_conv.weight", "conv.depthwise_conv.bias",
                                 "conv.norm.weight", "conv.norm.bias",
                                 "conv.norm.running_mean", "conv.norm.running_var",
                                 "conv.norm.num_batches_tracked"}) {
            std::snprintf(n, sizeof n, "bottleneck.blk.%u.%s", i, tail);
            if (!lib::require(m, n, "Hojo", err)) return false;
        }
    }
    // LLM layers: attn_norm(w), attn.q/k/v/o(w), attn.q_norm/k_norm(w),
    // ffn_norm(w), ffn.gate/up/down(w) = 11 each.
    for (uint32_t i = 0; i < c.llm.n_layers; ++i) {
        char n[128];
        for (const char* tail : {"attn_norm.weight", "attn.q.weight", "attn.k.weight",
                                 "attn.v.weight", "attn.o.weight", "attn.q_norm.weight",
                                 "attn.k_norm.weight", "ffn_norm.weight", "ffn.gate.weight",
                                 "ffn.up.weight", "ffn.down.weight"}) {
            std::snprintf(n, sizeof n, "llm.blk.%u.%s", i, tail);
            if (!lib::require(m, n, "Hojo", err)) return false;
        }
    }
    return true;
}
} // namespace starling::ggml::hojo
