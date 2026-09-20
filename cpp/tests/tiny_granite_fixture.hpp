// tiny_granite_fixture.hpp — the tiny synthesized granite GGUF shared by
// granite_stage_test.cpp and the trace tests. All-zero weights keep every
// activation exactly zero: LN(0)=0, softmax(0) uniform, argmax picks token 0
// — deterministic output and real, measurable stage graphs (1 encoder layer,
// 1 qformer layer, 1 LLM layer, chunk_seconds=1). CPU-only, no downloads.
#pragma once

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <vector>

// --------------------------------------------------------------------------- //
// Tiny synthesized granite GGUF (see the header comment). All-zero weights
// keep every activation exactly zero: LN(0)=0, softmax(0) uniform, argmax
// picks token 0 — deterministic output and real, measurable stage graphs.
// --------------------------------------------------------------------------- //
class TinyGraniteFixture {
public:
    std::filesystem::path path;
    ggml_context* gctx_ = nullptr;
    gguf_context* gf_ = nullptr;

    explicit TinyGraniteFixture(const std::filesystem::path& p) : path(p) {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
        gctx_ = ggml_init({64 << 20, nullptr, false});
        gf_ = gguf_init_empty();
        if (!gctx_ || !gf_) return;
        write_metadata();
        write_tensors();
        wrote_ = gguf_write_to_file(gf_, path.string().c_str(), /*only_meta=*/false);
    }
    ~TinyGraniteFixture() {
        if (gf_) gguf_free(gf_);
        if (gctx_) ggml_free(gctx_);
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
    }
    bool wrote() const { return wrote_; }

private:
    bool wrote_ = false;

    void kv_u32(const char* key, uint32_t v) { gguf_set_val_u32(gf_, key, v); }

    void write_metadata() {
        gguf_set_val_str(gf_, "general.architecture", "granite");
        gguf_set_val_u32(gf_, "starling.format_version", 1);
        // Encoder: 1 conformer block at the stock hidden width (the loader
        // pins enc.hidden == heads*head_dim == conv_inner/2), small ffn,
        // tiny block-local attention context.
        kv_u32("granite.enc.layers", 1);
        kv_u32("granite.enc.ffn_dim", 8);
        kv_u32("granite.enc.conv_kernel", 3);
        kv_u32("granite.enc.context_size", 4);
        kv_u32("granite.enc.output_dim", 4);
        // Projector: 2-frame windows -> 1 token each (num_queries ==
        // window/downsample), 1 qformer layer.
        kv_u32("granite.proj.window_size", 2);
        kv_u32("granite.proj.downsample_rate", 1);
        kv_u32("granite.proj.num_queries", 2);
        kv_u32("granite.proj.qformer_layers", 1);
        kv_u32("granite.proj.qformer_heads", 2);
        kv_u32("granite.proj.qformer_intermediate", 8);
        kv_u32("granite.proj.output_dim", 8);  // == llm.hidden
        // Tiny bias-free Qwen trunk (hidden == heads*head_dim == 8).
        kv_u32("granite.llm.hidden_size", 8);
        kv_u32("granite.llm.num_layers", 1);
        kv_u32("granite.llm.num_heads", 1);
        kv_u32("granite.llm.num_kv_heads", 1);
        kv_u32("granite.llm.head_dim", 8);
        kv_u32("granite.llm.intermediate_size", 8);
        kv_u32("granite.llm.vocab_size", 16);
        kv_u32("granite.llm.max_cache_len", 128);
        // Chunk policy: 1 s chunks (max_new_tokens 40 -> token-limited
        // chunk max(0.1, (40-32)/5) = 1.6 s), small decode budgets.
        kv_u32("granite.max_new_tokens", 40);
        gguf_set_val_f64(gf_, "granite.chunk_seconds", 1.0);
        kv_u32("granite.audio_token_id", 3);
        kv_u32("granite.pad_token_id", 0);
        kv_u32("granite.bos_token_id", 2);
        kv_u32("granite.eos_token_id", 2);
        const int64_t prefix[1] = {1};
        const int64_t suffix[1] = {2};
        gguf_set_arr_data(gf_, "granite.prompt_prefix", GGUF_TYPE_INT64, prefix, 1);
        gguf_set_arr_data(gf_, "granite.prompt_suffix", GGUF_TYPE_INT64, suffix, 1);
        // GPT-2 byte-level BPE table (decode-only): 16 printable tokens.
        gguf_set_val_str(gf_, "tokenizer.ggml.model", "gpt2");
        const char* tokens[16] = {"a", "b", "c", "d", "e", "f", "g", "h",
                                  "i", "j", "k", "l", "m", "n", "o", "p"};
        gguf_set_arr_str(gf_, "tokenizer.ggml.tokens", tokens, 16);
    }

    void add_tensor(const char* name, ggml_type type, std::initializer_list<int64_t> ne,
                    float value) {
        std::vector<int64_t> dims(ne);
        ggml_tensor* t = ggml_new_tensor(gctx_, type, (int) dims.size(), dims.data());
        ggml_set_name(t, name);
        if (type == GGML_TYPE_F32) {
            float* p = (float*) t->data;
            for (int64_t i = 0; i < ggml_nelements(t); ++i) p[i] = value;
        } else {
            ggml_bf16_t* p = (ggml_bf16_t*) t->data;
            const ggml_bf16_t v = ggml_fp32_to_bf16(value);
            for (int64_t i = 0; i < ggml_nelements(t); ++i) p[i] = v;
        }
        gguf_add_tensor(gf_, t);
    }
    // Linear weight layout is ggml's [in, out]; norms are [n].
    void bf16(const char* name, std::initializer_list<int64_t> ne, float v = 0.0f) {
        add_tensor(name, GGML_TYPE_BF16, ne, v);
    }

    void write_tensors() {
        constexpr int64_t H = 1024;    // encoder hidden (fixed by the loader)
        constexpr int64_t C = 2048;    // enc conv_inner (== 2*hidden, fixed)
        constexpr int64_t F = 8;       // encoder/qformer ffn width
        constexpr int64_t CS = 4;      // encoder attention context
        constexpr int64_t QH = 1024;   // projector hidden
        constexpr int64_t L = 8;       // llm hidden
        constexpr int64_t V = 16;      // llm vocab

        // Mel constants (F32, shapes checked by the frontend).
        add_tensor("audio.mel_filters", GGML_TYPE_F32, {80, 257}, 0.0f);
        add_tensor("audio.mel_window", GGML_TYPE_F32, {512}, 1.0f);

        // Encoder.
        bf16("enc.input_linear.weight", {160, H});
        bf16("enc.input_linear.bias", {H});
        bf16("enc.out.weight", {H, 4});
        bf16("enc.out.bias", {4});
        bf16("enc.out_mid.weight", {4, H});
        bf16("enc.out_mid.bias", {H});
        for (const char* ff : {"ff1", "ff2"}) {
            const std::string p = std::string("enc.blk.0.") + ff;
            bf16((p + "_norm.weight").c_str(), {H}, 1.0f);
            bf16((p + "_norm.bias").c_str(), {H});
            bf16((p + "_up.weight").c_str(), {H, F});
            bf16((p + "_up.bias").c_str(), {F});
            bf16((p + "_down.weight").c_str(), {F, H});
            bf16((p + "_down.bias").c_str(), {H});
        }
        bf16("enc.blk.0.attn_norm.weight", {H}, 1.0f);
        bf16("enc.blk.0.attn_norm.bias", {H});
        bf16("enc.blk.0.attn_q.weight", {H, H});
        bf16("enc.blk.0.attn_kv.weight", {H, 2 * H});
        bf16("enc.blk.0.attn_o.weight", {H, H});
        bf16("enc.blk.0.attn_o.bias", {H});
        bf16("enc.blk.0.rel_pos_bias", {128, CS, CS});  // head_dim x ctx x ctx
        bf16("enc.blk.0.conv_norm.weight", {H}, 1.0f);
        bf16("enc.blk.0.conv_norm.bias", {H});
        bf16("enc.blk.0.conv_up.weight", {H, 2 * C});
        bf16("enc.blk.0.conv_up.bias", {2 * C});
        bf16("enc.blk.0.conv_depth.weight", {C, 3});
        bf16("enc.blk.0.bn_weight", {C}, 1.0f);
        bf16("enc.blk.0.bn_bias", {C});
        bf16("enc.blk.0.bn_mean", {C});
        bf16("enc.blk.0.bn_var", {C}, 1.0f);
        bf16("enc.blk.0.conv_down.weight", {C, H});
        bf16("enc.blk.0.conv_down.bias", {H});
        bf16("enc.blk.0.post_norm.weight", {H}, 1.0f);
        bf16("enc.blk.0.post_norm.bias", {H});

        // Projector.
        bf16("proj.query", {QH, 2});
        bf16("proj.qformer_ln.weight", {QH}, 1.0f);
        bf16("proj.qformer_ln.bias", {QH});
        for (const char* attn : {"self", "cross"}) {
            const std::string p = std::string("proj.blk.0.") + attn;
            for (const char* w : {"q", "k", "v", "out"}) {
                bf16((p + "_" + w + ".weight").c_str(), {QH, QH});
                bf16((p + "_" + w + ".bias").c_str(), {QH});
            }
            bf16((p + "_ln.weight").c_str(), {QH}, 1.0f);
            bf16((p + "_ln.bias").c_str(), {QH});
        }
        bf16("proj.blk.0.ff_up.weight", {QH, F});
        bf16("proj.blk.0.ff_up.bias", {F});
        bf16("proj.blk.0.ff_down.weight", {F, QH});
        bf16("proj.blk.0.ff_down.bias", {QH});
        bf16("proj.blk.0.ff_ln.weight", {QH}, 1.0f);
        bf16("proj.blk.0.ff_ln.bias", {QH});
        bf16("proj.out.weight", {QH, L});
        bf16("proj.out.bias", {L});

        // LLM trunk (bias-free, untied head).
        bf16("llm.embed.weight", {L, V});
        bf16("llm.lm_head.weight", {L, V});
        bf16("llm.final_norm.weight", {L}, 1.0f);
        bf16("llm.blk.0.attn_norm.weight", {L}, 1.0f);
        bf16("llm.blk.0.attn.q.weight", {L, L});
        bf16("llm.blk.0.attn.k.weight", {L, L});
        bf16("llm.blk.0.attn.v.weight", {L, L});
        bf16("llm.blk.0.attn.o.weight", {L, L});
        bf16("llm.blk.0.ffn_norm.weight", {L}, 1.0f);
        bf16("llm.blk.0.ffn.gate.weight", {L, L});
        bf16("llm.blk.0.ffn.up.weight", {L, L});
        bf16("llm.blk.0.ffn.down.weight", {L, L});
    }
};

