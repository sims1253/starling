#include "loader.hpp"
#include <cstdio>
#include <vector>

#include "lib/loader_kit.hpp"
namespace starling::ggml::s1 {
namespace {
using lib::f32;
using lib::f64;
using lib::str;
} // namespace

bool S1Model::load(const char* path, std::string& err) {
    if (!loader.load(path)) {
        err = loader.last_error();
        return false;
    }
    auto& c = config;
    const auto& m = loader;
#define U(field, key) do { if (!lib::u32(m, "s1." key, field, field, err)) return false; } while (0)
#define F(field, key) field = f32(m, "s1." key, field)
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
        if (m.kv_int("s1.llm.tied_embeddings", tied)) c.llm.tied_embeddings = tied != 0;
        if (m.kv_int("s1.llm.has_qk_norm", qn)) c.llm.has_qk_norm = qn != 0;
    }
    // Token ids + generation budget.
    {
        uint32_t t;
#define T(field, key) do { if (!lib::u32(m, "s1." key, (uint32_t) field, t, err)) return false; field = (int32_t) t; } while (0)
        T(c.eos_token_id, "eos_token_id");
        T(c.eos2_token_id, "eos2_token_id");
        T(c.pad_token_id, "pad_token_id");
        T(c.max_input_tokens, "max_input_tokens");
#undef T
    }
    F(c.max_new_tokens_input_factor, "max_new_tokens_input_factor");
    F(c.max_new_tokens_fixed, "max_new_tokens_fixed");
    if (std::string v; m.kv_str("s1.styling_values", v) && !v.empty()) {
        c.styling_values.clear();
        size_t start = 0;
        while (start <= v.size()) {
            size_t bar = v.find('|', start);
            if (bar == std::string::npos) bar = v.size();
            if (bar > start) c.styling_values.push_back(v.substr(start, bar - start));
            if (bar == v.size()) break;
            start = bar + 1;
        }
    }
    if (std::string v; m.kv_str("s1.structure_values", v) && !v.empty()) {
        c.structure_values.clear();
        size_t start = 0;
        while (start <= v.size()) {
            size_t bar = v.find('|', start);
            if (bar == std::string::npos) bar = v.size();
            if (bar > start) c.structure_values.push_back(v.substr(start, bar - start));
            if (bar == v.size()) break;
            start = bar + 1;
        }
    }
    if (std::string v; m.kv_str("s1.context_values", v) && !v.empty()) {
        c.context_values.clear();
        size_t start = 0;
        while (start <= v.size()) {
            size_t bar = v.find('|', start);
            if (bar == std::string::npos) bar = v.size();
            if (bar > start) c.context_values.push_back(v.substr(start, bar - start));
            if (bar == v.size()) break;
            start = bar + 1;
        }
    }
    std::vector<int64_t> a;
    if (m.kv_arr_int("s1.prompt_prefix", a))
        for (auto v : a) c.prompt_prefix.push_back((int32_t) v);
    a.clear();
    if (m.kv_arr_int("s1.prompt_suffix", a))
        for (auto v : a) c.prompt_suffix.push_back((int32_t) v);

    // --- Validate untrusted GGUF metadata (mirror qwen3/loader.cpp). ---
    if (!lib::check_gguf_header(m, "s1", "S1", {"bf16_exact"}, err))
        return false;
#define POS(v, name) do { if (!(v)) { err = "S1 GGUF " name " must be positive"; return false; } } while (0)
    POS(c.llm.n_layers, "llm.num_layers");
    POS(c.llm.hidden, "llm.hidden_size");
    POS(c.llm.n_heads, "llm.num_heads");
    POS(c.llm.n_kv_heads, "llm.num_kv_heads");
    POS(c.llm.head_dim, "llm.head_dim");
    POS(c.llm.intermediate, "llm.intermediate_size");
    POS(c.llm.vocab, "llm.vocab_size");
    POS(c.llm.max_cache, "llm.max_cache_len");
#undef POS
    // Qwen3 allows head_dim != hidden/n_heads (s1: 16 heads * 128 = 2x the
    // 1024 hidden); the decode stack takes both dims independently.
    if (c.llm.n_heads % c.llm.n_kv_heads != 0) {
        err = "S1 GGUF n_heads must be divisible by n_kv_heads";
        return false;
    }
    if (!c.llm.tied_embeddings) {
        err = "S1 GGUF requires tied embeddings (llm.embed doubles as lm_head)";
        return false;
    }
    if (!c.llm.has_qk_norm) {
        err = "S1 GGUF requires qk_norm (Qwen3 trunk with per-head q/k norm)";
        return false;
    }
    if (c.prompt_prefix.empty() || c.prompt_suffix.empty()) {
        err = "S1 GGUF missing s1.prompt_prefix/prompt_suffix template arrays";
        return false;
    }
    if (c.eos_token_id < 0 || c.eos_token_id >= (int32_t) c.llm.vocab ||
        c.eos2_token_id < 0 || c.eos2_token_id >= (int32_t) c.llm.vocab) {
        err = "S1 GGUF eos ids out of vocabulary range";
        return false;
    }

    // Require every expected tensor so a structural change fails loudly:
    // embed/final norm + 11 tensors per Qwen3 layer (same layout as the
    // qwen3-ASR trunk).
    for (const char* n : {"llm.embed.weight", "llm.final_norm.weight"})
        if (!lib::require(m, n, "S1", err)) return false;
    for (uint32_t i = 0; i < c.llm.n_layers; ++i) {
        char n[128];
        for (const char* tail : {"attn_norm.weight", "attn.q.weight", "attn.k.weight",
                                 "attn.v.weight", "attn.o.weight",
                                 "attn.q_norm.weight", "attn.k_norm.weight",
                                 "ffn_norm.weight", "ffn.gate.weight",
                                 "ffn.up.weight", "ffn.down.weight"}) {
            std::snprintf(n, sizeof n, "llm.blk.%u.%s", i, tail);
            if (!lib::require(m, n, "S1", err)) return false;
        }
    }
    return true;
#undef U
#undef F
}

} // namespace starling::ggml::s1
