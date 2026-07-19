// llm.cpp — MOSS (Qwen3-style) text decoder on the Starling ggml runtime.
//
// One layer per run_graph call. Decode/prefill share the same layer builder;
// prefill is S>1, decode is S==1 with a KV cache carried in LlmState.
//
// Correctness contract: byte-exact bf16 vs the Transformers golden path on
// CUDA (see docs/ggml-moss-spec.md). CPU bf16 GEMMs are not bit-identical to
// cuBLAS and are a fallback only.

#include "llm.hpp"

#include "runtime/backend.hpp"
#include "runtime/graph.hpp"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace starling::ggml::moss {
namespace {

// ---------------------------------------------------------------------------
// Small tensor helpers. Convention: keep activations bf16 between ops (the
// reference model runs bf16), but do elementwise math (add/mul) in f32 and
// cast back, matching PyTorch's bf16 autocast semantics closely enough to
// stay bit-exact with the golden path.
// ---------------------------------------------------------------------------

ggml_tensor* bf(ggml_context* c, ggml_tensor* x) {
    return x->type == GGML_TYPE_BF16 ? x : ggml_cast(c, x, GGML_TYPE_BF16);
}
ggml_tensor* ff(ggml_context* c, ggml_tensor* x) {
    return x->type == GGML_TYPE_F32 ? x : ggml_cast(c, x, GGML_TYPE_F32);
}
ggml_tensor* wb(ggml_context* c, const MossModel& m, const std::string& n) {
    return clone_weight(c, m.loader, n.c_str());
}
// Linear: y = W x, result bf16.
ggml_tensor* lin(ggml_context* c, const MossModel& m, ggml_tensor* x, const std::string& n) {
    return bf(c, ggml_mul_mat(c, wb(c, m, n), bf(c, x)));
}
ggml_tensor* addb(ggml_context* c, ggml_tensor* a, ggml_tensor* b) {
    return bf(c, ggml_add(c, ff(c, a), ff(c, b)));
}
ggml_tensor* mulb(ggml_context* c, ggml_tensor* a, ggml_tensor* b) {
    return bf(c, ggml_mul(c, ff(c, a), ff(c, b)));
}
// RMSNorm in f32, then scale by the (bf16) weight.
ggml_tensor* rms(ggml_context* c, const MossModel& m, ggml_tensor* x, const std::string& n, float eps) {
    ggml_tensor* y = ggml_rms_norm(c, ff(c, x), eps);
    y = bf(c, y);
    return mulb(c, y, bf(c, wb(c, m, n)));
}

std::vector<ggml_bf16_t> tobf(const std::vector<float>& x) {
    std::vector<ggml_bf16_t> r(x.size());
    for (size_t i = 0; i < x.size(); ++i) r[i] = ggml_fp32_to_bf16(x[i]);
    return r;
}

// Append `add` new rows of k/v (src layout [heads, add, D] f32) to the cache
// (layout [heads, old+add, D] bf16).
void append_kv(std::vector<ggml_bf16_t>& dst, const std::vector<float>& src,
               int64_t old, int64_t add, int D, int heads) {
    std::vector<ggml_bf16_t> out((size_t)D * (old + add) * heads);
    for (int h = 0; h < heads; ++h) {
        if (old) std::copy_n(dst.data() + (size_t)h * D * old,
                             (size_t)D * old,
                             out.data() + (size_t)h * D * (old + add));
        for (int64_t s = 0; s < add; ++s)
            for (int d = 0; d < D; ++d)
                out[((size_t)h * (old + add) + old + s) * D + d] =
                    ggml_fp32_to_bf16(src[((size_t)h * add + s) * D + d]);
    }
    dst.swap(out);
}

int32_t argmax_low(const std::vector<float>& x) {
    int32_t best = 0;
    for (int32_t i = 1; i < (int32_t)x.size(); ++i)
        if (x[i] > x[best]) best = i;
    return best;
}

// ---------------------------------------------------------------------------
// Staged layer-0 parity probe.
// STARLING_MOSS_L0_STAGE=<stage> makes layer 0 of a 107-token prefill return
// the selected intermediate as the graph's REAL output (no capture side
// channels, which have proven untrustworthy), dumps it to
// <STARLING_MOSS_STAGE_DIR>/moss_stage_<stage>.f32 (default /tmp), and exits.
// Stages: n qn kn qr kr mask sc0 pr0 attn xmid down
// ---------------------------------------------------------------------------

struct L0Stage {
    const char* name = nullptr;          // nullptr = disabled
    const char* dir  = "/tmp";
    bool active(int li, int64_t S) const { return name && li == 0 && S == 107; }
    bool is(const char* s) const { return name && std::strcmp(name, s) == 0; }
};

L0Stage l0_stage() {
    L0Stage st;
    st.name = std::getenv("STARLING_MOSS_L0_STAGE");
    if (const char* d = std::getenv("STARLING_MOSS_STAGE_DIR")) st.dir = d;
    if (st.name && !*st.name) st.name = nullptr;
    return st;
}

[[noreturn]] void stage_exit(const L0Stage& st, const std::vector<float>& data) {
    std::string fn = std::string(st.dir) + "/moss_stage_" + st.name + ".f32";
    bool ok = false;
    if (FILE* f = std::fopen(fn.c_str(), "wb")) {
        const size_t want = data.size();
        const size_t got = std::fwrite(data.data(), sizeof(float), want, f);
        const int fc = std::fclose(f);
        if (got == want && fc == 0) {
            std::fprintf(stderr, "MOSS L0 probe: wrote %s (%zu floats)\n", fn.c_str(), got);
            ok = true;
        } else {
            std::fprintf(stderr, "MOSS L0 probe: FAILED to write %s (wrote %zu/%zu floats, fclose=%d)\n",
                         fn.c_str(), got, want, fc);
        }
    } else {
        std::fprintf(stderr, "MOSS L0 probe: FAILED to open %s\n", fn.c_str());
    }
    std::exit(ok ? 0 : 1);
}

// ---------------------------------------------------------------------------
// One transformer layer. xh is [hidden, S] bf16 (column per token). Returns
// the layer output x ([hidden, S] f32), plus this step's k/v ([D, S, KV]
// f32, head-major) for the KV cache.
// ---------------------------------------------------------------------------

struct LayerOut { std::vector<float> x, k, v; };

bool layer(const MossModel& m, int li, const std::vector<ggml_bf16_t>& xh,
           int64_t S, int64_t past, const LayerKvCache* cache, LayerOut& o) {
    const auto& lc = m.config.llm;
    const int D = lc.head_dim, H = lc.n_heads, KV = lc.n_kv_heads;
    const int64_t K = past + S;
    const L0Stage stage = l0_stage();
    const bool st0 = stage.active(li, S);

    // RoPE cos/sin tables, [D, S] bf16, duplicated halves (Qwen3 rotary).
    std::vector<ggml_bf16_t> cs((size_t)D * S), sn((size_t)D * S);
    for (int64_t s = 0; s < S; ++s) {
        for (int i = 0; i < D / 2; ++i) {
            float inv = 1.0f / std::pow(lc.rope_theta, (2.0f * i) / D);
            float a = (float)(past + s) * inv;
            cs[s * D + i] = cs[s * D + i + D / 2] = ggml_fp32_to_bf16(std::cos(a));
            sn[s * D + i] = sn[s * D + i + D / 2] = ggml_fp32_to_bf16(std::sin(a));
        }
    }

    // Causal mask [K, S] f32: 0 where allowed, -inf where masked.
    // (Row qi covers keys j <= past+qi.) Kept f32 end-to-end: the earlier
    // bf16-mask experiment corrupted far worse.
    std::vector<float> mask((size_t)K * S);
    const float neg = -3.3895313892515355e38f;
    for (int64_t qi = 0; qi < S; ++qi)
        for (int64_t j = 0; j < K; ++j)
            mask[qi * K + j] = (j <= past + qi) ? 0.0f : neg;

    bool ok = run_graph([&](ggml_context* c) -> ggml_tensor* {
        int64_t xne[2] = {lc.hidden, S};
        ggml_tensor* x = graph_input_tensor(c, GGML_TYPE_BF16, 2, xne,
                                            xh.data(), xh.size() * sizeof(xh[0]));
        ggml_tensor* r = x;  // residual

        const std::string p = "llm.blk." + std::to_string(li) + ".";

        ggml_tensor* n = rms(c, m, x, p + "attn_norm.weight", lc.rms_norm_eps);
        if (st0 && stage.is("n")) return ff(c, n);

        ggml_tensor* q = lin(c, m, n, p + "attn.q.weight");
        ggml_tensor* k = lin(c, m, n, p + "attn.k.weight");
        ggml_tensor* v = lin(c, m, n, p + "attn.v.weight");
        q = ggml_reshape_3d(c, q, D, H, S);
        k = ggml_reshape_3d(c, k, D, KV, S);
        v = ggml_reshape_3d(c, v, D, KV, S);

        q = rms(c, m, q, p + "attn.q_norm.weight", lc.rms_norm_eps);
        k = rms(c, m, k, p + "attn.k_norm.weight", lc.rms_norm_eps);
        if (st0 && stage.is("qn")) return ff(c, q);
        if (st0 && stage.is("kn")) return ff(c, k);

        q = ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));  // [D, S, H]
        k = ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3));  // [D, S, KV]
        v = ggml_cont(c, ggml_permute(c, v, 0, 2, 1, 3));  // [D, S, KV]
        if (st0 && stage.is("qc")) return ff(c, q);

        int64_t rne[2] = {D, S};
        ggml_tensor* ct = graph_input_tensor(c, GGML_TYPE_BF16, 2, rne,
                                             cs.data(), cs.size() * sizeof(cs[0]));
        ggml_tensor* st = graph_input_tensor(c, GGML_TYPE_BF16, 2, rne,
                                             sn.data(), sn.size() * sizeof(sn[0]));

        // Rotary embedding, rotate-half formulation, in f32 then back to bf16.
        auto rope = [&](ggml_tensor* z, int heads) {
            ggml_tensor* lo = ggml_view_3d(c, z, D / 2, S, heads,
                                           z->nb[1], z->nb[2], 0);
            ggml_tensor* hi = ggml_view_3d(c, z, D / 2, S, heads,
                                           z->nb[1], z->nb[2],
                                           (size_t)(D / 2) * z->nb[0]);
            ggml_tensor* rot = ggml_concat(c, ggml_scale(c, ff(c, hi), -1.0f),
                                           ff(c, lo), 0);
            rot = bf(c, rot);
            if (st0 && heads == H && stage.is("rot")) return ff(c, rot);
            if (st0 && heads == H && stage.is("mulz")) return ff(c, mulb(c, z, ct));
            if (st0 && heads == H && stage.is("mulr")) return ff(c, mulb(c, rot, st));
            return addb(c, mulb(c, z, ct), mulb(c, rot, st));
        };
        q = rope(q, H);
        k = rope(k, KV);
        if (st0 && stage.is("qr")) return ff(c, q);
        if (st0 && stage.is("kr")) return ff(c, k);

        // Prepend cached k/v from previous steps (decode only).
        ggml_tensor* kall = k, *vall = v;
        if (past) {
            int64_t cne[3] = {D, past, KV};
            ggml_tensor* pk = graph_input_tensor(c, GGML_TYPE_BF16, 3, cne,
                                                 cache->k.data(),
                                                 cache->k.size() * sizeof(cache->k[0]));
            ggml_tensor* pv = graph_input_tensor(c, GGML_TYPE_BF16, 3, cne,
                                                 cache->v.data(),
                                                 cache->v.size() * sizeof(cache->v[0]));
            kall = bf(c, ggml_concat(c, ff(c, pk), ff(c, k), 1));
            vall = bf(c, ggml_concat(c, ff(c, pv), ff(c, v), 1));
        }

        int64_t mne[2] = {K, S};
        ggml_tensor* mt = graph_input_tensor(c, GGML_TYPE_F32, 2, mne,
                                             mask.data(), mask.size() * sizeof(mask[0]));
        if (st0 && stage.is("mask")) return mt;

        // Per-head attention: scores = k^T q / sqrt(D), causal softmax,
        // context = probs v. Heads concatenated along the feature dim.
        const float scale = 1.0f / std::sqrt((float)D);
        ggml_tensor* joined = nullptr;
        for (int h = 0; h < H; ++h) {
            int kvh = h / (H / KV);
            ggml_tensor* qh = ggml_view_2d(c, q, D, S, q->nb[1], (size_t)h * q->nb[2]);
            ggml_tensor* kh = ggml_view_2d(c, kall, D, K, kall->nb[1], (size_t)kvh * kall->nb[2]);
            ggml_tensor* vh = ggml_view_2d(c, vall, D, K, vall->nb[1], (size_t)kvh * vall->nb[2]);
            ggml_tensor* sc = bf(c, ggml_mul_mat(c, kh, qh));          // [K, S]
            sc = bf(c, ggml_scale(c, ff(c, sc), scale));
            if (st0 && stage.is("sc0") && h == 0) return ff(c, sc);
            ggml_tensor* pr = bf(c, ggml_soft_max_ext(c, ff(c, sc), ff(c, mt), 1.0f, 0.0f));
            if (st0 && stage.is("pr0") && h == 0) return ff(c, pr);
            ggml_tensor* vt = ggml_cont(c, ggml_transpose(c, vh));     // [K, D]
            ggml_tensor* co = bf(c, ggml_mul_mat(c, vt, pr));          // [D, S]
            joined = joined ? bf(c, ggml_concat(c, ff(c, joined), ff(c, co), 0)) : co;
        }
        if (st0 && stage.is("ctx")) return ff(c, joined);

        ggml_tensor* a = lin(c, m, joined, p + "attn.o.weight");       // [hidden, S]
        if (st0 && stage.is("attn")) return ff(c, a);
        x = addb(c, r, a);
        r = x;
        if (st0 && stage.is("xmid")) return ff(c, x);

        n = rms(c, m, x, p + "ffn_norm.weight", lc.rms_norm_eps);
        ggml_tensor* g = lin(c, m, n, p + "ffn.gate.weight");
        ggml_tensor* u = lin(c, m, n, p + "ffn.up.weight");
        ggml_tensor* si = bf(c, ggml_silu(c, ff(c, g)));
        ggml_tensor* z = mulb(c, si, u);
        ggml_tensor* dn = lin(c, m, z, p + "ffn.down.weight");
        if (st0 && stage.is("down")) return ff(c, dn);
        x = addb(c, r, dn);

        // Output: concat(flat x, flat k, flat v) so one readback carries all.
        ggml_tensor* flatx = ggml_reshape_1d(c, ff(c, x), lc.hidden * S);
        ggml_tensor* flatk = ggml_reshape_1d(c, ff(c, k), D * KV * S);
        ggml_tensor* flatv = ggml_reshape_1d(c, ff(c, v), D * KV * S);
        return ggml_concat(c, ggml_concat(c, flatx, flatk, 0), flatv, 0);
    }, o.x);

    if (ok && st0) stage_exit(stage, o.x);

    if (ok) {
        size_t nx = (size_t)lc.hidden * S, nkv = (size_t)D * KV * S;
        std::vector<float> all = std::move(o.x);
        o.x.assign(all.begin(), all.begin() + nx);
        o.k.assign(all.begin() + nx, all.begin() + nx + nkv);
        o.v.assign(all.begin() + nx + nkv, all.end());
    }
    return ok;
}

// ---------------------------------------------------------------------------
// Full stack: embeds -> n_layers -> final norm -> lm_head (tied embeddings).
// `input` is [hidden, S] f32 (token-major), logits are for the LAST token.
// ---------------------------------------------------------------------------

bool forward(const MossModel& m, const std::vector<float>& input, int64_t S,
             LlmState& state, std::vector<float>& logits, std::string& e) {
    std::vector<ggml_bf16_t> x = tobf(input);
    if (state.layers.empty()) state.layers.resize(m.config.llm.n_layers);

    for (int li = 0; li < (int)m.config.llm.n_layers; ++li) {
        LayerOut o;
        if (!layer(m, li, x, S, state.length, &state.layers[li], o)) {
            e = "MOSS LLM layer graph failed at " + std::to_string(li);
            return false;
        }
        append_kv(state.layers[li].k, o.k, state.length, S,
                  m.config.llm.head_dim, m.config.llm.n_kv_heads);
        append_kv(state.layers[li].v, o.v, state.length, S,
                  m.config.llm.head_dim, m.config.llm.n_kv_heads);
        for (size_t z = 0; z < o.x.size(); ++z) {
            if (!std::isfinite(o.x[z])) {
                e = "non-finite MOSS hidden at layer " + std::to_string(li) +
                    " index " + std::to_string(z);
                return false;
            }
        }
        // STARLING_MOSS_DUMP_LAYERS=<prefix> dumps each
        // layer's output hidden as <prefix>_<li>.f32 (prefill only).
        if (S > 1) {
            if (const char* dp = std::getenv("STARLING_MOSS_DUMP_LAYERS")) {
                std::string fn = std::string(dp) + "_" + std::to_string(li) + ".f32";
                if (FILE* fp = std::fopen(fn.c_str(), "wb")) {
                    std::fwrite(o.x.data(), sizeof(float), o.x.size(), fp);
                    std::fclose(fp);
                }
            }
        }
        x = tobf(o.x);
    }
    state.length += S;

    // Logits for the last token: final RMSNorm, then tied-embedding lm_head.
    std::vector<ggml_bf16_t> last(m.config.llm.hidden);
    std::copy(x.end() - last.size(), x.end(), last.begin());
    bool ok = run_graph([&](ggml_context* c) -> ggml_tensor* {
        int64_t ne[2] = {m.config.llm.hidden, 1};
        ggml_tensor* t = graph_input_tensor(c, GGML_TYPE_BF16, 2, ne,
                                            last.data(), last.size() * sizeof(last[0]));
        t = rms(c, m, t, "llm.final_norm.weight", m.config.llm.rms_norm_eps);
        return ff(c, ggml_mul_mat(c, wb(c, m, "llm.embed.weight"), t));
    }, logits);
    if (!ok) e = "MOSS lm_head graph failed";
    return ok;
}

} // namespace

bool llm_prefill(const MossModel& m, const InputsEmbeds& i, int32_t maxc,
                 PrefillResult& o, std::string& e) {
    if (i.n_tokens <= 0 || i.width != (int64_t)m.config.llm.hidden || i.n_tokens > maxc) {
        e = "invalid MOSS prefill shape/cache";
        return false;
    }
    ensure_weights_realized(m.loader);
    if (!forward(m, i.data, i.n_tokens, o.state, o.logits, e)) return false;
    o.first_token = argmax_low(o.logits);
    // STARLING_MOSS_DUMP_LOGITS=<file> dumps prefill logits.
    if (const char* fp = std::getenv("STARLING_MOSS_DUMP_LOGITS")) {
        if (FILE* f = std::fopen(fp, "wb")) {
            std::fwrite(o.logits.data(), sizeof(float), o.logits.size(), f);
            std::fclose(f);
        }
    }
    return true;
}

bool greedy_generate(const MossModel& m, const InputsEmbeds& i,
                     const GenerateOptions& op, GenerateResult& o, std::string& e) {
    if (i.n_tokens + op.max_new_tokens > op.max_cache_len) {
        e = "MOSS generation exceeds cache";
        return false;
    }
    PrefillResult p;
    if (!llm_prefill(m, i, op.max_cache_len, p, e)) return false;
    o.prefill_logits = p.logits;
    o.ids.push_back(p.first_token);
    int32_t prev = p.first_token;
    for (int n = 1; n < op.max_new_tokens; ++n) {
        // Embed the single previous token.
        InputsEmbeds one;
        one.n_tokens = 1;
        one.width = m.config.llm.hidden;
        std::vector<int32_t> id = {prev};
        bool ok = run_graph([&](ggml_context* c) -> ggml_tensor* {
            int64_t ne[1] = {1};
            ggml_tensor* t = graph_input_tensor(c, GGML_TYPE_I32, 1, ne,
                                                id.data(), sizeof(int32_t));
            return ff(c, ggml_get_rows(c, clone_weight(c, m.loader, "llm.embed.weight"), t));
        }, one.data);
        if (!ok) { e = "decode embedding lookup failed"; return false; }

        std::vector<float> logits;
        if (!forward(m, one.data, 1, p.state, logits, e)) return false;
        prev = argmax_low(logits);
        o.ids.push_back(prev);
        if (prev == op.eos_token_id) { o.hit_eos = true; break; }
    }
    // STARLING_MOSS_DUMP_IDS=<file> dumps generated ids (i32).
    if (const char* fp = std::getenv("STARLING_MOSS_DUMP_IDS")) {
        if (FILE* f = std::fopen(fp, "wb")) {
            std::fwrite(o.ids.data(), sizeof(int32_t), o.ids.size(), f);
            std::fclose(f);
        }
    }
    return true;
}

} // namespace starling::ggml::moss
