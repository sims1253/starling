// qwen_decode.cpp — shared Qwen-trunk text decoder (the moss = Qwen3 and
// ark = Qwen2.5 bindings) on the Starling ggml runtime.
//
// Decode/prefill share the same layer builder; prefill is S>1, decode is S==1
// with the KV carried device-side (host LlmState on the probe path only).
//
// Correctness contract: byte-exact bf16 vs the Transformers golden path on
// CUDA. CPU bf16 GEMMs are not bit-identical to cuBLAS and are a fallback
// only.

#include "qwen_decode.hpp"

#include "runtime/backend.hpp"
#include "runtime/graph.hpp"
#include "runtime/graph_builder.hpp"
#include "runtime/lru_cache.hpp"
#include "graph_helpers.hpp"
#include "device_cache.hpp"
#include "mask_rope.hpp"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace starling::ggml::lib {
namespace {

// ---------------------------------------------------------------------------
// Small tensor helpers (graph_helpers.hpp). Convention: keep activations bf16
// between ops (the reference model runs bf16), but do elementwise math
// (add/mul) in f32 and cast back, matching PyTorch's bf16 autocast semantics
// closely enough to stay bit-exact with the golden path.
// ---------------------------------------------------------------------------

// getenv("<spec.env><suffix>") — the per-model env surface (K-step width,
// debug gates, dumps).
const char* env(const QwenDecodeSpec& spec, const char* suffix) {
    std::string name(spec.env);
    name += suffix;
    return std::getenv(name.c_str());
}

// ---------------------------------------------------------------------------
// Granite-family spec hooks (graph_helpers.hpp for the f32-math/bf16-round
// discipline). Every helper is skip-when-default, so moss/ark keep the exact
// historical op sequence — their graphs are unchanged by these extensions.
// ---------------------------------------------------------------------------

// The lm_head weight name: tied models reuse the embedding table; granite
// carries a separate llm.lm_head.weight.
const char* lm_head_name(const QwenDecodeSpec& s) {
    return s.tied_lm_head ? "llm.embed.weight" : "llm.lm_head.weight";
}

// Attention softmax scale: explicit (granite's attention_multiplier replaces
// 1/sqrt(D)) or the historical default.
float attn_scale(const QwenDecodeSpec& s, int D) {
    return s.attention_scale > 0.0f ? s.attention_scale
                                    : 1.0f / std::sqrt((float)D);
}

// Residual add r + m*y (granite's residual_multiplier); 1.0 keeps addb.
ggml_tensor* residual_add(ggml_context* c, const QwenDecodeSpec& s,
                          ggml_tensor* r, ggml_tensor* y) {
    if (s.residual_multiplier == 1.0f) return addb(c, r, y);
    // Mirrors stock modeling_granite: t = bf16(y * m); out = bf16(r + t).
    return addb(c, r, bf16(c, ggml_scale(c, f32(c, y), s.residual_multiplier)));
}

// Granite's embedding_multiplier: applied to the whole inputs_embeds at
// prefill and to the embed lookup at decode (modeling_granite.py:397 applies
// it to the passed-in embeds, audio rows included).
ggml_tensor* apply_embed_mul(ggml_context* c, const QwenDecodeSpec& s,
                             ggml_tensor* x) {
    if (s.embedding_multiplier == 1.0f) return x;
    return bf16(c, ggml_scale(c, f32(c, x), s.embedding_multiplier));
}

// Granite's logits_scaling: logits / s after the lm_head GEMM
// (modeling_granite.py:497). 1/8 is a power of two, so the ggml_scale by the
// reciprocal is bit-exact vs the reference's division.
ggml_tensor* apply_logits_scaling(ggml_context* c, const QwenDecodeSpec& s,
                                  ggml_tensor* logits) {
    if (s.logits_scaling == 1.0f) return logits;
    return ggml_scale(c, logits, 1.0f / s.logits_scaling);
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


// ---------------------------------------------------------------------------
// Staged layer-0 parity probe.
// <env>_L0_STAGE=<stage> makes layer 0 of a 107-token prefill return
// the selected intermediate as the graph's REAL output (no capture side
// channels, which have proven untrustworthy), dumps it to
// <env>_STAGE_DIR>/<stage_prefix><stage>.f32 (default /tmp), and exits.
// Stages: n qn kn qr kr mask sc0 pr0 attn xmid down
// ---------------------------------------------------------------------------

struct L0Stage {
    const char* name = nullptr;          // nullptr = disabled
    const char* dir  = "/tmp";
    bool active(int li, int64_t S) const { return name && li == 0 && S == 107; }
    bool is(const char* s) const { return name && std::strcmp(name, s) == 0; }
};

L0Stage l0_stage(const QwenDecodeSpec& spec) {
    L0Stage st;
    st.name = env(spec, "_L0_STAGE");
    if (const char* d = env(spec, "_STAGE_DIR")) st.dir = d;
    if (st.name && !*st.name) st.name = nullptr;
    return st;
}

[[noreturn]] void stage_exit(const QwenDecodeSpec& spec, const L0Stage& st,
                             const std::vector<float>& data) {
    std::string fn = std::string(st.dir) + "/" + spec.stage_prefix + st.name + ".f32";
    bool ok = false;
    if (FILE* f = std::fopen(fn.c_str(), "wb")) {
        const size_t want = data.size();
        const size_t got = std::fwrite(data.data(), sizeof(float), want, f);
        const int fc = std::fclose(f);
        if (got == want && fc == 0) {
            std::fprintf(stderr, "%s L0 probe: wrote %s (%zu floats)\n",
                         spec.label, fn.c_str(), got);
            ok = true;
        } else {
            std::fprintf(stderr, "%s L0 probe: FAILED to write %s (wrote %zu/%zu floats, fclose=%d)\n",
                         spec.label, fn.c_str(), got, want, fc);
        }
    } else {
        std::fprintf(stderr, "%s L0 probe: FAILED to open %s\n", spec.label, fn.c_str());
    }
    std::exit(ok ? 0 : 1);
}

// ---------------------------------------------------------------------------
// One transformer layer. xh is [hidden, S] bf16 (column per token). Returns
// the layer output x ([hidden, S] f32), plus this step's k/v ([D, S, KV]
// f32, head-major) for the KV cache.
// ---------------------------------------------------------------------------

struct LayerOut { std::vector<float> x, k, v; };

// Legacy per-layer path (one run_graph per layer). Kept for the divergence-
// localization probes (<env>_L0_STAGE / <env>_DUMP_LAYERS); the default hot
// path is the whole-model graph below.
bool layer_legacy(const QwenDecodeCtx& m, int li, const std::vector<ggml_bf16_t>& xh,
                  int64_t S, int64_t past, const LayerKvCache* cache, LayerOut& o) {
    const auto& lc = m.dims;
    const int D = lc.head_dim, H = lc.n_heads, KV = lc.n_kv_heads;
    const int64_t K = past + S;
    const L0Stage stage = l0_stage(m.spec);
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

        ggml_tensor* n = rms(c, m.loader, x, p + "attn_norm.weight", lc.rms_norm_eps);
        if (st0 && stage.is("n")) return ff(c, n);

        ggml_tensor* q = lin(c, m.loader, n, p + "attn.q.weight");
        ggml_tensor* k = lin(c, m.loader, n, p + "attn.k.weight");
        ggml_tensor* v = lin(c, m.loader, n, p + "attn.v.weight");
        q = ggml_reshape_3d(c, q, D, H, S);
        k = ggml_reshape_3d(c, k, D, KV, S);
        v = ggml_reshape_3d(c, v, D, KV, S);

        if (!m.spec.qkv_bias && m.spec.qk_norm) {
            q = rms(c, m.loader, q, p + "attn.q_norm.weight", lc.rms_norm_eps);
            k = rms(c, m.loader, k, p + "attn.k_norm.weight", lc.rms_norm_eps);
        }
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

        // Per-head attention: scores = k^T q * scale, causal softmax,
        // context = probs v. Heads concatenated along the feature dim.
        const float scale = attn_scale(m.spec, D);
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

        ggml_tensor* a = lin(c, m.loader, joined, p + "attn.o.weight");       // [hidden, S]
        if (st0 && stage.is("attn")) return ff(c, a);
        x = residual_add(c, m.spec, r, a);
        r = x;
        if (st0 && stage.is("xmid")) return ff(c, x);

        n = rms(c, m.loader, x, p + "ffn_norm.weight", lc.rms_norm_eps);
        ggml_tensor* g = lin(c, m.loader, n, p + "ffn.gate.weight");
        ggml_tensor* u = lin(c, m.loader, n, p + "ffn.up.weight");
        ggml_tensor* si = bf(c, ggml_silu(c, ff(c, g)));
        ggml_tensor* z = mulb(c, si, u);
        ggml_tensor* dn = lin(c, m.loader, z, p + "ffn.down.weight");
        if (st0 && stage.is("down")) return ff(c, dn);
        x = residual_add(c, m.spec, r, dn);

        // Output: concat(flat x, flat k, flat v) so one readback carries all.
        ggml_tensor* flatx = ggml_reshape_1d(c, ff(c, x), lc.hidden * S);
        ggml_tensor* flatk = ggml_reshape_1d(c, ff(c, k), D * KV * S);
        ggml_tensor* flatv = ggml_reshape_1d(c, ff(c, v), D * KV * S);
        return ggml_concat(c, ggml_concat(c, flatx, flatk, 0), flatv, 0);
    }, o.x);

    if (ok && st0) stage_exit(m.spec, stage, o.x);

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

bool forward_legacy(const QwenDecodeCtx& m, const std::vector<float>& input, int64_t S,
                     LlmState& state, std::vector<float>& logits, std::string& e) {
    // Granite's embedding multiplier applies to the layer-0 input (prefill:
    // the merged embeds; decode: the looked-up row). The host f32 multiply +
    // tobf round is bit-identical to the in-graph f32-scale + bf16-cast.
    std::vector<float> entry = input;
    if (m.spec.embedding_multiplier != 1.0f)
        for (auto& v : entry) v *= m.spec.embedding_multiplier;
    std::vector<ggml_bf16_t> x = tobf(entry);
    if (state.layers.empty()) state.layers.resize(m.dims.n_layers);

    for (int li = 0; li < (int)m.dims.n_layers; ++li) {
        LayerOut o;
        if (!layer_legacy(m, li, x, S, state.length, &state.layers[li], o)) {
            e = std::string(m.spec.label) + " LLM layer graph failed at " + std::to_string(li);
            return false;
        }
        append_kv(state.layers[li].k, o.k, state.length, S,
                  m.dims.head_dim, m.dims.n_kv_heads);
        append_kv(state.layers[li].v, o.v, state.length, S,
                  m.dims.head_dim, m.dims.n_kv_heads);
        for (size_t z = 0; z < o.x.size(); ++z) {
            if (!std::isfinite(o.x[z])) {
                e = "non-finite " + std::string(m.spec.label) + " hidden at layer " +
                    std::to_string(li) + " index " + std::to_string(z);
                return false;
            }
        }
        // <env>_DUMP_LAYERS=<prefix> dumps each layer's output hidden as
        // <prefix>_<li>.f32 (prefill only).
        if (S > 1) {
            if (const char* dp = env(m.spec, "_DUMP_LAYERS")) {
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

    // Logits for the last token: final RMSNorm, then the lm_head (tied to the
    // embedding table unless the spec carries an untied llm.lm_head.weight).
    std::vector<ggml_bf16_t> last(m.dims.hidden);
    std::copy(x.end() - last.size(), x.end(), last.begin());
    bool ok = run_graph([&](ggml_context* c) -> ggml_tensor* {
        int64_t ne[2] = {m.dims.hidden, 1};
        ggml_tensor* t = graph_input_tensor(c, GGML_TYPE_BF16, 2, ne,
                                            last.data(), last.size() * sizeof(last[0]));
        t = rms(c, m.loader, t, "llm.final_norm.weight", m.dims.rms_norm_eps);
        ggml_tensor* lg = ggml_mul_mat(c, wb(c, m.loader, lm_head_name(m.spec)), t);
        return ff(c, apply_logits_scaling(c, m.spec, lg));
    }, logits);
    if (!ok) e = std::string(m.spec.label) + " lm_head graph failed";
    return ok;
}

// ===========================================================================
// Whole-model graphs + device-resident KV cache.
//
// One prefill graph (one-shot, built per utterance) and one decode-step graph
// per token (one-shot, exact-width KV; captured on GPU). All layers + final
// norm + lm_head live in a SINGLE ggml cgraph, so the hidden state flows
// device->device between layers (no host tobf round-trip) and there is exactly
// one graph alloc per step instead of one per layer.
//
// The KV cache lives in a persistent device context (per-layer [D, max_cache,
// KV] bf16, zero-initialized) referenced as fixed graph leaves (like loader
// weights); each step's k/v are written in-graph via view + cpy (the llama.cpp
// static-cache pattern), so there is no per-step cache upload. RoPE cos/sin
// are precomputed once as a device table [D, max_cache] and selected in-graph
// via ggml_get_rows on a position index.
//
// The per-layer op order/dtype discipline (bf/ff casts, f32 elementwise, f32
// RMSNorm->bf16->bf16 weight, rotate-half RoPE in f32, soft_max_ext with f32
// mask) is byte-identical to layer_legacy.
// ===========================================================================

bool debug_probe_active(const QwenDecodeSpec& spec) {
    return env(spec, "_L0_STAGE") || env(spec, "_DUMP_LAYERS");
}

// Per-S captured prefill graph + its stable host input pool.
struct PrefillReplayEntry {
    int64_t S = 0;
    GraphInputPool pool;
    std::unique_ptr<ReplayGraph> graph;
    ggml_bf16_t* xb_buf = nullptr;  // stable pool backing for input #0 (varying)
};

struct PrefillCache {
    LruCache<int64_t, PrefillReplayEntry> by_S;
    explicit PrefillCache(size_t cap) : by_S(cap) {}
    void clear() { by_S.clear(); }
};

// K-step multistep decode graph (captured): K chained decode steps, one
// device<->host sync per K tokens.
struct KStepGraph {
    int K = 0;
    int64_t start_past = 0;
    std::unique_ptr<ReplayGraph> rg;
    size_t in_prev_tok = 0;
    std::vector<size_t> in_pos, in_mask;
    std::vector<int32_t> host_pos;              // [K]
    std::vector<std::vector<float>> host_mask;   // [K]
    std::vector<float> cap_tokens;               // [K] (i32 reinterpreted as f32 bytes)
};

// ONE captured K-step graph per K (start_past is a runtime input, so a single
// graph serves every decode step and every utterance -> perfect capture
// amortization).
struct KStepKey { int K;
    bool operator==(const KStepKey& o) const { return K == o.K; } };
struct KStepKeyHash { size_t operator()(const KStepKey& k) const noexcept { return (size_t)k.K; } };

// Per-spec process-global decode caches, keyed by spec address (each model
// bundle owns one static spec; first requester sizes the device cache).
// Starling inference is process-serial (one Backend), so — like the rest of
// the engine's caches — this is not internally locked.
struct SpecState {
    std::unique_ptr<DeviceCache> device_cache;
    std::unique_ptr<PrefillCache> prefill_cache;
    std::unordered_map<KStepKey, std::unique_ptr<KStepGraph>, KStepKeyHash> kstep;
    std::once_flag device_once, prefill_once, kstep_once;
};

std::unordered_map<const QwenDecodeSpec*, SpecState>& spec_states() {
    static std::unordered_map<const QwenDecodeSpec*, SpecState> m;
    return m;
}
SpecState& state_for(const QwenDecodeSpec& spec) {
    return spec_states()[&spec];
}

// Process-global device-resident KV cache + precomputed RoPE tables, one per
// spec; zeroed at the start of each utterance. The KV tensors live in a
// persistent ggml_context allocated on the backend buffer; graphs reference
// them (and the RoPE tables) as fixed leaves. Freed by the registered
// decode-cache-clearer BEFORE backend teardown.
DeviceCache* get_device_cache(const QwenDecodeCtx& m, std::string& e) {
    SpecState& st = state_for(m.spec);
    // Capture the map-slot POINTER, not the local reference: the shutdown
    // clearer runs long after this frame is gone, and a by-reference capture
    // of `st` dereferences a dead stack slot (the historical exit-time
    // "double free or corruption" in the decode-cache clearers).
    SpecState* stp = &st;
    std::call_once(st.device_once, [stp] {
        register_decode_cache_clearer([stp] { stp->device_cache.reset(); });
    });
    if (st.device_cache) return st.device_cache.get();
    const auto& lc = m.dims;
    st.device_cache = std::unique_ptr<DeviceCache>(new DeviceCache());
    if (!st.device_cache->init((int) lc.n_layers, (int) lc.head_dim, (int) lc.n_kv_heads, (int) lc.max_cache, lc.rope_theta, global_backend().handle(), e)) {
        st.device_cache.reset();
        return nullptr;
    }
    return st.device_cache.get();
}

// Append one transformer layer's ops to ctx. x_in is [hidden, S] bf16
// (device-resident, flows straight from the previous layer). Writes this step's
// k/v into the device cache in-graph and assembles kall/vall for attention.
//   kv_mode 0 = prefill exact (cpy slots [0,S), attend to new k/v)
//   kv_mode 1 = decode exact-width (cpy slot `past`, attend [0, past+S))
//   kv_mode 2 = decode full-capacity (set_rows slot `past`, attend [0, max_cache))
// idx_past is the runtime i32[1] write index used only by mode 2.
// cs/sn are [D, S] bf16 (RoPE rows). mask is f32 [K, S] (K = past+S for modes
// 0/1, max_cache for mode 2).
ggml_tensor* append_layer_new(ggml_context* c, const QwenDecodeCtx& m, int li,
                              ggml_tensor* x_in, int64_t S, int64_t past,
                              ggml_tensor* cache_k, ggml_tensor* cache_v,
                              ggml_tensor* cs, ggml_tensor* sn,
                              ggml_tensor* mask, int kv_mode,
                              ggml_tensor* idx_past) {
    const auto& lc = m.dims;
    const int D = lc.head_dim, H = lc.n_heads, KV = lc.n_kv_heads;
    const int64_t K = (kv_mode == 2) ? (int64_t)lc.max_cache : (past + S);
    const std::string p = "llm.blk." + std::to_string(li) + ".";
    ggml_tensor* r = x_in;

    ggml_tensor* n = rms(c, m.loader, x_in, p + "attn_norm.weight", lc.rms_norm_eps);
    // Projection family per spec (see QwenDecodeSpec): Qwen2.5 takes biased
    // q/k/v by BASE name and has no q_norm/k_norm; Qwen3 takes bias-free full
    // names plus per-head q_norm/k_norm after the reshape.
    ggml_tensor* q, *k, *v;
    if (m.spec.qkv_bias) {
        q = linear_bf16(c, m.loader, n, p + "attn.q", true);
        k = linear_bf16(c, m.loader, n, p + "attn.k", true);
        v = linear_bf16(c, m.loader, n, p + "attn.v", true);
    } else {
        q = lin(c, m.loader, n, p + "attn.q.weight");
        k = lin(c, m.loader, n, p + "attn.k.weight");
        v = lin(c, m.loader, n, p + "attn.v.weight");
    }
    q = ggml_reshape_3d(c, q, D, H, S);
    k = ggml_reshape_3d(c, k, D, KV, S);
    v = ggml_reshape_3d(c, v, D, KV, S);
    if (!m.spec.qkv_bias && m.spec.qk_norm) {
        q = rms(c, m.loader, q, p + "attn.q_norm.weight", lc.rms_norm_eps);
        k = rms(c, m.loader, k, p + "attn.k_norm.weight", lc.rms_norm_eps);
    }
    q = ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));  // [D, S, H]
    k = ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3));  // [D, S, KV]
    v = ggml_cont(c, ggml_permute(c, v, 0, 2, 1, 3));  // [D, S, KV]

    // RoPE (rotate-half, f32 math). cs/sn are the SAME bf16 values as the
    // legacy host table; only their source differs (device table vs host input).
    auto rope = [&](ggml_tensor* z, int heads) {
        ggml_tensor* lo = ggml_view_3d(c, z, D / 2, S, heads,
                                       z->nb[1], z->nb[2], 0);
        ggml_tensor* hi = ggml_view_3d(c, z, D / 2, S, heads,
                                       z->nb[1], z->nb[2],
                                       (size_t)(D / 2) * z->nb[0]);
        ggml_tensor* rot = ggml_concat(c, ggml_scale(c, ff(c, hi), -1.0f),
                                       ff(c, lo), 0);
        rot = bf(c, rot);
        return addb(c, mulb(c, z, cs), mulb(c, rot, sn));
    };
    q = rope(q, H);
    k = rope(k, KV);

    // KV cache: write this step's k/v into the device cache in-graph and
    // assemble kall/vall. Depending on the cpy/set_rows result (via attention)
    // forces the write to execute before the read.
    ggml_tensor* kall;
    ggml_tensor* vall;
    if (kv_mode == 0) {            // prefill exact
        ggml_tensor* kview = ggml_view_3d(c, cache_k, D, S, KV,
                                          cache_k->nb[1], cache_k->nb[2], 0);
        ggml_tensor* vview = ggml_view_3d(c, cache_v, D, S, KV,
                                          cache_v->nb[1], cache_v->nb[2], 0);
        kall = ggml_cpy(c, k, kview);  // [D, S, KV], slots [0,S)
        vall = ggml_cpy(c, v, vview);
    } else if (kv_mode == 2) {     // decode full-capacity (captured, dynamic slot)
        // set_rows writes ff(k) (f32) into the bf16 cache at slot `past`; the
        // values are bf16-representable so the f32->bf16 round is exact. The
        // result is a view of the WHOLE cache (slot `past` now updated), which
        // we use directly as kall -> the set_rows executes before attention.
        kall = ggml_set_rows(c, cache_k, ff(c, k), idx_past);
        vall = ggml_set_rows(c, cache_v, ff(c, v), idx_past);
    } else {                       // decode exact-width
        ggml_tensor* kslot = ggml_view_3d(c, cache_k, D, 1, KV,
                                          cache_k->nb[1], cache_k->nb[2],
                                          (size_t)past * cache_k->nb[1]);
        ggml_tensor* vslot = ggml_view_3d(c, cache_v, D, 1, KV,
                                          cache_v->nb[1], cache_v->nb[2],
                                          (size_t)past * cache_v->nb[1]);
        ggml_tensor* knew = ggml_cpy(c, k, kslot);  // writes slot `past`
        ggml_tensor* vnew = ggml_cpy(c, v, vslot);
        ggml_tensor* kprev = ggml_view_3d(c, cache_k, D, past, KV,
                                          cache_k->nb[1], cache_k->nb[2], 0);
        ggml_tensor* vprev = ggml_view_3d(c, cache_v, D, past, KV,
                                          cache_v->nb[1], cache_v->nb[2], 0);
        // CUDA concat is F32-only (concat.cu GGML_ASSERT); cast via ff/bf like
        // the legacy concat over the cached+new KV. The concat depends on the
        // cpy results, which forces the in-graph cache writes to execute first.
        kall = past ? bf(c, ggml_concat(c, ff(c, kprev), ff(c, knew), 1)) : knew;
        vall = past ? bf(c, ggml_concat(c, ff(c, vprev), ff(c, vnew), 1)) : vnew;
    }

    // Attention. Two formulations, switchable for an A/B exactness gate:
    //   - batched (default): one batched mul_mat over all heads using ggml's
    //     native GQA broadcast (ne12 % ne02 == 0, r2 = H/KV). Math per head is
    //     unchanged; produces [hidden, S] via permute+reshape.
    //   - per-head (<env>_PERHEAD=1): the H-iteration loop, identical op
    //     sequence to layer_legacy.
    const float scale = attn_scale(m.spec, D);
    ggml_tensor* joined;
    if (!env(m.spec, "_PERHEAD")) {
        // scores = K^T Q / sqrt(D) over all heads (GQA broadcast KV->H).
        ggml_tensor* sc = ggml_mul_mat(c, kall, q);                 // [K, S, H]
        sc = bf(c, ggml_scale(c, ff(c, sc), scale));
        ggml_tensor* pr = bf(c, ggml_soft_max_ext(c, ff(c, sc), ff(c, mask), 1.0f, 0.0f));
        // context = V^T @ probs: permute vall [D,K,KV] -> [K,D,KV], GQA broadcast.
        ggml_tensor* vt = ggml_cont(c, ggml_permute(c, vall, 1, 0, 2, 3));  // [K, D, KV]
        ggml_tensor* co = ggml_mul_mat(c, vt, pr);                 // [D, S, H]
        // heads -> features: [D,S,H] -> [D,H,S] -> [hidden=D*H, S].
        joined = ggml_reshape_2d(c, ggml_cont(c, ggml_permute(c, co, 0, 2, 1, 3)),
                                 (int64_t)D * H, S);                  // [hidden, S]
        joined = bf(c, joined);
    } else {
        joined = nullptr;
        for (int h = 0; h < H; ++h) {
            int kvh = h / (H / KV);
            ggml_tensor* qh = ggml_view_2d(c, q, D, S, q->nb[1], (size_t)h * q->nb[2]);
            ggml_tensor* kh = ggml_view_2d(c, kall, D, K, kall->nb[1], (size_t)kvh * kall->nb[2]);
            ggml_tensor* vh = ggml_view_2d(c, vall, D, K, vall->nb[1], (size_t)kvh * vall->nb[2]);
            ggml_tensor* sc = bf(c, ggml_mul_mat(c, kh, qh));          // [K, S]
            sc = bf(c, ggml_scale(c, ff(c, sc), scale));
            ggml_tensor* pr = bf(c, ggml_soft_max_ext(c, ff(c, sc), ff(c, mask), 1.0f, 0.0f));
            ggml_tensor* vtt = ggml_cont(c, ggml_transpose(c, vh));    // [K, D]
            ggml_tensor* co = bf(c, ggml_mul_mat(c, vtt, pr));         // [D, S]
            joined = joined ? bf(c, ggml_concat(c, ff(c, joined), ff(c, co), 0)) : co;
        }
    }

    ggml_tensor* a = lin(c, m.loader, joined, p + "attn.o.weight");
    ggml_tensor* x = residual_add(c, m.spec, r, a);
    r = x;
    n = rms(c, m.loader, x, p + "ffn_norm.weight", lc.rms_norm_eps);
    ggml_tensor* g = lin(c, m.loader, n, p + "ffn.gate.weight");
    ggml_tensor* u = lin(c, m.loader, n, p + "ffn.up.weight");
    ggml_tensor* si = bf(c, ggml_silu(c, ff(c, g)));
    ggml_tensor* z = mulb(c, si, u);
    ggml_tensor* dn = lin(c, m.loader, z, p + "ffn.down.weight");
    x = residual_add(c, m.spec, r, dn);
    return x;  // [hidden, S] bf16
}

// Build (or fetch) the captured prefill graph for prompt length S.
// On GPU the prefill graph is captured into a per-S ReplayGraph (prompt
// length S fully determines every shape). Inputs are inputs_embeds (bf16
// [hidden, S], the only varying input) + position (i32 [S], constant for S) +
// causal mask (f32 [S, S], constant for S). The KV write path is unchanged
// (kv_mode=0 cpy into slots [0, S); the captured cpys execute on replay,
// writing the same device cache the decode K-step graph reads). CPU keeps the
// one-shot build. Cached in a bounded LRU (runtime/lru_cache.hpp) of size
// STARLING_REPLAY_CACHE_SIZE (default 16) — without the bound, each distinct
// prompt length would pin its own captured graph + private gallocr until
// exit (the unbounded-cache OOM bug). Cleared via register_decode_cache_clearer.
PrefillReplayEntry* get_or_build_prefill(const QwenDecodeCtx& m, int64_t S,
                                         std::string& e) {
    SpecState& st = state_for(m.spec);
    SpecState* stp = &st;  // see get_device_cache: no dangling by-ref captures
    std::call_once(st.prefill_once, [stp] {
        register_decode_cache_clearer([stp] { stp->prefill_cache.reset(); });
    });
    if (!st.prefill_cache)
        st.prefill_cache = std::make_unique<PrefillCache>(replay_cache_size());
    const auto& lc = m.dims;
    DeviceCache* dc = get_device_cache(m, e);
    if (!dc) return nullptr;

    // get_or_init places the entry (stable address) first, then builds: the
    // ReplayGraph build lambda captures the stable pool pointers. On a miss at
    // capacity the LRU prompt length is evicted (its captured graph freed)
    // before this one is inserted.
    return st.prefill_cache->by_S.get_or_init(S,
        [&](PrefillReplayEntry& entry) -> PrefillReplayEntry& {
            entry.S = S;
            // xb (varying per utterance) -- filled each replay into this stable buffer.
            entry.xb_buf = reinterpret_cast<ggml_bf16_t*>(
                entry.pool.alloc_bytes((size_t)S * lc.hidden * sizeof(ggml_bf16_t)));
            // pos (constant for S = [0, S)) and mask (constant for S = causal).
            int32_t* pos_buf = entry.pool.alloc_i32((size_t)S);
            for (int64_t i = 0; i < S; ++i) pos_buf[i] = (int32_t)i;
            std::vector<float> mask = build_causal_mask(S, 0);
            float* mask_buf = entry.pool.alloc_f32((size_t)S * S);
            std::memcpy(mask_buf, mask.data(), (size_t)S * S * sizeof(float));

            entry.graph = std::make_unique<ReplayGraph>(global_backend(),
                [&](ggml_context* c) -> ggml_tensor* {
                    int64_t xne[2] = {lc.hidden, S};
                    ggml_tensor* x = graph_input_tensor(c, GGML_TYPE_BF16, 2, xne,
                        entry.xb_buf, (size_t)S * lc.hidden * sizeof(ggml_bf16_t));
                    x = apply_embed_mul(c, m.spec, x);
                    int64_t pne[1] = {S};
                    ggml_tensor* pos_t = graph_input_tensor(c, GGML_TYPE_I32, 1, pne,
                        pos_buf, (size_t)S * sizeof(int32_t));
                    ggml_tensor* cs = ggml_get_rows(c, dc->rope_cos, pos_t);  // [D, S]
                    ggml_tensor* sn = ggml_get_rows(c, dc->rope_sin, pos_t);
                    int64_t mne[2] = {S, S};
                    ggml_tensor* mt = graph_input_tensor(c, GGML_TYPE_F32, 2, mne,
                        mask_buf, (size_t)S * S * sizeof(float));
                    for (int li = 0; li < (int)lc.n_layers; ++li)
                        x = append_layer_new(c, m, li, x, S, 0, dc->k[li], dc->v[li],
                                             cs, sn, mt, /*kv_mode=*/0, nullptr);
                    // Final norm + lm_head on the LAST token only.
                    ggml_tensor* last = ggml_view_2d(c, x, lc.hidden, 1, x->nb[1],
                                                     (size_t)(S - 1) * x->nb[1]);
                    ggml_tensor* n = rms(c, m.loader, last, "llm.final_norm.weight", lc.rms_norm_eps);
                    ggml_tensor* lg = ggml_mul_mat(c, wb(c, m.loader, lm_head_name(m.spec)), n);
                    return ff(c, apply_logits_scaling(c, m.spec, lg));
                });
            return entry;
        });
}

bool forward_prefill(const QwenDecodeCtx& m, const std::vector<float>& input,
                     int64_t S, LlmState& state,
                     std::vector<float>& logits, std::string& e) {
    const auto& lc = m.dims;
    ensure_weights_realized(m.loader);
    DeviceCache* dc = get_device_cache(m, e);
    if (!dc) return false;
    dc->zero();  // fresh utterance
    state.length = 0;
    if (state.layers.empty()) state.layers.resize(lc.n_layers);

    // CPU backend keeps the one-shot build (capture is GPU-only, like the
    // encoder + K-step decode).
    if (!global_backend().is_gpu()) {
        std::vector<ggml_bf16_t> xb = tobf(input);  // [hidden*S] bf16 host
        std::vector<int32_t> pos((size_t)S);
        for (int64_t i = 0; i < S; ++i) pos[(size_t)i] = (int32_t)i;
        std::vector<float> mask = build_causal_mask(S, 0);
        bool ok = run_graph([&](ggml_context* c) -> ggml_tensor* {
            int64_t xne[2] = {lc.hidden, S};
            ggml_tensor* x = graph_input_tensor(c, GGML_TYPE_BF16, 2, xne,
                                                xb.data(), xb.size() * sizeof(xb[0]));
            x = apply_embed_mul(c, m.spec, x);
            int64_t pne[1] = {S};
            ggml_tensor* pos_t = graph_input_tensor(c, GGML_TYPE_I32, 1, pne,
                                                    pos.data(), pos.size() * sizeof(int32_t));
            ggml_tensor* cs = ggml_get_rows(c, dc->rope_cos, pos_t);  // [D, S]
            ggml_tensor* sn = ggml_get_rows(c, dc->rope_sin, pos_t);
            int64_t mne[2] = {S, S};
            ggml_tensor* mt = graph_input_tensor(c, GGML_TYPE_F32, 2, mne,
                                                 mask.data(), mask.size() * sizeof(float));
            for (int li = 0; li < (int)lc.n_layers; ++li)
                x = append_layer_new(c, m, li, x, S, 0, dc->k[li], dc->v[li],
                                     cs, sn, mt, /*kv_mode=*/0, nullptr);
            ggml_tensor* last = ggml_view_2d(c, x, lc.hidden, 1, x->nb[1],
                                             (size_t)(S - 1) * x->nb[1]);
            ggml_tensor* n = rms(c, m.loader, last, "llm.final_norm.weight", lc.rms_norm_eps);
            ggml_tensor* lg = ggml_mul_mat(c, wb(c, m.loader, lm_head_name(m.spec)), n);
            return ff(c, apply_logits_scaling(c, m.spec, lg));
        }, logits);
        if (!ok) { e = std::string(m.spec.label) + " prefill graph failed"; return false; }
        state.length = S;
        return true;
    }

    // GPU: captured per-S ReplayGraph. Only inputs_embeds varies; pos + mask are
    // constant for S (held in the stable pool, re-uploaded each replay).
    PrefillReplayEntry* pe = get_or_build_prefill(m, S, e);
    if (!pe) return false;
    const size_t nx = (size_t)S * lc.hidden;
    for (size_t i = 0; i < nx; ++i)
        pe->xb_buf[i] = ggml_fp32_to_bf16(input[i]);
    for (size_t i = 0; i < pe->graph->n_inputs(); ++i)
        pe->graph->set_input(i, pe->graph->input_host(i), pe->graph->input_nbytes(i));
    if (!pe->graph->compute(logits)) { e = std::string(m.spec.label) + " prefill replay failed"; return false; }
    state.length = S;
    return true;
}

// Whole-model decode-step graph (S=1): embed(prev) -> layers -> lm_head.
// Exact-width KV (one-shot per step). Reads slots [0, past), writes slot
// `past`, attention over [0, past).
bool forward_decode(const QwenDecodeCtx& m, int32_t prev_token, int64_t past,
                    LlmState& state, std::vector<float>& logits,
                    std::string& e) {
    const auto& lc = m.dims;
    DeviceCache* dc = get_device_cache(m, e);
    if (!dc) return false;
    const int64_t S = 1;
    const int64_t K = past + S;
    // <env>_FULLCAP=1 selects full-capacity attention (the captured-graph
    // shape): attention over [0, max_cache) with an additive f32 mask, used
    // to A/B-gate whether padded softmax flips any token.
    const bool fullcap = env(m.spec, "_FULLCAP") != nullptr;
    const int kv_mode = fullcap ? 2 : 1;
    std::vector<int32_t> pos = {(int32_t)past};
    std::vector<int32_t> idx_past = {(int32_t)past};
    std::vector<float> mask_exact((size_t)K, 0.0f);          // decode: all keys valid
    std::vector<float> mask_full((size_t)lc.max_cache, 0.0f);// 0 valid, -inf beyond past
    if (fullcap) {
        const float neg = -3.3895313892515355e38f;
        for (int64_t j = past + 1; j < (int64_t)lc.max_cache; ++j)
            mask_full[(size_t)j] = neg;
    }
    const std::vector<float>& mask = fullcap ? mask_full : mask_exact;
    const int64_t mask_w = fullcap ? (int64_t)lc.max_cache : K;

    bool ok = run_graph([&](ggml_context* c) -> ggml_tensor* {
        int64_t one[1] = {1};
        ggml_tensor* id_t = graph_input_tensor(c, GGML_TYPE_I32, 1, one,
                                               &prev_token, sizeof(int32_t));
        ggml_tensor* x = ggml_get_rows(c, clone_weight(c, m.loader, "llm.embed.weight"), id_t);
        x = apply_embed_mul(c, m.spec, x);
        ggml_tensor* pos_t = graph_input_tensor(c, GGML_TYPE_I32, 1, one,
                                                pos.data(), sizeof(int32_t));
        ggml_tensor* cs = ggml_get_rows(c, dc->rope_cos, pos_t);  // [D, 1]
        ggml_tensor* sn = ggml_get_rows(c, dc->rope_sin, pos_t);
        int64_t mne[2] = {mask_w, 1};
        ggml_tensor* mt = graph_input_tensor(c, GGML_TYPE_F32, 2, mne,
                                             mask.data(), mask.size() * sizeof(float));
        ggml_tensor* idx_t = fullcap
            ? graph_input_tensor(c, GGML_TYPE_I32, 1, one, idx_past.data(), sizeof(int32_t))
            : nullptr;
        for (int li = 0; li < (int)lc.n_layers; ++li)
            x = append_layer_new(c, m, li, x, S, past, dc->k[li], dc->v[li],
                                 cs, sn, mt, kv_mode, idx_t);
        ggml_tensor* n = rms(c, m.loader, x, "llm.final_norm.weight", lc.rms_norm_eps);
        ggml_tensor* lg = ggml_mul_mat(c, wb(c, m.loader, lm_head_name(m.spec)), n);
        return ff(c, apply_logits_scaling(c, m.spec, lg));
    }, logits);
    if (!ok) { e = std::string(m.spec.label) + " decode graph failed"; return false; }
    state.length = past + S;
    return true;
}

// ===========================================================================
// K-step multistep decode (captured ReplayGraph).
//
// Captures K consecutive decode steps into ONE ReplayGraph with the per-step
// state chained IN-GRAPH (output token -> get_rows(embed) -> next step's
// input), so there is ONE device<->host sync per K steps instead of per step.
// KV writes land at baked slots [start_past, start_past+K) (the graph is built
// per start_past); attention is EXACT-WIDTH per step (no padding, no full-
// capacity mask) so the softmax reduction order is byte-identical to the one-
// step decode.
//
// Graph cache keyed on K: start_past is a runtime input, so ONE graph per K
// serves every decode step and every utterance; graphs are cached
// process-globally and reused across reps / same-prompt runs.
// ===========================================================================

int kstep_K(const QwenDecodeSpec& spec) {
    int v = 4;
    if (const char* s = env(spec, "_KSTEP")) {
        int e = std::atoi(s);
        if (e >= 1) v = e;
    }
    // kGraphSize=32768 bounds the K-step graph (~2680 nodes/step) to K<=12;
    // clamp to 8 (no steady-state gain beyond it — the decode is compute-bound).
    if (v > 8) v = 8;
    return v;
}

// Build (or fetch) the single full-capacity K-step graph for K. start_past is
// a runtime input, so this graph is reused for every decode-step batch.
KStepGraph* get_or_build_kstep(const QwenDecodeCtx& m, int K, std::string& e) {
    SpecState& st = state_for(m.spec);
    SpecState* stp = &st;  // see get_device_cache: no dangling by-ref captures
    std::call_once(st.kstep_once, [stp] {
        register_decode_cache_clearer([stp] { stp->kstep.clear(); });
    });
    KStepKey key{K};
    auto it = st.kstep.find(key);
    if (it != st.kstep.end()) return it->second.get();

    const auto& lc = m.dims;
    DeviceCache* dc = get_device_cache(m, e);
    if (!dc) return nullptr;

    auto kg = std::unique_ptr<KStepGraph>(new KStepGraph());
    kg->K = K;
    kg->start_past = 0;
    // Per-replay host backing for the runtime inputs (overwritten each replay).
    kg->host_pos.assign((size_t)K, 0);
    kg->host_mask.assign((size_t)K, std::vector<float>((size_t)lc.max_cache, 0.0f));
    kg->in_pos.resize((size_t)K);
    kg->in_mask.resize((size_t)K);
    kg->cap_tokens.assign((size_t)K, 0.0f);

    KStepGraph* raw = kg.get();
    const int64_t mc = lc.max_cache;
    raw->rg = std::unique_ptr<ReplayGraph>(new ReplayGraph(global_backend(),
        [&](ggml_context* c) -> ggml_tensor* {
            int64_t one[1] = {1};
            size_t idx = 0;
            // 0: runtime prev-token (the token entering step 0).
            int32_t zero_tok = 0;
            ggml_tensor* prev_tok_t = graph_input_tensor(c, GGML_TYPE_I32, 1, one,
                                                          &zero_tok, sizeof(int32_t));
            raw->in_prev_tok = idx++;
            // 1..K: runtime per-step position (= start_past + j); also the
            // set_rows write index for that step's KV slot.
            std::vector<ggml_tensor*> pos_t((size_t)K), mask_t((size_t)K);
            for (int j = 0; j < K; ++j) {
                pos_t[(size_t)j] = graph_input_tensor(c, GGML_TYPE_I32, 1, one,
                                                       &raw->host_pos[(size_t)j], sizeof(int32_t));
                raw->in_pos[(size_t)j] = idx++;
            }
            // K+1..2K: runtime per-step full-capacity masks [max_cache, 1].
            for (int j = 0; j < K; ++j) {
                int64_t mw[2] = {mc, 1};
                mask_t[(size_t)j] = graph_input_tensor(c, GGML_TYPE_F32, 2, mw,
                                                       raw->host_mask[(size_t)j].data(),
                                                       raw->host_mask[(size_t)j].size() * sizeof(float));
                raw->in_mask[(size_t)j] = idx++;
            }

            ggml_tensor* embed_w = clone_weight(c, m.loader, "llm.embed.weight");
            // Untied models carry a separate lm_head; tied models keep using
            // the embedding clone (identical single node, moss/ark unchanged).
            ggml_tensor* head_w = m.spec.tied_lm_head
                ? embed_w
                : clone_weight(c, m.loader, "llm.lm_head.weight");

            // Chain K steps in-graph: tok = prev-token; each step's argmax feeds
            // the next step's embed (get_rows), all on device.
            ggml_tensor* tok = prev_tok_t;
            std::vector<ggml_tensor*> tok_nodes;
            tok_nodes.reserve((size_t)K);
            for (int j = 0; j < K; ++j) {
                ggml_tensor* x = ggml_get_rows(c, embed_w, tok);                     // [hidden, 1]
                x = apply_embed_mul(c, m.spec, x);
                ggml_tensor* cs = ggml_get_rows(c, dc->rope_cos, pos_t[(size_t)j]);  // [D, 1]
                ggml_tensor* sn = ggml_get_rows(c, dc->rope_sin, pos_t[(size_t)j]);
                for (int li = 0; li < (int)lc.n_layers; ++li)
                    x = append_layer_new(c, m, li, x, /*S=*/1, /*past=*/0,
                                         dc->k[li], dc->v[li], cs, sn, mask_t[(size_t)j],
                                         /*kv_mode=*/2, pos_t[(size_t)j]);
                ggml_tensor* n = rms(c, m.loader, x, "llm.final_norm.weight", lc.rms_norm_eps);
                // Argmax is invariant under the positive logits_scaling, so the
                // K-step graph skips the division (no logits are read back).
                ggml_tensor* logits = ggml_mul_mat(c, head_w, n);                  // [vocab, 1]
                ggml_tensor* tj = ggml_argmax(c, logits);                            // i32 [1]
                // CUDA concat is F32-only; token ids < 2^24 so (float)tok is exact.
                tok_nodes.push_back(ggml_cast(c, tj, GGML_TYPE_F32));
                tok = tj;  // chain: next step embeds this step's argmax token
            }

            // Ring of K token ids (f32), captured for a single readback.
            ggml_tensor* ring = tok_nodes[0];
            for (int j = 1; j < K; ++j)
                ring = ggml_concat(c, ring, tok_nodes[j], 0);                       // f32 [K]
            capture_graph_output(ring, &raw->cap_tokens);
            return ring;
        }));

    if (!raw->rg) { e = std::string(m.spec.label) + " K-step graph build failed"; return nullptr; }
    it = st.kstep.emplace(key, std::move(kg)).first;
    return it->second.get();
}

// Run one K-step replay from `past` (the slot the first step writes). Appends
// up to K tokens, stopping at EOS / max_new_tokens. `prev` in=out (token
// entering / last emitted); `past` advances by the steps actually consumed.
bool run_kstep(const QwenDecodeCtx& m, int32_t& prev, int64_t& past, int K,
               int32_t eos, int max_new_tokens, std::vector<int32_t>& ids,
               bool& hit_eos, std::string& e) {
    KStepGraph* kg = get_or_build_kstep(m, K, e);
    if (!kg) return false;
    const int64_t mc = m.dims.max_cache;
    const float neg = -3.3895313892515355e38f;
    // Pack runtime inputs: prev-token, per-step positions, per-step masks.
    kg->rg->set_input(kg->in_prev_tok, &prev, sizeof(int32_t));
    for (int j = 0; j < K; ++j) {
        int64_t boundary = past + j;  // step j writes slot (past+j); keys [0, past+j]
        kg->host_pos[(size_t)j] = (int32_t)(past + j);
        // Load-bearing invariant: no decode step's device index -- neither the
        // set_rows KV write target nor the get_rows RoPE source -- may reach
        // max_cache. greedy_generate's tail-cap (Kk = min(K, remaining)) makes
        // this hold; this guard turns any future regression of that bound into
        // a detectable error instead of silent device-memory corruption. (The
        // OOB write lands in ggml's buffer padding and does NOT fault, so a
        // crash/CUDA-error gate alone cannot catch this class of bug.)
        if (kg->host_pos[(size_t)j] < 0 || kg->host_pos[(size_t)j] >= (int32_t)mc) {
            e = std::string(m.spec.label) + " K-step position out of bounds (pos=" +
                std::to_string(kg->host_pos[(size_t)j]) +
                ", max_cache=" + std::to_string(mc) + ")";
            return false;
        }
        std::vector<float>& mk = kg->host_mask[(size_t)j];
        for (int64_t s = 0; s < boundary + 1 && s < mc; ++s) mk[(size_t)s] = 0.0f;
        for (int64_t s = boundary + 1; s < mc; ++s) mk[(size_t)s] = neg;
        kg->rg->set_input(kg->in_pos[(size_t)j], &kg->host_pos[(size_t)j], sizeof(int32_t));
        kg->rg->set_input(kg->in_mask[(size_t)j], mk.data(), mk.size() * sizeof(float));
    }
    std::vector<float> out;
    if (!kg->rg->compute_with_captures(out)) { e = std::string(m.spec.label) + " K-step replay failed"; return false; }

    hit_eos = false;
    for (int j = 0; j < K; ++j) {
        if ((int)ids.size() >= max_new_tokens) break;
        int32_t tok = (int32_t)kg->cap_tokens[(size_t)j];
        ids.push_back(tok);
        prev = tok;
        past += 1;
        if (tok == eos) { hit_eos = true; break; }
    }
    return true;
}

} // namespace

size_t prefill_replay_cache_size(const QwenDecodeSpec& spec) {
    SpecState& st = state_for(spec);
    return st.prefill_cache ? st.prefill_cache->by_S.size() : 0;
}

bool llm_prefill(const QwenDecodeCtx& m, const InputsEmbeds& i, int32_t maxc,
                 PrefillResult& o, std::string& e) {
    if (i.n_tokens <= 0 || i.width != (int64_t)m.dims.hidden || i.n_tokens > maxc) {
        e = std::string("invalid ") + m.spec.label + " prefill shape/cache";
        return false;
    }
    ensure_weights_realized(m.loader);
    const bool dbg = debug_probe_active(m.spec);
    bool ok = dbg ? forward_legacy(m, i.data, i.n_tokens, o.state, o.logits, e)
                  : forward_prefill(m, i.data, i.n_tokens, o.state, o.logits, e);
    if (!ok) return false;
    o.first_token = argmax_low(o.logits);
    // <env>_DUMP_LOGITS=<file> dumps prefill logits.
    if (const char* fp = env(m.spec, "_DUMP_LOGITS")) {
        if (FILE* f = std::fopen(fp, "wb")) {
            std::fwrite(o.logits.data(), sizeof(float), o.logits.size(), f);
            std::fclose(f);
        }
    }
    return true;
}

bool greedy_generate(const QwenDecodeCtx& m, const InputsEmbeds& i,
                     const GenerateParams& op, GenerateResult& o, std::string& e) {
    if (i.n_tokens + op.max_new_tokens > op.max_cache_len) {
        e = std::string(m.spec.label) + " generation exceeds cache";
        return false;
    }
    const bool dbg = debug_probe_active(m.spec);
    const bool timing = env(m.spec, "_TIMING") != nullptr;
    if (!dbg) {
        // Default hot path: whole-model prefill + whole-model decode graphs.
        LlmState state;
        double t_pf = 0.0;
        if (timing) {
            auto t0 = std::chrono::steady_clock::now();
            bool ok = forward_prefill(m, i.data, i.n_tokens, state, o.prefill_logits, e);
            auto t1 = std::chrono::steady_clock::now();
            t_pf = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (!ok) return false;
        } else {
            if (!forward_prefill(m, i.data, i.n_tokens, state, o.prefill_logits, e))
                return false;
        }
        int32_t prev = argmax_low(o.prefill_logits);
        o.ids.push_back(prev);
        const bool use_kstep = global_backend().is_gpu() &&
                               !env(m.spec, "_NOKSTEP");
        double dec_sum = 0.0; int dec_n = 0;
        if (use_kstep) {
            const int K = kstep_K(m.spec);
            for (int n = 1; n < op.max_new_tokens;) {
                bool hit = false;
                size_t before = o.ids.size();
                // Cap this block's step count to the remaining token budget.
                // The outer cache check only bounds the LAST NEEDED token
                // position (n_tokens + max_new_tokens - 1 <= max_cache - 1);
                // without this cap, the final K-step block still runs all K
                // decode steps and its wasted tail steps (those past
                // max_new_tokens) write KV slots and read RoPE rows at indices
                // past + j >= max_cache -> out-of-bounds device access
                // (set_rows KV write / get_rows RoPE read). With the cap, every
                // step is a REAL consumed step, so the bound on needed positions
                // also bounds every device index. Proof sketch: at block start,
                // past == state.length == n_tokens + (ids.size() - 1) (prefill
                // sets length=n_tokens and each consumed step advances both
                // length and ids by 1). With Kk = min(K, remaining) and remaining
                // = max_new_tokens - ids.size(), the block's last position is
                // past + Kk - 1 <= n_tokens + max_new_tokens - 2 <= max_cache - 2
                // < max_cache for both the get_rows RoPE source and the set_rows
                // KV target. Steady-state blocks keep Kk == K (byte-identical to
                // before); only the tail block shrinks, and a smaller-K graph is
                // fetched/built from the existing per-K cache. Kk >= 1 here
                // because the loop only runs while ids.size() < max_new_tokens.
                const int remaining = op.max_new_tokens - (int)o.ids.size();
                const int Kk = std::min(K, remaining);
                double s0 = timing ? (double)std::chrono::steady_clock::now().time_since_epoch().count() : 0.0;
                if (!run_kstep(m, prev, state.length, Kk, op.eos_token_id,
                               op.max_new_tokens, o.ids, hit, e))
                    return false;
                if (timing) {
                    double s1 = (double)std::chrono::steady_clock::now().time_since_epoch().count();
                    int produced = (int)(o.ids.size() - before);
                    dec_sum += (s1 - s0) * 1e-6; dec_n += produced;
                }
                if (hit) { o.hit_eos = true; break; }
                n = (int)o.ids.size();
            }
        } else {
            for (int n = 1; n < op.max_new_tokens; ++n) {
                std::vector<float> dl;
                double s0 = timing ? (double)std::chrono::steady_clock::now().time_since_epoch().count() : 0.0;
                if (!forward_decode(m, prev, state.length, state, dl, e)) return false;
                prev = argmax_low(dl);
                o.ids.push_back(prev);
                if (timing) {
                    double s1 = (double)std::chrono::steady_clock::now().time_since_epoch().count();
                    dec_sum += (s1 - s0) * 1e-6; ++dec_n;
                }
                if (prev == op.eos_token_id) { o.hit_eos = true; break; }
            }
        }
        if (timing) {
            std::fprintf(stderr, "%s_TIMING prefill=%.1fms decode_tokens=%d avg=%.2fms/tok (kstep=%d)\n",
                         m.spec.label, t_pf, dec_n, dec_n ? dec_sum / dec_n : 0.0, (int)use_kstep);
        }
    } else {
        // Debug/probe path: per-layer run_graph loop (layer_legacy) so the
        // <env>_L0_STAGE / <env>_DUMP_LAYERS divergence-local probes keep
        // working unchanged.
        PrefillResult p;
        if (!llm_prefill(m, i, op.max_cache_len, p, e)) return false;
        o.prefill_logits = p.logits;
        o.ids.push_back(p.first_token);
        int32_t prev = p.first_token;
        for (int n = 1; n < op.max_new_tokens; ++n) {
            InputsEmbeds one;
            one.n_tokens = 1;
            one.width = m.dims.hidden;
            std::vector<int32_t> id = {prev};
            bool ok = run_graph([&](ggml_context* c) -> ggml_tensor* {
                int64_t ne[1] = {1};
                ggml_tensor* t = graph_input_tensor(c, GGML_TYPE_I32, 1, ne,
                                                    id.data(), sizeof(int32_t));
                return ff(c, ggml_get_rows(c, clone_weight(c, m.loader, "llm.embed.weight"), t));
            }, one.data);
            if (!ok) { e = "decode embedding lookup failed"; return false; }

            std::vector<float> logits;
            if (!forward_legacy(m, one.data, 1, p.state, logits, e)) return false;
            prev = argmax_low(logits);
            o.ids.push_back(prev);
            if (prev == op.eos_token_id) { o.hit_eos = true; break; }
        }
    }
    // <env>_DUMP_IDS=<file> dumps generated ids (i32).
    if (const char* fp = env(m.spec, "_DUMP_IDS")) {
        if (FILE* f = std::fopen(fp, "wb")) {
            std::fwrite(o.ids.data(), sizeof(int32_t), o.ids.size(), f);
            std::fclose(f);
        }
    }
    return true;
}

} // namespace starling::ggml::lib
