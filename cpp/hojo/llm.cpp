// llm.cpp — Hojo Qwen3-4B decoder (beam-4) on the Starling ggml runtime.
//
// Ported from higgs/llm.cpp (the freshest Qwen3-with-qk_norm). Differences from
// higgs: 36 layers / hidden 2560 / GQA 32/8 / intermediate 9728 / rope_theta
// 5e6 / vocab 151670 (config-driven, no code change), and the decode is BEAM-4
// (not greedy). The Qwen3 op order/dtype discipline (bf/ff casts, f32 RMSNorm,
// rotate-half RoPE in f32, q_norm/k_norm per head) is byte-identical to higgs.
//
// KV cache: the prompt KV is computed ONCE (prefill writes device-resident
// cache slots [0, prefix)) and each beam step runs an exact-width S=1 forward
// over the beam's cached rows [0, past) plus the one new token. Hojo keeps its
// own numeric disciplines: RoPE cos/sin stay HOST f32 tables fed as f32 graph
// inputs, the causal mask stays f32, activations stay bf16 between ops
// (bf/ff/addb/mulb). Only WHERE k/v come from changed (cache tensors instead
// of recompute); the cached values are the same bf16 values a fresh full
// forward produces (bf16->bf16 cpy is bit-exact).
//
// Beam search: HF-compatible (num_beams, repetition_penalty, length_penalty,
// do_sample=False). The golden's gen_ids is the winning beam; match it.
#include "llm.hpp"
#include "runtime/backend.hpp"
#include "runtime/graph.hpp"
#include "runtime/graph_builder.hpp"
#include "runtime/lru_cache.hpp"
#include "lib/graph_helpers.hpp"
#include "lib/device_cache.hpp"
#include "lib/mask_rope.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace starling::ggml::hojo {
namespace {

// Small tensor helpers shared with the other engines (lib/graph_helpers.hpp).
using lib::bf;
using lib::ff;
using lib::build_causal_mask;
using lib::build_rope_tables;
using lib::RopeTables;
using lib::addb;
using lib::mulb;
using lib::tobf;
using lib::DeviceCache;
ggml_tensor* wb(ggml_context* c, const HojoModel& m, const std::string& n) {
    return lib::wb(c, m.loader, n);
}
// nn.Linear (Qwen3 attention has NO q/k/v/o bias).
ggml_tensor* lin(ggml_context* c, const HojoModel& m, ggml_tensor* x, const std::string& n) {
    return lib::lin(c, m.loader, x, n);
}
// RMSNorm in f32, then scale by the (bf16) weight.
ggml_tensor* rms(ggml_context* c, const HojoModel& m, ggml_tensor* x, const std::string& n,
                 float eps) {
    return lib::rms(c, m.loader, x, n, eps);
}

// ---------------------------------------------------------------------------
// Device-resident per-beam KV caches.
//
// Constraint (beam search + KV cache): the per-step top-2B candidate ranking
// can REORDER beams — a surviving beam's ancestor tokens may come from a
// different beam — so the cached rows for a beam's ancestry must follow the
// beam. Design: ONE full DeviceCache per beam (B <= 4 -> 4 x 576 MB bf16 at
// max_cache 4096, affordable next to the 12 GB weights), with beam ownership
// of the buffers expressed as a pointer permutation per step:
//   - a parent beam's buffer follows its FIRST surviving child (no copy);
//   - every further child of the same parent gets an unclaimed buffer plus a
//     row copy of slots [0, past) from the parent, executed inside that
//     child's next decode graph (the copy result IS the k/v the attention
//     consumes, which also orders the copy before the read).
// Copy sources are always claimed (moved) buffers and copy targets are always
// unclaimed ones, so no copy can clobber another copy's source. All beams at
// one step share the same `past`, so a sibling decode writing slot `past`
// into a moved buffer never intersects the [0, past) copy range. A single
// shared store with per-beam row indirection would halve the VRAM but needs
// a gather in attention; rejected as harder to keep byte-exact.
//
// Stale rows are never read (exact-width attention covers [0, past+1) only),
// so the caches are NOT zeroed between utterances. DeviceCache's bf16 RoPE
// tables are unused: hojo feeds f32 host RoPE rows as graph inputs.
// ---------------------------------------------------------------------------
std::vector<std::unique_ptr<DeviceCache>> g_beam_caches;
std::once_flag g_beam_caches_once;
void register_beam_cache_clearer_once() {
    std::call_once(g_beam_caches_once, [] {
        register_decode_cache_clearer([] { g_beam_caches.clear(); });
    });
}
// Lazily allocate (or return) the first `count` per-beam caches. Single-config
// per process (the shared global-backend assumption); a mismatched config is
// an error, never a silent reuse.
std::vector<DeviceCache*> get_beam_caches(const HojoModel& m, int count, std::string& e) {
    register_beam_cache_clearer_once();
    const auto& lc = m.config.llm;
    if (!g_beam_caches.empty()) {
        const DeviceCache& c = *g_beam_caches[0];
        if (c.n_layers != (int) lc.n_layers || c.D != (int) lc.head_dim ||
            c.KV != (int) lc.n_kv_heads || c.max_cache != (int) lc.max_cache) {
            e = "Hojo KV cache dims changed within process";
            return {};
        }
    }
    while ((int) g_beam_caches.size() < count) {
        auto dc = std::unique_ptr<DeviceCache>(new DeviceCache());
        if (!dc->init((int) lc.n_layers, (int) lc.head_dim, (int) lc.n_kv_heads,
                      (int) lc.max_cache, (float) lc.rope_theta,
                      global_backend().handle(), e)) {
            g_beam_caches.clear();
            return {};
        }
        g_beam_caches.push_back(std::move(dc));
    }
    std::vector<DeviceCache*> out;
    for (int b = 0; b < count; ++b) out.push_back(g_beam_caches[(size_t) b].get());
    return out;
}

// Append one Qwen3 layer over x_in [hidden, S]. cs/sn are [D, S] f32 graph
// inputs (RoPE rows for positions [0,S)); mask is f32 [K, S] with
// K = past + S. kv_mode 0 = prefill (write slots [0,S), attend over S);
// kv_mode 1 = decode exact-width (write slot `past`, attend over [0, past+S)).
// kprev/vprev (mode 1 only) are [D, past, KV] bf16 tensors holding the beam's
// cached rows — either a plain view of the destination cache or the result of
// a re-parenting cpy (see decode_step). Exact-width decode exists because
// full-capacity attention changes the softmax reduction width and is not
// byte-stable.
ggml_tensor* append_layer_cached(ggml_context* c, const HojoModel& m, int li,
                                 ggml_tensor* x_in, int64_t S, int64_t past,
                                 ggml_tensor* cache_k, ggml_tensor* cache_v,
                                 ggml_tensor* cs, ggml_tensor* sn,
                                 ggml_tensor* mask, int kv_mode,
                                 ggml_tensor* kprev, ggml_tensor* vprev) {
    const auto& lc = m.config.llm;
    const int D = lc.head_dim, H = lc.n_heads, KV = lc.n_kv_heads;
    const std::string p = "llm.blk." + std::to_string(li) + ".";
    ggml_tensor* r = x_in;
    ggml_tensor* n = rms(c, m, x_in, p + "attn_norm.weight", (float) lc.rms_norm_eps);
    ggml_tensor* q = lin(c, m, n, p + "attn.q.weight");
    ggml_tensor* k = lin(c, m, n, p + "attn.k.weight");
    ggml_tensor* v = lin(c, m, n, p + "attn.v.weight");
    q = ggml_reshape_3d(c, q, D, H, S);
    k = ggml_reshape_3d(c, k, D, KV, S);
    v = ggml_reshape_3d(c, v, D, KV, S);
    // qk_norm: RMSNorm per head over head_dim.
    q = rms(c, m, q, p + "attn.q_norm.weight", (float) lc.rms_norm_eps);
    k = rms(c, m, k, p + "attn.k_norm.weight", (float) lc.rms_norm_eps);
    q = ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));  // [D, S, H]
    k = ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3));  // [D, S, KV]
    v = ggml_cont(c, ggml_permute(c, v, 0, 2, 1, 3));  // [D, S, KV]
    // RoPE (rotate-half, f32 math on f32 cs/sn — the hojo discipline).
    auto rope = [&](ggml_tensor* z, int heads) {
        ggml_tensor* lo = ggml_view_3d(c, z, D / 2, S, heads, z->nb[1], z->nb[2], 0);
        ggml_tensor* hi = ggml_view_3d(c, z, D / 2, S, heads, z->nb[1], z->nb[2],
                                       (size_t)(D / 2) * z->nb[0]);
        ggml_tensor* rot = ggml_concat(c, ggml_scale(c, ff(c, hi), -1.0f), ff(c, lo), 0);
        rot = bf(c, rot);
        return addb(c, mulb(c, z, cs), mulb(c, rot, sn));
    };
    q = rope(q, H);
    k = rope(k, KV);
    // GQA: do NOT explicitly repeat k/v (no bf16 ggml_repeat CUDA kernel);
    // ggml_mul_mat broadcasts KV->H heads natively when H%KV==0.
    ggml_tensor* kall;
    ggml_tensor* vall;
    if (kv_mode == 0) {            // prefill: cpy this step's k/v into slots [0,S)
        ggml_tensor* kview = ggml_view_3d(c, cache_k, D, S, KV,
                                          cache_k->nb[1], cache_k->nb[2], 0);
        ggml_tensor* vview = ggml_view_3d(c, cache_v, D, S, KV,
                                          cache_v->nb[1], cache_v->nb[2], 0);
        kall = ggml_cpy(c, k, kview);  // [D, S, KV]
        vall = ggml_cpy(c, v, vview);
    } else {                       // decode: cpy k/v into slot `past`, concat cached rows
        ggml_tensor* kslot = ggml_view_3d(c, cache_k, D, 1, KV,
                                          cache_k->nb[1], cache_k->nb[2],
                                          (size_t) past * cache_k->nb[1]);
        ggml_tensor* vslot = ggml_view_3d(c, cache_v, D, 1, KV,
                                          cache_v->nb[1], cache_v->nb[2],
                                          (size_t) past * cache_v->nb[1]);
        ggml_tensor* knew = ggml_cpy(c, k, kslot);
        ggml_tensor* vnew = ggml_cpy(c, v, vslot);
        // CUDA concat is F32-only; the f32 round-trip is exact for bf16 values.
        // Consuming the cpy results forces the cache writes to execute first.
        kall = bf(c, ggml_concat(c, ff(c, kprev), ff(c, knew), 1));
        vall = bf(c, ggml_concat(c, ff(c, vprev), ff(c, vnew), 1));
    }
    const float scale = 1.0f / std::sqrt((float) D);
    ggml_tensor* sc = ggml_mul_mat(c, kall, q);                        // [K, S, H]
    sc = bf(c, ggml_scale(c, ff(c, sc), scale));
    ggml_tensor* pr = bf(c, ggml_soft_max_ext(c, ff(c, sc), ff(c, mask), 1.0f, 0.0f));
    ggml_tensor* vt = ggml_cont(c, ggml_permute(c, vall, 1, 0, 2, 3));  // [K, D, KV]
    ggml_tensor* co = ggml_mul_mat(c, vt, pr);                         // [D, S, H]
    ggml_tensor* joined = ggml_reshape_2d(c, ggml_cont(c, ggml_permute(c, co, 0, 2, 1, 3)),
                                          (int64_t) D * H, S);         // [hidden, S]
    joined = bf(c, joined);
    ggml_tensor* a = lin(c, m, joined, p + "attn.o.weight");
    ggml_tensor* x = addb(c, r, a);
    r = x;
    n = rms(c, m, x, p + "ffn_norm.weight", (float) lc.rms_norm_eps);
    ggml_tensor* g = lin(c, m, n, p + "ffn.gate.weight");
    ggml_tensor* u = lin(c, m, n, p + "ffn.up.weight");
    ggml_tensor* si = bf(c, ggml_silu(c, ff(c, g)));
    ggml_tensor* z = mulb(c, si, u);
    ggml_tensor* dn = lin(c, m, z, p + "ffn.down.weight");
    x = addb(c, r, dn);
    return x;  // [hidden, S] bf16
}

// ---------------------------------------------------------------------------
// Prefill: compute the prompt KV once into beam cache 0, return the
// last-position logits (f32, vocab). `prefix` is the f32 inputs_embeds
// [hidden, S].
// ---------------------------------------------------------------------------

// Shared prefill graph body (leaves are the graph inputs x/cs/sn/mt).
ggml_tensor* build_prefill_graph(ggml_context* c, const HojoModel& m, DeviceCache* dc,
                                 int64_t S, ggml_tensor* x, ggml_tensor* cs,
                                 ggml_tensor* sn, ggml_tensor* mt) {
    const auto& lc = m.config.llm;
    for (uint32_t li = 0; li < lc.n_layers; ++li)
        x = append_layer_cached(c, m, (int) li, x, S, 0, dc->k[li], dc->v[li],
                                cs, sn, mt, /*kv_mode=*/0, nullptr, nullptr);
    // Final norm + lm_head on the LAST token only.
    ggml_tensor* last = ggml_view_2d(c, x, lc.hidden, 1, x->nb[1],
                                     (size_t)(S - 1) * x->nb[1]);
    ggml_tensor* nrm = rms(c, m, last, "llm.final_norm.weight", (float) lc.rms_norm_eps);
    return ff(c, ggml_mul_mat(c, wb(c, m, "llm.lm_head.weight"), nrm));
}

// GPU: the prefill graph is captured into a per-S ReplayGraph (prompt length S
// fully determines every shape). Only inputs_embeds varies per utterance; the
// f32 RoPE rows [0,S) and the causal mask are constant for S. Bounded LRU so
// each distinct prompt length does not pin its own captured graph + private
// gallocr until exit. Cleared via register_decode_cache_clearer.
struct PrefillReplayEntry {
    int64_t S = 0;
    GraphInputPool pool;
    std::unique_ptr<ReplayGraph> graph;
    ggml_bf16_t* xb_buf = nullptr;  // stable pool backing for input #0 (varying)
    float* cs_buf = nullptr;        // constant for S
    float* sn_buf = nullptr;
};
namespace {
struct PrefillCache {
    LruCache<int64_t, PrefillReplayEntry> by_S;
    explicit PrefillCache(size_t cap) : by_S(cap) {}
};
std::unique_ptr<PrefillCache> g_prefill_cache;
std::once_flag g_prefill_once;
void register_prefill_clearer_once() {
    std::call_once(g_prefill_once, [] {
        register_decode_cache_clearer([] { g_prefill_cache.reset(); });
    });
}
} // namespace

PrefillReplayEntry* get_or_build_prefill(const HojoModel& m, DeviceCache* dc, int64_t S,
                                         const RopeTables& rope, std::string& e) {
    register_prefill_clearer_once();
    if (!g_prefill_cache)
        g_prefill_cache = std::make_unique<PrefillCache>(replay_cache_size());
    const auto& lc = m.config.llm;
    // get_or_init places the entry (stable address) first, then builds: the
    // ReplayGraph build lambda captures the stable pool pointers.
    return g_prefill_cache->by_S.get_or_init(S,
        [&](PrefillReplayEntry& entry) -> PrefillReplayEntry& {
            entry.S = S;
            entry.xb_buf = reinterpret_cast<ggml_bf16_t*>(
                entry.pool.alloc_bytes((size_t) S * lc.hidden * sizeof(ggml_bf16_t)));
            // cs/sn values depend only on (D, rope_theta, position), so an entry
            // reused across utterances never goes stale.
            entry.cs_buf = entry.pool.alloc_f32((size_t) rope.D * S);
            entry.sn_buf = entry.pool.alloc_f32((size_t) rope.D * S);
            for (int64_t t = 0; t < S; ++t)
                for (int d = 0; d < rope.D; ++d) {
                    entry.cs_buf[(size_t) t * rope.D + d] = rope.cos[(size_t) t * rope.D + d];
                    entry.sn_buf[(size_t) t * rope.D + d] = rope.sin[(size_t) t * rope.D + d];
                }
            std::vector<float> mask = build_causal_mask(S, 0);
            float* mask_buf = entry.pool.alloc_f32((size_t) S * S);
            std::memcpy(mask_buf, mask.data(), (size_t) S * S * sizeof(float));
            entry.graph = std::make_unique<ReplayGraph>(global_backend(),
                [&](ggml_context* c) -> ggml_tensor* {
                    int64_t xne[2] = {lc.hidden, S};
                    ggml_tensor* x = graph_input_tensor(c, GGML_TYPE_BF16, 2, xne,
                        entry.xb_buf, (size_t) S * lc.hidden * sizeof(ggml_bf16_t));
                    int64_t rne[2] = {rope.D, S};
                    ggml_tensor* cs = graph_input_tensor(c, GGML_TYPE_F32, 2, rne,
                        entry.cs_buf, (size_t) rope.D * S * sizeof(float));
                    ggml_tensor* sn = graph_input_tensor(c, GGML_TYPE_F32, 2, rne,
                        entry.sn_buf, (size_t) rope.D * S * sizeof(float));
                    int64_t mne[2] = {S, S};
                    ggml_tensor* mt = graph_input_tensor(c, GGML_TYPE_F32, 2, mne,
                        mask_buf, (size_t) S * S * sizeof(float));
                    return build_prefill_graph(c, m, dc, S, x, cs, sn, mt);
                });
            return entry;
        });
}

bool prefill_cached(const HojoModel& m, const std::vector<float>& prefix_embeds,
                    int64_t S, const RopeTables& rope, std::vector<float>& logits,
                    std::string& e) {
    const auto& lc = m.config.llm;
    ensure_weights_realized(m.loader);
    std::vector<DeviceCache*> caches = get_beam_caches(m, 1, e);
    if (caches.empty()) return false;
    DeviceCache* dc = caches[0];
    if (!global_backend().is_gpu()) {
        // CPU keeps the one-shot build (capture is GPU-only).
        std::vector<ggml_bf16_t> xb((size_t) S * lc.hidden);
        for (int64_t t = 0; t < S; ++t)
            for (int64_t d = 0; d < (int64_t) lc.hidden; ++d)
                xb[(size_t) t * lc.hidden + d] =
                    ggml_fp32_to_bf16(prefix_embeds[(size_t) t * lc.hidden + d]);
        std::vector<float> cs_host((size_t) rope.D * S), sn_host((size_t) rope.D * S);
        for (int64_t t = 0; t < S; ++t)
            for (int d = 0; d < rope.D; ++d) {
                cs_host[(size_t) t * rope.D + d] = rope.cos[(size_t) t * rope.D + d];
                sn_host[(size_t) t * rope.D + d] = rope.sin[(size_t) t * rope.D + d];
            }
        std::vector<float> mask = build_causal_mask(S, 0);
        bool ok = run_graph([&](ggml_context* c) -> ggml_tensor* {
            int64_t xne[2] = {lc.hidden, S};
            ggml_tensor* x = graph_input_tensor(c, GGML_TYPE_BF16, 2, xne,
                                                xb.data(), xb.size() * sizeof(xb[0]));
            int64_t rne[2] = {rope.D, S};
            ggml_tensor* cs = graph_input_tensor(c, GGML_TYPE_F32, 2, rne,
                                                 cs_host.data(), cs_host.size() * sizeof(float));
            ggml_tensor* sn = graph_input_tensor(c, GGML_TYPE_F32, 2, rne,
                                                 sn_host.data(), sn_host.size() * sizeof(float));
            int64_t mne[2] = {S, S};
            ggml_tensor* mt = graph_input_tensor(c, GGML_TYPE_F32, 2, mne,
                                                 mask.data(), mask.size() * sizeof(float));
            return build_prefill_graph(c, m, dc, S, x, cs, sn, mt);
        }, logits);
        if (!ok) { e = "Hojo prefill graph failed"; return false; }
        return true;
    }
    PrefillReplayEntry* pe = get_or_build_prefill(m, dc, S, rope, e);
    if (!pe || !pe->graph) { e = "Hojo prefill graph build failed"; return false; }
    const size_t nx = (size_t) S * lc.hidden;
    for (size_t z = 0; z < nx; ++z)
        pe->xb_buf[z] = ggml_fp32_to_bf16(prefix_embeds[z]);
    for (size_t k = 0; k < pe->graph->n_inputs(); ++k)
        pe->graph->set_input(k, pe->graph->input_host(k), pe->graph->input_nbytes(k));
    if (!pe->graph->compute(logits)) { e = "Hojo prefill replay failed"; return false; }
    return true;
}

// ---------------------------------------------------------------------------
// Decode step (one beam, S=1): feed `tok` at slot `past`, attend exactly
// [0, past+1), return the last-position logits. If `src` != nullptr the beam
// was re-parented: the graph first cpys slots [0, past) from `src` into `dst`
// and the attention consumes the cpy results, so the copy executes first.
// Each beam gets its own step graph: bundling all beams into one graph does
// not reduce step time (per-beam kernel launches + host ranking dominate,
// not graph setup).
// ---------------------------------------------------------------------------

bool decode_step(const HojoModel& m, DeviceCache* dst, DeviceCache* src,
                 int32_t tok, int64_t past, const RopeTables& rope,
                 std::vector<float>& logits, std::string& e) {
    const auto& lc = m.config.llm;
    const int64_t S = 1;
    const int64_t K = past + 1;
    // Both the cache slot write and the RoPE row read are indexed by `past`;
    // an out-of-bounds write into ggml buffer padding does NOT fault, so gate
    // it explicitly (beam_generate's n_tokens + max_new_tokens <= max_cache_len
    // check keeps every real step below this bound).
    if (past < 0 || past >= (int64_t) lc.max_cache) {
        e = "Hojo decode position out of bounds (past=" + std::to_string(past) +
            ", max_cache=" + std::to_string(lc.max_cache) + ")";
        return false;
    }
    std::vector<float> cs_host((size_t) rope.D), sn_host((size_t) rope.D);
    for (int d = 0; d < rope.D; ++d) {
        cs_host[(size_t) d] = rope.cos[(size_t) past * rope.D + d];
        sn_host[(size_t) d] = rope.sin[(size_t) past * rope.D + d];
    }
    std::vector<float> mask((size_t) K, 0.0f);  // decode: all keys valid
    int32_t tok_in = tok;
    bool ok = run_graph([&](ggml_context* c) -> ggml_tensor* {
        int64_t one[1] = {1};
        ggml_tensor* id_t = graph_input_tensor(c, GGML_TYPE_I32, 1, one,
                                               &tok_in, sizeof(int32_t));
        // Embed lookup with the same f32->bf16 rounding the host path applies
        // when building inputs_embeds.
        ggml_tensor* x = bf(c, ff(c, ggml_get_rows(c,
            clone_weight(c, m.loader, "llm.embed.weight"), id_t)));
        int64_t rne[2] = {rope.D, 1};
        ggml_tensor* cs = graph_input_tensor(c, GGML_TYPE_F32, 2, rne,
                                             cs_host.data(), cs_host.size() * sizeof(float));
        ggml_tensor* sn = graph_input_tensor(c, GGML_TYPE_F32, 2, rne,
                                             sn_host.data(), sn_host.size() * sizeof(float));
        int64_t mne[2] = {K, 1};
        ggml_tensor* mt = graph_input_tensor(c, GGML_TYPE_F32, 2, mne,
                                             mask.data(), mask.size() * sizeof(float));
        for (uint32_t li = 0; li < lc.n_layers; ++li) {
            ggml_tensor* kprev;
            ggml_tensor* vprev;
            if (src) {
                // Re-parenting copy: rows [0, past) of the parent's cache. The
                // cpy result (dst view) is what the attention consumes below.
                kprev = ggml_cpy(c,
                    ggml_view_3d(c, src->k[li], lc.head_dim, past, lc.n_kv_heads,
                                 src->k[li]->nb[1], src->k[li]->nb[2], 0),
                    ggml_view_3d(c, dst->k[li], lc.head_dim, past, lc.n_kv_heads,
                                 dst->k[li]->nb[1], dst->k[li]->nb[2], 0));
                vprev = ggml_cpy(c,
                    ggml_view_3d(c, src->v[li], lc.head_dim, past, lc.n_kv_heads,
                                 src->v[li]->nb[1], src->v[li]->nb[2], 0),
                    ggml_view_3d(c, dst->v[li], lc.head_dim, past, lc.n_kv_heads,
                                 dst->v[li]->nb[1], dst->v[li]->nb[2], 0));
            } else {
                kprev = ggml_view_3d(c, dst->k[li], lc.head_dim, past, lc.n_kv_heads,
                                     dst->k[li]->nb[1], dst->k[li]->nb[2], 0);
                vprev = ggml_view_3d(c, dst->v[li], lc.head_dim, past, lc.n_kv_heads,
                                     dst->v[li]->nb[1], dst->v[li]->nb[2], 0);
            }
            x = append_layer_cached(c, m, (int) li, x, S, past, dst->k[li], dst->v[li],
                                    cs, sn, mt, /*kv_mode=*/1, kprev, vprev);
        }
        ggml_tensor* nrm = rms(c, m, x, "llm.final_norm.weight", (float) lc.rms_norm_eps);
        return ff(c, ggml_mul_mat(c, wb(c, m, "llm.lm_head.weight"), nrm));
    }, logits);
    if (!ok) { e = "Hojo decode graph failed"; return false; }
    return true;
}

// Append one Qwen3 layer over x_in [hidden, S] (inputs_embeds for this forward).
// The KV is NOT cached (each greedy step is a fresh forward). cs/sn are [D, K] f32
// graph inputs (RoPE rows for positions [0,K)); mask is f32 [K, S].
ggml_tensor* append_layer(ggml_context* c, const HojoModel& m, int li,
                          ggml_tensor* x_in, int64_t S,
                          ggml_tensor* cs, ggml_tensor* sn,
                          ggml_tensor* mask) {
    const auto& lc = m.config.llm;
    const int D = lc.head_dim, H = lc.n_heads, KV = lc.n_kv_heads;
    const std::string p = "llm.blk." + std::to_string(li) + ".";
    ggml_tensor* r = x_in;
    ggml_tensor* n = rms(c, m, x_in, p + "attn_norm.weight", (float) lc.rms_norm_eps);
    ggml_tensor* q = lin(c, m, n, p + "attn.q.weight");
    ggml_tensor* k = lin(c, m, n, p + "attn.k.weight");
    ggml_tensor* v = lin(c, m, n, p + "attn.v.weight");
    q = ggml_reshape_3d(c, q, D, H, S);
    k = ggml_reshape_3d(c, k, D, KV, S);
    v = ggml_reshape_3d(c, v, D, KV, S);
    // qk_norm: RMSNorm per head over head_dim.
    q = rms(c, m, q, p + "attn.q_norm.weight", (float) lc.rms_norm_eps);
    k = rms(c, m, k, p + "attn.k_norm.weight", (float) lc.rms_norm_eps);
    q = ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));  // [D, S, H]
    k = ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3));  // [D, S, KV]
    v = ggml_cont(c, ggml_permute(c, v, 0, 2, 1, 3));  // [D, S, KV]
    // RoPE (rotate-half, f32).
    auto rope = [&](ggml_tensor* z, int heads) {
        ggml_tensor* lo = ggml_view_3d(c, z, D / 2, S, heads, z->nb[1], z->nb[2], 0);
        ggml_tensor* hi = ggml_view_3d(c, z, D / 2, S, heads, z->nb[1], z->nb[2],
                                       (size_t)(D / 2) * z->nb[0]);
        ggml_tensor* rot = ggml_concat(c, ggml_scale(c, ff(c, hi), -1.0f), ff(c, lo), 0);
        rot = bf(c, rot);
        return addb(c, mulb(c, z, cs), mulb(c, rot, sn));
    };
    q = rope(q, H);
    k = rope(k, KV);
    // GQA: do NOT explicitly repeat k/v. ggml_repeat has NO CUDA kernel for
    // BF16 (only F32/F16), and using it forces the scheduler onto CPU and
    // mis-sizes a pipeline buffer (the 232 GB crash). Instead, rely on
    // ggml_mul_mat's NATIVE GQA broadcast: pass k=[D,S,KV] / v=[D,S,KV] and
    // q=[D,S,H] directly; mul_mat broadcasts KV->H heads when H%KV==0.
    // Identical to higgs/llm.cpp (which is byte-exact vs the golden).
    const float scale = 1.0f / std::sqrt((float) D);
    ggml_tensor* sc = ggml_mul_mat(c, k, q);                 // [S, S, H]
    sc = bf(c, ggml_scale(c, ff(c, sc), scale));
    ggml_tensor* pr = bf(c, ggml_soft_max_ext(c, ff(c, sc), ff(c, mask), 1.0f, 0.0f));
    ggml_tensor* vt = ggml_cont(c, ggml_permute(c, v, 1, 0, 2, 3));  // [S, D, H]
    ggml_tensor* co = ggml_mul_mat(c, vt, pr);                       // [D, S, H]
    ggml_tensor* joined = ggml_reshape_2d(c, ggml_cont(c, ggml_permute(c, co, 0, 2, 1, 3)),
                                          (int64_t) D * H, S);       // [hidden, S]
    joined = bf(c, joined);
    ggml_tensor* a = lin(c, m, joined, p + "attn.o.weight");
    ggml_tensor* x = addb(c, r, a);
    r = x;
    n = rms(c, m, x, p + "ffn_norm.weight", (float) lc.rms_norm_eps);
    ggml_tensor* g = lin(c, m, n, p + "ffn.gate.weight");
    ggml_tensor* u = lin(c, m, n, p + "ffn.up.weight");
    ggml_tensor* si = bf(c, ggml_silu(c, ff(c, g)));
    ggml_tensor* z = mulb(c, si, u);
    ggml_tensor* dn = lin(c, m, z, p + "ffn.down.weight");
    x = addb(c, r, dn);
    return x;  // [hidden, S] bf16
}

// Fresh full forward over [prefix + extra_tokens] (NO cache). Retained for the
// greedy/diagnostic path only; beam search uses prefill_cached + decode_step.
bool forward_logits(const HojoModel& m, const std::vector<float>& prefix_embeds,
                    int64_t prefix_len, const std::vector<int32_t>& extra_tokens,
                    const RopeTables& rope, std::vector<float>& logits,
                    std::string& e) {
    const auto& lc = m.config.llm;
    const int64_t hidden = lc.hidden;
    const int64_t n_extra = (int64_t) extra_tokens.size();
    const int64_t S = prefix_len + n_extra;
    if (S <= 0) { e = "Hojo forward: empty sequence"; return false; }
    // Build the full inputs_embeds: prefix (bf16) + embed(extra_tokens).
    std::vector<ggml_bf16_t> xb((size_t) S * hidden);
    for (int64_t t = 0; t < prefix_len; ++t)
        for (int64_t d = 0; d < hidden; ++d)
            xb[(size_t) t * hidden + d] = ggml_fp32_to_bf16(prefix_embeds[(size_t) t * hidden + d]);
    // Look up extra token embeds.
    std::vector<ggml_bf16_t> extra_emb;
    if (n_extra > 0) {
        std::vector<float> extra_f;
        bool ok = run_graph([&](ggml_context* c) -> ggml_tensor* {
            int64_t ne[1] = {n_extra};
            ggml_tensor* it = graph_input_tensor(c, GGML_TYPE_I32, 1, ne,
                extra_tokens.data(), extra_tokens.size() * sizeof(int32_t));
            return ggml_cast(c,
                ggml_get_rows(c, clone_weight(c, m.loader, "llm.embed.weight"), it),
                GGML_TYPE_F32);
        }, extra_f);
        if (!ok) { e = "Hojo embed lookup failed"; return false; }
        extra_emb.resize(extra_f.size());
        for (size_t i = 0; i < extra_f.size(); ++i) extra_emb[i] = ggml_fp32_to_bf16(extra_f[i]);
        for (int64_t t = 0; t < n_extra; ++t)
            for (int64_t d = 0; d < hidden; ++d)
                xb[(size_t)(prefix_len + t) * hidden + d] =
                    extra_emb[(size_t) t * hidden + d];
    }
    std::vector<float> mask = build_causal_mask(S, 0);  // [S, S], past=0
    // RoPE rows for positions [0, S).
    std::vector<float> cs_host((size_t) rope.D * S), sn_host((size_t) rope.D * S);
    for (int64_t t = 0; t < S; ++t)
        for (int d = 0; d < rope.D; ++d) {
            cs_host[(size_t) t * rope.D + d] = rope.cos[(size_t) t * rope.D + d];
            sn_host[(size_t) t * rope.D + d] = rope.sin[(size_t) t * rope.D + d];
        }
    bool ok = run_graph([&](ggml_context* c) -> ggml_tensor* {
        int64_t xne[2] = {hidden, S};
        ggml_tensor* x = graph_input_tensor(c, GGML_TYPE_BF16, 2, xne,
            xb.data(), xb.size() * sizeof(ggml_bf16_t));
        int64_t rne[2] = {rope.D, S};
        ggml_tensor* cs = graph_input_tensor(c, GGML_TYPE_F32, 2, rne,
            cs_host.data(), cs_host.size() * sizeof(float));
        ggml_tensor* sn = graph_input_tensor(c, GGML_TYPE_F32, 2, rne,
            sn_host.data(), sn_host.size() * sizeof(float));
        int64_t mne[2] = {S, S};
        ggml_tensor* mt = graph_input_tensor(c, GGML_TYPE_F32, 2, mne,
            mask.data(), mask.size() * sizeof(float));
        for (uint32_t li = 0; li < lc.n_layers; ++li)
            x = append_layer(c, m, (int) li, x, S, cs, sn, mt);
        // Final norm + lm_head on the LAST token only.
        ggml_tensor* last = ggml_view_2d(c, x, hidden, 1, x->nb[1],
                                         (size_t)(S - 1) * x->nb[1]);
        ggml_tensor* nrm = rms(c, m, last, "llm.final_norm.weight", (float) lc.rms_norm_eps);
        return ff(c, ggml_mul_mat(c, wb(c, m, "llm.lm_head.weight"), nrm));
    }, logits);
    if (!ok) { e = "Hojo forward graph failed"; return false; }
    return true;
}

// HF repetition penalty: for each token in `input_ids`, if its logit > 0 divide
// by penalty, else multiply. Applied to logits BEFORE softmax.
void apply_repetition_penalty(std::vector<float>& logits,
                              const std::vector<int32_t>& input_ids,
                              double penalty) {
    for (int32_t tok : input_ids) {
        if (tok < 0 || (size_t) tok >= logits.size()) continue;
        if (logits[(size_t) tok] > 0.0f) logits[(size_t) tok] /= (float) penalty;
        else logits[(size_t) tok] *= (float) penalty;
    }
}

// Beam hypothesis for beam search (active beams only).
struct Beam {
    std::vector<int32_t> tokens;  // generated tokens (excl. prefix)
    float score = 0.0f;           // cumulative log-prob sum
};

} // namespace

// ---- Greedy (num_beams==1 or diagnostic) ----
bool greedy_generate(const HojoModel& m, const InputsEmbeds& i,
                     const GenerateOptions& op, GenerateResult& o, std::string& e) {
    if (i.n_tokens <= 0 || i.width != (int64_t) m.config.llm.hidden) {
        e = "invalid Hojo prefill shape";
        return false;
    }
    ensure_weights_realized(m.loader);
    const int32_t eos = op.eos_token_id;
    const int32_t max_new = (int32_t) op.max_new_tokens;
    RopeTables rope = build_rope_tables((int) m.config.llm.head_dim, (float) m.config.llm.rope_theta, (int)(i.n_tokens + max_new + 4));
    // Prefill.
    std::vector<int32_t> extra;
    if (!forward_logits(m, i.data, i.n_tokens, extra, rope, o.prefill_logits, e)) return false;
    int32_t prev = 0;
    {
        float mx = -std::numeric_limits<float>::infinity();
        for (size_t k = 0; k < o.prefill_logits.size(); ++k)
            if (o.prefill_logits[k] > mx) { mx = o.prefill_logits[k]; prev = (int32_t) k; }
    }
    o.ids.push_back(prev);
    if (prev == eos) { o.hit_eos = true; return true; }
    while ((int) o.ids.size() < max_new) {
        std::vector<float> lg;
        if (!forward_logits(m, i.data, i.n_tokens, o.ids, rope, lg, e)) return false;
        apply_repetition_penalty(lg, o.ids, op.repetition_penalty);
        int32_t next = 0;
        float mx = -std::numeric_limits<float>::infinity();
        for (size_t k = 0; k < lg.size(); ++k)
            if (lg[k] > mx) { mx = lg[k]; next = (int32_t) k; }
        o.ids.push_back(next);
        if (next == eos) { o.hit_eos = true; break; }
    }
    return true;
}

// ---- Beam search (HF-compatible, num_beams, repetition_penalty, length_penalty) ----
bool beam_generate(const HojoModel& m, const InputsEmbeds& i,
                   const GenerateOptions& op, GenerateResult& o, std::string& e) {
    if (op.num_beams <= 1 || std::getenv("STARLING_HOJO_GREEDY")) return greedy_generate(m, i, op, o, e);
    if (i.n_tokens <= 0 || i.width != (int64_t) m.config.llm.hidden) {
        e = "invalid Hojo prefill shape";
        return false;
    }
    // KV bound: the highest position any decode step touches (cache slot write
    // AND RoPE row read) is n_tokens + max_new_tokens - 2.
    if ((int64_t) i.n_tokens + op.max_new_tokens > op.max_cache_len) {
        e = "Hojo generation exceeds cache";
        return false;
    }
    ensure_weights_realized(m.loader);
    const int32_t eos = op.eos_token_id;
    const int32_t max_new = (int32_t) op.max_new_tokens;
    const int B = (int) op.num_beams;
    const int vocab = (int) m.config.llm.vocab;
    // num_beams comes from untrusted GGUF metadata; the candidate ranking does
    // std::partial_sort(..., begin + 2*B, end) over a `vocab`-sized list, so
    // reject B outside [1, vocab/2] to keep begin + 2*B within bounds. Beam
    // search is only validated against the beam-4 golden reference, so cap B
    // at 4 (checked before any beam_logp / B*vocab allocation below).
    const int max_beams = 4;
    if (B < 1 || vocab < 2 || B > vocab / 2 || B > max_beams) {
        e = "invalid Hojo beam count " + std::to_string(B) +
            " (must be in [1, " + std::to_string(std::min(vocab / 2, max_beams)) +
            "])";
        return false;
    }
    const float lp = (float) op.length_penalty;
    const float penalty = (float) op.repetition_penalty;
    RopeTables rope = build_rope_tables((int) m.config.llm.head_dim, (float) m.config.llm.rope_theta, (int)(i.n_tokens + max_new + 4));

    // Per-beam device KV caches. Cache 0 receives the prompt KV; beam_cache[b]
    // holds beam b's ancestry rows ([0, n_tokens + len(tokens)-1)).
    std::vector<DeviceCache*> beam_cache = get_beam_caches(m, B, e);
    if ((int) beam_cache.size() != B) return false;
    std::vector<DeviceCache*> beam_copy_src(B, nullptr);

    // Prefill (step 0): prompt KV ONCE into cache 0 (== beam_cache[0]); logits
    // over vocab at the last prefix position.
    if (!prefill_cached(m, i.data, i.n_tokens, rope, o.prefill_logits, e)) return false;
    // The lm_head output dim must match config vocab: log_softmax below indexes
    // the logits vector with `vocab`, so a mismatch would read out of bounds.
    // (lm_head shape is constant, so this one check covers every later forward.)
    if ((int) o.prefill_logits.size() != vocab) {
        e = "Hojo lm_head vocab dim (" + std::to_string(o.prefill_logits.size()) +
            ") != config vocab (" + std::to_string(vocab) + ")";
        return false;
    }

    // ---- Helpers mirroring HF 4.57.x beam search semantics. ----
    // log_softmax over the full vocab (f32).
    auto log_softmax = [&](const std::vector<float>& logits, std::vector<float>& out) {
        out.assign(vocab, 0.0f);
        float mx = *std::max_element(logits.begin(), logits.end());
        float s = 0.0f;
        for (int k = 0; k < vocab; ++k) s += std::exp(logits[k] - mx);
        float lse = std::log(s) + mx;
        for (int k = 0; k < vocab; ++k) out[k] = logits[k] - lse;
    };
    // HF RepetitionPenaltyLogitsProcessor. IMPORTANT: 4.57.x applies the
    // processor to log_softmax(logits), NOT to the raw logits. log-probs are
    // always <= 0, so the (v < 0) branch (v *= penalty) is taken, making
    // already-generated tokens more negative. The gather+scatter semantics
    // apply the penalty ONCE per UNIQUE token (duplicate input_ids scatter the
    // same value), so deduplicate -- otherwise a token seen k times gets
    // penalty**k, which derails long highly-repetitive decodes.
    auto rep_penalty = [&](std::vector<float>& logp, const std::vector<int32_t>& ids) {
        std::vector<int32_t> uniq = ids;
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        for (int32_t tok : uniq) {
            if (tok < 0 || tok >= vocab) continue;
            float v = logp[(size_t) tok];
            logp[(size_t) tok] = (v < 0.0f) ? v * penalty : v / penalty;
        }
    };
    // Finished hypotheses (BeamHypotheses): kept sorted by length-normalized
    // score descending, capped at B entries (worst_score = back()).
    struct Finished { std::vector<int32_t> tokens; float norm; };
    std::vector<Finished> finished;
    auto worst_finished = [&]() -> float {
        return ((int) finished.size() >= B) ? finished.back().norm
                                            : -std::numeric_limits<float>::infinity();
    };
    auto add_finished = [&](const std::vector<int32_t>& toks, float raw_score) {
        float gen_len = (float) toks.size();
        Finished f; f.tokens = toks; f.norm = raw_score / std::pow(gen_len, lp);
        auto it = std::lower_bound(
            finished.begin(), finished.end(), f,
            [](const Finished& a, const Finished& b) { return a.norm > b.norm; });
        finished.insert(it, f);
        if ((int) finished.size() > B) finished.pop_back();
    };

    // Reassign the per-beam KV buffers after a selection round.
    // parent_slot[s] = the OLD beam whose history new beam s continues
    // (-1 = inactive slot). A parent's buffer follows its first surviving
    // child by pointer; further children of the same parent copy rows
    // [0, past) from it inside their next decode graph (beam_copy_src).
    // Copy sources are claimed buffers, copy targets unclaimed ones, so no
    // copy clobbers a source.
    auto reassign_caches = [&](const std::vector<int>& parent_slot) {
        std::vector<DeviceCache*> next_cache(B, nullptr);
        std::vector<DeviceCache*> next_copy(B, nullptr);
        std::vector<std::vector<int>> children(B);
        for (int s = 0; s < B; ++s)
            if (parent_slot[s] >= 0) children[parent_slot[s]].push_back(s);
        std::vector<int> first_child(B, -1);
        for (int v = 0; v < B; ++v) {
            if (children[v].empty()) continue;
            int i0 = children[v][0];
            for (int s : children[v])
                if (s == v) { i0 = s; break; }
            first_child[v] = i0;
            next_cache[i0] = beam_cache[v];
        }
        std::vector<DeviceCache*> unclaimed;
        for (int v = 0; v < B; ++v)
            if (first_child[v] < 0) unclaimed.push_back(beam_cache[v]);
        for (int v = 0; v < B; ++v) {
            if (first_child[v] < 0) continue;
            for (int s : children[v]) {
                if (s == first_child[v]) continue;
                next_cache[s] = unclaimed.back();
                unclaimed.pop_back();
                next_copy[s] = beam_cache[v];
            }
        }
        for (int s = 0; s < B; ++s)
            if (!next_cache[s]) { next_cache[s] = unclaimed.back(); unclaimed.pop_back(); }
        beam_cache = std::move(next_cache);
        beam_copy_src = std::move(next_copy);
    };

    // Active beams: cumulative raw logprob. Init beam 0 = 0, others = -inf
    // (HF: only beam 0 expands at step 0).
    std::vector<Beam> beams(B);
    for (int b = 0; b < B; ++b) beams[b].score = -std::numeric_limits<float>::infinity();
    beams[0].score = 0.0f;

    // ---- Step 0: rank beam-0 log-probs, pick first tokens. ----
    {
        std::vector<float> pf_logp;
        log_softmax(o.prefill_logits, pf_logp);
        std::vector<std::pair<float, int32_t>> ranked;  // (raw = score0 + logp, tok)
        ranked.reserve((size_t) vocab);
        for (int k = 0; k < vocab; ++k) ranked.push_back({beams[0].score + pf_logp[k], k});
        std::partial_sort(ranked.begin(), ranked.begin() + 2 * B, ranked.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        int picked = 0;
        std::vector<int> parent_slot(B, -1);
        for (int rank = 0; rank < 2 * B; ++rank) {
            float raw = ranked[rank].first;
            int32_t tok = ranked[rank].second;
            if (tok == eos) {
                if (rank < B) add_finished({tok}, raw);  // eos in top-num_beams -> finished
            } else if (picked < B) {
                beams[picked].tokens = {tok};
                beams[picked].score = raw;
                parent_slot[picked] = 0;  // only beam 0 expands at step 0
                ++picked;
            }
        }
        reassign_caches(parent_slot);  // fan the prompt KV out to the beams
    }

    // Early-stop: no running beam can still beat the worst finished hypothesis
    // (HF _check_earlystop_heuristic with early_stopping=False). best_running_raw
    // is the best active beam's cumulative raw score; gen_count = generated
    // length (decoder_prompt_len excluded since we track only generated tokens).
    auto is_done = [&](float best_running_raw, int gen_count) -> bool {
        if ((int) finished.size() < B) return false;
        float highest_attainable = best_running_raw / std::pow((float) gen_count, lp);
        return worst_finished() >= highest_attainable;
    };

    // ---- Decode steps: per beam an exact-width S=1 forward over the cached
    // ancestry + the beam's last token. ----
    int step = 0;  // beams currently hold (step+1) generated tokens
    while ((step + 1) < max_new) {
        std::vector<std::vector<float>> beam_logp(B);
        bool any_active = false;
        for (int b = 0; b < B; ++b) {
            if (beams[b].tokens.empty()) continue;  // unfilled slot
            any_active = true;
            std::vector<float> raw_lg;
            // Feed ONLY the last token: every earlier position's KV already
            // sits in beam_cache[b] (prompt from the prefill, ancestors from
            // prior steps and/or the re-parenting copy).
            if (!decode_step(m, beam_cache[b], beam_copy_src[b],
                             beams[b].tokens.back(),
                             (int64_t) i.n_tokens + (int64_t) beams[b].tokens.size() - 1,
                             rope, raw_lg, e))
                return false;
            log_softmax(raw_lg, beam_logp[b]);
            rep_penalty(beam_logp[b], beams[b].tokens);
        }
        if (!any_active) break;
        // Candidate cumulative raw scores = beam.score + logp[tok]. Top-2B.
        const int gen_len = step + 2;  // generated tokens after this step's pick
        std::vector<std::tuple<float, int, int32_t>> ranked;  // (raw, beam, tok)
        ranked.reserve((size_t) B * (size_t) vocab);
        for (int b = 0; b < B; ++b) {
            if (beam_logp[b].empty()) continue;
            const float bs = beams[b].score;
            const float* lp_ = beam_logp[b].data();
            for (int k = 0; k < vocab; ++k)
                ranked.push_back({bs + lp_[k], b, (int32_t) k});
        }
        std::partial_sort(ranked.begin(), ranked.begin() + 2 * B, ranked.end(),
                          [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });
        // Select: eos ranked < num_beams -> finished; top-B non-eos -> running
        // beams (eos continuations are effectively -inf for the next step).
        std::vector<Beam> next_beams(B);
        for (int b = 0; b < B; ++b) next_beams[b].score = -std::numeric_limits<float>::infinity();
        int picked = 0;
        float best_running_raw = -std::numeric_limits<float>::infinity();
        std::vector<int> parent_slot(B, -1);
        for (int rank = 0; rank < 2 * B; ++rank) {
            const auto [raw, b, tok] = ranked[rank];
            if (tok == eos) {
                if (rank < B) {
                    std::vector<int32_t> toks = beams[b].tokens;
                    toks.push_back(tok);
                    add_finished(toks, raw);  // raw includes the eos logprob
                }
            } else if (picked < B) {
                next_beams[picked].tokens = beams[b].tokens;
                next_beams[picked].tokens.push_back(tok);
                next_beams[picked].score = raw;
                parent_slot[picked] = b;
                if (raw > best_running_raw) best_running_raw = raw;
                ++picked;
            }
        }
        beams = std::move(next_beams);
        reassign_caches(parent_slot);
        if (picked == 0) break;  // no open beam can continue
        if (is_done(best_running_raw, gen_len)) break;
        step = step + 1;
    }

    // ---- Final selection: best finished by length-normalized score. ----
    o.hit_eos = !finished.empty();
    if (finished.empty()) {
        int best = 0;
        for (int b = 1; b < B; ++b)
            if (beams[b].score > beams[best].score) best = b;
        o.ids = beams[best].tokens;
    } else {
        // `finished` is sorted by norm desc; front() is the winner.
        o.ids = finished.front().tokens;
    }
    return true;
}
} // namespace starling::ggml::hojo
