// llm.cpp — Higgs Qwen3-1.7B text decoder on the Starling ggml runtime.
//
// Ported from cpp/moss/llm.cpp (also a Qwen3-family trunk with qk_norm). ark's
// llm.cpp is the WRONG reference (Qwen2.5 WITHOUT qk_norm). The Higgs LLM
// differs from moss only in: (a) SEPARATE lm_head (llm.lm_head.weight, NOT tied
// to embed), (b) ASR stops on EITHER <|endoftext|> (151643) OR <|im_end|>
// (151645) — config.EOS_TOKEN_IDS, (c) 28 layers / GQA 16/8 / intermediate 6144
// (config-driven, no code change). The Qwen3 op order/dtype discipline
// (bf/ff casts, f32 elementwise, f32 RMSNorm->bf16->bf16 weight, rotate-half
// RoPE in f32, soft_max_ext with f32 mask, q_norm/k_norm per head) is
// byte-identical to moss.
//
// Structure (mirrors moss): device-resident KV cache (DeviceCache, per-layer
// [D, max_cache, KV] bf16 + precomputed RoPE tables), one per-S prefill
// ReplayGraph, one per-K K-step decode ReplayGraph (chained K steps in-graph so
// there is one device<->host sync per K tokens), and the one-shot CPU fallbacks.
//
// Correctness contract: byte-exact bf16 vs the Transformers golden path on CUDA.
#include "llm.hpp"

#include "runtime/backend.hpp"
#include "runtime/graph.hpp"
#include "runtime/graph_builder.hpp"
#include "runtime/lru_cache.hpp"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace starling::ggml::higgs {
namespace {

// ---------------------------------------------------------------------------
// Small tensor helpers. Convention: keep activations bf16 between ops (the
// reference model runs bf16), but do elementwise math (add/mul) in f32 and cast
// back, matching PyTorch's bf16 autocast semantics closely enough to stay
// bit-exact with the golden path. Identical discipline to moss/llm.cpp.
// ---------------------------------------------------------------------------

ggml_tensor* bf(ggml_context* c, ggml_tensor* x) {
    return x->type == GGML_TYPE_BF16 ? x : ggml_cast(c, x, GGML_TYPE_BF16);
}
ggml_tensor* ff(ggml_context* c, ggml_tensor* x) {
    return x->type == GGML_TYPE_F32 ? x : ggml_cast(c, x, GGML_TYPE_F32);
}
ggml_tensor* wb(ggml_context* c, const HiggsModel& m, const std::string& n) {
    return clone_weight(c, m.loader, n.c_str());
}
// Linear: y = W x, result bf16. (Qwen3 attention has NO q/k/v/o bias; the
// projector/encoder linears use the bias-aware helper in audio_encoder.cpp.)
ggml_tensor* lin(ggml_context* c, const HiggsModel& m, ggml_tensor* x, const std::string& n) {
    return bf(c, ggml_mul_mat(c, wb(c, m, n), bf(c, x)));
}
ggml_tensor* addb(ggml_context* c, ggml_tensor* a, ggml_tensor* b) {
    return bf(c, ggml_add(c, ff(c, a), ff(c, b)));
}
ggml_tensor* mulb(ggml_context* c, ggml_tensor* a, ggml_tensor* b) {
    return bf(c, ggml_mul(c, ff(c, a), ff(c, b)));
}
// RMSNorm in f32, then scale by the (bf16) weight. (Qwen3: no RMSNorm bias.)
ggml_tensor* rms(ggml_context* c, const HiggsModel& m, ggml_tensor* x, const std::string& n,
                 float eps) {
    ggml_tensor* y = ggml_rms_norm(c, ff(c, x), eps);
    y = bf(c, y);
    return mulb(c, y, bf(c, wb(c, m, n)));
}

std::vector<ggml_bf16_t> tobf(const std::vector<float>& x) {
    std::vector<ggml_bf16_t> r(x.size());
    for (size_t i = 0; i < x.size(); ++i) r[i] = ggml_fp32_to_bf16(x[i]);
    return r;
}

int32_t argmax_low(const std::vector<float>& x) {
    int32_t best = 0;
    for (int32_t i = 1; i < (int32_t) x.size(); ++i)
        if (x[i] > x[best]) best = i;
    return best;
}

// ---------------------------------------------------------------------------
// Process-global device-resident KV cache + precomputed RoPE tables. One per
// process; zeroed at the start of each utterance. The KV tensors live in a
// persistent ggml_context allocated on the backend buffer; graphs reference them
// (and the RoPE tables) as fixed leaves. Freed by the registered decode-cache-
// clearer BEFORE backend teardown. Identical mechanism to moss's DeviceCache.
// ---------------------------------------------------------------------------

struct DeviceCache {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor*> k, v;    // [n_layers], each [D, max_cache, KV] bf16
    ggml_tensor* rope_cos = nullptr;  // [D, max_pos] bf16
    ggml_tensor* rope_sin = nullptr;  // [D, max_pos] bf16
    int max_cache = 0, max_pos = 0;
    int n_layers = 0, D = 0, KV = 0;

    bool init(const LlmConfig& lc, ggml_backend_t backend, std::string& e);
    void zero();
    ~DeviceCache() {
        if (shutting_down()) return;  // driver gone -> leak (fine at exit)
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
    }
};

bool DeviceCache::init(const LlmConfig& lc, ggml_backend_t backend, std::string& e) {
    n_layers = (int) lc.n_layers;
    D = (int) lc.head_dim;
    KV = (int) lc.n_kv_heads;
    max_cache = (int) lc.max_cache;
    max_pos = (int) lc.max_cache;  // decode positions stay < max_cache

    const size_t n_tensors = 2 * (size_t) n_layers + 2;
    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (n_tensors + 8),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx = ggml_init(params);
    if (!ctx) { e = "DeviceCache: ggml_init failed"; return false; }

    int64_t kv_ne[3] = {D, max_cache, KV};
    k.resize(n_layers);
    v.resize(n_layers);
    for (int i = 0; i < n_layers; ++i) {
        k[i] = ggml_new_tensor(ctx, GGML_TYPE_BF16, 3, kv_ne);
        v[i] = ggml_new_tensor(ctx, GGML_TYPE_BF16, 3, kv_ne);
    }
    int64_t rope_ne[2] = {D, max_pos};
    rope_cos = ggml_new_tensor(ctx, GGML_TYPE_BF16, 2, rope_ne);
    rope_sin = ggml_new_tensor(ctx, GGML_TYPE_BF16, 2, rope_ne);

    buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { e = "DeviceCache: backend alloc failed"; return false; }

    // Precompute the RoPE cos/sin tables with the f32 std::pow-based formula
    // (Qwen3 rotary, duplicated halves, rounded to bf16). theta = 1e6 for higgs.
    std::vector<ggml_bf16_t> cos_t((size_t) D * max_pos), sin_t((size_t) D * max_pos);
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < D / 2; ++i) {
            float inv = 1.0f / std::pow(lc.rope_theta, (2.0f * i) / D);
            float a = (float) p * inv;
            ggml_bf16_t c = ggml_fp32_to_bf16(std::cos(a));
            ggml_bf16_t s = ggml_fp32_to_bf16(std::sin(a));
            cos_t[(size_t) p * D + i] = cos_t[(size_t) p * D + i + D / 2] = c;
            sin_t[(size_t) p * D + i] = sin_t[(size_t) p * D + i + D / 2] = s;
        }
    }
    ggml_backend_tensor_set(rope_cos, cos_t.data(), 0, cos_t.size() * sizeof(ggml_bf16_t));
    ggml_backend_tensor_set(rope_sin, sin_t.data(), 0, sin_t.size() * sizeof(ggml_bf16_t));

    zero();
    return true;
}

void DeviceCache::zero() {
    std::vector<ggml_bf16_t> z((size_t) D * max_cache * KV, ggml_bf16_t{0});
    for (int i = 0; i < n_layers; ++i) {
        ggml_backend_tensor_set(k[i], z.data(), 0, z.size() * sizeof(ggml_bf16_t));
        ggml_backend_tensor_set(v[i], z.data(), 0, z.size() * sizeof(ggml_bf16_t));
    }
}

std::unique_ptr<DeviceCache> g_device_cache;
std::once_flag g_device_cache_once;
void register_device_cache_clearer_once() {
    std::call_once(g_device_cache_once, [] {
        register_decode_cache_clearer([] { g_device_cache.reset(); });
    });
}
DeviceCache* get_device_cache(const HiggsModel& m, std::string& e) {
    register_device_cache_clearer_once();
    if (g_device_cache) return g_device_cache.get();
    g_device_cache = std::unique_ptr<DeviceCache>(new DeviceCache());
    if (!g_device_cache->init(m.config.llm, global_backend().handle(), e)) {
        g_device_cache.reset();
        return nullptr;
    }
    return g_device_cache.get();
}

// Causal additive mask [K, S] f32: 0 where allowed, -3.3895313892515355e38 beyond
// (row qi covers keys j <= past+qi). Same constant + layout as moss.
std::vector<float> build_causal_mask(int64_t S, int64_t past) {
    const int64_t K = past + S;
    std::vector<float> mask((size_t) K * S);
    const float neg = -3.3895313892515355e38f;
    for (int64_t qi = 0; qi < S; ++qi)
        for (int64_t j = 0; j < K; ++j)
            mask[(size_t) qi * K + j] = (j <= past + qi) ? 0.0f : neg;
    return mask;
}

// Append one Qwen3 transformer layer's ops to ctx. x_in is [hidden, S] bf16
// (device-resident, flows straight from the previous layer). Writes this step's
// k/v into the device cache in-graph and assembles kall/vall for attention.
//   kv_mode 0 = prefill exact (cpy slots [0,S), attend to new k/v)
//   kv_mode 1 = decode exact-width (cpy slot `past`, attend [0, past+S))
//   kv_mode 2 = decode full-capacity (set_rows slot `past`, attend [0, max_cache))
// idx_past is the runtime i32[1] write index used only by mode 2.
// cs/sn are [D, S] bf16 (RoPE rows). mask is f32 [K, S] (K = past+S for modes
// 0/1, max_cache for mode 2). Identical to moss's append_layer_new.
ggml_tensor* append_layer(ggml_context* c, const HiggsModel& m, int li,
                          ggml_tensor* x_in, int64_t S, int64_t past,
                          ggml_tensor* cache_k, ggml_tensor* cache_v,
                          ggml_tensor* cs, ggml_tensor* sn,
                          ggml_tensor* mask, int kv_mode,
                          ggml_tensor* idx_past) {
    const auto& lc = m.config.llm;
    const int D = lc.head_dim, H = lc.n_heads, KV = lc.n_kv_heads;
    const int64_t K = (kv_mode == 2) ? (int64_t) lc.max_cache : (past + S);
    const std::string p = "llm.blk." + std::to_string(li) + ".";
    ggml_tensor* r = x_in;

    ggml_tensor* n = rms(c, m, x_in, p + "attn_norm.weight", lc.rms_norm_eps);
    ggml_tensor* q = lin(c, m, n, p + "attn.q.weight");
    ggml_tensor* k = lin(c, m, n, p + "attn.k.weight");
    ggml_tensor* v = lin(c, m, n, p + "attn.v.weight");
    q = ggml_reshape_3d(c, q, D, H, S);
    k = ggml_reshape_3d(c, k, D, KV, S);
    v = ggml_reshape_3d(c, v, D, KV, S);
    // Qwen3 qk_norm: RMSNorm per head over head_dim after the q/k projection.
    q = rms(c, m, q, p + "attn.q_norm.weight", lc.rms_norm_eps);
    k = rms(c, m, k, p + "attn.k_norm.weight", lc.rms_norm_eps);
    q = ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));  // [D, S, H]
    k = ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3));  // [D, S, KV]
    v = ggml_cont(c, ggml_permute(c, v, 0, 2, 1, 3));  // [D, S, KV]

    // RoPE (rotate-half, f32 math). cs/sn are bf16 device-table rows.
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

    // KV cache: write this step's k/v into the device cache in-graph and assemble
    // kall/vall. Depending on the cpy/set_rows result (via attention) forces the
    // write to execute before the read.
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
        kall = ggml_set_rows(c, cache_k, ff(c, k), idx_past);
        vall = ggml_set_rows(c, cache_v, ff(c, v), idx_past);
    } else {                       // decode exact-width
        ggml_tensor* kslot = ggml_view_3d(c, cache_k, D, 1, KV,
                                          cache_k->nb[1], cache_k->nb[2],
                                          (size_t) past * cache_k->nb[1]);
        ggml_tensor* vslot = ggml_view_3d(c, cache_v, D, 1, KV,
                                          cache_v->nb[1], cache_v->nb[2],
                                          (size_t) past * cache_v->nb[1]);
        ggml_tensor* knew = ggml_cpy(c, k, kslot);
        ggml_tensor* vnew = ggml_cpy(c, v, vslot);
        ggml_tensor* kprev = ggml_view_3d(c, cache_k, D, past, KV,
                                          cache_k->nb[1], cache_k->nb[2], 0);
        ggml_tensor* vprev = ggml_view_3d(c, cache_v, D, past, KV,
                                          cache_v->nb[1], cache_v->nb[2], 0);
        // CUDA concat is F32-only (concat.cu GGML_ASSERT); cast via ff/bf.
        kall = past ? bf(c, ggml_concat(c, ff(c, kprev), ff(c, knew), 1)) : knew;
        vall = past ? bf(c, ggml_concat(c, ff(c, vprev), ff(c, vnew), 1)) : vnew;
    }

    // Attention: batched over all heads using ggml's native GQA broadcast
    // (ne12 % ne02 == 0, r2 = H/KV). Per-head math unchanged from moss.
    const float scale = 1.0f / std::sqrt((float) D);
    ggml_tensor* sc = ggml_mul_mat(c, kall, q);                 // [K, S, H]
    sc = bf(c, ggml_scale(c, ff(c, sc), scale));
    ggml_tensor* pr = bf(c, ggml_soft_max_ext(c, ff(c, sc), ff(c, mask), 1.0f, 0.0f));
    ggml_tensor* vt = ggml_cont(c, ggml_permute(c, vall, 1, 0, 2, 3));  // [K, D, KV]
    ggml_tensor* co = ggml_mul_mat(c, vt, pr);                  // [D, S, H]
    // heads -> features: [D,S,H] -> [D,H,S] -> [hidden=D*H, S].
    ggml_tensor* joined = ggml_reshape_2d(c, ggml_cont(c, ggml_permute(c, co, 0, 2, 1, 3)),
                                          (int64_t) D * H, S);  // [hidden, S]
    joined = bf(c, joined);

    ggml_tensor* a = lin(c, m, joined, p + "attn.o.weight");
    ggml_tensor* x = addb(c, r, a);
    r = x;
    n = rms(c, m, x, p + "ffn_norm.weight", lc.rms_norm_eps);
    ggml_tensor* g = lin(c, m, n, p + "ffn.gate.weight");
    ggml_tensor* u = lin(c, m, n, p + "ffn.up.weight");
    ggml_tensor* si = bf(c, ggml_silu(c, ff(c, g)));
    ggml_tensor* z = mulb(c, si, u);
    ggml_tensor* dn = lin(c, m, z, p + "ffn.down.weight");
    x = addb(c, r, dn);
    return x;  // [hidden, S] bf16
}

// Whole-model prefill graph: embeds -> 28 layers -> final norm -> lm_head (last
// token only). One-shot on CPU; on GPU captured into a per-S ReplayGraph (prompt
// length S fully determines every shape). Inputs: inputs_embeds (bf16 [hidden, S],
// the only varying input) + position (i32 [S], constant for S) + causal mask
// (f32 [S, S], constant for S). The KV write path is kv_mode=0 (cpy into slots
// [0, S)). Cached in a bounded LRU (runtime/lru_cache.hpp).
struct PrefillReplayEntry {
    int64_t S = 0;
    GraphInputPool pool;
    std::unique_ptr<ReplayGraph> graph;
    ggml_bf16_t* xb_buf = nullptr;  // stable pool backing for input #0 (varying)
};
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

PrefillReplayEntry* get_or_build_prefill(const HiggsModel& m, int64_t S, std::string& e) {
    register_prefill_clearer_once();
    if (!g_prefill_cache)
        g_prefill_cache = std::make_unique<PrefillCache>(replay_cache_size());
    const auto& lc = m.config.llm;
    DeviceCache* dc = get_device_cache(m, e);
    if (!dc) return nullptr;

    return g_prefill_cache->by_S.get_or_init(S,
        [&](PrefillReplayEntry& entry) -> PrefillReplayEntry& {
            entry.S = S;
            entry.xb_buf = reinterpret_cast<ggml_bf16_t*>(
                entry.pool.alloc_bytes((size_t) S * lc.hidden * sizeof(ggml_bf16_t)));
            int32_t* pos_buf = entry.pool.alloc_i32((size_t) S);
            for (int64_t i = 0; i < S; ++i) pos_buf[i] = (int32_t) i;
            std::vector<float> mask = build_causal_mask(S, 0);
            float* mask_buf = entry.pool.alloc_f32((size_t) S * S);
            std::memcpy(mask_buf, mask.data(), (size_t) S * S * sizeof(float));

            entry.graph = std::make_unique<ReplayGraph>(global_backend(),
                [&](ggml_context* c) -> ggml_tensor* {
                    int64_t xne[2] = {lc.hidden, S};
                    ggml_tensor* x = graph_input_tensor(c, GGML_TYPE_BF16, 2, xne,
                        entry.xb_buf, (size_t) S * lc.hidden * sizeof(ggml_bf16_t));
                    int64_t pne[1] = {S};
                    ggml_tensor* pos_t = graph_input_tensor(c, GGML_TYPE_I32, 1, pne,
                        pos_buf, (size_t) S * sizeof(int32_t));
                    ggml_tensor* cs = ggml_get_rows(c, dc->rope_cos, pos_t);  // [D, S]
                    ggml_tensor* sn = ggml_get_rows(c, dc->rope_sin, pos_t);
                    int64_t mne[2] = {S, S};
                    ggml_tensor* mt = graph_input_tensor(c, GGML_TYPE_F32, 2, mne,
                        mask_buf, (size_t) S * S * sizeof(float));
                    for (int li = 0; li < (int) lc.n_layers; ++li)
                        x = append_layer(c, m, li, x, S, 0, dc->k[li], dc->v[li],
                                         cs, sn, mt, /*kv_mode=*/0, nullptr);
                    // Final norm + lm_head on the LAST token only. Higgs uses a
                    // SEPARATE lm_head (llm.lm_head.weight), NOT tied to embed.
                    ggml_tensor* last = ggml_view_2d(c, x, lc.hidden, 1, x->nb[1],
                                                     (size_t)(S - 1) * x->nb[1]);
                    ggml_tensor* nrm = rms(c, m, last, "llm.final_norm.weight", lc.rms_norm_eps);
                    return ff(c, ggml_mul_mat(c, wb(c, m, "llm.lm_head.weight"), nrm));
                });
            return entry;
        });
}

bool forward_prefill(const HiggsModel& m, const std::vector<float>& input, int64_t S,
                     std::vector<float>& logits, std::string& e) {
    const auto& lc = m.config.llm;
    ensure_weights_realized(m.loader);
    DeviceCache* dc = get_device_cache(m, e);
    if (!dc) return false;
    dc->zero();  // fresh utterance

    // CPU backend: one-shot build (capture is GPU-only).
    if (!global_backend().is_gpu()) {
        std::vector<ggml_bf16_t> xb = tobf(input);
        std::vector<int32_t> pos((size_t) S);
        for (int64_t i = 0; i < S; ++i) pos[(size_t) i] = (int32_t) i;
        std::vector<float> mask = build_causal_mask(S, 0);
        bool ok = run_graph([&](ggml_context* c) -> ggml_tensor* {
            int64_t xne[2] = {lc.hidden, S};
            ggml_tensor* x = graph_input_tensor(c, GGML_TYPE_BF16, 2, xne,
                                                xb.data(), xb.size() * sizeof(xb[0]));
            int64_t pne[1] = {S};
            ggml_tensor* pos_t = graph_input_tensor(c, GGML_TYPE_I32, 1, pne,
                                                    pos.data(), pos.size() * sizeof(int32_t));
            ggml_tensor* cs = ggml_get_rows(c, dc->rope_cos, pos_t);
            ggml_tensor* sn = ggml_get_rows(c, dc->rope_sin, pos_t);
            int64_t mne[2] = {S, S};
            ggml_tensor* mt = graph_input_tensor(c, GGML_TYPE_F32, 2, mne,
                                                 mask.data(), mask.size() * sizeof(float));
            for (int li = 0; li < (int) lc.n_layers; ++li)
                x = append_layer(c, m, li, x, S, 0, dc->k[li], dc->v[li],
                                 cs, sn, mt, /*kv_mode=*/0, nullptr);
            ggml_tensor* last = ggml_view_2d(c, x, lc.hidden, 1, x->nb[1],
                                             (size_t)(S - 1) * x->nb[1]);
            ggml_tensor* nrm = rms(c, m, last, "llm.final_norm.weight", lc.rms_norm_eps);
            return ff(c, ggml_mul_mat(c, wb(c, m, "llm.lm_head.weight"), nrm));
        }, logits);
        if (!ok) { e = "Higgs prefill graph failed"; return false; }
        return true;
    }

    // GPU: captured per-S ReplayGraph. Only inputs_embeds varies.
    PrefillReplayEntry* pe = get_or_build_prefill(m, S, e);
    if (!pe) return false;
    const size_t nx = (size_t) S * lc.hidden;
    for (size_t i = 0; i < nx; ++i)
        pe->xb_buf[i] = ggml_fp32_to_bf16(input[i]);
    for (size_t i = 0; i < pe->graph->n_inputs(); ++i)
        pe->graph->set_input(i, pe->graph->input_host(i), pe->graph->input_nbytes(i));
    if (!pe->graph->compute(logits)) { e = "Higgs prefill replay failed"; return false; }
    return true;
}

// Whole-model decode-step graph (S=1): embed(prev) -> 28 layers -> lm_head.
// Exact-width KV (one-shot per step). Reads slots [0, past), writes slot `past`,
// attention over [0, past). Used on CPU and as the non-Kstep GPU fallback.
bool forward_decode(const HiggsModel& m, int32_t prev_token, int64_t past,
                    std::vector<float>& logits, std::string& e) {
    const auto& lc = m.config.llm;
    DeviceCache* dc = get_device_cache(m, e);
    if (!dc) return false;
    const int64_t S = 1;
    const int64_t K = past + S;
    // Full-capacity attention (captured-graph shape): attention over [0, max_cache)
    // with an additive f32 mask, used to gate whether padded softmax flips a token.
    const bool fullcap = std::getenv("STARLING_HIGGS_FULLCAP") != nullptr;
    const int kv_mode = fullcap ? 2 : 1;
    std::vector<int32_t> pos = {(int32_t) past};
    std::vector<int32_t> idx_past = {(int32_t) past};
    std::vector<float> mask_exact((size_t) K, 0.0f);
    std::vector<float> mask_full((size_t) lc.max_cache, 0.0f);
    if (fullcap) {
        const float neg = -3.3895313892515355e38f;
        for (int64_t j = past + 1; j < (int64_t) lc.max_cache; ++j)
            mask_full[(size_t) j] = neg;
    }
    const std::vector<float>& mask = fullcap ? mask_full : mask_exact;
    const int64_t mask_w = fullcap ? (int64_t) lc.max_cache : K;

    bool ok = run_graph([&](ggml_context* c) -> ggml_tensor* {
        int64_t one[1] = {1};
        ggml_tensor* id_t = graph_input_tensor(c, GGML_TYPE_I32, 1, one,
                                               &prev_token, sizeof(int32_t));
        ggml_tensor* x = ggml_get_rows(c, clone_weight(c, m.loader, "llm.embed.weight"), id_t);
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
        for (int li = 0; li < (int) lc.n_layers; ++li)
            x = append_layer(c, m, li, x, S, past, dc->k[li], dc->v[li],
                             cs, sn, mt, kv_mode, idx_t);
        ggml_tensor* nrm = rms(c, m, x, "llm.final_norm.weight", lc.rms_norm_eps);
        return ff(c, ggml_mul_mat(c, wb(c, m, "llm.lm_head.weight"), nrm));
    }, logits);
    if (!ok) { e = "Higgs decode graph failed"; return false; }
    return true;
}

// ===========================================================================
// K-step multistep decode (captured ReplayGraph). Mirrors moss's K-step design.
// Captures K consecutive decode steps into ONE ReplayGraph with the per-step
// state chained IN-GRAPH (output token -> get_rows(embed) -> next step's input),
// so there is ONE device<->host sync per K steps. KV writes land at baked slots
// [start_past, start_past+K) (start_past is a runtime input); attention is
// EXACT-WIDTH per step (no padding) so the softmax reduction order is byte-
// identical to the one-step decode. Graph cache keyed on K (one graph per K,
// reused across all decode steps + utterances).
// ===========================================================================

struct HiggsKStepGraph {
    int K = 0;
    std::unique_ptr<ReplayGraph> rg;
    size_t in_prev_tok = 0;
    std::vector<size_t> in_pos, in_mask;
    std::vector<int32_t> host_pos;              // [K]
    std::vector<std::vector<float>> host_mask;  // [K]
    std::vector<float> cap_tokens;              // [K] (i32 reinterpreted as f32 bytes)
};

struct HiggsKKey { int K;
    bool operator==(const HiggsKKey& o) const { return K == o.K; } };
struct HiggsKKeyHash { size_t operator()(const HiggsKKey& k) const noexcept { return (size_t) k.K; } };
std::unordered_map<HiggsKKey, std::unique_ptr<HiggsKStepGraph>, HiggsKKeyHash> g_kstep_cache;
std::once_flag g_kstep_clearer_once;
void ensure_kstep_clearer() {
    std::call_once(g_kstep_clearer_once, [] {
        register_decode_cache_clearer([] { g_kstep_cache.clear(); });
    });
}

int higgs_kstep_K() {
    int v = 4;
    if (const char* env = std::getenv("STARLING_HIGGS_KSTEP")) {
        int e = std::atoi(env);
        if (e >= 1) v = e;
    }
    if (v > 8) v = 8;  // kGraphSize bounds the K-step graph; clamp to 8.
    return v;
}

// Build (or fetch) the single full-capacity K-step graph for K.
HiggsKStepGraph* get_or_build_kstep(const HiggsModel& m, int K, std::string& e) {
    ensure_kstep_clearer();
    HiggsKKey key{K};
    auto it = g_kstep_cache.find(key);
    if (it != g_kstep_cache.end()) return it->second.get();

    const auto& lc = m.config.llm;
    DeviceCache* dc = get_device_cache(m, e);
    if (!dc) return nullptr;

    auto kg = std::unique_ptr<HiggsKStepGraph>(new HiggsKStepGraph());
    kg->K = K;
    kg->host_pos.assign((size_t) K, 0);
    kg->host_mask.assign((size_t) K, std::vector<float>((size_t) lc.max_cache, 0.0f));
    kg->in_pos.resize((size_t) K);
    kg->in_mask.resize((size_t) K);
    kg->cap_tokens.assign((size_t) K, 0.0f);

    HiggsKStepGraph* raw = kg.get();
    const int64_t mc = lc.max_cache;
    raw->rg = std::unique_ptr<ReplayGraph>(new ReplayGraph(global_backend(),
        [&](ggml_context* c) -> ggml_tensor* {
            int64_t one[1] = {1};
            size_t idx = 0;
            int32_t zero_tok = 0;
            ggml_tensor* prev_tok_t = graph_input_tensor(c, GGML_TYPE_I32, 1, one,
                                                          &zero_tok, sizeof(int32_t));
            raw->in_prev_tok = idx++;
            std::vector<ggml_tensor*> pos_t((size_t) K), mask_t((size_t) K);
            for (int j = 0; j < K; ++j) {
                pos_t[(size_t) j] = graph_input_tensor(c, GGML_TYPE_I32, 1, one,
                                                       &raw->host_pos[(size_t) j], sizeof(int32_t));
                raw->in_pos[(size_t) j] = idx++;
            }
            for (int j = 0; j < K; ++j) {
                int64_t mw[2] = {mc, 1};
                mask_t[(size_t) j] = graph_input_tensor(c, GGML_TYPE_F32, 2, mw,
                                                       raw->host_mask[(size_t) j].data(),
                                                       raw->host_mask[(size_t) j].size() * sizeof(float));
                raw->in_mask[(size_t) j] = idx++;
            }

            ggml_tensor* embed_w = clone_weight(c, m.loader, "llm.embed.weight");
            // SEPARATE lm_head for higgs.
            ggml_tensor* lm_head_w = clone_weight(c, m.loader, "llm.lm_head.weight");

            // Chain K steps in-graph: tok = prev-token; each step's argmax feeds
            // the next step's embed (get_rows), all on device.
            ggml_tensor* tok = prev_tok_t;
            std::vector<ggml_tensor*> tok_nodes;
            tok_nodes.reserve((size_t) K);
            for (int j = 0; j < K; ++j) {
                ggml_tensor* x = ggml_get_rows(c, embed_w, tok);                     // [hidden, 1]
                ggml_tensor* cs = ggml_get_rows(c, dc->rope_cos, pos_t[(size_t) j]);  // [D, 1]
                ggml_tensor* sn = ggml_get_rows(c, dc->rope_sin, pos_t[(size_t) j]);
                for (int li = 0; li < (int) lc.n_layers; ++li)
                    x = append_layer(c, m, li, x, /*S=*/1, /*past=*/0,
                                     dc->k[li], dc->v[li], cs, sn, mask_t[(size_t) j],
                                     /*kv_mode=*/2, pos_t[(size_t) j]);
                ggml_tensor* nrm = rms(c, m, x, "llm.final_norm.weight", lc.rms_norm_eps);
                ggml_tensor* logits = ggml_mul_mat(c, lm_head_w, nrm);             // [vocab, 1]
                ggml_tensor* tj = ggml_argmax(c, logits);                           // i32 [1]
                // CUDA concat is F32-only; token ids < 2^24 so (float)tok is exact.
                tok_nodes.push_back(ggml_cast(c, tj, GGML_TYPE_F32));
                tok = tj;  // chain: next step embeds this step's argmax token
            }

            ggml_tensor* ring = tok_nodes[0];
            for (int j = 1; j < K; ++j)
                ring = ggml_concat(c, ring, tok_nodes[j], 0);                       // f32 [K]
            capture_graph_output(ring, &raw->cap_tokens);
            return ring;
        }));

    if (!raw->rg) { e = "Higgs K-step graph build failed"; return nullptr; }
    it = g_kstep_cache.emplace(key, std::move(kg)).first;
    return it->second.get();
}

// Run one K-step replay from `past` (the slot the first step writes). Appends up
// to K tokens, stopping at EOS / im_end / max_new_tokens. `prev` in=out (token
// entering / last emitted); `past` advances by the steps actually consumed.
// Returns true if at least one token was generated (caller continues) or false
// on error; `hit_eos` set true when EOS/im_end reached.
bool run_kstep(const HiggsModel& m, int32_t& prev, int64_t& past, int K,
               int32_t eos, int32_t im_end, int max_new_tokens,
               std::vector<int32_t>& ids, bool& hit_eos, std::string& e) {
    HiggsKStepGraph* kg = get_or_build_kstep(m, K, e);
    if (!kg) return false;
    const int64_t mc = m.config.llm.max_cache;
    const float neg = -3.3895313892515355e38f;
    kg->rg->set_input(kg->in_prev_tok, &prev, sizeof(int32_t));
    for (int j = 0; j < K; ++j) {
        int64_t boundary = past + j;
        kg->host_pos[(size_t) j] = (int32_t)(past + j);
        // Load-bearing invariant: no decode step's device index may reach
        // max_cache (greedy_generate's tail-cap Kk = min(K, remaining) enforces
        // it; this guard turns a regression into a detectable error instead of
        // silent device-memory corruption — the OOB write lands in ggml buffer
        // padding and does NOT fault).
        if (kg->host_pos[(size_t) j] < 0 || kg->host_pos[(size_t) j] >= (int32_t) mc) {
            e = "Higgs K-step position out of bounds (pos=" +
                std::to_string(kg->host_pos[(size_t) j]) +
                ", max_cache=" + std::to_string(mc) + ")";
            return false;
        }
        std::vector<float>& mk = kg->host_mask[(size_t) j];
        for (int64_t s = 0; s < boundary + 1 && s < mc; ++s) mk[(size_t) s] = 0.0f;
        for (int64_t s = boundary + 1; s < mc; ++s) mk[(size_t) s] = neg;
        kg->rg->set_input(kg->in_pos[(size_t) j], &kg->host_pos[(size_t) j], sizeof(int32_t));
        kg->rg->set_input(kg->in_mask[(size_t) j], mk.data(), mk.size() * sizeof(float));
    }
    std::vector<float> out;
    if (!kg->rg->compute_with_captures(out)) { e = "Higgs K-step replay failed"; return false; }

    hit_eos = false;
    for (int j = 0; j < K; ++j) {
        if ((int) ids.size() >= max_new_tokens) break;
        int32_t tok = (int32_t) kg->cap_tokens[(size_t) j];
        ids.push_back(tok);
        prev = tok;
        past += 1;
        if (tok == eos || tok == im_end) { hit_eos = true; break; }
    }
    return true;
}

// True if `tok` is an ASR stop token (either eos or im_end).
inline bool is_stop(int32_t tok, int32_t eos, int32_t im_end) {
    return tok == eos || tok == im_end;
}

} // namespace

size_t prefill_replay_cache_size() {
    return g_prefill_cache ? g_prefill_cache->by_S.size() : 0;
}

bool greedy_generate(const HiggsModel& m, const InputsEmbeds& i,
                     const GenerateOptions& op, GenerateResult& o, std::string& e) {
    if (i.n_tokens <= 0 || i.width != (int64_t) m.config.llm.hidden) {
        e = "invalid Higgs prefill shape";
        return false;
    }
    if (i.n_tokens + op.max_new_tokens > op.max_cache_len) {
        e = "Higgs generation exceeds cache";
        return false;
    }
    ensure_weights_realized(m.loader);
    const bool timing = std::getenv("STARLING_HIGGS_TIMING") != nullptr;

    double t_pf = 0.0;
    auto now = [&]() { return std::chrono::steady_clock::now(); };
    auto ms = [&](auto t0, auto t1) {
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    };
    // Prefill.
    {
        auto t0 = now();
        if (!forward_prefill(m, i.data, i.n_tokens, o.prefill_logits, e)) return false;
        auto t1 = now();
        t_pf = ms(t0, t1);
    }
    int32_t prev = argmax_low(o.prefill_logits);
    o.ids.push_back(prev);
    if (is_stop(prev, op.eos_token_id, op.im_end_id)) { o.hit_eos = true; return true; }

    const bool use_kstep = global_backend().is_gpu() &&
                           !std::getenv("STARLING_HIGGS_NOKSTEP");
    double dec_sum = 0.0; int dec_n = 0;
    int64_t past = i.n_tokens;
    if (use_kstep) {
        const int K = higgs_kstep_K();
        while ((int) o.ids.size() < op.max_new_tokens) {
            bool hit = false;
            size_t before = o.ids.size();
            // Tail-cap (Wave G): cap this block's step count to the remaining
            // token budget so no device index reaches max_cache.
            const int remaining = op.max_new_tokens - (int) o.ids.size();
            const int Kk = std::min(K, remaining);
            auto s0 = now();
            if (!run_kstep(m, prev, past, Kk, op.eos_token_id, op.im_end_id,
                           op.max_new_tokens, o.ids, hit, e))
                return false;
            auto s1 = now();
            int produced = (int)(o.ids.size() - before);
            dec_sum += ms(s0, s1); dec_n += produced;
            if (hit) { o.hit_eos = true; break; }
        }
    } else {
        while ((int) o.ids.size() < op.max_new_tokens) {
            std::vector<float> dl;
            auto s0 = now();
            if (!forward_decode(m, prev, past, dl, e)) return false;
            prev = argmax_low(dl);
            o.ids.push_back(prev);
            auto s1 = now();
            dec_sum += ms(s0, s1); ++dec_n;
            if (is_stop(prev, op.eos_token_id, op.im_end_id)) { o.hit_eos = true; break; }
        }
    }
    if (timing) {
        std::fprintf(stderr,
            "HIGGS_TIMING prefill=%.1fms decode_tokens=%d avg=%.2fms/tok (kstep=%d)\n",
            t_pf, dec_n, dec_n ? dec_sum / dec_n : 0.0, (int) use_kstep);
    }
    if (const char* fp = std::getenv("STARLING_HIGGS_DUMP_IDS")) {
        if (FILE* f = std::fopen(fp, "wb")) {
            std::fwrite(o.ids.data(), sizeof(int32_t), o.ids.size(), f);
            std::fclose(f);
        }
    }
    return true;
}

} // namespace starling::ggml::higgs
