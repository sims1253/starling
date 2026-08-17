// projector.cpp — the granite-speech BLIP2 Q-Former projector on the Starling
// ggml runtime.
//
// Forward (transformers GraniteSpeechEncoderProjector + Blip2QFormerModel, the
// eager bf16 path the reference runs): pad the encoder output to a multiple of
// window_size 15, view as [hidden, 15, nblk] windows; LayerNorm(eps 1e-12) the
// 3 learned queries; 2 BERT-style layers, each self-attention (16 heads x 64,
// scale 0.125, unmasked, qkv biased) + output dense + output LayerNorm(residual),
// cross-attention (Q = the 3 query rows, KV = the window) + same output
// structure, then the query FF (dense 1024->4096, erf GELU, dense 4096->1024,
// LayerNorm(residual)); finally flatten (block-major: position b*3+q) and
// Linear(1024->2048, bias) into the decoder's embedding space.
//
// Layer 0's self-attention runs on the shared [hidden, 3] queries (batch 1) —
// row-independent math makes this byte-exact vs the reference's broadcast;
// from the cross-attention on, every window carries its own rows.
#include "projector.hpp"

#include "lib/graph_helpers.hpp"
#include "runtime/backend.hpp"
#include "ggml.h"

#include <cmath>
#include <string>

namespace starling::ggml::granite {
namespace {

using lib::bf16;
using lib::f32;

// BERT attention block over qh [hidden, S, B(, batch dims beyond 2 are the
// tensor's ne2)]. kvh == nullptr selects self-attention (k/v from qh); else
// cross-attention with k/v from kvh [hidden, S_kv, B_kv]. Returns the dense
// output + output LayerNorm(residual): [hidden, S, B].
ggml_tensor* bert_attention(ggml_context* c, const ModelLoader& ml,
                            const std::string& base, float ln_eps,
                            ggml_tensor* qh, ggml_tensor* kvh,
                            int64_t H, int64_t hd) {
    ggml_tensor* kv_src = kvh ? kvh : qh;
    ggml_tensor* q = lib::linear_bf16(c, ml, qh, base + "q", true);
    ggml_tensor* k = lib::linear_bf16(c, ml, kv_src, base + "k", true);
    ggml_tensor* v = lib::linear_bf16(c, ml, kv_src, base + "v", true);
    const int64_t S = q->ne[1], SKV = k->ne[1];
    // [hidden, S, B] -> [hd, H, S, B] for the batched head matmuls.
    auto heads = [&](ggml_tensor* z, int64_t S_) {
        z = ggml_reshape_4d(c, z, hd, H, S_, z->ne[2]);
        return ggml_cont(c, ggml_permute(c, z, 0, 2, 1, 3));  // [hd, S, H, B]
    };
    ggml_tensor* qh_ = heads(q, S), * kh_ = heads(k, SKV), * vh_ = heads(v, SKV);
    // scores [SKV, S, H, B]; the Q side broadcasts over the KV batch when the
    // query is shared (layer 0: Bq == 1 < Bk) — ggml's native mul_mat repeat.
    ggml_tensor* sc = bf16(c, ggml_mul_mat(c, kh_, qh_));
    // BLIP2 eager attention multiplies by `scaling` (= head_dim**-0.5 = 0.125,
    // an exact power of two) and softmaxes WITHOUT an fp32 upcast.
    const float scaling = std::pow((float) hd, -0.5f);
    sc = bf16(c, ggml_scale(c, f32(c, sc), scaling));
    ggml_tensor* pr = bf16(c, ggml_soft_max_ext(c, f32(c, sc), nullptr, 1.0f, 0.0f));
    ggml_tensor* vt = ggml_cont(c, ggml_permute(c, vh_, 1, 0, 2, 3));  // [SKV, hd, H, B]
    ggml_tensor* co = bf16(c, ggml_mul_mat(c, vt, pr));                // [hd, S, H, B]
    // heads -> features: [hd, S, H, B] -> [hd, H, S, B] -> [hidden, S, B].
    co = ggml_cont(c, ggml_permute(c, co, 0, 2, 1, 3));                // [hd, H, S, B]
    const int64_t B = co->ne[3];
    co = ggml_reshape_3d(c, co, H * hd, S, B);
    ggml_tensor* o = lib::linear_bf16(c, ml, co, base + "out", true);
    // LN(input + out): the OUTPUT operand comes first — ggml broadcasts the
    // second addend onto the first, and layer 0's shared query batch (Bq=1)
    // must repeat onto the windowed output (B=nblk). Value-commutative.
    return lib::layer_norm_bf16(c, ml, lib::addb(c, o, qh), base + "ln", ln_eps);
}

} // namespace

ggml_tensor* build_projector(ggml_context* c, const GraniteModel& m,
                             ggml_tensor* enc, const EncScratch& s) {
    const auto& pc = m.config.projector;
    const auto& ml = m.loader;
    const int64_t hidden = pc.hidden;
    const int64_t T = enc->ne[1];
    const int64_t nblk = (T + pc.window_size - 1) / pc.window_size;
    const int64_t T15 = nblk * pc.window_size;

    // Zero-pad the trailing frames (the reference pads BEFORE windowing; the
    // cross-attention attends to the padded rows unmasked, exactly as stock).
    ggml_tensor* xp = enc;
    if (T15 > T) {
        int64_t zne[2] = {hidden, T15 - T};
        ggml_tensor* zp = graph_input_tensor(c, GGML_TYPE_F32, 2, zne,
                                             s.zeros_proj.data(),
                                             s.zeros_proj.size() * sizeof(float));
        xp = bf16(c, ggml_concat(c, f32(c, enc), zp, 1));  // CUDA concat is f32-only
    }
    ggml_tensor* xw = ggml_reshape_3d(c, xp, hidden, pc.window_size, nblk);

    // The 3 learned queries, stored (3, 1024) -> [hidden, num_queries], pass
    // through the qformer's input LayerNorm (eps 1e-12).
    ggml_tensor* query = lib::weight(c, ml, "proj.query");
    query = ggml_reshape_2d(c, query, hidden, pc.num_queries);
    ggml_tensor* h = lib::layer_norm_bf16(c, ml, query, "proj.qformer_ln",
                                          pc.layer_norm_eps);
    // Broadcast the shared queries to per-window rows up front: the reference
    // runs layer 0's self-attention on the batch-1 query tensor, but that math
    // is row-independent, so tiling identical rows is value-exact — and it
    // keeps every qformer matmul on matching batch dims (ggml's mul_mat only
    // broadcasts the KV side). The repeat runs in f32 (the CUDA REPEAT kernel
    // is f32/f16-only); the bf16->f32->bf16 round-trip is exact.
    h = ggml_reshape_3d(c, h, hidden, pc.num_queries, 1);
    if (nblk > 1) {
        ggml_tensor* tmpl = ggml_view_3d(c, xw, hidden, pc.num_queries, nblk,
                                         xw->nb[1], xw->nb[2], 0);
        h = bf16(c, ggml_repeat(c, f32(c, h), tmpl));
    }

    for (uint32_t i = 0; i < pc.qformer_layers; ++i) {
        const std::string p = "proj.blk." + std::to_string(i) + ".";
        h = bert_attention(c, ml, p + "self_", pc.layer_norm_eps, h, nullptr,
                           pc.qformer_heads, pc.hidden / pc.qformer_heads);
        h = bert_attention(c, ml, p + "cross_", pc.layer_norm_eps, h, xw,
                           pc.qformer_heads, pc.hidden / pc.qformer_heads);
        ggml_tensor* t = lib::linear_bf16(c, ml, h, p + "ff_up", true);
        t = lib::gelu_erf_bf16(c, t);
        t = lib::linear_bf16(c, ml, t, p + "ff_down", true);
        h = lib::layer_norm_bf16(c, ml, lib::addb(c, h, t), p + "ff_ln",
                                 pc.layer_norm_eps);
    }

    // (d, q, b) storage flattens to position b*num_queries+q — the reference's
    // view(B, nblocks * (window/downsample), -1) ordering.
    h = ggml_reshape_2d(c, h, hidden, nblk * pc.num_queries);
    return lib::linear_bf16(c, ml, h, "proj.out", true);  // [output_dim, N]
}

} // namespace starling::ggml::granite
