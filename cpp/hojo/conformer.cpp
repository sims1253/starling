// conformer.cpp — WeNet Conformer bottleneck for Hojo-ASR-V1.
//
// The bottleneck (hojo_asr.wenet.transformer.encoder.ConformerEncoder):
//   LinearNoSubsampling: Linear(2048->2560) + LayerNorm(eps=1e-5) + Dropout(0).
//     pos_enc = RelPositionalEncoding: forward scales x by sqrt(2560) and
//     returns pos_emb = pe[:, 0:T] (the baked [1,5000,2560] buffer, NOT scaled).
//   2 ConformerEncoderLayer (normalize_before=True), block order:
//     1. r += 0.5 * ffn_macaron(norm_ff_macaron(r))
//     2. r += relpos_mha(norm_mha(r), pos_emb)
//     3. r += conv_module(norm_conv(r))
//     4. r += 0.5 * ffn(norm_ff(r))
//     5. r = norm_final(r)
//   then after_norm (LayerNorm).
//
// RelPositionMultiHeadedAttention (attention.py, NO rel_shift — the comment in
// the reference confirms rel_shift is removed for ASR):
//   q,k,v = linear_q/k/v(x) -> heads [dk, H, T] -> [dk, T, H]
//   p = linear_pos(pos_emb) -> heads
//   q_with_bias_u = q + pos_bias_u ; q_with_bias_v = q + pos_bias_v
//   matrix_ac = q_with_bias_u @ k^T  ;  matrix_bd = q_with_bias_v @ p^T
//   scores = (ac + bd) / sqrt(dk) ; softmax(mask) ; @ v ; linear_out
//
// ConvolutionModule (convolution.py): pointwise_conv1 (Conv1d D->2D k1) ->
// GLU(dim=1) -> depthwise_conv (Conv1d D->D k15 groups=D, symmetric pad) ->
// BatchNorm1d (INFERENCE fold: scale=g/sqrt(var+eps), shift=b-mean*scale) ->
// Swish -> pointwise_conv2 (Conv1d D->D k1). No mask_pad effect (all valid).
//
// Correctness-first: one-shot run_graph. BatchNorm folds host-side (the parakeet
// conformer proved this is byte-exact in eval mode).
#include "conformer.hpp"
#include "runtime/backend.hpp"
#include "runtime/graph.hpp"
#include "runtime/graph_builder.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace starling::ggml::hojo {
namespace {

ggml_tensor* weight(ggml_context* c, const ModelLoader& ml, const std::string& n) {
    return clone_weight(c, ml, n.c_str());
}
ggml_tensor* f32(ggml_context* c, ggml_tensor* x) {
    return x->type == GGML_TYPE_F32 ? x : ggml_cast(c, x, GGML_TYPE_F32);
}
ggml_tensor* linear_b(ggml_context* c, const ModelLoader& ml, ggml_tensor* x,
                      const std::string& n, bool bias) {
    ggml_tensor* y = ggml_mul_mat(c, weight(c, ml, n + ".weight"), f32(c, x));
    if (bias) y = ggml_add(c, f32(c, y), f32(c, weight(c, ml, n + ".bias")));
    return f32(c, y);
}
ggml_tensor* layer_norm(ggml_context* c, const ModelLoader& ml, ggml_tensor* x,
                        const std::string& n, float eps) {
    ggml_tensor* y = ggml_norm(c, f32(c, x), eps);
    y = ggml_mul(c, y, f32(c, weight(c, ml, n + ".weight")));
    y = ggml_add(c, y, f32(c, weight(c, ml, n + ".bias")));
    return f32(c, y);
}
// Swish == SiLU.
ggml_tensor* swish(ggml_context* c, ggml_tensor* x) {
    return f32(c, ggml_silu(c, f32(c, x)));
}

std::vector<float> read_f32(const ModelLoader& ml, const char* name) {
    ggml_tensor* t = ml.tensor(name);
    if (!t) return {};
    ensure_weights_realized(ml);
    size_t n = (size_t) ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> raw(n);
        ggml_backend_tensor_get(t, raw.data(), 0, n * sizeof(ggml_bf16_t));
        for (size_t i = 0; i < n; ++i) out[i] = ggml_bf16_to_fp32(raw[i]);
    }
    return out;
}

// Rel-pos MHA (NO rel_shift) over x [D, T]. pos_emb is [D, T] (the pe slice,
// already extracted). Returns [D, T]. mask is nullptr (single utterance, all
// valid -> forward_attention with mask all-ones is a no-op).
ggml_tensor* relpos_mha(ggml_context* c, const ModelLoader& ml,
                        const std::string& pre, ggml_tensor* x,
                        ggml_tensor* pos_emb, int H, int dk, int T) {
    const float scale = 1.0f / std::sqrt((float) dk);
    ggml_tensor* q = linear_b(c, ml, x, pre + "linear_q", true);  // [D, T]
    ggml_tensor* k = linear_b(c, ml, x, pre + "linear_k", true);
    ggml_tensor* v = linear_b(c, ml, x, pre + "linear_v", true);
    ggml_tensor* p = linear_b(c, ml, pos_emb, pre + "linear_pos", false);  // [D, T]
    auto to_heads = [&](ggml_tensor* t) {
        t = ggml_reshape_3d(c, t, dk, H, T);
        return ggml_cont(c, ggml_permute(c, t, 0, 2, 1, 3));  // [dk, T, H]
    };
    ggml_tensor* qh = to_heads(q), * kh = to_heads(k), * vh = to_heads(v);
    ggml_tensor* ph = to_heads(p);  // [dk, T, H] (T positions)
    // pos_bias_u/v: [dk, H] -> broadcast over T.
    ggml_tensor* bu = ggml_reshape_3d(c, weight(c, ml, pre + "pos_bias_u"), dk, 1, H);
    ggml_tensor* bv = ggml_reshape_3d(c, weight(c, ml, pre + "pos_bias_v"), dk, 1, H);
    ggml_tensor* qu = ggml_add(c, qh, bu);  // [dk, T, H]
    ggml_tensor* qv = ggml_add(c, qh, bv);
    // matrix_ac = kh^T @ qu -> [T(pos=k), T(query), H]; NO rel_shift on bd.
    ggml_tensor* ac = ggml_mul_mat(c, kh, qu);  // [T, T, H]
    ggml_tensor* bd = ggml_mul_mat(c, ph, qv);  // [T(pos), T, H]
    ggml_tensor* scores = ggml_add(c, f32(c, ac), f32(c, bd));
    scores = ggml_scale(c, scores, scale);
    // softmax over the last dim (keys); no mask (all valid).
    ggml_tensor* prob = ggml_soft_max_ext(c, f32(c, scores), nullptr, 1.0f, 0.0f);
    ggml_tensor* vt = ggml_cont(c, ggml_permute(c, vh, 1, 0, 2, 3));  // [T, dk, H]
    ggml_tensor* co = ggml_mul_mat(c, vt, prob);                      // [dk, T, H]
    co = ggml_cont(c, ggml_permute(c, co, 0, 2, 1, 3));               // [dk, H, T]
    ggml_tensor* joined = ggml_reshape_2d(c, co, (int64_t) dk * H, T); // [D, T]
    return linear_b(c, ml, joined, pre + "linear_out", true);
}

// Pre-computed BatchNorm1d inference-fold params (constants: depend only on
// weights). Pre-computed OUTSIDE the graph build to avoid device I/O
// (read_f32/ensure_weights_realized) inside run_graph, which deadlocks on the
// backend mutex.
struct BnFold {
    std::vector<float> scale, shift;  // [D] each
};

BnFold precompute_bn_fold(const ModelLoader& ml, const std::string& pre,
                          int D, float eps) {
    std::vector<float> g  = read_f32(ml, (pre + "conv.norm.weight").c_str());
    std::vector<float> bb = read_f32(ml, (pre + "conv.norm.bias").c_str());
    std::vector<float> mn = read_f32(ml, (pre + "conv.norm.running_mean").c_str());
    std::vector<float> vr = read_f32(ml, (pre + "conv.norm.running_var").c_str());
    BnFold out;
    out.scale.resize(D);
    out.shift.resize(D);
    for (int i = 0; i < D; ++i) {
        out.scale[i] = g[i] / std::sqrt(vr[i] + eps);
        out.shift[i] = bb[i] - mn[i] * out.scale[i];
    }
    return out;
}

// ConvolutionModule: pointwise1 (D->2D) -> GLU -> depthwise k15 groups=D ->
// BatchNorm(inference fold) -> Swish -> pointwise2 (D->D).
ggml_tensor* conv_module(ggml_context* c, const ModelLoader& ml,
                         const std::string& pre, ggml_tensor* x, int D, int T,
                         int K, const BnFold& bn, GraphInputPool& pool) {
    ggml_tensor* pw1w = weight(c, ml, pre + "conv.pointwise_conv1.weight");
    pw1w = ggml_reshape_2d(c, pw1w, D, 2 * D);
    ggml_tensor* pw1b = f32(c, weight(c, ml, pre + "conv.pointwise_conv1.bias"));
    ggml_tensor* y = ggml_add(c, ggml_mul_mat(c, pw1w, x), pw1b);  // [2D, T]
    ggml_tensor* a = ggml_view_2d(c, y, D, T, y->nb[1], 0);
    ggml_tensor* b = ggml_view_2d(c, y, D, T, y->nb[1], (size_t) D * y->nb[0]);
    ggml_tensor* glu = ggml_mul(c, ggml_cont(c, a),
                                ggml_sigmoid(c, ggml_cont(c, b)));  // [D, T]
    // depthwise_conv via ggml_conv_2d_dw_direct (mapped to 1D: W=T, H=1).
    ggml_tensor* glu_tc = ggml_cont(c, ggml_transpose(c, glu));  // [T, D]
    ggml_tensor* dw_w = weight(c, ml, pre + "conv.depthwise_conv.weight");
    dw_w = ggml_reshape_4d(c, dw_w, K, 1, 1, D);
    ggml_tensor* nb = ggml_reshape_4d(c, glu_tc, glu_tc->ne[0], 1, glu_tc->ne[1], 1);
    const int pad = (K - 1) / 2;
    ggml_tensor* dw = ggml_conv_2d_dw_direct(c, dw_w, nb,
                                             /*s0*/1, /*s1*/1, /*p0*/pad, /*p1*/0,
                                             /*d0*/1, /*d1*/1);
    dw = ggml_reshape_2d(c, dw, T, D);  // [T, D]
    ggml_tensor* dwt = ggml_cont(c, ggml_transpose(c, dw));  // [D, T]
    ggml_tensor* dwb = f32(c, weight(c, ml, pre + "conv.depthwise_conv.bias"));
    dwt = ggml_add(c, dwt, dwb);
    // BatchNorm1d inference fold (pre-computed scale/shift fed as graph inputs).
    int64_t d_ne[1] = {D};
    float* sc = pool.alloc_f32(D);
    float* sh = pool.alloc_f32(D);
    std::memcpy(sc, bn.scale.data(), (size_t) D * sizeof(float));
    std::memcpy(sh, bn.shift.data(), (size_t) D * sizeof(float));
    ggml_tensor* scale = graph_input_tensor(c, GGML_TYPE_F32, 1, d_ne,
                                            sc, (size_t) D * sizeof(float));
    ggml_tensor* shift = graph_input_tensor(c, GGML_TYPE_F32, 1, d_ne,
                                            sh, (size_t) D * sizeof(float));
    ggml_tensor* normed = ggml_add(c, ggml_mul(c, dwt, scale), shift);  // [D, T]
    normed = swish(c, normed);
    ggml_tensor* pw2w = weight(c, ml, pre + "conv.pointwise_conv2.weight");
    pw2w = ggml_reshape_2d(c, pw2w, D, D);
    ggml_tensor* pw2b = f32(c, weight(c, ml, pre + "conv.pointwise_conv2.bias"));
    return ggml_add(c, ggml_mul_mat(c, pw2w, normed), pw2b);  // [D, T]
}

// One ConformerEncoderLayer. x_in [D, T], pos_emb [D, T]. Returns [D, T].
ggml_tensor* conformer_layer(ggml_context* c, const HojoModel& m,
                             int li, int64_t T, ggml_tensor* x,
                             ggml_tensor* pos_emb, const BnFold& bn,
                             GraphInputPool& pool) {
    const auto& bc = m.config.bottleneck;
    const ModelLoader& ml = m.loader;
    const int D = bc.output_size, H = bc.attention_heads;
    const int dk = D / H, K = bc.cnn_module_kernel;
    const float eps = (float) bc.norm_eps;
    const std::string p = "bottleneck.blk." + std::to_string(li) + ".";
    // 1. macaron FFN (0.5).
    ggml_tensor* r = x;
    ggml_tensor* n = layer_norm(c, ml, x, p + "norm_ff_macaron", eps);
    ggml_tensor* h = linear_b(c, ml, n, p + "ffn_macaron.w_1", true);
    h = swish(c, h);
    h = linear_b(c, ml, h, p + "ffn_macaron.w_2", true);
    x = ggml_add(c, f32(c, r), ggml_scale(c, f32(c, h), 0.5f));
    // 2. rel-pos MHA.
    r = x;
    n = layer_norm(c, ml, x, p + "norm_mha", eps);
    ggml_tensor* a = relpos_mha(c, ml, p + "mha.", n, pos_emb, H, dk, (int) T);
    x = ggml_add(c, f32(c, r), f32(c, a));
    // 3. conv module.
    r = x;
    n = layer_norm(c, ml, x, p + "norm_conv", eps);
    ggml_tensor* cv = conv_module(c, ml, p, n, D, (int) T, K, bn, pool);
    x = ggml_add(c, f32(c, r), f32(c, cv));
    // 4. FFN (0.5).
    r = x;
    n = layer_norm(c, ml, x, p + "norm_ff", eps);
    h = linear_b(c, ml, n, p + "ffn.w_1", true);
    h = swish(c, h);
    h = linear_b(c, ml, h, p + "ffn.w_2", true);
    x = ggml_add(c, f32(c, r), ggml_scale(c, f32(c, h), 0.5f));
    // 5. norm_final.
    return layer_norm(c, ml, x, p + "norm_final", eps);
}

} // namespace

bool encode_bottleneck(const HojoModel& model, const TowerOutput& tower,
                       BottleneckOutput& out, std::string& err) {
    ensure_weights_realized(model.loader);
    const auto& bc = model.config.bottleneck;
    if (tower.width != (int64_t) bc.input_size || tower.n_speech <= 0) {
        err = "invalid Hojo tower output for bottleneck";
        return false;
    }
    const int64_t T = tower.n_speech;
    const int64_t in_dim = bc.input_size;    // 2048
    const int64_t D = bc.output_size;        // 2560
    const float xscale = std::sqrt((float) D);

    // Pre-compute the BatchNorm fold params for each layer BEFORE the graph
    // build (read_f32 does device I/O that deadlocks inside run_graph's mutex).
    std::vector<BnFold> bn_folds(bc.num_blocks);
    for (uint32_t li = 0; li < bc.num_blocks; ++li) {
        bn_folds[li] = precompute_bn_fold(
            model.loader, "bottleneck.blk." + std::to_string(li) + ".",
            (int) D, 1e-5f);
    }

    // Build the pos_emb slice [D, T] from the baked pe buffer (pe is [1,5000,D]).
    std::vector<float> pos_emb_host((size_t) T * D, 0.0f);
    {
        ggml_tensor* pe = model.loader.tensor("bottleneck.pos_enc.pe");
        if (!pe || pe->type != GGML_TYPE_F32) {
            err = "Hojo bottleneck pos_enc.pe missing or not F32";
            return false;
        }
        ensure_weights_realized(model.loader);
        std::vector<float> pe_host((size_t) ggml_nelements(pe));
        ggml_backend_tensor_get(pe, pe_host.data(), 0, pe_host.size() * sizeof(float));
        const int64_t pe_T = pe->ne[1];
        for (int64_t t = 0; t < T && t < pe_T; ++t)
            for (int64_t d = 0; d < D; ++d)
                pos_emb_host[(size_t) t * D + d] = pe_host[(size_t) t * D + d];
    }

    std::vector<float> tower_in = tower.data;  // [T, 2048] token-major
    std::vector<float> body_out;
    GraphInputPool pool;
    bool ok = run_graph([&](ggml_context* c) -> ggml_tensor* {
        // LinearNoSubsampling: Linear(2048->2560) + LayerNorm + xscale.
        int64_t tne[2] = {in_dim, T};
        ggml_tensor* tin = graph_input_tensor(c, GGML_TYPE_F32, 2, tne,
            tower_in.data(), tower_in.size() * sizeof(float));
        ggml_tensor* x = linear_b(c, model.loader, tin, "bottleneck.embed.out.0", true);
        x = layer_norm(c, model.loader, x, "bottleneck.embed.out.1", (float) bc.norm_eps);
        x = ggml_scale(c, x, xscale);
        int64_t pne[2] = {D, T};
        ggml_tensor* pos_emb = graph_input_tensor(c, GGML_TYPE_F32, 2, pne,
            pos_emb_host.data(), pos_emb_host.size() * sizeof(float));
        for (uint32_t li = 0; li < bc.num_blocks; ++li)
            x = conformer_layer(c, model, (int) li, T, x, pos_emb, bn_folds[li], pool);
        return layer_norm(c, model.loader, x, "bottleneck.after_norm",
                          (float) bc.norm_eps);
    }, body_out);
    if (!ok) { err = "Hojo bottleneck graph failed"; return false; }

    if (const char* dp = std::getenv("STARLING_HOJO_DUMP_BOTTLENECK")) {
        if (FILE* f = std::fopen(dp, "wb")) {
            std::fwrite(body_out.data(), sizeof(float), body_out.size(), f);
            std::fclose(f);
        }
    }
    out.data = std::move(body_out);
    out.n_tokens = T;
    out.width = D;
    return true;
}
} // namespace starling::ggml::hojo
