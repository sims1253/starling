// conformer.cpp — FastConformer encoder layer graph build.
//
// Starling-authored port of NeMo's ConformerLayer + ConformerConvolution. The
// numerics are load-bearing for byte-exactness vs the encoder golden; every
// detail below mirrors the proven parakeet.cpp CPU reference exactly:
//
//   - macaron order FFN1(0.5) -> attn -> conv -> FFN2(0.5) -> norm_out
//   - LayerNorm eps 1e-5 over ne0 (channel dim)
//   - conv module: pointwise1 (cast F32->F16, mul_mat) -> GLU -> depthwise
//     (transpose -> reshape -> conv_2d_dw_direct, p=(k-1)/2) -> BN fold (host
//     scale/shift) -> SiLU -> pointwise2 (cast F32->F16, mul_mat)
//   - batch_norm fold: scale = g/sqrt(var+1e-5), shift = b - mean*scale
//   - FFN biases are absent for parakeet-tdt-0.6b-v3 (clone_weight_opt + skip)

#include "conformer.hpp"

#include "runtime/backend.hpp"  // clone_weight, clone_weight_opt, graph_input_tensor, weight_to_host_f32

#include "ggml.h"

#include "ggml-backend.h"  // ggml_backend_tensor_get (D2H, works for host + device tensors)
#include "runtime/graph.hpp"    // global_backend

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace starling::ggml::parakeet {

namespace {

ggml_tensor* clone_weight_s(ggml_context* ctx, const ModelLoader& ml,
                            const std::string& name) {
    return clone_weight(ctx, ml, name.c_str());
}
ggml_tensor* clone_weight_opt_s(ggml_context* ctx, const ModelLoader& ml,
                                const std::string& name) {
    return clone_weight_opt(ctx, ml, name.c_str());
}

// Build the ConformerConvolution sub-graph (everything AFTER norm_conv) on the
// conv input `c` (= norm_conv(residual)), ne [D, T]. Returns the conv output
// [D, T]. Mirrors parakeet.cpp conformer.cpp:178-289 (scalar B=1 path) exactly.
ggml_tensor* build_conv_module(ggml_context* ctx, const ModelLoader& ml,
                               const std::string& pre, ggml_tensor* c,
                               int D, int T, int K, int valid_len,
                               const std::string& conv_norm_type,
                               GraphInputPool& pool) {
    const float bn_eps = 1e-5f;
    const int pad = (K - 1) / 2;  // symmetric padding (offline model)

    // -- pointwise_conv1 (Conv1d d->2d, k=1): 1x1 conv == linear over channels.
    ggml_tensor* pw1w = clone_weight_s(ctx, ml, pre + "conv.pointwise_conv1.weight");
    // The GGUF stores conv weights F16 already; a same-dtype cast would emit a
    // full-tensor CPY node per pass. Cast only when actually needed.
    if (pw1w->type != GGML_TYPE_F16) pw1w = ggml_cast(ctx, pw1w, GGML_TYPE_F16);
    pw1w = ggml_reshape_2d(ctx, pw1w, D, 2 * D);  // [in=d, out=2d]
    ggml_tensor* pw1b = clone_weight_opt_s(ctx, ml, pre + "conv.pointwise_conv1.bias");
    ggml_tensor* y = ggml_mul_mat(ctx, pw1w, c);  // [2d, T]
    if (pw1b) y = ggml_add(ctx, y, pw1b);

    // -- GLU over channel dim (NeMo F.glu(x, dim=1)). The two views share y's
    // row stride; SIGMOID/MUL consume strided rows directly, so no cont()
    // materialization is needed (2 fewer dispatches per layer).
    ggml_tensor* a = ggml_view_2d(ctx, y, D, T, y->nb[1], 0);
    ggml_tensor* b = ggml_view_2d(ctx, y, D, T, y->nb[1], (size_t)D * y->nb[0]);
    ggml_tensor* glu = ggml_mul(ctx, a, ggml_sigmoid(ctx, b));  // [d, T]

    // -- pad_mask: zero padded time positions before depthwise conv.
    if (valid_len < T) {
        float* md = pool.alloc_f32(T);
        for (int t = 0; t < T; ++t) md[t] = (t < valid_len) ? 1.0f : 0.0f;
        int64_t tm_ne[2] = {1, T};
        ggml_tensor* tmask = graph_input_tensor(ctx, GGML_TYPE_F32, 2, tm_ne,
                                 md, (size_t)T * sizeof(float));
        glu = ggml_mul(ctx, glu, tmask);
    }

    // -- depthwise_conv (Conv1d d->d, k=K, groups=d).
    // CPU keeps the reference whcn path: the CPU cwhn kernel asserts a kernel
    // stride layout our permuted view does not satisfy, and CPU has no
    // dispatch tax to save. GPU takes the channels-first path below.
    if (!global_backend().is_gpu()) {
        ggml_tensor* glu_tc = ggml_cont(ctx, ggml_transpose(ctx, glu));  // [T, C]
        ggml_tensor* dww_c = clone_weight_s(ctx, ml, pre + "conv.depthwise_conv.weight");
        dww_c = ggml_reshape_4d(ctx, dww_c, K, 1, 1, D);  // [K,1,1,C]
        ggml_tensor* nb_c = ggml_reshape_4d(ctx, glu_tc,
                               glu_tc->ne[0], 1, glu_tc->ne[1], 1);  // [T,1,C,1]
        ggml_tensor* rc = ggml_conv_2d_dw_direct(ctx, dww_c, nb_c,
                                                 1, 1, pad, 0, 1, 1);
        rc = ggml_reshape_2d(ctx, rc, T, D);
        ggml_tensor* dwt_c = ggml_cont(ctx, ggml_transpose(ctx, rc));  // [C, T]
        ggml_tensor* dwb_c = clone_weight_opt_s(ctx, ml, pre + "conv.depthwise_conv.bias");
        if (dwb_c) dwt_c = ggml_add(ctx, dwt_c, dwb_c);
        // batch_norm fold (same host math as the GPU path below).
        float* sc_c = pool.alloc_f32(D);
        float* sh_c = pool.alloc_f32(D);
        {
            auto read_f32 = [&](const std::string& nm, std::vector<float>& dstv) {
                ggml_tensor* t = ml.tensor(nm.c_str());
                if (!t) { dstv.clear(); return; }
                size_t n = (size_t)ggml_nelements(t);
                dstv.resize(n);
                ggml_backend_tensor_get(t, dstv.data(), 0, n * sizeof(float));
            };
            std::vector<float> g, bb, m, var;
            read_f32(pre + "conv.batch_norm.weight", g);
            read_f32(pre + "conv.batch_norm.bias", bb);
            read_f32(pre + "conv.batch_norm.running_mean", m);
            read_f32(pre + "conv.batch_norm.running_var", var);
            for (int cc = 0; cc < D; ++cc) {
                sc_c[cc] = g[cc] / std::sqrt(var[cc] + bn_eps);
                sh_c[cc] = bb[cc] - m[cc] * sc_c[cc];
            }
        }
        int64_t dn[1] = {D};
        ggml_tensor* sc_t = graph_input_tensor(ctx, GGML_TYPE_F32, 1, dn, sc_c, (size_t)D * sizeof(float));
        ggml_tensor* sh_t = graph_input_tensor(ctx, GGML_TYPE_F32, 1, dn, sh_c, (size_t)D * sizeof(float));
        ggml_tensor* normed_c = ggml_add(ctx, ggml_mul(ctx, dwt_c, sc_t), sh_t);
        normed_c = ggml_silu(ctx, normed_c);
        ggml_tensor* pw2_c = clone_weight_s(ctx, ml, pre + "conv.pointwise_conv2.weight");
        if (pw2_c->type != GGML_TYPE_F16) pw2_c = ggml_cast(ctx, pw2_c, GGML_TYPE_F16);
        pw2_c = ggml_reshape_2d(ctx, pw2_c, D, D);
        ggml_tensor* cout_c = ggml_mul_mat(ctx, pw2_c, normed_c);
        ggml_tensor* pw2b_c = clone_weight_opt_s(ctx, ml, pre + "conv.pointwise_conv2.bias");
        if (pw2b_c) cout_c = ggml_add(ctx, cout_c, pw2b_c);
        return cout_c;  // [D, T]
    }
    // (GPU) channels-first (cwhn) path.
    // ggml_conv_2d_dw_direct natively consumes a cwhn input ([W=T,H=1,C,1]
    // with C contiguous, built by free reshape+permute views of glu [C,T])
    // and produces a cwhn result viewable as [C,T] directly — no transpose,
    // no cont, on either the input or the output side. The kernel must be
    // c-fastest ([C per tap]); built once per layer and cached.
    ggml_tensor* dww_t = nullptr;
    {
        auto& cache = ml.cache<DwwTransposedCache>();
        if (!cache) cache = std::make_unique<DwwTransposedCache>();
        // layer index from the caller's pre string "encoder.layers.N."
        const char* ls = pre.c_str() + sizeof("encoder.layers.") - 1;
        int li = 0;
        while (*ls >= '0' && *ls <= '9') li = li * 10 + (*ls++ - '0');
        if (cache->by_layer.size() <= (size_t)li)
            cache->by_layer.resize(li + 1);
        auto& slot = cache->by_layer[li];
        if (slot.empty()) {
            ggml_tensor* src_w = ml.tensor((pre + "conv.depthwise_conv.weight").c_str());
            if (!src_w) { /* unreachable: clone_weight below asserts */ }
            if (!src_w->buffer) ensure_weights_realized(ml);
            src_w = ml.tensor((pre + "conv.depthwise_conv.weight").c_str());
            const size_t n = (size_t)ggml_nelements(src_w);
            GGML_ASSERT(src_w->type == GGML_TYPE_F16);
            std::vector<ggml_fp16_t> raw(n);
            ggml_backend_tensor_get(src_w, raw.data(), 0, n * sizeof(ggml_fp16_t));
            slot.assign((size_t)K * D, 0);
            // (k, c): src at c*K + k  ->  dst at k*D + c
            for (int c = 0; c < D; ++c)
                for (int k = 0; k < K; ++k)
                    slot[(size_t)k * D + c] = raw[(size_t)c * K + k];
        }
        int64_t dwt_ne[2] = {D, K};  // memory: (c, k) at c + k*D
        dww_t = graph_input_tensor(ctx, GGML_TYPE_F16, 2, dwt_ne,
                                   slot.data(), slot.size() * sizeof(ggml_fp16_t));
        mark_graph_input_persistent(dww_t);
    }
    // ggml_permute(ctx, a, ax0..ax3): ne[ax_i] = a->ne[i] (ax_i = where
    // source dim i LANDS). Kernel [D,K,1,1] -> [K,1,1,D]: D->3, K->0.
    ggml_tensor* knl = ggml_permute(ctx, ggml_reshape_4d(ctx, dww_t, D, K, 1, 1),
                                    3, 0, 1, 2);  // ne [K,1,1,D]; c stride = 1 elem
    // Input [D,1,T,1] -> [T,1,D,1]: D->2, T->0; the H=1 dim lands on axis3 so
    // nb1 picks up the big N stride (is_contiguous_channels wants nb1 > nb0).
    ggml_tensor* nb_in = ggml_permute(
        ctx, ggml_reshape_4d(ctx, glu, D, 1, T, 1), 2, 3, 0, 1);  // [T,1,D,1] cwhn
    ggml_tensor* r = ggml_conv_2d_dw_direct(ctx, knl, nb_in,
                                            /*s0*/1, /*s1*/1, /*p0*/pad, /*p1*/0,
                                            /*d0*/1, /*d1*/1);
    // r is [OW=T, OH=1, C, 1] with cwhn strides: element (c, t) at t*D + c.
    ggml_tensor* dwt = ggml_view_2d(ctx, r, D, T, (size_t)D * sizeof(float), 0);
    ggml_tensor* dwb = clone_weight_opt_s(ctx, ml, pre + "conv.depthwise_conv.bias");
    if (dwb) dwt = ggml_add(ctx, dwt, dwb);                      // broadcast [C] over T

    // -- norm (between depthwise conv and SiLU).
    ggml_tensor* normed;
    if (conv_norm_type == "layer_norm") {
        // Streaming variant (not the parakeet-tdt target; included for parity).
        const float ln_eps = 1e-5f;
        ggml_tensor* g  = clone_weight_s(ctx, ml, pre + "conv.batch_norm.weight");
        ggml_tensor* bb = clone_weight_s(ctx, ml, pre + "conv.batch_norm.bias");
        normed = ggml_norm(ctx, dwt, ln_eps);
        normed = ggml_mul(ctx, normed, g);
        normed = ggml_add(ctx, normed, bb);
    } else {
        // batch_norm (inference): fold into per-channel scale/shift constants:
        //   scale = g / sqrt(var+eps); shift = b - mean*scale. Computed host-side.
        // NOTE: we read the BN params DIRECTLY from the loader's tensor ->data
        // (already realized to a backend buffer at load) instead of calling
        // weight_to_host_f32 — that helper calls ensure_weights_realized ->
        // global_backend(), which locks the SAME global mutex run_graph holds
        // (deadlock). On CPU the loader zero-copies the mmap'd GGUF, so ->data
        // is a valid host pointer. The GGUF stores BN params as F32, but we
        // guard against F16 to be safe.
        float* sc = pool.alloc_f32(D);
        float* sh = pool.alloc_f32(D);
        // Read a weight's f32 contents to host. Uses ggml_backend_tensor_get so
        // it works whether the weight lives in host (CPU backend, mmap'd GGUF)
        // or device (GPU backend, realized + uploaded) memory. A raw ->data
        // memcpy would dereference a device pointer on GPU and segfault.
        // (global_backend() is safe to call here: it only lazy-creates + locks
        // briefly; it does NOT re-enter run_graph's g_backend_mutex because
        // global_backend() is called BEFORE the build lambda runs and the
        // backend already exists by this point.)
        auto read_f32 = [&](const std::string& nm, std::vector<float>& dst) {
            ggml_tensor* t = ml.tensor(nm.c_str());
            if (!t) { dst.clear(); return; }
            size_t n = (size_t)ggml_nelements(t);
            dst.resize(n);
            if (t->type == GGML_TYPE_F32) {
                ggml_backend_tensor_get(t, dst.data(), 0, n * sizeof(float));
            } else if (t->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> raw(n);
                ggml_backend_tensor_get(t, raw.data(), 0, n * sizeof(ggml_fp16_t));
                ggml_fp16_to_fp32_row(raw.data(), dst.data(), n);
            } else {
                GGML_ASSERT(false && "unsupported batch_norm dtype");
            }
        };
        std::vector<float> g, bb, m, var;
        read_f32(pre + "conv.batch_norm.weight", g);
        read_f32(pre + "conv.batch_norm.bias", bb);
        read_f32(pre + "conv.batch_norm.running_mean", m);
        read_f32(pre + "conv.batch_norm.running_var", var);
        for (int cc = 0; cc < D; ++cc) {
            sc[cc] = g[cc] / std::sqrt(var[cc] + bn_eps);
            sh[cc] = bb[cc] - m[cc] * sc[cc];
        }
        int64_t d_ne[1] = {D};
        ggml_tensor* scale = graph_input_tensor(ctx, GGML_TYPE_F32, 1, d_ne,
                                 sc, (size_t)D * sizeof(float));
        ggml_tensor* shift = graph_input_tensor(ctx, GGML_TYPE_F32, 1, d_ne,
                                 sh, (size_t)D * sizeof(float));
        normed = ggml_add(ctx, ggml_mul(ctx, dwt, scale), shift);  // [C, T]
    }

    // -- SiLU (Swish), then pointwise_conv2 (Conv1d d->d, k=1).
    normed = ggml_silu(ctx, normed);
    ggml_tensor* pw2w = clone_weight_s(ctx, ml, pre + "conv.pointwise_conv2.weight");
    if (pw2w->type != GGML_TYPE_F16) pw2w = ggml_cast(ctx, pw2w, GGML_TYPE_F16);
    pw2w = ggml_reshape_2d(ctx, pw2w, D, D);  // [in=d, out=d]
    ggml_tensor* pw2b = clone_weight_opt_s(ctx, ml, pre + "conv.pointwise_conv2.bias");
    ggml_tensor* cout = ggml_mul_mat(ctx, pw2w, normed);  // [d, T]
    if (pw2b) cout = ggml_add(ctx, cout, pw2b);
    return cout;  // [D, T]
}

} // namespace

ggml_tensor* ConformerLayer::build_graph(ggml_context* ctx, ggml_tensor* xt,
                                         int T, ggml_tensor* pe, int pos_len,
                                         int valid_len,
                                         GraphInputPool& pool,
                                         ggml_tensor* ph) const {
    const int D = d_model_;
    const int K = conv_kernel_;
    const float ln_eps = 1e-5f;  // LayerNorm eps (NeMo nn.LayerNorm default)

    const std::string pre = "encoder.layers." + std::to_string(layer_idx_) + ".";
    const ModelLoader& ml = ml_;

    // LayerNorm over the channel dim (ne0 = D), affine. Input ne [D, T].
    auto layer_norm = [&](ggml_tensor* in, const std::string& nm) {
        ggml_tensor* g = clone_weight_s(ctx, ml, pre + nm + ".weight");  // [D]
        ggml_tensor* b = clone_weight_s(ctx, ml, pre + nm + ".bias");    // [D]
        ggml_tensor* y = ggml_norm(ctx, in, ln_eps);                     // normalize over ne0
        y = ggml_mul(ctx, y, g);                                         // broadcast [D] over T
        y = ggml_add(ctx, y, b);
        return y;
    };
    // nn.Linear: ggml weight ne = [in, out]. in ne [in, T] -> [out, T].
    // FFN biases are ABSENT for parakeet-tdt-0.6b-v3 (clone_weight_opt + skip).
    auto linear = [&](ggml_tensor* in, const std::string& nm, bool bias) {
        ggml_tensor* W = clone_weight_s(ctx, ml, pre + nm + ".weight");
        ggml_tensor* y = ggml_mul_mat(ctx, W, in);
        if (bias) {
            ggml_tensor* B = clone_weight_opt_s(ctx, ml, pre + nm + ".bias");
            if (B) y = ggml_add(ctx, y, B);
        }
        return y;
    };
    // ConformerFeedForward: linear1(d->ff) -> SiLU -> linear2(ff->d). in [D, T].
    auto feed_forward = [&](ggml_tensor* in, const std::string& ff) {
        ggml_tensor* h = linear(in, ff + ".linear1", /*bias*/ true);  // [FF, T]
        h = ggml_silu(ctx, h);                                         // Swish == SiLU
        h = linear(h, ff + ".linear2", /*bias*/ true);                 // [D, T]
        return h;
    };

    // === Stage A: r = x + 0.5 * FFN1(norm_ff1(x)). ===
    ggml_tensor* h1 = layer_norm(xt, "norm_feed_forward1");
    h1 = feed_forward(h1, "feed_forward1");
    h1 = ggml_scale(ctx, h1, 0.5f);          // fc_factor
    ggml_tensor* r = ggml_add(ctx, xt, h1);  // [D, T]

    // === Stage B: r = r + self_attn(norm_self_att(r)). ===
    ggml_tensor* attn_in = layer_norm(r, "norm_self_att");
    ggml_tensor* attn_out = attn_.build_graph(ctx, attn_in, T, pe, pos_len,
                                              valid_len, pool, ph);  // [D, T]
    r = ggml_add(ctx, r, attn_out);

    // === Stage C: r = r + conv(norm_conv(r)). ===
    ggml_tensor* c = layer_norm(r, "norm_conv");  // [D, T]
    ggml_tensor* conv_out = build_conv_module(ctx, ml, pre, c, D, T, K,
                                              valid_len, conv_norm_type_, pool);
    r = ggml_add(ctx, r, conv_out);

    // === Stage D: r = r + 0.5 * FFN2(norm_ff2(r)); out = norm_out(r). ===
    ggml_tensor* h2 = layer_norm(r, "norm_feed_forward2");
    h2 = feed_forward(h2, "feed_forward2");
    h2 = ggml_scale(ctx, h2, 0.5f);
    r = ggml_add(ctx, r, h2);
    r = layer_norm(r, "norm_out");
    return r;  // [D, T]
}

} // namespace starling::ggml::parakeet
