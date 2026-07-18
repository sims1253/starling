// relpos_attention.cpp — FastConformer RelPositionMultiHeadAttention graph build.
//
// Starling-authored port of NeMo's RelPositionMultiHeadAttention. The numerics
// are load-bearing for byte-exactness vs the encoder golden; every detail below
// mirrors the proven parakeet.cpp CPU reference exactly (the bd skew, the
// head-split reshape/permute order, the v^T transpose, the soft_max_ext scale).
//
// The CPU path is the byte-identical reference. The GPU flash-attn path is NOT
// emitted here — Phase 1b targets CPU validation; the GPU perf path lands later.

#include "relpos_attention.hpp"

#include "runtime/backend.hpp"  // clone_weight, clone_weight_opt, graph_input_tensor

#include "ggml.h"

#include <cmath>
#include <string>

namespace starling::ggml::parakeet {

namespace {
// std::string overload of clone_weight so the pre+suffix concatenation reads
// cleanly at each call site.
ggml_tensor* clone_weight_s(ggml_context* ctx, const ModelLoader& ml,
                            const std::string& name) {
    return clone_weight(ctx, ml, name.c_str());
}
} // namespace

ggml_tensor* RelPosAttention::build_graph(ggml_context* ctx, ggml_tensor* xt,
                                          int T, ggml_tensor* pe, int pos_len,
                                          int valid_len,
                                          GraphInputPool& pool) const {
    const int D  = d_model_;
    const int H  = n_heads_;
    const int dk = d_head_;
    const float scale = 1.0f / std::sqrt((float)dk);
    (void)pos_len;  // pos_len == 2T-1 asserted by caller; not used otherwise.

    const std::string pre = "encoder.layers." + std::to_string(layer_idx_) +
                            ".self_attn.";
    const ModelLoader& ml = ml_;

    // ---- linear projections (nn.Linear: ggml W ne=[in,out]) ----
    // parakeet-tdt-0.6b-v3 has NO bias on the attention linears — clone_weight
    // asserts presence, so only the weight is referenced and no add is emitted.
    auto linear = [&](const char* w, const char* b, ggml_tensor* in) {
        ggml_tensor* W = clone_weight_s(ctx, ml, pre + w);
        ggml_tensor* y = ggml_mul_mat(ctx, W, in);  // [out, *]
        if (b && ml.tensor((pre + b).c_str())) {
            ggml_tensor* B = clone_weight_s(ctx, ml, pre + b);
            y = ggml_add(ctx, y, B);  // broadcast [out] over cols
        }
        return y;
    };
    ggml_tensor* q = linear("linear_q.weight", "linear_q.bias", xt);  // [D, T]
    ggml_tensor* k = linear("linear_k.weight", "linear_k.bias", xt);  // [D, T]
    ggml_tensor* v = linear("linear_v.weight", "linear_v.bias", xt);  // [D, T]
    ggml_tensor* p = linear("linear_pos.weight", nullptr, pe);        // [D, P]

    // ---- split into heads: [D, *] -> [dk, H, *] -> [dk, *, H] ----
    auto to_heads = [&](ggml_tensor* t, int n) {
        t = ggml_reshape_3d(ctx, t, dk, H, n);                  // [dk, H, n]
        t = ggml_cont(ctx, ggml_permute(ctx, t, 0, 2, 1, 3));   // [dk, n, H]
        return t;
    };
    ggml_tensor* qh = to_heads(q, T);        // [dk, T, H]
    ggml_tensor* kh = to_heads(k, T);        // [dk, T, H]
    ggml_tensor* vh = to_heads(v, T);        // [dk, T, H]
    ggml_tensor* ph = to_heads(p, pos_len);  // [dk, P, H]

    // ---- pos_bias_u/v: ne [dk, H] -> [dk, 1, H] to broadcast over T ----
    ggml_tensor* bu = clone_weight_s(ctx, ml, pre + "pos_bias_u");  // [dk, H]
    ggml_tensor* bv = clone_weight_s(ctx, ml, pre + "pos_bias_v");  // [dk, H]
    bu = ggml_reshape_3d(ctx, bu, dk, 1, H);
    bv = ggml_reshape_3d(ctx, bv, dk, 1, H);
    ggml_tensor* qu = ggml_add(ctx, qh, bu);  // [dk, T, H]
    ggml_tensor* qv = ggml_add(ctx, qh, bv);  // [dk, T, H]

    // ---- bd = p^T @ qv -> [P(pos), T(query), H], then rel_shift -> [T,T,H] ----
    // The skew is load-bearing — mirror parakeet.cpp relpos_attention.cpp:94-104
    // EXACTLY (pad by 1 on dim0, reshape, view with offset nb[1], reshape back,
    // view with offset 0, cont).
    ggml_tensor* bd = ggml_mul_mat(ctx, ph, qv);                     // [P, T, H]
    bd = ggml_pad_ext(ctx, bd, /*lp0*/1, /*rp0*/0, 0, 0, 0, 0, 0, 0); // [P+1=2T, T, H]
    bd = ggml_reshape_3d(ctx, bd, T, 2 * T, H);                       // [T, 2T, H]
    bd = ggml_view_3d(ctx, bd, T, 2 * T - 1, H,
                      bd->nb[1], bd->nb[2], bd->nb[1]);              // [T, 2T-1, H]
    bd = ggml_cont(ctx, bd);
    bd = ggml_reshape_3d(ctx, bd, 2 * T - 1, T, H);                  // [2T-1, T, H]
    bd = ggml_view_3d(ctx, bd, T, T, H, bd->nb[1], bd->nb[2], 0);
    bd = ggml_cont(ctx, bd);                                          // [T_k, T_q, H]

    // ---- manual path (CPU reference, byte-identical) ----
    // ac = q_u @ k^T : ggml_mul_mat([dk,T,H],[dk,T,H]) -> [T_k, T_q, H]
    ggml_tensor* ac = ggml_mul_mat(ctx, kh, qu);            // [T(key), T(query), H]
    ggml_tensor* scores = ggml_add(ctx, ac, bd);            // [T_k, T_q, H]
    // For the offline full-context model on a non-padded single clip, the
    // additive mask is uniformly 0 -> omit it (soft_max_ext with mask=null).
    // When valid_len < T we apply the standard key-validity mask (-inf beyond
    // valid_len) so padded keys get zero attention.
    ggml_tensor* mask = nullptr;
    const bool mask_is_trivial = (valid_len >= T);
    if (!mask_is_trivial) {
        float* mask_host = pool.alloc_f32((size_t)T * T);
        {
            float* md = mask_host;
            const float ninf = -INFINITY;
            for (int qi = 0; qi < T; ++qi) {
                for (int kj = 0; kj < T; ++kj) {
                    md[(size_t)qi * T + kj] = (kj < valid_len) ? 0.0f : ninf;
                }
            }
        }
        int64_t mask_ne[2] = {T, T};
        mask = graph_input_tensor(ctx, GGML_TYPE_F32, 2, mask_ne,
                                  mask_host,
                                  (size_t)T * T * sizeof(float));
    }
    ggml_tensor* attn = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f); // [T_k, T_q, H]
    // context = attn @ v -> [dk, T_q, H]
    ggml_tensor* vtk = ggml_cont(ctx, ggml_permute(ctx, vh, 1, 0, 2, 3)); // [T_k, dk, H]
    ggml_tensor* ctxh = ggml_mul_mat(ctx, vtk, attn);                      // [dk, T_q, H]
    // concat heads: [dk, T, H] -> [dk, H, T] -> [D, T]
    ggml_tensor* merged = ggml_cont(ctx, ggml_permute(ctx, ctxh, 0, 2, 1, 3)); // [dk, H, T]
    merged = ggml_reshape_2d(ctx, merged, D, T);                              // [D, T]

    // Zero the context for PADDED query rows (NeMo masks padded query rows fully
    // -> output reduces to linear_out.bias). Apply a query-row mask [1, T].
    if (valid_len < T) {
        float* qmask_host = pool.alloc_f32(T);
        for (int qi = 0; qi < T; ++qi)
            qmask_host[qi] = (qi < valid_len) ? 1.0f : 0.0f;
        int64_t qm_ne[2] = {1, T};
        ggml_tensor* qmask = graph_input_tensor(ctx, GGML_TYPE_F32, 2, qm_ne,
                                 qmask_host,
                                 (size_t)T * sizeof(float));
        merged = ggml_mul(ctx, merged, qmask);  // broadcast over D
    }

    // ---- output projection (no bias for parakeet-tdt-0.6b-v3) ----
    return linear("linear_out.weight", "linear_out.bias", merged);  // [D, T]
}

} // namespace starling::ggml::parakeet
