// subsampling.cpp — FastConformer ConvSubsampling graph build.
//
// Starling-authored port of NeMo's ConvSubsampling (dw_striding). The exact
// data flow mirrors the proven parakeet.cpp reference bit-for-bit:
//
//   1. host-transpose mel (feat-major) -> time-major x [F, T, 1, 1] (input #0)
//   2. stage 1: full Conv2d(1->C, k=3, s=2, p=1) + bias + ReLU -> [F/2, T/2, C, 1]
//   3. stage 2/3: depthwise(k=3,s=2,p=1,groups=C) + cont + bias +
//      pointwise Conv2d(C->C, k=1) + bias + ReLU
//   4. flatten via permute(0,2,1,3) -> [F', C, T', 1] -> reshape_2d(C*F', T')
//   5. length mask (zero padded valid frames >= valid_out before the Linear)
//   6. Linear: mul_mat(pre_encode.out.weight [C*F', d_model]) + bias -> [d_model, T']
//
// xscaling is OFF for parakeet-tdt-0.6b-v3 — handled in the encoder, not here.

#include "subsampling.hpp"

#include "runtime/backend.hpp"  // clone_weight, graph_input_tensor

#include "ggml.h"

#include <cstring>

namespace starling::ggml::parakeet {

ggml_tensor* Subsampling::build_graph(ggml_context* ctx,
                                      const std::vector<float>& mel,
                                      int n_mels, int T,
                                      GraphInputPool& pool,
                                      int& out_Tp, int& out_valid,
                                      int in_valid_frames) const {
    const int C = conv_channels_;
    const int F = n_mels;
    const ModelLoader& ml = ml_;

    // Host-transpose mel to time-major [T, F] in pool-owned storage, then feed
    // as a 4-D graph input [W=feat, H=T, IC=1, N=1] (ggml conv data layout).
    float* x_host = pool.alloc_f32((size_t)F * T);
    for (int t = 0; t < T; ++t)
        for (int f = 0; f < F; ++f)
            x_host[(size_t)t * F + f] = mel[(size_t)f * T + t];

    int64_t x_ne[4] = {F, T, 1, 1};
    ggml_tensor* x = graph_input_tensor(ctx, GGML_TYPE_F32, 4, x_ne,
                                        x_host, (size_t)F * T * sizeof(float));

    // ---- Stage 1: full Conv2d(1 -> C, k=3, s=2, p=1) + bias + ReLU ----
    // conv.0.weight: torch [C,1,3,3] -> ggml ne [3,3,1,C] = [KW,KH,IC,OC].
    ggml_tensor* w0 = clone_weight(ctx, ml, "encoder.pre_encode.conv.0.weight");
    ggml_tensor* b0 = clone_weight(ctx, ml, "encoder.pre_encode.conv.0.bias");
    x = ggml_conv_2d(ctx, w0, x,
                     /*s0*/2, /*s1*/2, /*p0*/1, /*p1*/1, /*d0*/1, /*d1*/1);
    // Add bias broadcast over channels: reshape bias to [1,1,C,1].
    x = ggml_add(ctx, x, ggml_reshape_4d(ctx, b0, 1, 1, C, 1));
    x = ggml_relu(ctx, x);

    // ---- Stages 2 & 3: depthwise(k=3,s=2,p=1,groups=C) + cont + bias +
    //                    pointwise(C->C, k=1) + bias + ReLU ----
    struct StageW { const char* dw_w; const char* dw_b; const char* pw_w; const char* pw_b; };
    const StageW stages[2] = {
        { "encoder.pre_encode.conv.2.weight", "encoder.pre_encode.conv.2.bias",
          "encoder.pre_encode.conv.3.weight", "encoder.pre_encode.conv.3.bias" },
        { "encoder.pre_encode.conv.5.weight", "encoder.pre_encode.conv.5.bias",
          "encoder.pre_encode.conv.6.weight", "encoder.pre_encode.conv.6.bias" },
    };
    for (int si = 0; si < 2; ++si) {
        const StageW& s = stages[si];
        // Depthwise: weight torch [C,1,3,3] -> ggml ne [3,3,1,C].
        // ggml_conv_2d_dw_direct expects a:[KW,KH,1,C], b:[W,H,C,N].
        ggml_tensor* dww = clone_weight(ctx, ml, s.dw_w);
        ggml_tensor* dwb = clone_weight(ctx, ml, s.dw_b);
        x = ggml_conv_2d_dw_direct(ctx, dww, x,
                                   /*s0*/2, /*s1*/2, /*p0*/1, /*p1*/1,
                                   /*d0*/1, /*d1*/1);
        // dw_direct keeps WHCN; make it contiguous so the bias add and the
        // following pointwise conv see a standard layout.
        x = ggml_cont(ctx, x);
        x = ggml_add(ctx, x, ggml_reshape_4d(ctx, dwb, 1, 1, C, 1));

        // Pointwise: weight torch [C,C,1,1] -> ggml ne [1,1,C,C] = [KW,KH,IC,OC].
        ggml_tensor* pww = clone_weight(ctx, ml, s.pw_w);
        ggml_tensor* pwb = clone_weight(ctx, ml, s.pw_b);
        x = ggml_conv_2d(ctx, pww, x,
                         /*s0*/1, /*s1*/1, /*p0*/0, /*p1*/0, /*d0*/1, /*d1*/1);
        x = ggml_add(ctx, x, ggml_reshape_4d(ctx, pwb, 1, 1, C, 1));
        x = ggml_relu(ctx, x);
    }

    // x is now ne [F'=OW, T'=OH, C, 1]. NeMo flatten:
    //   [B,C,T',F'].transpose(1,2).reshape(B,T',C*F')
    // -> per time t, the vector is channel-major: idx = c*F' + f.
    const int Fp = (int)x->ne[0]; // F'
    const int Tp = (int)x->ne[1]; // T'
    // Want contiguous [F', C, T', 1] so flat = t*(C*F') + c*F' + f.
    // current dims (0,1,2,3) = (F', T', C, 1); permute to (F', C, T', 1).
    ggml_tensor* xp = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
    ggml_tensor* flat = ggml_reshape_2d(ctx, xp, (int64_t)C * Fp, Tp); // [C*F', T']

    // --- Length masking (faithful to NeMo MaskedConvSequential) ---
    // The valid output frames never read masked input frames (centred kernel),
    // so we can run the conv stack spatially and zero the flattened conv output
    // at frames >= valid_out before the Linear.
    const int valid_out = valid_out_len(T, in_valid_frames);
    if (valid_out < Tp) {
        float* outmask = pool.alloc_f32(Tp);
        for (int t = 0; t < Tp; ++t) outmask[t] = (t < valid_out) ? 1.0f : 0.0f;
        int64_t mk_ne[2] = {1, Tp};
        ggml_tensor* mask = graph_input_tensor(ctx, GGML_TYPE_F32, 2, mk_ne,
                                outmask, (size_t)Tp * sizeof(float));
        flat = ggml_mul(ctx, flat, mask);
    }

    // ---- Linear out: torch [d_model, C*F'] -> ggml ne [C*F', d_model]. ----
    ggml_tensor* ow = clone_weight(ctx, ml, "encoder.pre_encode.out.weight");
    ggml_tensor* ob = clone_weight(ctx, ml, "encoder.pre_encode.out.bias");
    ggml_tensor* y = ggml_mul_mat(ctx, ow, flat); // [d_model, T']
    y = ggml_add(ctx, y, ob);                     // broadcast bias [d_model] over T'

    out_Tp = Tp;
    out_valid = (valid_out > Tp) ? Tp : valid_out;
    return y; // ne [d_model, T'] contiguous -> row-major [T', d_model].
}

} // namespace starling::ggml::parakeet
