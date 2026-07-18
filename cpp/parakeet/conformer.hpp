// conformer.hpp — a single FastConformer encoder layer (NeMo ConformerLayer).
//
// Starling-authored port. The layer mirrors NeMo's ConformerLayer.forward in
// the macaron order used by parakeet-tdt-0.6b-v3:
//
//   r = x
//   r = r + 0.5 * feed_forward1(norm_feed_forward1(r))   # FFN1 (half-step)
//   r = r + self_attn(norm_self_att(r), pos_emb)          # RelPosAttention
//   r = r + conv(norm_conv(r))                            # ConformerConvolution
//   r = r + 0.5 * feed_forward2(norm_feed_forward2(r))    # FFN2 (half-step)
//   out = norm_out(r)                                     # final LN, no residual
//
// FFN: linear1(d->ff) -> SiLU -> linear2(ff->d). The ConformerConvolution
// module (batch_norm variant, the offline default) folds the inference-time BN
// into per-channel scale/shift constants computed host-side, then SiLU then a
// 1x1 Linear.
//
// Layout convention (matches the rest of the encoder):
//   x       row-major [T, d_model]      (ne0=D fastest)
//   pos_emb row-major [2T-1, d_model]
//   out     row-major [T, d_model]
//
// `valid_len` is the number of non-padding frames (frames >= valid_len are
// center-pad). For the validation fixtures valid_len == T (no masking).

#pragma once

#include "config.hpp"
#include "relpos_attention.hpp"
#include "runtime/graph_builder.hpp"
#include "runtime/model_loader.hpp"

#include <string>

struct ggml_context;
struct ggml_tensor;

namespace starling::ggml::parakeet {

class ConformerLayer {
public:
    ConformerLayer(const ModelLoader& ml, const Config& cfg, int layer_idx)
        : ml_(ml),
          attn_(ml, cfg, layer_idx),
          layer_idx_(layer_idx),
          d_model_((int)cfg.d_model),
          n_heads_((int)cfg.n_heads),
          ff_dim_((int)cfg.ff_dim),
          conv_kernel_((int)cfg.conv_kernel),
          conv_norm_type_(cfg.conv_norm_type) {}

    // GRAPH-BUILDER: append the WHOLE conformer layer (FFN1 + MHSA + conv +
    // FFN2 + norm_out) to a SHARED graph `ctx`. xt is the layer input [D, T] and
    // pe is the positional-encoding tensor [D, 2T-1], both ALREADY in the graph.
    // Returns the layer output [D, T]. Host-built BN-fold constants / masks are
    // fed via graph_input_tensor and registered into `pool` (must outlive the
    // compute).
    ggml_tensor* build_graph(ggml_context* ctx, ggml_tensor* xt, int T,
                             ggml_tensor* pe, int pos_len, int valid_len,
                             GraphInputPool& pool) const;

private:
    const ModelLoader& ml_;
    RelPosAttention attn_;
    int layer_idx_;
    int d_model_;
    int n_heads_;
    int ff_dim_;
    int conv_kernel_;
    std::string conv_norm_type_;  // "batch_norm" (offline) for parakeet-tdt
};

} // namespace starling::ggml::parakeet
