// relpos_attention.hpp — Transformer-XL relative-position multi-head self-
// attention (NeMo RelPositionMultiHeadAttention) for the FastConformer encoder.
//
// Starling-authored port. The graph build mirrors the proven parakeet.cpp CPU
// reference bit-for-bit:
//
//   q/k/v = linear_{q,k,v}(x)            # x is norm_self_att(r), no pos bias
//   p     = linear_pos(pe)               # pos applied to the pe table
//   heads split: [D, n] -> [dk, n, H]
//   qu = qh + pos_bias_u, qv = qh + pos_bias_v
//   bd = p^T @ qv                         # [P, T, H]
//   bd = rel_shift_skew(bd)               # -> [T_k, T_q, H]  (load-bearing)
//   scores = (k^T @ qu) + bd              # [T_k, T_q, H]
//   attn = soft_max_ext(scores, scale)    # mask=null for full attention
//   ctx  = (cont(permute(v, 1,0,2,3)))^T @ attn
//   merged = cont(permute(ctx, 0,2,1,3)) reshaped to [D, T]
//   out = linear_out(merged)              # no bias if absent
//
// Layout convention (matches the rest of the encoder):
//   xt   row-major [T, d_model]      (ne0=D fastest)
//   pe   row-major [2T-1, d_model]
//   out  row-major [T, d_model]
//
// valid_len is the number of non-padding frames. For the validation fixtures
// valid_len == T (no padding), so the mask is trivial and omitted.

#pragma once

#include "config.hpp"
#include "runtime/graph_builder.hpp"
#include "runtime/model_loader.hpp"

struct ggml_context;
struct ggml_tensor;

namespace starling::ggml::parakeet {

class RelPosAttention {
public:
    RelPosAttention(const ModelLoader& ml, const Config& cfg, int layer_idx)
        : ml_(ml),
          layer_idx_(layer_idx),
          d_model_((int)cfg.d_model),
          n_heads_((int)cfg.n_heads),
          d_head_((int)cfg.d_model / (int)cfg.n_heads) {}

    // GRAPH-BUILDER: append MHSA ops to a SHARED graph `ctx`. xt is the
    // normalized attention input [D, T], pe is the positional-encoding tensor
    // [D, 2T-1]. Returns the attention output [D, T]. Host masks (omitted when
    // trivial) are fed via graph_input_tensor and registered into `pool` (must
    // outlive the compute). CPU path is the byte-identical reference.
    ggml_tensor* build_graph(ggml_context* ctx, ggml_tensor* xt, int T,
                             ggml_tensor* pe, int pos_len, int valid_len,
                             GraphInputPool& pool) const;

private:
    const ModelLoader& ml_;
    int layer_idx_;
    int d_model_;
    int n_heads_;
    int d_head_;
};

} // namespace starling::ggml::parakeet
