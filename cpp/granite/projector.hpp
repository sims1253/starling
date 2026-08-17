#pragma once
#include "loader.hpp"
#include "ggml.h"
#include <cstdint>

struct ggml_context;
struct ggml_tensor;

namespace starling::ggml::granite {

// Host-side constant scratch for one encoder graph (zero padding, block mask,
// the ones vector for the BatchNorm invstd). Built per input length; the
// buffers back f32 graph inputs and must outlive the compute.
struct EncScratch {
    std::vector<float> zeros7;    // [conv_inner * 7] depthwise left+right pad
    std::vector<float> zeros_q;   // [hidden * pad]      attention q pad rows
    std::vector<float> zeros_kv;  // [hidden*2 * pad]    attention kv pad rows
    std::vector<float> blk_mask;  // [CS*CS*nblk] additive mask (last block only)
    std::vector<float> ones_bn;   // [conv_inner] 1.0 for invstd = 1/sqrt(var+eps)
    std::vector<float> eps_bn;    // [conv_inner] BatchNorm eps broadcast
    std::vector<float> zeros_proj;// [proj.hidden * pad15] projector window pad
    int64_t pad = 0;              // attention pad (context_size remainder)
    int64_t nblocks = 0;          // attention blocks (nblk)
};

// BLIP2 Q-Former projector: pad `enc` [hidden, T] to a multiple of
// window_size, window into [hidden, window, nblk], run the 2 BERT-style
// qformer layers over the 3 learned queries (self-attn + cross-attn to the
// windows + erf-GELU FF), flatten to [hidden, nblk*num_queries] and project to
// the decoder width. Returns [output_dim, N] bf16; sets *N_out.
ggml_tensor* build_projector(ggml_context* c, const GraniteModel& m,
                             ggml_tensor* enc, const EncScratch& s);

} // namespace starling::ggml::granite
