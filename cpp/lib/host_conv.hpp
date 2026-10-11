// host_conv.hpp — the Whisper-style K=3 Conv1d front-end on the host (ark,
// voxtral CPU paths), blocked, vectorized and threaded.
//
// Numerics are those of the scalar reference loop it replaces, bit for bit:
//   acc = (double) bias[oc];
//   for c in [0, IC): for k in [0, 3): acc += (double) w[oc,c,k] * (double) xpad[c, t*s+k];
//   y[oc, t] = (float) acc;
// Every output keeps that exact sequential summation order (the blocking
// only changes WHICH outputs are computed together, never the order of the
// terms within one), and the products are exact in f64 (bf16/f32 operands
// have <= 24-bit significands), so fused multiply-add and separate
// multiply+add round identically. Zero padding is materialized as explicit
// +0 terms, as in the reference. Results do not depend on the thread count.
#pragma once

#include "ggml.h"
#include <cstdint>
#include <functional>
#include <vector>

namespace starling::ggml::lib {

// Conv1d (kernel 3, stride `stride`) over a feat-major [IC, L] bf16 input
// (element (c, t) at c*L + t), zero-padded with `pad_l` columns before and
// `pad_r` after. `w` is [OC, IC, 3] f32 (HF order), `bias` [OC] f32. Writes
// the oc-contiguous [OC, OL] output (element (oc, t) at oc*OL + t),
// OL = (L + pad_l + pad_r - 3) / stride + 1, then applies `post` to each
// finished run of one output row inside the worker threads (nullptr: none).
// Returns OL.
using ConvPost = std::function<void(float* row, int64_t n)>;
int64_t host_conv1d_k3(const float* w, const float* bias, int64_t OC, int64_t IC,
                       const ggml_bf16_t* x, int64_t L, int64_t pad_l, int64_t pad_r,
                       int64_t stride, std::vector<float>& y, int n_threads,
                       const ConvPost& post = nullptr);

} // namespace starling::ggml::lib
