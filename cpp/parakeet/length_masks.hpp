// length_masks.hpp — per-call length masks of a T-bucketed encoder graph.
//
// The GPU encoder caches one replay graph per mel-length bucket (encoder.cpp)
// and runs shorter inputs zero-padded to the bucket. These graph inputs carry
// the call's true lengths so the padded frames never reach the real ones:
// every cross-frame op (the stride-2 subsampling convs, attention keys, the
// depthwise conv) sees exactly what the unpadded graph sees. Their host data
// is rewritten before every replay; a null tensor means "build the static
// masks from valid_len" (CPU path).
#pragma once

struct ggml_tensor;

namespace starling::ggml::parakeet {

struct LengthMasks {
    ggml_tensor* stage1 = nullptr;  // [1, T1]: 1 for subsampling stage-1 frames of the true input
    ggml_tensor* stage2 = nullptr;  // [1, T2]: same after stage 2
    ggml_tensor* time = nullptr;    // [1, T']: 1 for encoder frames < valid_len
    ggml_tensor* key = nullptr;     // [T', T']: 0 for keys < valid_len, -inf beyond
};

} // namespace starling::ggml::parakeet
