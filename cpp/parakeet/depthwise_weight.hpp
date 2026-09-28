#pragma once

#include "runtime/backend.hpp"
#include "runtime/graph.hpp"

#include "ggml.h"

#include <cstring>

namespace starling::ggml::parakeet {

// The pinned ggml-cuda depthwise kernel accepts F32 weights only.
inline ggml_tensor* depthwise_weight_for_backend(ggml_context* ctx, ggml_tensor* weight) {
    if (weight->type == GGML_TYPE_F16 &&
        std::strncmp(global_backend().device_name(), "CUDA", 4) == 0)
        return ggml_cast(ctx, weight, GGML_TYPE_F32);
    return weight;
}

} // namespace starling::ggml::parakeet
