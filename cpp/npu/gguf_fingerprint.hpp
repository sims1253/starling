// Copyright 2026 The Starling Authors. Apache-2.0.
#pragma once

#include "model_fingerprint.hpp"

#include <string>

namespace starling::npu {

// The model hash (model_fingerprint.hpp) of a Qwen3-ASR GGUF file, read
// tensor by tensor: what the engine computes from the same GGUF once loaded.
bool gguf_model_hash(const std::string& path, uint64_t& hash, std::string& err);

}  // namespace starling::npu
