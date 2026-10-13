// Copyright 2026 The Starling Authors. Apache-2.0.
#pragma once

#include <cstdint>
#include <string>

namespace starling::npu {

// encoder_contract.hpp's model fingerprint of a Qwen3-ASR GGUF file.
bool gguf_fingerprint(const std::string& path, uint64_t& hash, std::string& err);

}  // namespace starling::npu
