// Copyright 2026 The Starling Authors. Apache-2.0.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace starling::npu {

enum class LoadMode;

std::string probe_compile(const std::string& tflite, const std::string& artifact, const std::string& function,
                          const std::vector<uint64_t>& in_bytes, const std::vector<uint64_t>& out_bytes,
                          uint64_t pad, bool& ok, const std::string& key = "", uint64_t model_hash = 0);
std::string probe_run(const std::string& artifact, LoadMode mode, const std::vector<std::string>& inputs,
                      const std::string& out_prefix, int runs, bool& ok);

}  // namespace starling::npu
