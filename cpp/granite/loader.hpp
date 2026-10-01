#pragma once
#include "config.hpp"
#include "kv_factors.hpp"
#include "runtime/model_loader.hpp"
#include <string>

namespace starling::ggml::granite {
struct GraniteModel {
    Config config;
    ModelLoader loader;
    KVFactors kv_factors;  // issue #59 research path (STARLING_GRANITE_KVFACT)
    bool load(const char* gguf_path, std::string& err);
};
} // namespace starling::ggml::granite
