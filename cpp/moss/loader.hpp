#pragma once
#include "config.hpp"
#include "runtime/model_loader.hpp"
#include <string>

namespace starling::ggml::moss {
struct MossModel {
    Config config;
    ModelLoader loader;
    bool load(const char* gguf_path, std::string& err);
};
} // namespace starling::ggml::moss
