// loader.hpp — audex model bundle over the shared GGUF ModelLoader.
#pragma once

#include "config.hpp"
#include "runtime/model_loader.hpp"

#include <string>

namespace starling::ggml::audex {

struct AudexModel {
    Config config;
    ModelLoader loader;
    bool load(const char* gguf_path, std::string& err);
};

} // namespace starling::ggml::audex
