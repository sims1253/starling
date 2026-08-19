// loader.hpp — S1-mini GGUF loader shell.
#pragma once

#include "config.hpp"
#include "runtime/model_loader.hpp"

#include <string>

namespace starling::ggml::s1 {

struct S1Model {
    Config config;
    ModelLoader loader;
    bool load(const char* gguf_path, std::string& err);
};

} // namespace starling::ggml::s1
