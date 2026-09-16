#pragma once
#include "config.hpp"
#include "lib/qwen_decode.hpp"
#include "runtime/model_loader.hpp"
#include <string>

namespace starling::ggml::ark {
struct ArkModel {
    Config config;
    ModelLoader loader;
    bool load(const char* gguf_path, std::string& err);
    // kSpec patched with this model's suppression list, materialized once on
    // first use (decode_ctx is const, and banned_ids must point into storage
    // that outlives the model).
    mutable lib::QwenDecodeSpec decode_spec;
    mutable bool decode_spec_ready = false;
};
} // namespace starling::ggml::ark
