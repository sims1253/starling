// loader.hpp — Voxtral-Mini-4B-Realtime GGUF metadata loader (Phase 1).
//
// Reads every voxtral.* key, enforces positivity/divisibility guards, and
// requires every expected tensor. Phase 2 slots the encoder/decoder graphs
// behind this validated config; nothing here touches a backend.
#pragma once
#include "config.hpp"
#include "runtime/model_loader.hpp"
#include <string>

namespace starling::ggml::voxtral {
struct VoxtralModel {
    Config config;
    ModelLoader loader;
    bool load(const char* gguf_path, std::string& err);
};
} // namespace starling::ggml::voxtral
