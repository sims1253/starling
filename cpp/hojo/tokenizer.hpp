#pragma once
#include "runtime/model_loader.hpp"
#include "config.hpp"
#include "lib/bpe_tokenizer.hpp"
namespace starling::ggml::hojo {
// Qwen3 BPE byte-decoder: the shared lib::BpeTokenizer (this model's
// tokenizer is a standard Qwen3 BPE table in the GGUF; no hojo-specific
// behavior).
using Tokenizer = lib::BpeTokenizer;
} // namespace starling::ggml::hojo
