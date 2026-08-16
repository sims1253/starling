#pragma once
#include "lib/bpe_tokenizer.hpp"
namespace starling::ggml::ark {
// Qwen2.5 BPE byte-decoder: the shared lib::BpeTokenizer (this model's
// tokenizer is a standard Qwen2.5 BPE table in the GGUF; no ark-specific
// behavior).
using Tokenizer = lib::BpeTokenizer;
} // namespace starling::ggml::ark
