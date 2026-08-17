#pragma once
#include "loader.hpp"
#include "lib/bpe_tokenizer.hpp"
namespace starling::ggml::moss {
// Qwen BPE byte-decoder: the shared lib::BpeTokenizer (standard Qwen BPE
// table in the GGUF; no moss-specific behavior).
using Tokenizer = lib::BpeTokenizer;
}
