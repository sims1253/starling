// tokenizer.hpp — qwen3-asr's decoder tokenizer.
//
// Qwen3-ASR-1.7B uses the Qwen byte-level BPE (GPT-2 family) tokenizer; the
// GGUF carries the standard token list/scores/types/merges written by the
// converter, which is exactly what the shared BPE detokenizer reads.
#pragma once

#include "lib/bpe_tokenizer.hpp"

namespace starling::ggml::qwen3 {

using Tokenizer = starling::ggml::lib::BpeTokenizer;

} // namespace starling::ggml::qwen3
