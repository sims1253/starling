// tokenizer.hpp — granite's decoder tokenizer.
//
// granite-speech-4.1-2b uses a byte-level BPE (GPT-2 family) tokenizer; the
// GGUF carries the standard token list/scores/types/merges written by the
// converter, which is exactly what the shared BPE detokenizer reads.
#pragma once

#include "lib/bpe_tokenizer.hpp"

namespace starling::ggml::granite {

using Tokenizer = starling::ggml::lib::BpeTokenizer;

} // namespace starling::ggml::granite
