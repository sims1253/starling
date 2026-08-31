// tokenizer.hpp — audex's decoder tokenizer.
//
// Nemotron-Labs-Audex-2B uses a byte-level BPE (GPT-2 family) tokenizer
// (vocab 205312 incl. 74733 added tokens); the GGUF carries the standard
// token list/scores/types/merges written by the converter, which is exactly
// what the shared BPE detokenizer reads.
#pragma once

#include "lib/bpe_tokenizer.hpp"

namespace starling::ggml::audex {

using Tokenizer = starling::ggml::lib::BpeTokenizer;

} // namespace starling::ggml::audex
