// tokenizer.hpp — parakeet-tdt SentencePiece detokenizer (Phase 1c).
//
// Starling-authored port of parakeet.cpp's tokenizer.cpp:8-40. Detokenizes the
// greedy-decode id stream (including blanks) into text:
//   1. Concatenate pieces[id] for each id in [0, vocab_size); blank id
//      (vocab_size = 8192) and any out-of-range id contribute nothing.
//   2. Replace every U+2581 (▁, SentencePiece meta-space; UTF-8 0xE2 0x96 0x81)
//      with an ASCII space.
//   3. Strip a single leading space (SentencePiece decode_ids behaviour).

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace starling::ggml::parakeet {

// pieces: the SentencePiece piece strings (config.tokenizer_pieces), indexed by
//         id. ids in [0, pieces.size()) contribute pieces[id]; out-of-range ids
//         (incl. the blank id = pieces.size()) are skipped.
// Returns the detokenized UTF-8 text (no trailing newline).
std::string detokenize(const std::vector<std::string>& pieces,
                       const std::vector<int32_t>& ids);

} // namespace starling::ggml::parakeet
