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

// A word of detokenize()'s text and the encoder frames it was heard at
// (issue #357): text bytes [begin, end), from the frame of its first token to
// the end (frame + duration) of its last. Words are the whitespace-separated
// runs of the text, so they split exactly as the text does.
struct WordFrames {
    size_t begin = 0, end = 0;
    int32_t first_frame = 0, end_frame = 0;
};

// detokenize(pieces, ids) into `text`, and its words. `frames` and
// `durations` are parallel to `ids` (tdt.hpp TdtTiming).
std::vector<WordFrames> word_frames(const std::vector<std::string>& pieces,
                                    const std::vector<int32_t>& ids,
                                    const std::vector<int32_t>& frames,
                                    const std::vector<int32_t>& durations,
                                    std::string& text);

} // namespace starling::ggml::parakeet
