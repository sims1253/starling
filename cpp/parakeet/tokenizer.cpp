// tokenizer.cpp — parakeet-tdt SentencePiece detokenizer (Phase 1c).
//
// Starling-authored port of parakeet.cpp's tokenizer.cpp:8-40, bit-for-bit.

#include "tokenizer.hpp"

namespace starling::ggml::parakeet {

// U+2581 LOWER ONE EIGHTH BLOCK — SentencePiece meta-space marker.
// UTF-8 encoding: 0xE2 0x96 0x81 (3 bytes).
static constexpr unsigned char META_SPACE[3] = { 0xE2, 0x96, 0x81 };
static constexpr size_t META_SPACE_LEN = 3;

std::string detokenize(const std::vector<std::string>& pieces,
                       const std::vector<int32_t>& ids) {
    // Step 1: concatenate the piece strings for each id.
    std::string result;
    result.reserve(ids.size() * 4);
    for (int32_t id : ids) {
        if (id >= 0 && (size_t)id < pieces.size()) {
            result += pieces[(size_t)id];
        }
    }

    // Step 2: replace every occurrence of META_SPACE (▁) with a regular space.
    std::string out;
    out.reserve(result.size());
    for (size_t i = 0; i < result.size(); ) {
        if (i + META_SPACE_LEN <= result.size() &&
            (unsigned char)result[i]     == META_SPACE[0] &&
            (unsigned char)result[i + 1] == META_SPACE[1] &&
            (unsigned char)result[i + 2] == META_SPACE[2]) {
            out += ' ';
            i += META_SPACE_LEN;
        } else {
            out += result[i++];
        }
    }

    // Step 3: strip a single leading space (SentencePiece decode_ids behaviour).
    if (!out.empty() && out[0] == ' ') {
        out.erase(0, 1);
    }
    return out;
}

} // namespace starling::ggml::parakeet
