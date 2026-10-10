// tokenizer.cpp — parakeet-tdt SentencePiece detokenizer (Phase 1c).
//
// Starling-authored port of parakeet.cpp's tokenizer.cpp:8-40, bit-for-bit.

#include "tokenizer.hpp"

#include <algorithm>
#include <cctype>

namespace starling::ggml::parakeet {

// U+2581 LOWER ONE EIGHTH BLOCK — SentencePiece meta-space marker.
// UTF-8 encoding: 0xE2 0x96 0x81 (3 bytes).
static constexpr unsigned char META_SPACE[3] = { 0xE2, 0x96, 0x81 };
static constexpr size_t META_SPACE_LEN = 3;

// detokenize(); with `owner` set, also the index into `ids` of the token
// each output byte came from.
static std::string detokenize_owned(const std::vector<std::string>& pieces,
                                    const std::vector<int32_t>& ids,
                                    std::vector<size_t>* owner) {
    // Step 1: concatenate the piece strings for each id.
    std::string result;
    std::vector<size_t> from;
    result.reserve(ids.size() * 4);
    for (size_t k = 0; k < ids.size(); ++k) {
        const int32_t id = ids[k];
        if (id >= 0 && (size_t)id < pieces.size()) {
            result += pieces[(size_t)id];
            if (owner) from.resize(result.size(), k);
        }
    }

    // Step 2: replace every occurrence of META_SPACE (▁) with a regular space.
    std::string out;
    out.reserve(result.size());
    if (owner) owner->clear();
    for (size_t i = 0; i < result.size(); ) {
        if (owner) owner->push_back(from[i]);
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
        if (owner) owner->erase(owner->begin());
    }
    return out;
}

std::string detokenize(const std::vector<std::string>& pieces,
                       const std::vector<int32_t>& ids) {
    return detokenize_owned(pieces, ids, nullptr);
}

std::vector<WordFrames> word_frames(const std::vector<std::string>& pieces,
                                    const std::vector<int32_t>& ids,
                                    const std::vector<int32_t>& frames,
                                    const std::vector<int32_t>& durations,
                                    std::string& text) {
    std::vector<size_t> owner;
    text = detokenize_owned(pieces, ids, &owner);
    std::vector<WordFrames> words;
    if (frames.size() != ids.size() || durations.size() != ids.size()) return words;
    // Words split on whitespace, as the stream session's split_words does.
    auto space = [&](size_t i) { return std::isspace((unsigned char)text[i]) != 0; };
    for (size_t i = 0; i < text.size();) {
        if (space(i)) { ++i; continue; }
        WordFrames w;
        w.begin = i;
        w.first_frame = frames[owner[i]];
        w.end_frame = w.first_frame;
        for (; i < text.size() && !space(i); ++i)
            w.end_frame = std::max(w.end_frame, frames[owner[i]] + durations[owner[i]]);
        w.end = i;
        words.push_back(w);
    }
    return words;
}

} // namespace starling::ggml::parakeet
