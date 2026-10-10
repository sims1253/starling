// Word timestamps from the Parakeet TDT decode (issue #357): word_frames()
// splits detokenize()'s text into words exactly as the stream session does
// and times each one from its first token's frame to its last token's end.
#include "parakeet/tokenizer.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace starling::ggml::parakeet;

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}
}  // namespace

int main() {
    // A SentencePiece fixture vocabulary; id 8 is the blank (vocab size).
    const std::string sp = "\xE2\x96\x81";  // U+2581, the meta-space
    const std::vector<std::string> pieces = {
        sp + "He", sp + "has", sp + "gra", "ve", ",", sp + "doubt", "s", ".",
    };
    const int32_t blank = 8;
    // "He has grave, doubts." with blanks between tokens.
    const std::vector<int32_t> ids = {blank, 0, blank, 1, 2, blank, 3, 4, blank, 5, 6, 7, blank};
    const std::vector<int32_t> frames = {0, 4, 6, 8, 11, 13, 13, 15, 16, 20, 23, 24, 25};
    const std::vector<int32_t> durations = {4, 2, 2, 3, 2, 0, 2, 1, 4, 3, 1, 1, 1};
    std::string text;
    const auto words = word_frames(pieces, ids, frames, durations, text);
    check(text == detokenize(pieces, ids), "the text is detokenize()'s");
    check(text == "He has grave, doubts.", "fixture text");
    const std::vector<std::string> expect_words = {"He", "has", "grave,", "doubts."};
    const std::vector<std::pair<int32_t, int32_t>> expect_frames = {
        {4, 6}, {8, 11}, {11, 16}, {20, 25}};
    check(words.size() == expect_words.size(), "one entry per word");
    for (size_t i = 0; i < words.size() && i < expect_words.size(); ++i) {
        check(text.substr(words[i].begin, words[i].end - words[i].begin) == expect_words[i],
              "word bytes");
        check(words[i].first_frame == expect_frames[i].first, "word starts at its first token");
        check(words[i].end_frame == expect_frames[i].second,
              "word ends at its last token's frame + duration");
    }

    // No speech: empty text, no words.
    const std::vector<int32_t> blanks = {blank, blank};
    check(word_frames(pieces, blanks, {0, 1}, {1, 1}, text).empty() && text.empty(),
          "blank decode has no words");
    // Timing that does not cover the ids gives no words (the caller then
    // has text without times).
    check(word_frames(pieces, ids, {0}, {1}, text).empty() && !text.empty(),
          "mismatched timing gives no words");
    // A piece that is only the meta-space separates words without one.
    const std::vector<std::string> spaced = {sp, "a", "b"};
    const auto split = word_frames(spaced, {1, 0, 2}, {0, 1, 2}, {1, 1, 1}, text);
    check(text == "a b" && split.size() == 2 && split[1].first_frame == 2,
          "a lone meta-space splits words");

    if (failures) {
        std::fprintf(stderr, "PARAKEET WORD FRAMES FAILED (%d)\n", failures);
        return 1;
    }
    std::puts("PARAKEET WORD FRAMES OK");
    return 0;
}
