#include "moss_preview_copy.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    using starling::bench::moss_preview::PreviewCopyDrafter;
    using starling::bench::moss_preview::normalize_preview;
    if (normalize_preview("  HELLO,\tWORLD!\n") != "hello, world!" ||
        normalize_preview("A  B\r\nC") != "a b c") {
        std::fprintf(stderr, "preview normalization failed\n");
        return 1;
    }

    PreviewCopyDrafter copy({1, 2, 3, 4}, 4);
    if (copy.propose({}, 2) != std::vector<int32_t>{1, 2}) return 2;
    // Only the verified prefix is observed. The next source span advances
    // after an accepted proposal; a mismatch reduces K but stays bounded.
    if (copy.propose({1, 2}, 2) != std::vector<int32_t>{3, 4}) return 3;
    if (copy.propose({1, 2, 3, 9}, 2) != std::vector<int32_t>{4}) return 4;
    if (!copy.propose({1}, 2).empty()) return 5; // prefix rollback fails closed
    if (copy.propose({1, 2}, 2) != std::vector<int32_t>{3, 4}) return 6;

    PreviewCopyDrafter adaptive({1, 2, 3, 4, 5, 6}, 4);
    if (adaptive.propose({}, 4) != std::vector<int32_t>{1, 2}) return 7;
    if (adaptive.propose({1, 2}, 4) != std::vector<int32_t>{3, 4, 5}) return 8;
    if (adaptive.propose({1, 2, 3, 9}, 4) != std::vector<int32_t>{4}) return 9;

    PreviewCopyDrafter repeated({1, 2, 3}, 4);
    if (repeated.propose({10, 11, 10, 11}, 2) !=
        std::vector<int32_t>{10, 11}) return 10;
}
