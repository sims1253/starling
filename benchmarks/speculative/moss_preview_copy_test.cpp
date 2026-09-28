#include "moss_preview_copy.hpp"

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
}
