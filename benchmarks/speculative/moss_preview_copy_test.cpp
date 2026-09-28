#include "moss_preview_copy.hpp"

#include <cassert>
#include <vector>

int main() {
    using starling::bench::moss_preview::PreviewCopyDrafter;
    using starling::bench::moss_preview::normalize_preview;
    assert(normalize_preview("  HELLO,\tWORLD!\n") == "hello, world!");
    assert(normalize_preview("A  B\r\nC") == "a b c");

    PreviewCopyDrafter copy({1, 2, 3, 4}, 4);
    assert((copy.propose({}, 2) == std::vector<int32_t>{1, 2}));
    // Only the verified prefix is observed. The next source span advances
    // after an accepted proposal; a mismatch reduces K but stays bounded.
    assert((copy.propose({1, 2}, 2) == std::vector<int32_t>{3, 4}));
    assert((copy.propose({1, 2, 3, 9}, 2) == std::vector<int32_t>{4}));
    assert(copy.propose({1}, 2).empty()); // prefix rollback fails closed
}
