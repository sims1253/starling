#include "s1/copy_draft.hpp"

#include <cstdio>
#include <vector>

using starling::ggml::s1::CopyDrafter;

int main() {
    int failed = 0;
    const auto check = [&](bool ok, const char* name) {
        std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", name);
        failed += !ok;
    };

    CopyDrafter realign({10, 99, 11, 12, 13}, 4);
    check(realign.propose({10}, 4) == std::vector<int32_t>({99, 11}),
          "source draft starts after verified token");
    check(realign.propose({10, 11}, 4) == std::vector<int32_t>({12}),
          "rejection skips omitted filler and halves draft length");

    CopyDrafter expanding({1, 2, 3, 4, 5, 6, 7}, 4);
    check(expanding.propose({1}, 4) == std::vector<int32_t>({2, 3}),
          "initial draft is two tokens");
    check(expanding.propose({1, 2, 3, 4}, 4) ==
              std::vector<int32_t>({5, 6, 7}),
          "full acceptance expands draft length to three");

    CopyDrafter partial({1, 2, 3, 4, 5}, 4);
    check(partial.propose({1}, 4) == std::vector<int32_t>({2, 3}) &&
              partial.propose({1, 2, 9}, 4) == std::vector<int32_t>({3}),
          "partial acceptance keeps only verified prefix and shrinks K");

    CopyDrafter repeated({}, 4);
    check(repeated.propose({4, 5, 4}, 4) == std::vector<int32_t>({5, 4}),
          "repeated-prefix fallback reads only verified output");
    CopyDrafter unmatched({}, 4);
    check(unmatched.propose({7}, 4).empty(),
          "unmatched prefix has no fabricated future draft");
    return failed ? 1 : 0;
}
