#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace starling::ggml::granite {

// An alignment-only proposer. `draft` comes from the native CTC head; the
// verifier supplies `prefix` only after target tokens are committed. No target
// logits or future greedy IDs are available to this class.
class CtcProposer {
public:
    explicit CtcProposer(std::vector<int32_t> draft, int max_k)
        : draft_(std::move(draft)), max_k_(std::max(1, max_k)),
          chunk_(std::min(2, max_k_)) {}

    std::vector<int32_t> propose(const std::vector<int32_t>& prefix, int cap) {
        if (prefix.size() < observed_) return {}; // a new decode needs a new proposer
        if (observed_ != 0 && !previous_.empty()) {
            size_t accepted = 0;
            const size_t added = prefix.size() - observed_;
            while (accepted < previous_.size() && accepted < added &&
                   prefix[observed_ + accepted] == previous_[accepted])
                ++accepted;
            pos_ += accepted;
            if (accepted == previous_.size())
                chunk_ = std::min(max_k_, chunk_ + 1);
            else
                chunk_ = 1;
            // A target correction or bonus token may be ahead in the CTC
            // stream. Search only forward; punctuation/case often has no CTC
            // match, in which case the next verification may reject at j=0.
            if (added > accepted) {
                const int32_t last = prefix.back();
                const size_t end = std::min(draft_.size(), pos_ + size_t{40});
                const auto first = draft_.begin() + (ptrdiff_t)pos_;
                const auto match = std::find(first, draft_.begin() + (ptrdiff_t)end, last);
                if (match != draft_.begin() + (ptrdiff_t)end) {
                    pos_ = (size_t)(match - draft_.begin()) + 1;
                    stalled_ = 0;
                } else if (accepted == 0) {
                    if (++stalled_ >= 8 && pos_ < draft_.size()) {
                        ++pos_;
                        stalled_ = 0;
                    }
                } else {
                    stalled_ = 0;
                }
            } else {
                stalled_ = 0;
            }
        }
        observed_ = prefix.size();
        previous_.clear();
        if (cap <= 0 || pos_ >= draft_.size()) return {};
        const size_t count = std::min({(size_t)cap, (size_t)chunk_, draft_.size() - pos_});
        previous_.assign(draft_.begin() + (ptrdiff_t)pos_,
                         draft_.begin() + (ptrdiff_t)(pos_ + count));
        return previous_;
    }

    size_t draft_count() const { return draft_.size(); }

private:
    std::vector<int32_t> draft_;
    std::vector<int32_t> previous_;
    size_t pos_ = 0;
    size_t observed_ = 0;
    int max_k_ = 1;
    int chunk_ = 1;
    int stalled_ = 0;
};

} // namespace starling::ggml::granite
