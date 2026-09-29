// Token-only S1 draft proposal. The source and the verified output prefix are
// its only inputs; no target logits or future output IDs are visible.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace starling::ggml::s1 {

class CopyDrafter {
public:
    explicit CopyDrafter(std::vector<int32_t> source, int max_k,
                         size_t max_skip = 32)
        : source_(std::move(source)), max_k_(std::clamp(max_k, 1, 16)),
          k_(std::min(2, max_k_)), max_skip_(max_skip) {}

    std::vector<int32_t> propose(const std::vector<int32_t>& prefix, int cap) {
        if (prefix.size() < observed_) {
            position_ = 0;
            observed_ = 0;
            misses_ = 0;
            k_ = std::min(2, max_k_);
            last_draft_.clear();
            return {};
        }
        if (cap < 1) return {};
        if (!last_draft_.empty() && prefix.size() > observed_) {
            const size_t emitted = prefix.size() - observed_;
            size_t accepted = 0;
            while (accepted < emitted && accepted < last_draft_.size() &&
                   prefix[observed_ + accepted] == last_draft_[accepted])
                ++accepted;
            if (accepted == last_draft_.size()) k_ = std::min(max_k_, k_ + 1);
            else k_ = std::max(1, k_ / 2);
        }
        for (size_t i = observed_; i < prefix.size(); ++i) observe(prefix[i]);
        observed_ = prefix.size();

        const size_t count = (size_t)std::min(cap, k_);
        if (misses_ >= 3 || position_ >= source_.size())
            last_draft_ = lookup(prefix, count);
        else
            last_draft_.clear();
        if (last_draft_.empty() && misses_ < 3) {
            const size_t end = std::min(source_.size(), position_ + count);
            last_draft_.assign(source_.begin() + (std::ptrdiff_t)position_,
                               source_.begin() + (std::ptrdiff_t)end);
        }
        return last_draft_;
    }

private:
    void observe(int32_t emitted) {
        const size_t end = std::min(source_.size(), position_ + max_skip_ + 1);
        for (size_t i = position_; i < end; ++i) {
            if (source_[i] == emitted) {
                position_ = i + 1;
                misses_ = 0;
                return;
            }
        }
        ++misses_;
    }

    static std::vector<int32_t> lookup(const std::vector<int32_t>& prefix,
                                       size_t k) {
        if (prefix.size() < 2) return {};
        for (size_t n = std::min<size_t>(4, prefix.size() - 1); n > 0; --n) {
            const size_t earliest = prefix.size() > 512 ? prefix.size() - 512 : 0;
            for (size_t i = prefix.size() - n; i-- > earliest;) {
                if (std::equal(prefix.begin() + (std::ptrdiff_t)i,
                               prefix.begin() + (std::ptrdiff_t)(i + n),
                               prefix.end() - (std::ptrdiff_t)n)) {
                    const size_t end = std::min(prefix.size(), i + n + k);
                    return {prefix.begin() + (std::ptrdiff_t)(i + n),
                            prefix.begin() + (std::ptrdiff_t)end};
                }
            }
        }
        return {};
    }

    std::vector<int32_t> source_;
    std::vector<int32_t> last_draft_;
    size_t position_ = 0;
    size_t observed_ = 0;
    int max_k_;
    int k_;
    size_t max_skip_;
    int misses_ = 0;
};

} // namespace starling::ggml::s1
