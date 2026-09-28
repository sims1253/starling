// Research-only preview proposer for #313. The source is an already-known
// Parakeet transcript tokenized with MOSS BPE; the target supplies only its
// verified output prefix. No oracle target IDs, logits or audio are visible.
// The monotone alignment and adaptive K follow the S1 copy pilot (#345).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace starling::bench::moss_preview {

inline std::string normalize_preview(const std::string& text) {
    std::string out;
    bool space = false;
    for (unsigned char ch : text) {
        if (ch == ' ' || ch == '\n' || ch == '\r' || ch == '\t') {
            space = !out.empty();
            continue;
        }
        if (space) out.push_back(' ');
        space = false;
        out.push_back(ch >= 'A' && ch <= 'Z' ? char(ch - 'A' + 'a') : char(ch));
    }
    return out;
}

class PreviewCopyDrafter {
public:
    explicit PreviewCopyDrafter(std::vector<int32_t> source, int max_k,
                                size_t max_skip = 32)
        : source_(std::move(source)), max_k_(std::clamp(max_k, 1, 16)),
          k_(std::min(2, max_k_)), max_skip_(max_skip) {}

    std::vector<int32_t> propose(const std::vector<int32_t>& prefix, int cap) {
        if (prefix.size() < observed_) {
            // A new verified prefix must be aligned from its beginning.
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

        const size_t count = size_t(std::min(cap, k_));
        if (misses_ >= 3 || position_ >= source_.size())
            last_draft_ = lookup(prefix, count);
        else
            last_draft_.clear();
        if (last_draft_.empty()) {
            const size_t end = std::min(source_.size(), position_ + count);
            last_draft_.assign(source_.begin() + ptrdiff_t(position_),
                               source_.begin() + ptrdiff_t(end));
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
            for (size_t i = prefix.size() - n; i-- > 0;) {
                if (std::equal(prefix.begin() + ptrdiff_t(i),
                               prefix.begin() + ptrdiff_t(i + n),
                               prefix.end() - ptrdiff_t(n))) {
                    const size_t end = std::min(prefix.size(), i + n + k);
                    return {prefix.begin() + ptrdiff_t(i + n),
                            prefix.begin() + ptrdiff_t(end)};
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

} // namespace starling::bench::moss_preview
