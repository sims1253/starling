// stream_session.cpp — streaming dictation session implementation.
//
// Faithful C++ port of src/starling/server.py (StreamSession) and
// src/starling/stream_chunk.py (ChunkStreamer + stitch_words).

#include "stream_session.hpp"
#include "audio.hpp"

#include "runtime/call_abort.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>

namespace starling::serve {

namespace {
struct StreamQueueTimeout {};
}

// ---- word helpers ---------------------------------------------------------

std::string norm_word(const std::string& word) {
    // Lowercase, strip non-word chars (port of _norm() from stream_chunk.py).
    //
    // UTF-8 aware (issue #118): the previous byte-wise filter kept only bytes
    // where std::isalnum is true, which is false for every byte >= 0x80 in the
    // default C locale — Cyrillic/CJK words normalized to "" and runs of empty
    // keys then matched as overlap runs, deleting words at every streaming
    // window boundary. Conservatively decode UTF-8 code points, lowercase
    // ASCII A-Z only, keep code points >= 0x80 verbatim (no Unicode case
    // folding), and strip ASCII whitespace/punctuation as before. The only
    // property the overlap matcher needs: the key is deterministic per word
    // and identical across chunk boundaries, which verbatim bytes guarantee.
    std::string out;
    out.reserve(word.size());
    size_t i = 0;
    while (i < word.size()) {
        unsigned char uc = static_cast<unsigned char>(word[i]);
        if (uc < 0x80) {
            if (uc >= 'A' && uc <= 'Z') {
                out += static_cast<char>(uc - 'A' + 'a');
            } else if (std::isalnum(uc) || uc == '\'') {
                out += static_cast<char>(uc);
            }
            // else: strip punctuation
            ++i;
            continue;
        }
        // Multi-byte UTF-8: copy the whole code point verbatim. A truncated
        // or otherwise invalid sequence copies its lead byte alone — either
        // way the same input always produces the same key.
        size_t len = 0;
        if (uc >= 0xc2 && uc <= 0xdf) len = 2;
        else if (uc >= 0xe0 && uc <= 0xef) len = 3;
        else if (uc >= 0xf0 && uc <= 0xf4) len = 4;
        bool valid = len > 0 && i + len <= word.size();
        for (size_t j = 1; valid && j < len; ++j) {
            unsigned char cc = static_cast<unsigned char>(word[i + j]);
            if (cc < 0x80 || cc > 0xbf) valid = false;
        }
        if (valid) {
            out.append(word, i, len);
            i += len;
        } else {
            out += word[i];
            ++i;
        }
    }
    return out;
}

std::vector<std::string> split_words(const std::string& s) {
    std::vector<std::string> words;
    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) i++;
        if (i >= s.size()) break;
        size_t start = i;
        while (i < s.size() && !std::isspace(static_cast<unsigned char>(s[i]))) i++;
        words.emplace_back(s.substr(start, i - start));
    }
    return words;
}

std::string join_words(const std::vector<std::string>& words) {
    std::string out;
    for (size_t i = 0; i < words.size(); ++i) {
        if (i) out += ' ';
        out += words[i];
    }
    return out;
}

// Overlap alignment scores (issue #357; _STITCH_* in stream_chunk.py). The
// empty alignment scores 0; two matched words at the very edges score 4,
// three matches around four unmatched committed words score 2 (rejected).
constexpr int kStitchMatch = 2;
constexpr int kStitchMismatch = -1;
constexpr int kStitchGap = -1;
constexpr int kStitchMinScore = 3;

std::vector<std::string> stitch_words(
    const std::vector<std::string>& committed,
    const std::vector<std::string>& new_words,
    int max_overlap,
    int max_head,
    double expected_overlap) {
    if (committed.empty()) return new_words;
    if (new_words.empty()) return committed;

    // tail = last max_overlap words of committed; head = first max_head of new.
    const int tail_start =
        std::max(0, static_cast<int>(committed.size()) - max_overlap);
    const int head_end = std::min(max_head < 0 ? max_overlap : max_head,
                                  static_cast<int>(new_words.size()));
    // Normalized keys. An empty key (pure punctuation like "--") never
    // matches (issue #118 defense in depth): runs of them would align
    // unrelated boundary words.
    std::vector<std::string> a, b;
    for (int i = tail_start; i < static_cast<int>(committed.size()); ++i)
        a.push_back(norm_word(committed[i]));
    for (int i = 0; i < head_end; ++i) b.push_back(norm_word(new_words[i]));
    const int n = static_cast<int>(a.size());
    const int m = static_cast<int>(b.size());
    auto same = [&](int i, int j) { return !a[i].empty() && a[i] == b[j]; };

    // d[i][j]: best score of aligning a[i0, i) (any i0 <= i) with b[0, j).
    std::vector<std::vector<int>> d(n + 1, std::vector<int>(m + 1, 0));
    for (int j = 1; j <= m; ++j) d[0][j] = d[0][j - 1] + kStitchGap;
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            const int diag = d[i - 1][j - 1]
                + (same(i - 1, j - 1) ? kStitchMatch : kStitchMismatch);
            d[i][j] = std::max({diag, d[i - 1][j] + kStitchGap,
                                d[i][j - 1] + kStitchGap});
        }
    }
    // The suffix runs to the end of the tail; the prefix may end anywhere
    // (the smallest end column wins a tie).
    int best_j = 0, best = 0;
    for (int j = 1; j <= m; ++j) {
        if (d[n][j] > best) {
            best = d[n][j];
            best_j = j;
        }
    }
    if (best < kStitchMinScore) {
        std::vector<std::string> result = committed;
        result.insert(result.end(), new_words.begin(), new_words.end());
        return result;
    }
    std::vector<std::pair<int, int>> pairs;  // matched (tail, head) words
    int i = n;
    for (int j = best_j; i > 0 && j > 0;) {
        const bool eq = same(i - 1, j - 1);
        if (d[i][j] == d[i - 1][j - 1] + (eq ? kStitchMatch : kStitchMismatch)) {
            if (eq) pairs.emplace_back(i - 1, j - 1);
            --i;
            --j;
        } else if (d[i][j] == d[i - 1][j] + kStitchGap) {
            --i;
        } else {
            --j;
        }
    }
    std::reverse(pairs.begin(), pairs.end());
    // The alignment covers tail rows i..n-1.
    const int length = n - i;
    if (expected_overlap >= 0.0 && length > expected_overlap) {
        // Over committed words with a period, every matched pair also matches
        // one period further on: the alignment can start k periods later
        // (pairs moved past the tail drop out). Keep the k whose length is
        // closest to the estimate and that keeps a pair (the smallest k on a
        // tie).
        int period = 0;
        for (int p = 1; p <= length / 2 && !period; ++p) {
            bool periodic = true;
            for (int k = 0; k + p < length && periodic; ++k)
                periodic = a[i + k] == a[i + k + p];
            if (periodic) period = p;
        }
        if (period) {
            int best_k = 0;
            for (int k = 1; k <= (length - 1) / period; ++k) {
                if (pairs[0].first + k * period < n
                    && std::abs(length - k * period - expected_overlap)
                           < std::abs(length - best_k * period - expected_overlap))
                    best_k = k;
            }
            std::vector<std::pair<int, int>> moved;
            for (const auto& [r, c] : pairs)
                if (r + best_k * period < n) moved.emplace_back(r + best_k * period, c);
            pairs = std::move(moved);
        }
    }
    const auto [ci, cj] = pairs[(pairs.size() - 1) / 2];
    std::vector<std::string> result(committed.begin(),
                                    committed.begin() + tail_start + ci + 1);
    result.insert(result.end(), new_words.begin() + cj + 1, new_words.end());
    return result;
}

// ---- window plausibility (issue #357) --------------------------------------

namespace {
constexpr int kVadFramesPerSecond = 50;
constexpr double kVadFloorPercentile = 0.1;
constexpr double kVadFloorMaxDb = -45.0;
constexpr double kVadMarginDb = 15.0;
constexpr double kVadMinDb = -60.0;
}  // namespace

int max_plausible_words(double seconds) {
    return static_cast<int>(seconds * kMaxWordsPerSecond) + kMaxWordsSlack;
}

std::vector<uint8_t> voiced_frames(const float* samples, int64_t n, int sample_rate) {
    const int frame = sample_rate / kVadFramesPerSecond;
    const int64_t count = frame > 0 ? n / frame : 0;
    std::vector<uint8_t> flags(static_cast<size_t>(std::max<int64_t>(count, 0)), 0);
    if (count == 0) return flags;
    std::vector<double> db(static_cast<size_t>(count));
    for (int64_t f = 0; f < count; ++f) {
        double sum = 0.0;
        for (int k = 0; k < frame; ++k) {
            const double x = samples[f * frame + k];
            sum += x * x;
        }
        db[f] = 10.0 * std::log10(sum / frame + 1e-10);
    }
    std::vector<double> sorted = db;
    const auto nth = sorted.begin()
        + static_cast<std::ptrdiff_t>(kVadFloorPercentile * (count - 1));
    std::nth_element(sorted.begin(), nth, sorted.end());
    const double floor = std::min(*nth, kVadFloorMaxDb);
    const double threshold = std::max(floor + kVadMarginDb, kVadMinDb);
    for (int64_t f = 0; f < count; ++f) flags[f] = db[f] > threshold;
    return flags;
}

double voiced_seconds(const float* samples, int64_t n, int sample_rate) {
    const int frame = sample_rate / kVadFramesPerSecond;
    const auto flags = voiced_frames(samples, n, sample_rate);
    const auto voiced = std::count(flags.begin(), flags.end(), uint8_t{1});
    return static_cast<double>(voiced) * frame / sample_rate;
}

namespace {
// How many of a decode's `words` fall in buffer[lo, hi): its share of the
// decode's voiced frames (those starting in [lo, hi)); -1 without voiced
// frames (the audio cannot place the words). Port of
// _expected_words() in stream_chunk.py.
double expected_words(int64_t words, const std::vector<uint8_t>& flags, int64_t start,
                      int64_t lo, int64_t hi, int64_t frame) {
    const auto total = std::count(flags.begin(), flags.end(), uint8_t{1});
    if (total == 0) return -1.0;
    auto ceil_div = [](int64_t x, int64_t y) {
        return x >= 0 ? (x + y - 1) / y : -((-x) / y);
    };
    const int64_t size = static_cast<int64_t>(flags.size());
    const int64_t first = std::min(size, std::max<int64_t>(0, ceil_div(lo - start, frame)));
    const int64_t last = std::min(size, std::max(first, ceil_div(hi - start, frame)));
    const auto inside = std::count(flags.begin() + first, flags.begin() + last, uint8_t{1});
    return static_cast<double>(words * inside) / static_cast<double>(total);
}
}  // namespace

namespace {
// The back-to-back run of one phrase (1..max_n words, more than max_repeats
// copies) covering the most words: {start, phrase length, repeats}, the
// earliest, then the shortest phrase on a tie; {0, 0, 0} when there is none.
// Port of _longest_repeat() in stream_chunk.py.
std::array<size_t, 3> longest_repeat(const std::vector<std::string>& keys,
                                     size_t max_repeats, size_t max_n) {
    std::array<size_t, 3> best{0, 0, 0};
    const size_t len = keys.size();
    for (size_t i = 0; i < len; ++i) {
        for (size_t n = 1; n <= max_n; ++n) {
            if (i + n * (max_repeats + 1) > len) break;
            size_t reps = 1;
            while (i + (reps + 1) * n <= len
                   && std::equal(keys.begin() + i, keys.begin() + i + n,
                                 keys.begin() + i + reps * n)) {
                ++reps;
            }
            if (reps > max_repeats && reps * n > best[1] * best[2]) best = {i, n, reps};
        }
    }
    return best;
}

// Cut a run found by longest_repeat() to `keep` copies, in words and keys.
void cut_run(std::vector<std::string>& words, std::vector<std::string>& keys,
             const std::array<size_t, 3>& run, size_t keep) {
    const auto from = static_cast<std::ptrdiff_t>(run[0] + keep * run[1]);
    const auto to = static_cast<std::ptrdiff_t>(run[0] + run[2] * run[1]);
    words.erase(words.begin() + from, words.begin() + to);
    keys.erase(keys.begin() + from, keys.begin() + to);
}

std::vector<std::string> norm_keys(const std::vector<std::string>& words) {
    std::vector<std::string> keys;
    keys.reserve(words.size());
    for (const auto& w : words) keys.push_back(norm_word(w));
    return keys;
}

// A preview loop: one phrase repeated back to back more than this many
// times, covering at least kPreviewLoopMinWords words.
constexpr size_t kPreviewLoopMaxRepeats = 3;
constexpr size_t kPreviewLoopMinWords = 8;
}  // namespace

std::vector<std::string> suppress_loops(const std::vector<std::string>& words,
                                        double seconds, int max_repeats, int max_n) {
    const size_t bound = static_cast<size_t>(std::max(0, max_plausible_words(seconds)));
    std::vector<std::string> out = words;
    std::vector<std::string> keys = norm_keys(out);
    const size_t reps_max = static_cast<size_t>(std::max(0, max_repeats));
    while (out.size() > bound) {
        const auto run = longest_repeat(keys, reps_max, static_cast<size_t>(max_n));
        if (!run[1]) break;
        cut_run(out, keys, run, reps_max);
    }
    if (out.size() > bound) out.resize(bound);
    return out;
}

std::vector<std::string> suppress_preview_loops(const std::vector<std::string>& words,
                                                double seconds, int max_n) {
    std::vector<std::string> out = suppress_loops(words, seconds, 2, max_n);
    std::vector<std::string> keys = norm_keys(out);
    for (;;) {
        const auto run = longest_repeat(keys, kPreviewLoopMaxRepeats,
                                        static_cast<size_t>(max_n));
        if (!run[1] || run[1] * run[2] < kPreviewLoopMinWords) return out;
        cut_run(out, keys, run, 2);
    }
}

// ---- ChunkStreamer --------------------------------------------------------

std::optional<double> parse_double_strict(const std::string& text) {
    // Full-string parse: std::stod accepts partial parses ("3abc" -> 3) and
    // non-finite tokens ("nan", "inf"); the CLI must reject both (issue #146).
    try {
        std::size_t pos = 0;
        const double value = std::stod(text, &pos);
        // Allow trailing whitespace only; anything else is junk.
        while (pos < text.size()
               && std::isspace(static_cast<unsigned char>(text[pos]))) {
            ++pos;
        }
        if (pos != text.size()) return std::nullopt;
        if (!std::isfinite(value)) return std::nullopt;
        return value;
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<int> parse_int_strict(const std::string& text) {
    try {
        std::size_t pos = 0;
        const long value = std::stol(text, &pos);
        while (pos < text.size()
               && std::isspace(static_cast<unsigned char>(text[pos]))) {
            ++pos;
        }
        if (pos != text.size()) return std::nullopt;
        if (value < std::numeric_limits<int>::min()
            || value > std::numeric_limits<int>::max()) {
            return std::nullopt;
        }
        return static_cast<int>(value);
    } catch (...) {
        return std::nullopt;
    }
}

std::string stream_window_config_error(int sample_rate, double chunk_seconds,
                                       double overlap_seconds, double min_seconds,
                                       double partial_interval) {
    if (sample_rate <= 0) {
        return "stream sample rate must be positive (got "
               + std::to_string(sample_rate) + ")";
    }
    const struct {
        const char* name;
        double value;
    } fields[] = {
        {"stream chunk seconds", chunk_seconds},
        {"stream overlap seconds", overlap_seconds},
        {"stream min chunk seconds", min_seconds},
        {"stream partial interval seconds", partial_interval},
    };
    for (const auto& f : fields) {
        if (!std::isfinite(f.value)) {
            return std::string(f.name) + " must be a finite number";
        }
    }
    if (chunk_seconds < 0.0) {
        return "stream chunk seconds must be nonnegative "
               "(0 selects whole-buffer mode)";
    }
    if (overlap_seconds < 0.0) {
        return "stream overlap seconds must be nonnegative";
    }
    if (min_seconds < 0.0) {
        return "stream min chunk seconds must be nonnegative";
    }
    if (partial_interval < 0.0) {
        return "stream partial interval seconds must be nonnegative";
    }
    // The sample counters (chunk_, overlap_, min_) are ints: reject windows
    // whose sample counts do not fit, before the narrowing casts overflow.
    const double max_seconds =
        static_cast<double>(std::numeric_limits<int>::max()) / sample_rate;
    if (chunk_seconds > 0.0) {
        if (chunk_seconds * sample_rate < 1.0) {
            return "stream chunk seconds too small: the window is shorter "
                   "than one sample at this rate";
        }
        if (overlap_seconds >= chunk_seconds) {
            return "stream overlap seconds must be smaller than "
                   "stream chunk seconds";
        }
        if (chunk_seconds > max_seconds) {
            return "stream chunk seconds too large: the window does not fit "
                   "the sample counters";
        }
    }
    if (min_seconds > max_seconds) {
        return "stream min chunk seconds too large: the minimum does not fit "
               "the sample counters";
    }
    return "";
}

std::string preview_policy_error(int sample_rate, double min_seconds,
                                 double interval_seconds) {
    if (!std::isfinite(min_seconds) || !std::isfinite(interval_seconds))
        return "preview cadence must be finite";
    if (min_seconds < 0.0 || interval_seconds < 0.0)
        return "preview cadence must be nonnegative";
    if (sample_rate <= 0
        || min_seconds > static_cast<double>(std::numeric_limits<int>::max())
                             / sample_rate)
        return "first-partial minimum too large";
    return "";
}

void ChunkStreamer::set_preview_policy(double min_seconds, double interval_seconds) {
    const std::string err = preview_policy_error(sr_, min_seconds, interval_seconds);
    if (!err.empty()) throw std::invalid_argument(err);
    min_ = static_cast<int>(min_seconds * sr_);
    partial_interval_ = interval_seconds;
}

ChunkStreamer::ChunkStreamer(int sample_rate, double chunk_seconds,
                             double overlap_seconds, double min_seconds,
                             double partial_interval)
    : sr_(0),
      chunk_(0),
      overlap_(0),
      advance_(1),
      min_(0),
      partial_interval_(0.0),
      max_overlap_words_(0),
      lookback_(0),
      max_head_words_(0) {
    // Validate the whole configuration before deriving anything, then build
    // the members in dependency order (issue #146). The old member-init list
    // computed advance_ from the unclamped overlap_: a negative overlap was
    // baked into advance_ (every window advanced chunk+|overlap| samples,
    // skipping audio), and the body's late `overlap_ < 0` clamp could not fix
    // it — flush() could then hand the transcriber a negative-length window.
    if (chunk_seconds <= 0.0) {
        // 0 selects the legacy whole-buffer mode at the CLI; a chunker needs
        // a real window.
        throw std::invalid_argument(
            "stream chunk seconds must be positive to build a chunked stream");
    }
    const std::string err = stream_window_config_error(
        sample_rate, chunk_seconds, overlap_seconds, min_seconds,
        partial_interval);
    if (!err.empty()) throw std::invalid_argument(err);

    sr_ = sample_rate;
    chunk_ = static_cast<int>(chunk_seconds * sample_rate);
    // Normalize overlap before deriving advance_: clamp to [0, chunk/2] (the
    // documented cap; overlap >= chunk was already rejected above).
    overlap_ = static_cast<int>(
        std::max(0.0, std::min(overlap_seconds, chunk_seconds * 0.5))
        * sample_rate);
    advance_ = std::max(1, chunk_ - overlap_);
    min_ = static_cast<int>(min_seconds * sample_rate);
    partial_interval_ = partial_interval;
    max_overlap_words_ = std::max(8, static_cast<int>(overlap_seconds * 6) + 6);
    lookback_ = static_cast<int>(
        *std::max_element(std::begin(kRedecodeShifts), std::end(kRedecodeShifts))
        * overlap_);
    // A re-decoded window starts up to the lookback earlier and shares that
    // much more audio with the committed text (issue #357).
    max_head_words_ = 2 * max_overlap_words_;
}

std::vector<std::string> ChunkStreamer::stitched(const Decoded& d) const {
    double expected = -1.0;  // mean of the two sides' estimates
    if (last_.valid) {
        const int64_t lo = std::max(d.start, last_.start);
        const int64_t hi = std::min(d.end, last_.end);
        if (hi > lo) {
            const int64_t frame = std::max(1, sr_ / 50);
            double sum = 0.0;
            int count = 0;
            for (double e : {expected_words(last_.words, last_.flags, last_.start, lo, hi, frame),
                             expected_words(static_cast<int64_t>(d.words.size()), d.flags,
                                            d.start, lo, hi, frame)}) {
                if (e >= 0.0) {
                    sum += e;
                    ++count;
                }
            }
            if (count) expected = sum / count;
        }
    }
    // Stitch against the unfrozen tail only: the frozen prefix was already
    // reported stable, so no alignment may cut into it.
    std::vector<std::string> tail(committed_.begin() + frozen_, committed_.end());
    tail = stitch_words(tail, d.words, max_overlap_words_, max_head_words_, expected);
    std::vector<std::string> result(committed_.begin(), committed_.begin() + frozen_);
    result.insert(result.end(), tail.begin(), tail.end());
    return result;
}

void ChunkStreamer::commit(const Decoded& d) {
    committed_ = stitched(d);
    const int64_t reachable =
        static_cast<int64_t>(committed_.size()) - max_overlap_words_;
    frozen_ = std::max(frozen_, reachable);
    last_.valid = true;
    last_.start = d.start;
    last_.end = d.end;
    last_.words = static_cast<int64_t>(d.words.size());
    last_.flags = d.flags;
}

double ChunkStreamer::min_rate() const {
    if (rates_.empty()) return kMinWordsPerVoicedSecond;
    std::vector<double> sorted = rates_;
    std::sort(sorted.begin(), sorted.end());
    return std::max(kMinWordsPerVoicedSecond,
                    kRelativeMinRate * sorted[(sorted.size() - 1) / 2]);
}

int ChunkStreamer::verdict(size_t words, double seconds, double voiced) const {
    if (words > static_cast<size_t>(std::max(0, max_plausible_words(seconds)))) return 2;
    if (voiced >= kMinVoicedSeconds && static_cast<double>(words) < min_rate() * voiced)
        return 1;
    return 0;
}

std::optional<ChunkStreamer::Decoded> ChunkStreamer::decode_committed(
    const std::vector<float>& samples, int64_t start, int64_t end,
    const char* kind, const TranscribeFn& tx) {
    // An implausible result (too sparse for its voiced audio, or a loop) is
    // decoded again from the earlier starts in kRedecodeShifts until one is
    // plausible: a full window moves back whole (the next window's overlap
    // still covers its end), or ends earlier where the buffer holds no audio
    // before it (the take's first window); the flush tail grows backwards up
    // to one window (nothing else covers its end). A candidate replaces the
    // current one when it is plausible and the current one is not, or when
    // neither is plausible, it does not loop and it is denser by
    // kRedecodeMinGain. A busy re-decode keeps the best so far if it is
    // plausible, and otherwise leaves the window pending (nullopt): the audio
    // stays for a retry instead of committing known-wrong text.
    const int frame = sr_ / 50;
    auto voiced_of = [&](const std::vector<uint8_t>& flags) {
        return static_cast<double>(std::count(flags.begin(), flags.end(), uint8_t{1}))
               * frame / sr_;
    };
    call_kind_ = kind;
    auto text = tx(samples.data() + start, end - start);
    if (!text.has_value()) return std::nullopt;
    Decoded best{split_words(*text), start, end,
                 voiced_frames(samples.data() + start, end - start, sr_)};
    double best_voiced = voiced_of(best.flags);
    const int first = verdict(best.words.size(), static_cast<double>(end - start) / sr_,
                              best_voiced);
    int best_verdict = first;
    bool replaced = false;
    std::vector<std::pair<int64_t, int64_t>> tried{{start, end}};
    const bool full = end - start == chunk_;
    for (double shift : kRedecodeShifts) {
        if (first == 0 || best_verdict == 0) break;
        const int64_t back = static_cast<int64_t>(shift * overlap_);
        int64_t a, z;
        if (full) {
            a = start >= back ? start - back : start;
            z = end - back;
        } else {
            a = std::max<int64_t>({start - back, end - chunk_, 0});
            z = end;
        }
        if (a >= z || (!full && a >= start)
            || std::find(tried.begin(), tried.end(), std::make_pair(a, z)) != tried.end())
            continue;
        tried.emplace_back(a, z);
        call_kind_ = "redecode";
        auto alt = tx(samples.data() + a, z - a);
        if (!alt.has_value()) {
            if (best_verdict != 0) return std::nullopt;
            break;
        }
        Decoded cand{split_words(*alt), a, z, voiced_frames(samples.data() + a, z - a, sr_)};
        const double v_voiced = voiced_of(cand.flags);
        const int v = verdict(cand.words.size(), static_cast<double>(z - a) / sr_, v_voiced);
        bool better = false;
        if (v == 0 || (v == 1 && best_verdict == 2)) {
            better = true;
        } else if (v == 1) {
            better = static_cast<double>(cand.words.size()) * std::max(best_voiced, 1e-9)
                     >= kRedecodeMinGain * static_cast<double>(best.words.size())
                            * std::max(v_voiced, 1e-9);
        }
        if (better) {
            best = std::move(cand);
            best_verdict = v;
            best_voiced = v_voiced;
            replaced = true;
        }
    }
    if (replaced) ++redecodes_;
    if (best_verdict == 2) {
        best.words = suppress_loops(best.words, static_cast<double>(best.end - best.start) / sr_);
    } else if (best_voiced >= kMinVoicedSeconds) {
        rates_.push_back(static_cast<double>(best.words.size()) / best_voiced);
        if (rates_.size() > kRateHistory) rates_.erase(rates_.begin());
    }
    return best;
}

bool ChunkStreamer::finalize_full_windows(
    const std::vector<float>& samples, const TranscribeFn& tx, bool flushing) {
    bool did = false;
    while (static_cast<int64_t>(samples.size()) - boundary_ >= chunk_) {
        const int64_t end = boundary_ + chunk_;
        auto got = decode_committed(samples, boundary_, end,
                                    flushing ? "flush_window" : "window", tx);
        if (!got.has_value()) break;  // busy → stop, boundary unchanged
        commit(*got);
        // The next window overlaps the audio the committed text came from
        // by the full overlap, also when a re-decode ended earlier.
        boundary_ += advance_ - (end - got->end);
        did = true;
    }
    return did;
}

std::optional<std::string> ChunkStreamer::step(
    const std::vector<float>& samples, double now, const TranscribeFn& tx,
    const PendingFn& newer_pending) {
    // Window commits are required work: never throttled, never coalesced.
    // `now` is when the step began; the window decodes take real time, so
    // the preview decision below uses the clock after them (otherwise a
    // slow window would eat the gap the interval promises).
    const auto t_windows = std::chrono::steady_clock::now();
    const bool committed = finalize_full_windows(samples, tx, false);
    now += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_windows).count();
    if (committed) {
        // The committed text covers the buffer up to the window's end (all
        // audio received when the window filled), so it is as fresh as a
        // preview: it restarts the interval instead of forcing a preview of
        // the overlap right behind it (issue #357).
        emit_due_ = true;
        last_emit_ = now;
    }
    // Committed text the client has not seen yet (or nothing).
    auto committed_update = [this]() -> std::optional<std::string> {
        if (!emit_due_) return std::nullopt;
        emit_due_ = false;
        return join_words(committed_);
    };

    const int64_t tail_len = static_cast<int64_t>(samples.size()) - boundary_;
    if (tail_len >= chunk_) {  // a full window is still waiting for a retry
        return committed_update();
    }
    // Preview eligibility depends on the take, not on the tail: the tail
    // shrinks to the overlap after every window commit, and gating it on the
    // first-partial minimum stalled previews for (min - overlap) seconds
    // after each commit (issue #357).
    const bool eligible = tail_len > 0
        && rebased_ + static_cast<int64_t>(samples.size()) >= min_;
    // A window commit never forces a preview: the committed text is
    // emitted at once, and the tail waits for the (cost-stretched) interval.
    const bool throttled = (now - last_emit_) < effective_interval();
    if (!eligible || throttled) return committed_update();
    if (newer_pending && newer_pending()) {
        // Newer audio is already queued: this preview would be stale before
        // it finished. Skip it; the next step previews the newer audio and
        // carries any committed-text update along (emit_due_ stays set).
        ++coalesced_;
        return std::nullopt;
    }
    last_emit_ = now;

    call_kind_ = "preview";
    const auto t0 = std::chrono::steady_clock::now();
    auto text = tx(samples.data() + boundary_, tail_len);
    if (!text.has_value()) {
        // Busy on the tail.
        return committed_update();
    }
    last_preview_cost_ = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    emit_due_ = false;
    // A preview is shown as decoded (no re-decode: the next one replaces
    // it), but never with a decoding loop in it (issue #357).
    const int64_t n = static_cast<int64_t>(samples.size());
    return join_words(stitched(Decoded{
        suppress_preview_loops(split_words(*text), static_cast<double>(tail_len) / sr_),
        boundary_, n, voiced_frames(samples.data() + boundary_, tail_len, sr_)}));
}

std::optional<std::string> ChunkStreamer::flush(
    const std::vector<float>& samples, const TranscribeFn& tx) {
    constexpr int kMaxRetries = 5;
    for (int attempt = 0; attempt < kMaxRetries; ++attempt) {
        finalize_full_windows(samples, tx, true);
        int64_t tail_len = static_cast<int64_t>(samples.size()) - boundary_;
        if (tail_len == 0) return join_words(committed_);
        // Guard the tail's sign as well (issue #146): the window geometry is
        // validated at construction, but the transcriber contract (a
        // nonempty window inside the buffer) is enforced here regardless.
        if (tail_len > 0 && tail_len < chunk_) {
            auto got = decode_committed(samples, boundary_,
                                        static_cast<int64_t>(samples.size()),
                                        "flush_tail", tx);
            if (got.has_value()) {
                commit(*got);
                boundary_ = static_cast<int64_t>(samples.size());
                emit_due_ = false;
                return join_words(committed_);
            }
        }
        if (attempt + 1 < kMaxRetries)
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return std::nullopt;
}

void ChunkStreamer::reset() {
    committed_.clear();
    frozen_ = 0;
    boundary_ = 0;
    rebased_ = 0;
    last_emit_ = 0.0;
    emit_due_ = false;
    last_preview_cost_ = 0.0;
    coalesced_ = 0;
    redecodes_ = 0;
    rates_.clear();
    last_ = {};
    call_kind_ = "window";
}

// ---- StreamSession --------------------------------------------------------

constexpr int kStreamTrimMinSamples = kSampleRate;

StreamSession::StreamSession(StarlingServer* server) : server_(server) {
    const auto& cfg = server_->config();
    if (cfg.stream_chunk_seconds > 0.0) {
        chunker_ = std::make_unique<ChunkStreamer>(
            kSampleRate, cfg.stream_chunk_seconds, cfg.stream_overlap_seconds,
            cfg.min_chunk_seconds, cfg.partial_interval);
    }
    max_buffer_seconds_ = cfg.max_stream_seconds;
    // Engine identity for exact-tail reuse (S11): everything about the request
    // that can change a raw window result. The native server fixes the model
    // slug + gguf artifact (which encodes the quantization) per process and
    // selected backend after load (compile-time family before a lazy load);
    // the window/overlap policy shapes the very windows being keyed. The
    // preview cadence is not part of it: it decides when a window is
    // transcribed, never what the engine returns for it (issue #357).
    // std::to_string(double) is fixed-point, so the
    // string is deterministic. There is no language/normalization parameter
    // on the native streaming path (nothing extra to key on); a hypothetical
    // reload/re-config goes through set_engine_identity(), which invalidates.
    engine_id_ = cfg.model_slug + "|" + cfg.gguf_path + "|"
               + server_->backend_identity() + "|chunk="
               + std::to_string(cfg.stream_chunk_seconds)
               + "|overlap=" + std::to_string(cfg.stream_overlap_seconds);
}

TranscribeFn StreamSession::make_transcribe_fn(RequestContext* ctx) {
    return [this, ctx](const float* samples, int64_t n)
               -> std::optional<std::string> {
        std::string err;
        // In Granite fairness mode, a stream takes a FIFO ticket so an upload
        // yields after its current chunk. The default retry-on-busy contract
        // is unchanged for every other serving mode.
        const QueuePolicy policy = (server_->config().granite_chunk_fairness &&
                                    server_->config().model_slug == "granite")
            ? QueuePolicy::Block : QueuePolicy::SkipIfBusy;
        auto result = server_->transcribe_pcm(samples, n, ctx, &err,
                                              policy);
        if (!err.empty()) {
            // A blocking queue wait has already consumed its deadline. Do
            // not let ChunkStreamer::flush retry it five times with fresh
            // deadlines, or present it as a transient busy reply.
            if (err == "request timed out") throw StreamQueueTimeout{};
            // "server busy" or "cancelled" → return nullopt (retry without
            // advancing state, matching the Python StreamSession._tx behavior).
            if (err == "server busy" || err == "cancelled") return std::nullopt;
            // Other errors: also treat as busy (non-fatal in streaming).
            return std::nullopt;
        }
        return result.text;
    };
}

AppendOutcome StreamSession::append_pcm(const std::string& bytes) {
    if (overflow_) return AppendOutcome::Overflowed;  // capped: refuse until reset()
    if (take_invalid_) return AppendOutcome::TakeInvalid;  // rejected take: refuse until reset()
    const size_t nbytes = bytes.size();
    if (nbytes == 0) return AppendOutcome::Accepted;
    // Raw PCM is a sequence of whole int16 samples: an odd byte count means
    // a sample was split mid-frame at a transport boundary. The old code
    // silently dropped the dangling byte, hiding a misaligned client from
    // itself; reject the frame and invalidate the take instead (issue #145).
    if (nbytes % 2 == 1) {
        std::fprintf(stderr,
            "[starling-serve] dropping odd-length PCM chunk (len=%zu)\n",
            bytes.size());
        take_invalid_ = true;
        invalid_reason_ = "odd_pcm_length";
        invalidate_tail_result();  // an invalidated take never reuses (S11)
        return AppendOutcome::OddPcmLength;
    }
    size_t nsamples = nbytes / 2;
    if (nsamples == 0) return AppendOutcome::Accepted;
    // Cap the LIVE buffer (samples_ memory). Finalized audio is trimmed from
    // samples_, so a long dictation session without commits keeps memory
    // bounded while the cumulative audio grows freely.
    if (max_buffer_seconds_ > 0.0
        && unfinalized_seconds()
               + static_cast<double>(nsamples) / kSampleRate
             > max_buffer_seconds_) {
        overflow_ = true;
        return AppendOutcome::Overflowed;
    }
    const auto* src = reinterpret_cast<const int16_t*>(bytes.data());
    size_t old = samples_.size();
    samples_.resize(old + nsamples);
    for (size_t i = 0; i < nsamples; ++i) {
        samples_[old + i] = static_cast<float>(src[i]) / 32768.0f;
    }
    ++audio_rev_;  // any appended audio invalidates exact-tail reuse (S11)
    mark_take_start();
    maybe_trim_samples();
    return AppendOutcome::Accepted;
}

AppendOutcome StreamSession::append_wav(const std::string& bytes) {
    if (overflow_) return AppendOutcome::Overflowed;  // capped: refuse until reset()
    if (take_invalid_) return AppendOutcome::TakeInvalid;  // rejected take: refuse until reset()
    // Check for RIFF/WAVE header.
    if (bytes.size() < 12 || bytes.substr(0, 4) != "RIFF"
        || bytes.substr(8, 4) != "WAVE") {
        // Treat as raw PCM16.
        return append_pcm(bytes);
    }
    std::vector<float> decoded;
    int sr = 0;
    if (!audio::wav_bytes_to_float32(bytes, decoded, sr)) {
        std::fprintf(stderr,
            "[starling-serve] dropping malformed WAV chunk (len=%zu)\n",
            bytes.size());
        take_invalid_ = true;
        invalid_reason_ = "malformed_wav";
        invalidate_tail_result();  // an invalidated take never reuses (S11)
        return AppendOutcome::MalformedWav;
    }
    // No C++ resampler exists (the Python server resamples via scipy): a
    // non-16 kHz WAV must be rejected loudly, not dropped silently — the
    // client hears nothing back otherwise and blames the model (issue #145).
    if (sr != kSampleRate) {
        std::fprintf(stderr,
            "[starling-serve] dropping WAV chunk: sample rate %d != %d\n",
            sr, kSampleRate);
        take_invalid_ = true;
        invalid_reason_ = "sample_rate_mismatch";
        invalidate_tail_result();  // an invalidated take never reuses (S11)
        return AppendOutcome::RateMismatch;
    }
    if (!decoded.empty()) {
        if (max_buffer_seconds_ > 0.0
            && unfinalized_seconds()
                   + static_cast<double>(decoded.size()) / kSampleRate
                 > max_buffer_seconds_) {
            overflow_ = true;
            return AppendOutcome::Overflowed;
        }
        size_t old = samples_.size();
        samples_.resize(old + decoded.size());
        std::copy(decoded.begin(), decoded.end(), samples_.begin() + old);
        ++audio_rev_;  // any appended audio invalidates exact-tail reuse (S11)
        mark_take_start();
    }
    maybe_trim_samples();
    return AppendOutcome::Accepted;
}

void StreamSession::maybe_trim_samples() {
    if (!chunker_) return;
    // Keep the audio a re-decode may still read before the boundary.
    int64_t b = chunker_->retain_from();
    if (b <= 0 || b >= static_cast<int64_t>(samples_.size())) return;
    if (b < kStreamTrimMinSamples
        && b < static_cast<int64_t>(samples_.size()) / 2) return;
    samples_.erase(samples_.begin(),
                   samples_.begin() + static_cast<size_t>(b));
    trimmed_samples_ += b;
    // The chunker's boundary index now points into dropped territory;
    // rebase it so the chunker sees the trimmed buffer as fresh from index 0.
    chunker_->rebase(b);
}

// ---- exact streaming-tail result reuse (S11) -------------------------------
//
// Wrap the session's transcribe callback with the retained exact-tail entry.
// Every window call is keyed by (absolute sample start including the trimmed
// prefix, window length, audio revision, engine identity, transcribe-callback
// generation); a call whose key equals the retained SUCCESSFUL result's key is
// answered without invoking the engine — a successful preview followed by a
// commit on unchanged audio becomes one engine call instead of two, and
// duplicate preview snapshots of an unchanged buffer are coalesced into the
// retained result instead of re-running a stale generation. The commit path is
// never dropped or shortened by this: a hit returns the same complete raw text
// the engine produced for those exact bytes, and a miss runs the engine as
// before (a hit can even complete a commit while the engine is busy, because
// the exact answer is already known).
//
// Only complete successes are retained — empty text included. Busy, cancelled
// and failed calls return nullopt and never enter the entry; a stale retained
// entry cannot false-match later because audio_rev_ only moves forward.
TranscribeFn StreamSession::active_tx() {
    // Built once and cached (R29, issue #236): rebuilt only after
    // set_transcribe_fn()/set_engine_identity() drop the cache — the two
    // sole mutators of everything the wrapper snapshots below.
    if (wrapped_tx_) return wrapped_tx_;  // a copy: see the return below
    TranscribeFn inner = custom_tx_ ? custom_tx_ : make_transcribe_fn(nullptr);
    // Snapshot the generation together with the callback it belongs to
    // (PR #199 batch-2): reading tx_gen_ inside the lambda would pair the
    // callback captured here with whatever generation is current when the
    // chunker invokes the wrapper. One wrapper is invoked more than once per
    // step (full windows, then the tail; flush retries), and a
    // set_transcribe_fn() issued in between — e.g. from inside an earlier
    // window's callback — would make a LATER invocation of the OLD callback
    // retain its result under the NEW generation, where the new callback's
    // next identical window would replay it without running. Snapshotted
    // together, every invocation of this wrapper is keyed as the
    // (callback, generation) pair that existed at construction.
    const uint64_t tx_gen = tx_gen_;
    // engine_id joins the snapshot for the same reason (batch-4): reading it
    // inside the lambda would key a mid-step set_engine_identity() under the
    // new identity for a wrapper built against the old one.
    std::string engine_id = engine_id_;
    wrapped_tx_ = [this, inner = std::move(inner), tx_gen,
            engine_id = std::move(engine_id)](const float* p, int64_t n)
               -> std::optional<std::string> {
        StreamTailKey key;
        // p always points into samples_ (the chunker passes
        // samples.data() + boundary); with the trimmed prefix this is the
        // absolute take-time sample index, stable across buffer trims.
        key.abs_start = trimmed_samples_
                        + static_cast<int64_t>(p - samples_.data());
        key.length = n;
        key.audio_rev = audio_rev_;
        // Per-window cost note (R29, issue #236): the identity string is
        // copied into the key — and string-compared on the hit path — once
        // per transcribe call, i.e. per window, not per sample. An integer
        // identity (hash or generation counter from set_engine_identity)
        // could short-circuit that, but at one call per window the copy is
        // noise next to the inference it keys; accepted as-is.
        key.engine_id = engine_id;
        key.tx_gen = tx_gen;
        StreamCall call;
        call.kind = chunker_ ? chunker_->call_kind() : "window";
        call.abs_start = key.abs_start;
        call.length = n;
        call.t0_ms = take_ms();
        if (tail_valid_ && tail_key_ == key) {
            ++tail_cache_hits_;
            call.t1_ms = call.t0_ms;
            call.result = "reused";
            record_call(call);
            return tail_text_;  // exact-input reuse: engine not called
        }
        std::optional<std::string> result;
        bool preempted = false;
        try {
            if (preempt_ && std::strcmp(call.kind, "preview") == 0) {
                // A preview overtaken by required work stops at the engine's
                // next checkpoint (issue #357). A result that completes after
                // required work queued (past the last checkpoint, e.g. in the
                // TDT decoder, or an engine without checkpoints) is discarded
                // the same way: the work queued behind it supersedes it, so it
                // is neither published nor kept for exact-tail reuse.
                starling::ggml::CallAbortScope abort_scope(
                    [](void* self) {
                        auto* s = static_cast<StreamSession*>(self);
                        return s->preempt_ && s->preempt_();
                    },
                    this);
                result = inner(p, n);
                preempted = abort_scope.fired() || preempt_();
                if (preempted) result.reset();
            } else {
                result = inner(p, n);
            }
        } catch (const StreamQueueTimeout&) {
            call.t1_ms = take_ms();
            call.result = "timed_out";
            record_call(call);
            throw;
        }
        call.t1_ms = take_ms();
        call.result = result.has_value() ? "ok" : preempted ? "preempted" : "busy";
        record_call(call);
        if (result.has_value()) {
            // Retain exactly one entry: this success replaces any previous
            // one (bounded: one entry per session, never a growing cache).
            tail_valid_ = true;
            tail_key_ = key;
            tail_text_ = *result;
        }
        return result;
    };
    // Returned by value, deliberately: an engine callback may swap the fn or
    // the identity mid-step, which clears wrapped_tx_ — the caller's copy
    // keeps the (callback, generation, identity) snapshot it was built with
    // (see active_tx() in the header).
    return wrapped_tx_;
}

void StreamSession::invalidate_tail_result() {
    tail_valid_ = false;
    tail_text_.clear();
    tail_key_ = StreamTailKey{};
}

void StreamSession::set_engine_identity(std::string id) {
    if (id == engine_id_) return;
    // A different model/quant/backend/config can answer the same bytes
    // differently: the retained result is no longer an exact answer.
    invalidate_tail_result();
    engine_id_ = std::move(id);
    wrapped_tx_ = nullptr;  // the cached wrapper snapshots the old identity
}

std::optional<std::string> StreamSession::stream_step(
    double now, const PendingFn& newer_pending, const PreemptFn& preempt) {
    if (!chunker_) return std::nullopt;
    if (!terminal_error_.empty()) return std::nullopt;
    TranscribeFn tx = active_tx();
    // Installed for this step only: flushes and later steps without a
    // predicate never cancel.
    struct PreemptGuard {
        PreemptFn& slot;
        ~PreemptGuard() { slot = nullptr; }
    } guard{preempt_};
    preempt_ = preempt;
    try {
        return chunker_->step(samples_, now, tx, newer_pending);
    } catch (const StreamQueueTimeout&) {
        terminal_error_ = "request timed out";
        take_invalid_ = true;
        invalid_reason_ = "request_timed_out";
        invalidate_tail_result();
        return std::nullopt;
    }
}

std::optional<std::string> StreamSession::stream_flush() {
    if (!chunker_) return "";
    if (!terminal_error_.empty()) return std::nullopt;
    TranscribeFn tx = active_tx();
    // The stop section of the trace covers THIS flush only (a busy flush is
    // retried by a later commit, which starts a fresh section).
    flush_totals_ = StreamCallTotals{};
    flush_t0_ms_ = take_ms();
    flush_t1_ms_ = -1.0;
    flush_unfinalized_ =
        static_cast<int64_t>(samples_.size()) - chunker_->boundary();
    final_path_ = "";
    flushing_ = true;
    std::optional<std::string> out;
    try {
        out = chunker_->flush(samples_, tx);
    } catch (const StreamQueueTimeout&) {
        terminal_error_ = "request timed out";
        take_invalid_ = true;
        invalid_reason_ = "request_timed_out";
        invalidate_tail_result();
        out = std::nullopt;
    }
    flushing_ = false;
    flush_t1_ms_ = take_ms();
    if (out.has_value()) {
        final_path_ = flush_totals_.engine_calls > 0 ? "tail"
                    : flush_totals_.reused > 0       ? "reused"
                                                     : "committed";
    }
    return out;
}

void StreamSession::mark_empty_commit() {
    flush_totals_ = StreamCallTotals{};
    flush_t0_ms_ = flush_t1_ms_ = take_ms();
    flush_unfinalized_ = 0;
    final_path_ = "committed";
}

void StreamSession::reset() {
    samples_.clear();
    last_partial_ts_ = 0.0;
    trimmed_samples_ = 0;
    overflow_ = false;
    take_invalid_ = false;
    invalid_reason_.clear();
    terminal_error_.clear();
    if (chunker_) chunker_->reset();
    // A new take must never inherit the previous take's retained result.
    // Dropping the entry (below) covers the normal path; bumping audio_rev_
    // ENFORCES the monotonicity the keying relies on instead of leaving it
    // comment-only: absolute sample indices restart at 0 here, so without the
    // bump a later key could otherwise repeat a pre-reset
    // (abs_start, length, audio_rev) triple if any future path ever produced
    // one without an intervening append.
    ++audio_rev_;
    invalidate_tail_result();
    take_started_ = false;
    calls_.clear();
    calls_dropped_ = 0;
    totals_ = StreamCallTotals{};
    for (auto& t : by_kind_) t = StreamCallTotals{};
    flush_totals_ = StreamCallTotals{};
    flushing_ = false;
    covered_end_ = 0;
    flush_t0_ms_ = -1.0;
    flush_t1_ms_ = -1.0;
    flush_unfinalized_ = 0;
    final_path_ = "";
}

double StreamSession::buffered_seconds() const {
    return static_cast<double>(trimmed_samples_ + samples_.size()) / kSampleRate;
}

double StreamSession::live_seconds() const {
    return static_cast<double>(samples_.size()) / kSampleRate;
}

double StreamSession::unfinalized_seconds() const {
    const int64_t done = chunker_ ? std::min<int64_t>(chunker_->boundary(),
                                                      static_cast<int64_t>(samples_.size()))
                                  : 0;
    return static_cast<double>(static_cast<int64_t>(samples_.size()) - done) / kSampleRate;
}

// ---- stream instrumentation (issue #226) -----------------------------------

void StreamCallTotals::add(const StreamCall& c) {
    ++calls;
    const std::string r = c.result;
    if (r == "ok") {
        ++engine_calls;
        engine_samples += c.length;
        engine_ms += c.t1_ms - c.t0_ms;
    } else if (r == "reused") {
        ++reused;
    } else if (r == "preempted") {
        ++preempted;
        preempted_ms += c.t1_ms - c.t0_ms;
    } else {
        ++busy;
    }
}

double StreamSession::take_ms() const {
    if (!take_started_) return 0.0;
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - take_t0_).count();
}

void StreamSession::mark_take_start() {
    if (take_started_) return;
    take_started_ = true;
    take_t0_ = std::chrono::steady_clock::now();
}

namespace {
const char* const kCallKinds[] = {
    "window", "preview", "flush_window", "flush_tail", "redecode"};
}

void StreamSession::record_call(const StreamCall& c) {
    totals_.add(c);
    for (size_t k = 0; k < std::size(kCallKinds); ++k)
        if (std::string(c.kind) == kCallKinds[k]) by_kind_[k].add(c);
    if (flushing_) flush_totals_.add(c);
    const std::string r = c.result;
    if (r == "ok" || r == "reused")
        covered_end_ = std::max(covered_end_, c.abs_start + c.length);
    if (calls_.size() < kMaxStreamCalls) calls_.push_back(c);
    else ++calls_dropped_;
}

namespace {

std::string fmt3(double v) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.3f", v);
    return buf;
}

std::string seconds(int64_t samples) {
    return fmt3(static_cast<double>(samples) / kSampleRate);
}

std::string totals_json(const StreamCallTotals& t) {
    return "{\"calls\":" + std::to_string(t.calls)
         + ",\"engine_calls\":" + std::to_string(t.engine_calls)
         + ",\"engine_audio_s\":" + seconds(t.engine_samples)
         + ",\"engine_ms\":" + fmt3(t.engine_ms)
         + ",\"reused\":" + std::to_string(t.reused)
         + ",\"busy\":" + std::to_string(t.busy)
         + ",\"preempted\":" + std::to_string(t.preempted)
         + ",\"preempted_ms\":" + fmt3(t.preempted_ms) + "}";
}

} // namespace

std::string StreamSession::trace_preview_json() const {
    if (!chunker_) return "";
    return ",\"preview\":{\"min_s\":" + fmt3(chunker_->min_preview_seconds())
         + ",\"interval_s\":" + fmt3(chunker_->preview_interval())
         + ",\"effective_interval_s\":" + fmt3(chunker_->effective_interval())
         + ",\"coalesced\":" + std::to_string(chunker_->coalesced_previews())
         + "}";
}

std::string StreamSession::trace_partial_json() const {
    return "{\"v\":1,\"t_ms\":" + fmt3(take_ms())
         + ",\"audio_s\":" + seconds(trimmed_samples_
                                        + static_cast<int64_t>(samples_.size()))
         + ",\"covered_s\":" + seconds(covered_end_)
         + ",\"totals\":" + totals_json(totals_) + trace_preview_json() + "}";
}

std::string StreamSession::trace_final_json() const {
    std::string by_kind = "{";
    for (size_t k = 0; k < std::size(kCallKinds); ++k) {
        if (k) by_kind += ",";
        by_kind += "\"" + std::string(kCallKinds[k]) + "\":"
                 + totals_json(by_kind_[k]);
    }
    by_kind += "}";
    std::string list = "[";
    for (size_t i = 0; i < calls_.size(); ++i) {
        const auto& c = calls_[i];
        if (i) list += ",";
        list += "{\"kind\":\"" + std::string(c.kind)
              + "\",\"start_s\":" + seconds(c.abs_start)
              + ",\"end_s\":" + seconds(c.abs_start + c.length)
              + ",\"t0_ms\":" + fmt3(c.t0_ms)
              + ",\"t1_ms\":" + fmt3(c.t1_ms)
              + ",\"result\":\"" + c.result + "\"}";
    }
    list += "]";
    return "{\"v\":1,\"t_ms\":" + fmt3(take_ms())
         + ",\"audio_s\":" + seconds(trimmed_samples_
                                        + static_cast<int64_t>(samples_.size()))
         + ",\"covered_s\":" + seconds(covered_end_)
         + ",\"totals\":" + totals_json(totals_) + trace_preview_json()
         + ",\"by_kind\":" + by_kind
         + ",\"stop\":{\"path\":\"" + final_path_
         + "\",\"t0_ms\":" + fmt3(flush_t0_ms_)
         + ",\"t1_ms\":" + fmt3(flush_t1_ms_)
         + ",\"unfinalized_s\":" + seconds(flush_unfinalized_)
         + ",\"totals\":" + totals_json(flush_totals_) + "}"
         + ",\"calls_dropped\":" + std::to_string(calls_dropped_)
         + ",\"calls\":" + list + "}";
}

} // namespace starling::serve
