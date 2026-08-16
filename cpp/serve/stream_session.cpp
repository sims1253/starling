// stream_session.cpp — streaming dictation session implementation.
//
// Faithful C++ port of src/starling/server.py (StreamSession) and
// src/starling/stream_chunk.py (ChunkStreamer + stitch_words).

#include "stream_session.hpp"
#include "audio.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <thread>

namespace starling::serve {

// ---- word helpers ---------------------------------------------------------

std::string norm_word(const std::string& word) {
    // Lowercase, strip non-word chars (port of _norm() from stream_chunk.py).
    std::string out;
    out.reserve(word.size());
    for (char c : word) {
        unsigned char uc = static_cast<unsigned char>(c);
        if (uc >= 'A' && uc <= 'Z') {
            out += static_cast<char>(uc - 'A' + 'a');
        } else if (std::isalnum(uc) || c == '\'') {
            out += c;
        }
        // else: strip punctuation
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

// Find the longest common substring (in terms of word index runs) between
// a[0..na) and b[0..nb). This is the C++ equivalent of
// difflib.SequenceMatcher.find_longest_match(). Returns {a_start, b_start, size}.
struct Match { int a_start; int b_start; int size; };

static Match find_longest_match(
    const std::vector<std::string>& a, int a_lo, int a_hi,
    const std::vector<std::string>& b, int b_lo, int b_hi) {
    // O((a_hi-a_lo)*(b_hi-b_lo)) — fine for max_overlap=24.
    int best_i = a_lo, best_j = b_lo, best_k = 0;
    // b2j: map from word to list of positions in b (within [b_lo, b_hi)).
    // Simple approach: for each starting position in a, extend in b.
    int na = a_hi - a_lo;
    int nb = b_hi - b_lo;
    // dp approach: longest common extension from each (i,j) pair.
    // dp[i][j] = length of the longest common run ending at a[a_lo+i], b[b_lo+j].
    std::vector<std::vector<int>> dp(na + 1, std::vector<int>(nb + 1, 0));
    for (int i = 1; i <= na; ++i) {
        for (int j = 1; j <= nb; ++j) {
            if (a[a_lo + i - 1] == b[b_lo + j - 1]) {
                dp[i][j] = dp[i-1][j-1] + 1;
                if (dp[i][j] > best_k) {
                    best_k = dp[i][j];
                    best_i = a_lo + i - best_k;
                    best_j = b_lo + j - best_k;
                }
            }
        }
    }
    return {best_i, best_j, best_k};
}

std::vector<std::string> stitch_words(
    const std::vector<std::string>& committed,
    const std::vector<std::string>& new_words,
    int max_overlap,
    int min_match) {
    if (committed.empty()) return new_words;
    if (new_words.empty()) return committed;

    // tail = last max_overlap words of committed; head = first max_overlap of new.
    int tail_start = static_cast<int>(committed.size()) - max_overlap;
    if (tail_start < 0) tail_start = 0;
    int head_end = std::min(max_overlap, static_cast<int>(new_words.size()));

    // Normalize for matching.
    std::vector<std::string> a_norm, b_norm;
    for (int i = tail_start; i < static_cast<int>(committed.size()); ++i)
        a_norm.push_back(norm_word(committed[i]));
    for (int i = 0; i < head_end; ++i)
        b_norm.push_back(norm_word(new_words[i]));

    Match m = find_longest_match(
        a_norm, 0, static_cast<int>(a_norm.size()),
        b_norm, 0, static_cast<int>(b_norm.size()));

    if (m.size >= min_match) {
        // keep = committed up to the end of the matched run in committed.
        // The matched run in committed starts at tail_start + m.a_start,
        // length m.size.
        int keep = tail_start + m.a_start + m.size;
        int start = m.b_start + m.size;  // new words after the matched run.
        std::vector<std::string> result(committed.begin(),
                                        committed.begin() + keep);
        result.insert(result.end(),
                      new_words.begin() + start, new_words.end());
        return result;
    }
    // No match: concatenate.
    std::vector<std::string> result = committed;
    result.insert(result.end(), new_words.begin(), new_words.end());
    return result;
}

// ---- ChunkStreamer --------------------------------------------------------

ChunkStreamer::ChunkStreamer(int sample_rate, double chunk_seconds,
                             double overlap_seconds, double min_seconds,
                             double partial_interval)
    : sr_(sample_rate),
      chunk_(static_cast<int>(chunk_seconds * sample_rate)),
      overlap_(static_cast<int>(
          std::min(overlap_seconds, chunk_seconds * 0.5) * sample_rate)),
      advance_(std::max(1, chunk_ - overlap_)),
      min_(static_cast<int>(min_seconds * sample_rate)),
      partial_interval_(partial_interval),
      max_overlap_words_(std::max(8, static_cast<int>(overlap_seconds * 6) + 6)) {
    if (chunk_ < 1) chunk_ = 1;
    if (overlap_ < 0) overlap_ = 0;
}

bool ChunkStreamer::finalize_full_windows(
    const std::vector<float>& samples, const TranscribeFn& tx) {
    bool did = false;
    while (static_cast<int64_t>(samples.size()) - boundary_ >= chunk_) {
        int64_t start = boundary_;
        int64_t len = chunk_;
        auto text = tx(samples.data() + start, len);
        if (!text.has_value()) break;  // busy → stop, boundary unchanged
        committed_ = stitch_words(committed_, split_words(*text),
                                  max_overlap_words_);
        boundary_ += advance_;
        did = true;
    }
    return did;
}

std::optional<std::string> ChunkStreamer::step(
    const std::vector<float>& samples, double now, const TranscribeFn& tx) {
    bool finalized = finalize_full_windows(samples, tx);

    int64_t tail_len = static_cast<int64_t>(samples.size()) - boundary_;
    bool throttled = (now - last_emit_) < partial_interval_;
    if (!finalized && (throttled || tail_len < min_)) {
        return std::nullopt;
    }
    last_emit_ = now;

    if (tail_len >= min_) {
        auto text = tx(samples.data() + boundary_, tail_len);
        if (!text.has_value()) {
            // Busy on the tail.
            return finalized ? std::optional<std::string>(join_words(committed_))
                             : std::nullopt;
        }
        auto words = split_words(*text);
        auto combined = committed_;
        combined.insert(combined.end(), words.begin(), words.end());
        return join_words(combined);
    }
    return finalized ? std::optional<std::string>(join_words(committed_))
                     : std::nullopt;
}

std::string ChunkStreamer::flush(
    const std::vector<float>& samples, const TranscribeFn& tx) {
    constexpr int kMaxRetries = 5;
    constexpr double kBackoffS = 0.05;

    finalize_full_windows(samples, tx);
    int64_t tail_len = static_cast<int64_t>(samples.size()) - boundary_;
    if (tail_len > 0) {
        std::optional<std::string> text;
        for (int attempt = 0; attempt < kMaxRetries; ++attempt) {
            text = tx(samples.data() + boundary_, tail_len);
            if (text.has_value()) break;
            std::this_thread::sleep_for(
                std::chrono::microseconds(static_cast<int64_t>(kBackoffS * 1e6)));
        }
        if (text.has_value()) {
            committed_ = stitch_words(committed_, split_words(*text),
                                      max_overlap_words_);
            boundary_ = static_cast<int64_t>(samples.size());
        } else {
            std::fprintf(stderr,
                "[starling-serve] flush: dropped untranscribed tail (%lld samples) "
                "after %d retries\n",
                static_cast<long long>(tail_len), kMaxRetries);
        }
    }
    return join_words(committed_);
}

void ChunkStreamer::reset() {
    committed_.clear();
    boundary_ = 0;
    last_emit_ = 0.0;
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
}

TranscribeFn StreamSession::make_transcribe_fn(RequestContext* ctx) {
    return [this, ctx](const float* samples, int64_t n)
               -> std::optional<std::string> {
        std::string err;
        // Streaming chunks never wait behind queued requests: if the serial
        // queue is occupied, report busy and let the chunker retry later
        // (matching the Python StreamSession._tx behavior).
        auto result = server_->transcribe_pcm(samples, n, ctx, &err,
                                              QueuePolicy::SkipIfBusy);
        if (!err.empty()) {
            // "server busy" or "cancelled" → return nullopt (retry without
            // advancing state, matching the Python StreamSession._tx behavior).
            if (err == "server busy" || err == "cancelled") return std::nullopt;
            // Other errors: also treat as busy (non-fatal in streaming).
            return std::nullopt;
        }
        return result.text;
    };
}

void StreamSession::append_pcm(const std::string& bytes) {
    if (overflow_) return;  // capped: refuse audio until reset()
    const size_t nbytes = bytes.size();
    if (nbytes == 0) return;
    size_t nsamples = nbytes / 2;
    // Drop odd trailing byte.
    if (nbytes % 2 == 1) nsamples = (nbytes - 1) / 2;
    if (nsamples == 0) return;
    if (max_buffer_seconds_ > 0.0
        && buffered_seconds()
               + static_cast<double>(nsamples) / kSampleRate
             > max_buffer_seconds_) {
        overflow_ = true;
        return;
    }
    const auto* src = reinterpret_cast<const int16_t*>(bytes.data());
    size_t old = samples_.size();
    samples_.resize(old + nsamples);
    for (size_t i = 0; i < nsamples; ++i) {
        samples_[old + i] = static_cast<float>(src[i]) / 32768.0f;
    }
    maybe_trim_samples();
}

void StreamSession::append_wav(const std::string& bytes) {
    if (overflow_) return;  // capped: refuse audio until reset()
    // Check for RIFF/WAVE header.
    if (bytes.size() < 12 || bytes.substr(0, 4) != "RIFF"
        || bytes.substr(8, 4) != "WAVE") {
        // Treat as raw PCM16.
        append_pcm(bytes);
        return;
    }
    std::vector<float> decoded;
    int sr = 0;
    if (!audio::wav_bytes_to_float32(bytes, decoded, sr)) {
        std::fprintf(stderr,
            "[starling-serve] dropping malformed WAV chunk (len=%zu)\n",
            bytes.size());
        return;
    }
    // Resample if needed (simple: if sr != 16k, we can't resample in C++ easily;
    // assume 16k or let the engine handle it — the C API checks).
    if (sr != kSampleRate) {
        std::fprintf(stderr,
            "[starling-serve] dropping WAV chunk: sample rate %d != %d\n",
            sr, kSampleRate);
        return;
    }
    if (!decoded.empty()) {
        if (max_buffer_seconds_ > 0.0
            && buffered_seconds()
                   + static_cast<double>(decoded.size()) / kSampleRate
                 > max_buffer_seconds_) {
            overflow_ = true;
            return;
        }
        size_t old = samples_.size();
        samples_.resize(old + decoded.size());
        std::copy(decoded.begin(), decoded.end(), samples_.begin() + old);
    }
    maybe_trim_samples();
}

void StreamSession::maybe_trim_samples() {
    if (!chunker_) return;
    int64_t b = chunker_->boundary();
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

std::optional<std::string> StreamSession::stream_step(double now) {
    if (!chunker_) return std::nullopt;
    TranscribeFn tx = custom_tx_ ? custom_tx_ : make_transcribe_fn(nullptr);
    return chunker_->step(samples_, now, tx);
}

std::string StreamSession::stream_flush() {
    if (!chunker_) return "";
    TranscribeFn tx = custom_tx_ ? custom_tx_ : make_transcribe_fn(nullptr);
    return chunker_->flush(samples_, tx);
}

void StreamSession::reset() {
    samples_.clear();
    last_partial_ts_ = 0.0;
    trimmed_samples_ = 0;
    overflow_ = false;
    if (chunker_) chunker_->reset();
}

double StreamSession::buffered_seconds() const {
    return static_cast<double>(trimmed_samples_ + samples_.size()) / kSampleRate;
}

double StreamSession::live_seconds() const {
    return static_cast<double>(samples_.size()) / kSampleRate;
}

} // namespace starling::serve
