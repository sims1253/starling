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
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <thread>

namespace starling::serve {

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
            // Empty keys never participate in a match (issue #118 defense in
            // depth): words that normalize to "" (pure punctuation like "--")
            // would otherwise align as runs and drop unrelated boundary words.
            if (!a[a_lo + i - 1].empty() && a[a_lo + i - 1] == b[b_lo + j - 1]) {
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

ChunkStreamer::ChunkStreamer(int sample_rate, double chunk_seconds,
                             double overlap_seconds, double min_seconds,
                             double partial_interval)
    : sr_(0),
      chunk_(0),
      overlap_(0),
      advance_(1),
      min_(0),
      partial_interval_(0.0),
      max_overlap_words_(0) {
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
    if (tail_len >= chunk_) {  // a full window is still waiting for a retry
        return finalized ? std::optional<std::string>(join_words(committed_))
                         : std::nullopt;
    }
    bool throttled = (now - last_emit_) < partial_interval_;
    if (!finalized && (throttled || tail_len < min_)) {
        return std::nullopt;
    }
    last_emit_ = now;

    if (tail_len > 0 && tail_len >= min_) {
        auto text = tx(samples.data() + boundary_, tail_len);
        if (!text.has_value()) {
            // Busy on the tail.
            return finalized ? std::optional<std::string>(join_words(committed_))
                             : std::nullopt;
        }
        return join_words(stitch_words(committed_, split_words(*text),
                                       max_overlap_words_));
    }
    return finalized ? std::optional<std::string>(join_words(committed_))
                     : std::nullopt;
}

std::optional<std::string> ChunkStreamer::flush(
    const std::vector<float>& samples, const TranscribeFn& tx) {
    constexpr int kMaxRetries = 5;
    for (int attempt = 0; attempt < kMaxRetries; ++attempt) {
        finalize_full_windows(samples, tx);
        int64_t tail_len = static_cast<int64_t>(samples.size()) - boundary_;
        if (tail_len == 0) return join_words(committed_);
        // Guard the tail's sign as well (issue #146): the window geometry is
        // validated at construction, but the transcriber contract (a
        // nonempty window inside the buffer) is enforced here regardless.
        if (tail_len > 0 && tail_len < chunk_) {
            auto text = tx(samples.data() + boundary_, tail_len);
            if (text.has_value()) {
                committed_ = stitch_words(committed_, split_words(*text),
                                          max_overlap_words_);
                boundary_ = static_cast<int64_t>(samples.size());
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
    // Engine identity for exact-tail reuse (S11): everything about the request
    // that can change a raw window result. The native server fixes the model
    // slug + gguf artifact (which encodes the quantization) per process and
    // the backend at link time; the window/overlap policy shapes the very
    // windows being keyed. std::to_string(double) is fixed-point, so the
    // string is deterministic. There is no language/normalization parameter
    // on the native streaming path (nothing extra to key on); a hypothetical
    // reload/re-config goes through set_engine_identity(), which invalidates.
    engine_id_ = cfg.model_slug + "|" + cfg.gguf_path + "|"
               + starling_ggml_backend_name() + "|chunk="
               + std::to_string(cfg.stream_chunk_seconds)
               + "|overlap=" + std::to_string(cfg.stream_overlap_seconds)
               + "|min=" + std::to_string(cfg.min_chunk_seconds)
               + "|partial=" + std::to_string(cfg.partial_interval);
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
        && live_seconds()
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
            && live_seconds()
                   + static_cast<double>(decoded.size()) / kSampleRate
                 > max_buffer_seconds_) {
            overflow_ = true;
            return AppendOutcome::Overflowed;
        }
        size_t old = samples_.size();
        samples_.resize(old + decoded.size());
        std::copy(decoded.begin(), decoded.end(), samples_.begin() + old);
        ++audio_rev_;  // any appended audio invalidates exact-tail reuse (S11)
    }
    maybe_trim_samples();
    return AppendOutcome::Accepted;
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
    return [this, inner = std::move(inner), tx_gen,
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
        key.engine_id = engine_id;
        key.tx_gen = tx_gen;
        if (tail_valid_ && tail_key_ == key) {
            ++tail_cache_hits_;
            return tail_text_;  // exact-input reuse: engine not called
        }
        std::optional<std::string> result = inner(p, n);
        if (result.has_value()) {
            // Retain exactly one entry: this success replaces any previous
            // one (bounded: one entry per session, never a growing cache).
            tail_valid_ = true;
            tail_key_ = key;
            tail_text_ = *result;
        }
        return result;
    };
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
}

std::optional<std::string> StreamSession::stream_step(double now) {
    if (!chunker_) return std::nullopt;
    TranscribeFn tx = active_tx();
    return chunker_->step(samples_, now, tx);
}

std::optional<std::string> StreamSession::stream_flush() {
    if (!chunker_) return "";
    TranscribeFn tx = active_tx();
    return chunker_->flush(samples_, tx);
}

void StreamSession::reset() {
    samples_.clear();
    last_partial_ts_ = 0.0;
    trimmed_samples_ = 0;
    overflow_ = false;
    take_invalid_ = false;
    invalid_reason_.clear();
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
}

double StreamSession::buffered_seconds() const {
    return static_cast<double>(trimmed_samples_ + samples_.size()) / kSampleRate;
}

double StreamSession::live_seconds() const {
    return static_cast<double>(samples_.size()) / kSampleRate;
}

} // namespace starling::serve
