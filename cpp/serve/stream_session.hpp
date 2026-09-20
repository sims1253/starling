// stream_session.hpp — real-time streaming dictation session.
//
// Fixed-length overlapping windows bound each transcription call. Word matching
// reduces overlap duplication but cannot guarantee agreement across windows.
#pragma once

#include "server.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace starling::serve {

// Append `new_words` to `committed`, deduping the overlapping boundary words.
// Looks at the last/first `max_overlap` words of each side (the shared region)
// and aligns the longest common run there. If no run of >= min_match words is
// found, the two are simply concatenated (rare; a duplicated word reads better
// than a dropped one for dictation).
//
// This is a direct port of stitch_words() from src/starling/stream_chunk.py.
std::vector<std::string> stitch_words(
    const std::vector<std::string>& committed,
    const std::vector<std::string>& new_words,
    int max_overlap = 24,
    int min_match = 2);

// Split a string on whitespace into words (matching Python's str.split()).
std::vector<std::string> split_words(const std::string& s);

// Join words with a single space (matching Python's " ".join()).
std::string join_words(const std::vector<std::string>& words);

// Normalize a word for overlap matching: lowercase ASCII, strip ASCII
// punctuation/whitespace, keep non-ASCII (UTF-8) bytes verbatim. Port of
// _norm() from stream_chunk.py, conservatively Unicode-aware (issue #118):
// the key is deterministic per word and identical across chunk boundaries.
std::string norm_word(const std::string& word);

// Transcribe function: takes a window of mono float32 samples, returns text or
// std::nullopt if the transcriber is busy (should retry without advancing state).
using TranscribeFn = std::function<std::optional<std::string>(const float*, int64_t)>;

// ---- stream window config validation (issue #146) ---------------------------
// Strict numeric parse for CLI flags: the whole string must be one finite
// number. std::stod/std::stoi alone accept partial parses ("3abc" -> 3) and
// std::stod also accepts non-finite tokens ("nan", "inf"); both must be
// rejected before the values reach the chunker.
std::optional<double> parse_double_strict(const std::string& text);
std::optional<int> parse_int_strict(const std::string& text);

// Validate the stream window configuration: finite values, nonnegative
// overlap/min/partial-interval, and the window/overlap relationships (the
// window must span at least one sample and its sample counts must fit the
// int counters; overlap must stay below the chunk). Returns an empty string
// when valid, otherwise a human-readable error.
//
// `chunk_seconds == 0` is valid ONLY at the CLI level (it selects the legacy
// whole-buffer mode and no chunker is constructed); ChunkStreamer's
// constructor rejects it separately because a chunker needs a real window.
std::string stream_window_config_error(int sample_rate, double chunk_seconds,
                                       double overlap_seconds, double min_seconds,
                                       double partial_interval);

// Outcome of appending one binary audio frame to the session
// (StreamSession::append_pcm / append_wav). A rejection invalidates the take:
// the session refuses further audio and the WS layer reports the failure to
// the client (a structured error frame) instead of committing an incomplete
// capture as an ordinary successful final (issue #145). reset() clears the
// invalidation and re-arms audio acceptance.
enum class AppendOutcome {
    Accepted,      // audio appended (empty frames are accepted no-ops)
    MalformedWav,  // RIFF/WAVE frame the WAV decoder rejects
    RateMismatch,  // WAV decodes but its sample rate is not 16 kHz
    OddPcmLength,  // raw PCM16 with an odd byte count (a split sample)
    Overflowed,    // refused: the per-connection buffer cap tripped
    TakeInvalid,   // refused: the take was already invalidated
};

// ChunkStreamer: rolling fixed-window overlapping-chunk transcription state.
// Direct port of ChunkStreamer from src/starling/stream_chunk.py.
// Throws std::invalid_argument when the window configuration is invalid
// (see stream_window_config_error; chunk_seconds must be positive here).
class ChunkStreamer {
public:
    ChunkStreamer(int sample_rate, double chunk_seconds, double overlap_seconds,
                  double min_seconds, double partial_interval);

    // Advance streaming state for the current buffer. Finalizes any full
    // windows, then (throttled) transcribes the live tail for a responsive
    // partial. Returns the full text to emit, or std::nullopt.
    std::optional<std::string> step(const std::vector<float>& samples,
                                     double now, const TranscribeFn& tx);

    // Finalize all remaining audio (on commit) and return the full text.
    // After bounded busy retries, returns nullopt; retain audio and retry commit.
    std::optional<std::string> flush(const std::vector<float>& samples, const TranscribeFn& tx);

    void reset();

    // The sample index up to which audio is fully finalized (for buffer trimming).
    int64_t boundary() const { return boundary_; }

    // Adjust the boundary after samples are dropped from the front of the buffer.
    // Called by StreamSession::maybe_trim_samples to keep the chunker aligned
    // with the shifted samples_ buffer.
    void rebase(int64_t dropped) {
        boundary_ = std::max<int64_t>(0, boundary_ - dropped);
    }

private:
    bool finalize_full_windows(const std::vector<float>& samples, const TranscribeFn& tx);

    int sr_;
    int chunk_;           // chunk size in samples
    int overlap_;         // overlap in samples
    int advance_;         // chunk - overlap
    int min_;             // minimum samples for a partial
    double partial_interval_;
    int max_overlap_words_;

    std::vector<std::string> committed_;
    int64_t boundary_ = 0;  // sample index; audio before this is finalized
    double last_emit_ = 0.0;
};

// StreamSession: per-connection rolling audio buffer + streaming state.
// Direct port of StreamSession from src/starling/server.py.
class StreamSession {
public:
    explicit StreamSession(StarlingServer* server);

    // Append raw PCM16 bytes (little-endian int16 → float32).
    //
    // Raw PCM frames are defined as sequences of whole int16 samples: an odd
    // byte count means a sample was split mid-frame at a transport boundary.
    // The session keeps the take honest the same way it does for WAV rejects
    // (issue #145): the frame is refused and the take is invalidated rather
    // than silently dropping the dangling byte. The client re-sends on
    // whole-sample boundaries (e.g. even-length binary frames).
    AppendOutcome append_pcm(const std::string& bytes);
    // Append WAV bytes (decoded via dr_wav). Non-RIFF/WAVE payloads fall back
    // to append_pcm.
    AppendOutcome append_wav(const std::string& bytes);

    // True when a frame exceeded the per-connection buffer cap
    // (config.max_stream_seconds): the frame was refused and every further
    // append is a no-op until reset(). The cap bounds the LIVE rolling
    // buffer (memory): finalized windows are trimmed from the buffer, so
    // long dictation sessions without commits keep memory bounded without
    // tripping the cap. The WS layer reports overflow to the client as an
    // error frame.
    bool overflowed() const { return overflow_; }

    // True when a malformed-WAV / sample-rate / odd-PCM rejection invalidated
    // the current take (see AppendOutcome). While set, appends are refused
    // (TakeInvalid) and commit must not emit an ordinary successful final —
    // the buffered audio is incomplete, so the client must reset() and fall
    // back to its authoritative local WAV (issue #145). reset() clears it.
    // overflow_ is independent: it also refuses audio, but reports the
    // buffer-cap error instead and does not itself mark the take invalid.
    bool take_invalid() const { return take_invalid_; }
    // Short machine-readable code for the rejection that invalidated the
    // take ("malformed_wav", "sample_rate_mismatch", "odd_pcm_length").
    // Empty unless take_invalid().
    const std::string& invalid_reason() const { return invalid_reason_; }

    // Advance the chunked stream; returns text to emit as a partial, or nullopt.
    std::optional<std::string> stream_step(double now);

    // Finalize all buffered audio, or nullopt if busy; retain audio for retry.
    std::optional<std::string> stream_flush();

    void reset();

    double buffered_seconds() const;
    double live_seconds() const;

    // Build the chunked-streaming transcribe callback.
    TranscribeFn make_transcribe_fn(RequestContext* ctx);

    // Override the transcribe callback (unit tests inject a fake; production
    // uses the server-backed make_transcribe_fn).
    void set_transcribe_fn(TranscribeFn fn) { custom_tx_ = std::move(fn); }

    // ---- exact streaming-tail result reuse (S11) ----------------------------
    // A successful preview followed immediately by a commit used to call the
    // engine again on the byte-identical tail window (two whole-model calls
    // for one answer). The session retains the LAST successful raw window
    // result, keyed by complete identity, and replays it for any later call
    // with the same key — one engine callback becomes zero. Reuse is
    // exact-input only: the retained entry never crosses an audio append, a
    // reset, a take invalidation or an engine-identity change, and only
    // complete successes are retained (empty text IS a success; busy,
    // cancelled and failed calls never enter the entry).
    //
    // Engine identity: everything about the request that can change the raw
    // window output — model slug, gguf artifact (which encodes the
    // quantization), backend, and the stream window/overlap policy. The
    // native server fixes these per process; set_engine_identity() is the
    // invalidating hook for a reload/re-config (and for tests).
    void set_engine_identity(std::string id);
    const std::string& engine_identity() const { return engine_id_; }

    // Number of transcribe calls answered from the retained exact-tail entry
    // (the engine was not invoked). Observability for tests and for reporting
    // inference calls avoided, separately from any latency claims.
    int64_t tail_cache_hits() const { return tail_cache_hits_; }

private:
    void maybe_trim_samples();

    // Identity of one raw window result (S11): the exact sample window
    // (absolute start index including the trimmed prefix, and length), the
    // audio revision when the engine produced it, and the engine identity.
    // Equal keys mean the two calls saw byte-identical samples through the
    // same model/config/backend, so the earlier result answers the later one.
    struct StreamTailKey {
        int64_t abs_start = -1;   // absolute sample index of the window start
        int64_t length = 0;       // window length in samples
        uint64_t audio_rev = 0;   // audio revision at production time
        std::string engine_id;    // model/quant/backend/window-config identity
        bool operator==(const StreamTailKey& o) const {
            return abs_start == o.abs_start && length == o.length
                && audio_rev == o.audio_rev && engine_id == o.engine_id;
        }
    };

    // The transcribe callback actually used by stream_step/stream_flush: the
    // custom (test) or server callback wrapped with exact-tail reuse.
    TranscribeFn active_tx();

    // Drop the retained exact-tail entry (any event that could change the
    // answer for a future window: append, reset, take invalidation, engine
    // identity change).
    void invalidate_tail_result();

    StarlingServer* server_;
    std::vector<float> samples_;
    double last_partial_ts_ = 0.0;
    int64_t trimmed_samples_ = 0;
    double max_buffer_seconds_ = 0.0;  // from config; 0 = unlimited
    bool overflow_ = false;
    bool take_invalid_ = false;   // set by an append rejection (issue #145)
    std::string invalid_reason_;  // machine-readable code for the rejection
    TranscribeFn custom_tx_;  // when set, used instead of the server callback
    std::unique_ptr<ChunkStreamer> chunker_;

    // ---- exact streaming-tail result reuse (S11) ----------------------------
    // Bumped on every append that actually adds samples and NEVER reset: keys
    // recorded before a reset can never collide with keys recorded after it,
    // even though absolute sample indices restart at 0.
    uint64_t audio_rev_ = 0;
    std::string engine_id_;       // identity snapshot (see set_engine_identity)
    bool tail_valid_ = false;     // retained entry below is meaningful
    StreamTailKey tail_key_;      // key of the retained result (iff tail_valid_)
    std::string tail_text_;       // the raw window result ("" is a success)
    int64_t tail_cache_hits_ = 0; // calls answered from the retained entry
};

} // namespace starling::serve
