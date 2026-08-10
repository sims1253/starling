// stream_session.hpp — real-time streaming dictation session.
//
// A faithful C++ port of the Python streaming logic in src/starling/server.py
// (StreamSession) and src/starling/stream_chunk.py (ChunkStreamer).
//
// The rolling audio buffer is finalized in fixed-length overlapping windows:
//   - Each window is a constant mel length → cudagraph encoder reuse.
//   - Work is O(N): each second of audio is transcribed a bounded number of
//     times.
//   - The prompt per transcribe is bounded by the window, so the KV cache is
//     never exceeded regardless of total session length.
//   - Consecutive windows overlap, so boundary words are deduped via
//     stitch_words (longest-common-subsequence match on the overlap region).
#pragma once

#include "server.hpp"

#include <functional>
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

// Normalize a word for overlap matching: lowercase, strip punctuation.
// Direct port of _norm() from stream_chunk.py.
std::string norm_word(const std::string& word);

// Transcribe function: takes a window of mono float32 samples, returns text or
// std::nullopt if the transcriber is busy (should retry without advancing state).
using TranscribeFn = std::function<std::optional<std::string>(const float*, int64_t)>;

// ChunkStreamer: rolling fixed-window overlapping-chunk transcription state.
// Direct port of ChunkStreamer from src/starling/stream_chunk.py.
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
    // Retries the tail window a few times if the transcriber is busy.
    std::string flush(const std::vector<float>& samples, const TranscribeFn& tx);

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
    void append_pcm(const std::string& bytes);
    // Append WAV bytes (decoded via dr_wav).
    void append_wav(const std::string& bytes);

    // Advance the chunked stream; returns text to emit as a partial, or nullopt.
    std::optional<std::string> stream_step(double now);

    // Finalize all buffered audio on commit; returns the full text.
    std::string stream_flush();

    void reset();

    double buffered_seconds() const;
    double live_seconds() const;

    // Build the chunked-streaming transcribe callback.
    TranscribeFn make_transcribe_fn(RequestContext* ctx);

private:
    void maybe_trim_samples();

    StarlingServer* server_;
    std::vector<float> samples_;
    double last_partial_ts_ = 0.0;
    int64_t trimmed_samples_ = 0;
    std::unique_ptr<ChunkStreamer> chunker_;
};

} // namespace starling::serve
