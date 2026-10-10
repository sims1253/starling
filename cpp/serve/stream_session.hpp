// stream_session.hpp — real-time streaming dictation session.
//
// Fixed-length overlapping windows bound each transcription call. Word matching
// reduces overlap duplication but cannot guarantee agreement across windows.
#pragma once

#include "server.hpp"

#include <algorithm>
#include <chrono>
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

// Asked right before a preview would start: true when newer audio (or a
// control message) is already queued behind the current step, so the
// preview would be obsolete before it finished (issue #357).
using PendingFn = std::function<bool()>;

// Preview cadence bound (issue #357): the effective preview interval is
// at least the latest preview's engine time divided by this duty, so on a
// slow model or device previews take at most this fraction of wall time
// instead of running back to back. Window and finalization work is never
// throttled.
constexpr double kMaxPreviewDuty = 0.5;

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

// Validate a preview cadence (issue #357): finite, nonnegative, and a
// minimum whose sample count fits the counters. Empty string when valid.
std::string preview_policy_error(int sample_rate, double min_seconds,
                                 double interval_seconds);

// ChunkStreamer: rolling fixed-window overlapping-chunk transcription state.
// Direct port of ChunkStreamer from src/starling/stream_chunk.py.
// Throws std::invalid_argument when the window configuration is invalid
// (see stream_window_config_error; chunk_seconds must be positive here).
//
// The window geometry (chunk/overlap) decides what is committed; the
// preview cadence (min_seconds/partial_interval) only decides when the live
// tail is previewed (issue #357). `min_seconds` is the take's first-partial
// minimum: once the take holds that much audio, every nonempty tail is
// eligible, including the overlap right after a window commit.
class ChunkStreamer {
public:
    ChunkStreamer(int sample_rate, double chunk_seconds, double overlap_seconds,
                  double min_seconds, double partial_interval);

    // Replace the preview cadence (per connection); throws
    // std::invalid_argument on an invalid policy (preview_policy_error).
    void set_preview_policy(double min_seconds, double interval_seconds);

    // Advance streaming state for the current buffer. Finalizes any full
    // windows (always), then (throttled) transcribes the live tail for a
    // responsive partial. When `newer_pending` reports queued audio right
    // before the preview, the preview is skipped (coalesced): the next step
    // previews the newer audio instead, and a pending committed-text update
    // is carried over to it. Returns the full text to emit, or std::nullopt.
    std::optional<std::string> step(const std::vector<float>& samples,
                                     double now, const TranscribeFn& tx,
                                     const PendingFn& newer_pending = nullptr);

    // Finalize all remaining audio (on commit) and return the full text.
    // After bounded busy retries, returns nullopt; retain audio and retry commit.
    std::optional<std::string> flush(const std::vector<float>& samples, const TranscribeFn& tx);

    void reset();

    // The sample index up to which audio is fully finalized (for buffer trimming).
    int64_t boundary() const { return boundary_; }

    // How many leading words of every text this streamer returns from now
    // on are fixed: no later window, tail or flush can change them. Only
    // the last `max_overlap_words` committed words take part in stitching,
    // and stitching never reaches below this count, so a client may treat
    // these words as final while the take is still being spoken. Grows
    // monotonically; 0 after reset().
    int64_t stable_words() const { return frozen_; }

    // Adjust the boundary after samples are dropped from the front of the buffer.
    // Called by StreamSession::maybe_trim_samples to keep the chunker aligned
    // with the shifted samples_ buffer.
    void rebase(int64_t dropped) {
        boundary_ = std::max<int64_t>(0, boundary_ - dropped);
        rebased_ += dropped;
    }

    // Previews skipped because newer audio was already queued.
    int64_t coalesced_previews() const { return coalesced_; }
    double min_preview_seconds() const {
        return static_cast<double>(min_) / sr_;
    }
    double preview_interval() const { return partial_interval_; }
    // The interval actually applied: the configured one, stretched so the
    // latest preview's engine time is at most kMaxPreviewDuty of it.
    double effective_interval() const {
        return std::max(partial_interval_, last_preview_cost_ / kMaxPreviewDuty);
    }

    // Why the transcribe call in flight was made (issue #226): "window" (a
    // full window committed while recording), "preview" (the live tail),
    // "flush_window" / "flush_tail" (finalization on commit). Set before
    // every transcribe call; the session's call ledger reads it.
    const char* call_kind() const { return call_kind_; }

private:
    bool finalize_full_windows(const std::vector<float>& samples,
                               const TranscribeFn& tx, bool flushing);
    // committed_ with `new_words` stitched onto its unfrozen tail.
    std::vector<std::string> stitched(const std::vector<std::string>& new_words) const;
    // Stitches `new_words` into committed_ and advances frozen_.
    void commit(const std::vector<std::string>& new_words);

    int sr_;
    int chunk_;           // chunk size in samples
    int overlap_;         // overlap in samples
    int advance_;         // chunk - overlap
    int min_;             // minimum samples for a partial
    double partial_interval_;
    int max_overlap_words_;

    std::vector<std::string> committed_;
    int64_t frozen_ = 0;    // leading committed_ words stitching never touches
    int64_t boundary_ = 0;  // sample index; audio before this is finalized
    int64_t rebased_ = 0;   // samples dropped before index 0 (rebase())
    double last_emit_ = 0.0;
    bool emit_due_ = false;          // a commit changed the text, not yet emitted
    double last_preview_cost_ = 0.0; // seconds, latest successful preview
    int64_t coalesced_ = 0;
    const char* call_kind_ = "window";
};

// ---- stream call ledger (issue #226) ----------------------------------------
// One transcribe call made by a streaming take: why it ran, which original
// audio it covered (absolute take sample indices, stable across buffer
// trims), when it started and ended (ms since the take's first audio), and
// how it ended. Overlapping windows and repeated previews of the same audio
// each appear, so the ledger shows the real inference work per recorded
// second instead of the batch throughput of one pass.
struct StreamCall {
    const char* kind = "";    // ChunkStreamer::call_kind()
    int64_t abs_start = 0;    // first sample covered
    int64_t length = 0;       // samples covered
    double t0_ms = 0.0;       // call start, ms since the take's first audio
    double t1_ms = 0.0;       // call end
    // "ok" (engine ran), "reused" (exact-tail reuse; engine not called),
    // "busy" (engine busy or cancelled; state not advanced), "timed_out".
    const char* result = "";
};

// Per-kind totals over a take (engine calls only; reused calls cost nothing).
struct StreamCallTotals {
    int64_t calls = 0;          // every call, whatever its result
    int64_t engine_calls = 0;   // result "ok" (the engine produced text)
    int64_t engine_samples = 0; // audio the engine transcribed, overlap included
    double engine_ms = 0.0;     // wall time of those engine calls
    int64_t reused = 0;
    int64_t busy = 0;           // busy, cancelled and timed-out calls
    void add(const StreamCall& c);
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
    // A blocking server queue timeout is terminal for this take; reset() is
    // required before more audio. Empty for transient busy responses.
    const std::string& terminal_error() const { return terminal_error_; }

    // Advance the chunked stream; returns text to emit as a partial, or
    // nullopt. `newer_pending` coalesces obsolete previews (ChunkStreamer::step).
    std::optional<std::string> stream_step(double now,
                                           const PendingFn& newer_pending = nullptr);

    // Per-connection preview cadence (issue #357); throws
    // std::invalid_argument when invalid. No-op in whole-buffer mode.
    void set_preview_policy(double min_seconds, double interval_seconds) {
        if (chunker_) chunker_->set_preview_policy(min_seconds, interval_seconds);
    }
    const ChunkStreamer* chunker() const { return chunker_.get(); }

    // The chunker's stable word count (ChunkStreamer::stable_words); 0 in
    // the legacy whole-buffer mode, where every partial is a fresh guess.
    int64_t stable_words() const { return chunker_ ? chunker_->stable_words() : 0; }

    // Finalize all buffered audio, or nullopt if busy/terminal. A blocking
    // queue timeout sets terminal_error(); transient busy retains audio.
    std::optional<std::string> stream_flush();

    void reset();

    double buffered_seconds() const;
    double live_seconds() const;

    // Build the chunked-streaming transcribe callback.
    TranscribeFn make_transcribe_fn(RequestContext* ctx);

    // Override the transcribe callback (unit tests inject a fake; production
    // uses the server-backed make_transcribe_fn). Every swap bumps the
    // transcribe-callback generation, which is part of the exact-tail
    // retention key: a retained result can only ever answer a call made under
    // the callback that produced it.
    void set_transcribe_fn(TranscribeFn fn) {
        custom_tx_ = std::move(fn);
        ++tx_gen_;
        wrapped_tx_ = nullptr;  // else the cached wrapper pairs stale inputs
    }

    // ---- exact streaming-tail result reuse (S11) ----------------------------
    // A successful preview followed immediately by a commit used to call the
    // engine again on the byte-identical tail window (two whole-model calls
    // for one answer). The session retains exactly ONE entry — the most
    // recent successful raw window result; each later success replaces the
    // previous entry (bounded: one entry per session, never a growing cache).
    // The entry is keyed by complete identity and replays for any later call
    // with the same key — one engine callback becomes zero. Reuse is
    // exact-input only: the retained entry never crosses an audio append, a
    // reset, a take invalidation, an engine-identity change, or a
    // transcribe-callback swap (set_transcribe_fn), and only complete
    // successes are retained (empty text IS a success; busy, cancelled and
    // failed calls never enter the entry).
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

    // ---- stream instrumentation (issue #226) --------------------------------
    // Opt-in per-take metadata for WS clients that ask for it (`trace=1` on
    // the /stream URL); the default wire contract carries none of it. Times
    // are steady-clock ms since the take's first accepted audio.
    //
    // trace_partial_json(): the object attached to a partial — received
    // audio, the end of the audio the text reflects (`covered_s`), and the
    // running inference totals.
    // trace_final_json(): the object attached to the final — totals per call
    // kind, the stop section (work done after commit, bounded by the
    // unfinalized tail on the healthy path) and the call ledger.
    std::string trace_partial_json() const;
    std::string trace_final_json() const;
    // The ledger itself (bounded; see kMaxStreamCalls) and its totals.
    const std::vector<StreamCall>& calls() const { return calls_; }
    const StreamCallTotals& totals() const { return totals_; }
    // Totals of the calls made by the latest stream_flush().
    const StreamCallTotals& flush_totals() const { return flush_totals_; }
    // How the latest stream_flush() produced its text: "tail" (the engine
    // finalized only the unfinalized remainder), "reused" (the exact-tail
    // result answered it), "committed" (nothing was left to finalize).
    const char* final_path() const { return final_path_; }

    // Ledger bound: a 10-minute take at a 0.5 s preview cadence makes ~1300
    // calls; beyond this, calls still count in the totals but are dropped
    // from the list (reported as `calls_dropped`).
    static constexpr size_t kMaxStreamCalls = 20000;

private:
    void maybe_trim_samples();

    // Identity of one raw window result (S11): the exact sample window
    // (absolute start index including the trimmed prefix, and length), the
    // audio revision when the engine produced it, the engine identity, and
    // the transcribe-callback generation. Equal keys mean the two calls saw
    // byte-identical samples through the same model/config/backend under the
    // same callback, so the earlier result answers the later one.
    struct StreamTailKey {
        int64_t abs_start = -1;   // absolute sample index of the window start
        int64_t length = 0;       // window length in samples
        uint64_t audio_rev = 0;   // audio revision at production time
        std::string engine_id;    // model/quant/backend/window-config identity
        uint64_t tx_gen = 0;      // transcribe-callback generation (PR #199)
        bool operator==(const StreamTailKey& o) const {
            return abs_start == o.abs_start && length == o.length
                && audio_rev == o.audio_rev && engine_id == o.engine_id
                && tx_gen == o.tx_gen;
        }
    };

    // The transcribe callback actually used by stream_step/stream_flush: the
    // custom (test) or server callback wrapped with exact-tail reuse. Built
    // once and cached in wrapped_tx_ (R29, issue #236): stream_step and
    // stream_flush call it per step, and rebuilding the wrapper each time
    // re-constructed the server callback and allocated a fresh closure even
    // when the call was about to be answered from the retained entry. The
    // wrapper's only inputs are custom_tx_, tx_gen_ and engine_id_, whose
    // sole mutators (set_transcribe_fn, set_engine_identity) drop the cache —
    // a cached wrapper always pairs current inputs. Returned BY VALUE: an
    // engine callback may swap the fn mid-step, clearing this slot while the
    // in-flight copy keeps its snapshot.
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
    std::string terminal_error_;  // nonempty after a blocking queue timeout
    TranscribeFn custom_tx_;  // when set, used instead of the server callback
    TranscribeFn wrapped_tx_;  // cached active_tx() wrapper (see active_tx)
    std::unique_ptr<ChunkStreamer> chunker_;

    // ---- exact streaming-tail result reuse (S11) ----------------------------
    // Bumped on every append that actually adds samples and on reset(); it
    // never returns to a past value: keys recorded before a reset can never
    // collide with keys recorded after it, even though absolute sample
    // indices restart at 0 — reset() enforces this by bumping, not by relying
    // on the retained-entry lifecycle.
    uint64_t audio_rev_ = 0;
    // Bumped by every set_transcribe_fn() and never reset: the callback
    // generation is part of the retention key, so a swapped-in callback is
    // never answered by the previous callback's retained result. Keying (not
    // clearing on swap) keeps the guarantee order-independent: active_tx()
    // writes the retained entry only after the callback returns, so an
    // invalidation performed at swap time could be overwritten by an in-flight
    // call's result — a generation recorded per entry makes the later match
    // itself fail instead. active_tx() snapshots the callback and its
    // generation TOGETHER when it is built (PR #199 batch-2): a call in
    // flight across a mid-step swap runs its captured callback keyed by the
    // generation that callback had at construction, so it can neither be
    // answered by nor retain into the other generation's entries.
    uint64_t tx_gen_ = 0;
    std::string engine_id_;       // identity snapshot (see set_engine_identity)
    bool tail_valid_ = false;     // retained entry below is meaningful
    StreamTailKey tail_key_;      // key of the retained result (iff tail_valid_)
    std::string tail_text_;       // the raw window result ("" is a success)
    int64_t tail_cache_hits_ = 0; // calls answered from the retained entry

    // ---- stream instrumentation (issue #226) --------------------------------
    double take_ms() const;       // ms since take_t0_ (0 before any audio)
    std::string trace_preview_json() const;  // ",\"preview\":{...}" or ""
    void mark_take_start();       // latch take_t0_ on the first audio
    void record_call(const StreamCall& c);
    bool take_started_ = false;
    std::chrono::steady_clock::time_point take_t0_{};
    std::vector<StreamCall> calls_;
    int64_t calls_dropped_ = 0;
    StreamCallTotals totals_;
    StreamCallTotals by_kind_[4];  // window, preview, flush_window, flush_tail
    StreamCallTotals flush_totals_;
    bool flushing_ = false;       // stream_flush() in progress
    int64_t covered_end_ = 0;     // abs end of the latest successful call
    double flush_t0_ms_ = -1.0;   // latest stream_flush() start (-1: none)
    double flush_t1_ms_ = -1.0;
    int64_t flush_unfinalized_ = 0;  // samples past the boundary at flush start
    const char* final_path_ = "";
};

} // namespace starling::serve
