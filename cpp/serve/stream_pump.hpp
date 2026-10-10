// stream_pump.hpp — one WS /stream connection's worker (issue #357).
//
// The transport thread only reads frames and queues them; a worker thread
// owns the StreamSession and runs every engine call. Before each streaming
// step the worker drains everything queued, so a step always sees all audio
// received so far: previews are computed on the newest audio, and previews
// that would be obsolete before they finish are skipped instead of decoded
// one stale frame at a time. A preview that is already running is cancelled
// when required work queues up behind it: a reset, a commit after newer
// audio, or audio that completes a window. Window commits and finalization
// are never skipped, and no audio is dropped: when the queue is full the
// transport thread blocks (TCP backpressure) until the worker catches up.
#pragma once

#include "stream_session.hpp"

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

namespace starling::serve {

// The error frame for a refused binary audio frame (buffer cap, malformed
// WAV, sample-rate mismatch, odd PCM length, invalidated take).
std::string ws_append_error(AppendOutcome outcome, const StreamSession& session,
                            double max_stream_seconds);

class StreamPump {
public:
    // Returns false when the frame could not be written (peer gone or
    // stalled past the transport's write timeout).
    using SendFn = std::function<bool(const std::string&)>;

    struct Options {
        bool trace = false;              // attach the trace object (?trace=1)
        double max_stream_seconds = 0.0; // for the buffer-cap error text
        // Queued-but-unprocessed audio bound. Above it, push_audio() blocks
        // (a single larger frame is still admitted into an empty queue).
        size_t max_pending_bytes = 32u * 1024u * 1024u;
        // Queued-but-unprocessed event bound (audio and control frames
        // alike, so pings or empty frames cannot grow the queue either).
        size_t max_pending_events = 4096;
    };

    // `session` must outlive the pump; it is only touched by the worker
    // until the pump is closed. `send` must be thread-safe against the
    // transport's own sends (httplib serializes WebSocket writes) and must
    // not block forever (httplib's server write timeout bounds it). A
    // failed send closes the pump: the peer cannot receive results, so the
    // worker stops its engine work and a transport thread blocked in a
    // push for queue space is released instead of waiting on a worker that
    // keeps writing into a dead socket.
    StreamPump(StreamSession& session, Options options, SendFn send);
    ~StreamPump();  // close() + join
    StreamPump(const StreamPump&) = delete;
    StreamPump& operator=(const StreamPump&) = delete;

    // Transport side. Events are processed in arrival order.
    void push_audio(std::string bytes);
    void push_commit();
    void push_reset();
    // Answered in order: the pong follows every earlier frame's processing
    // (including the step owed for earlier audio), as with the former
    // synchronous loop.
    void push_ping();
    // True once close() ran or a send failed.
    bool is_closed();
    // Stop the worker after its current step (the engine calls of one
    // append or commit: at most the windows of the --max-stream-seconds
    // buffer, as in the former synchronous loop); queued events are
    // discarded (the peer is gone). Idempotent.
    void close();

    // Wait until every queued event has been processed (tests).
    void drain();

private:
    enum class Kind { Audio, Commit, Reset, Ping };
    struct Event {
        Kind kind;
        std::string bytes;
    };

    void push(Event ev);
    void send(const std::string& frame);  // send_, closing the pump on failure
    void run();
    bool newer_pending();
    // Polled by the engine during a running preview (PreemptFn).
    bool preempt_preview();
    bool handle_audio(const std::string& bytes);  // true: accepted
    void handle_commit();
    // `coalesce_preview`: commit due windows but skip the preview (more
    // queued audio follows in the same batch).
    void step(bool coalesce_preview = false);

    StreamSession& session_;
    Options opt_;
    SendFn send_;
    bool reject_error_sent_ = false;  // worker-only

    std::mutex mu_;
    std::condition_variable cv_;       // worker: events or close
    std::condition_variable space_cv_; // transport: queue space
    std::condition_variable idle_cv_;  // drain(): queue empty and idle
    std::deque<Event> queue_;
    size_t pending_bytes_ = 0;
    size_t pending_work_ = 0;          // queued events other than pings
    // For preempt_preview(): queued audio (samples, estimated from the
    // frame size), queued resets and commits, and the audio queued ahead
    // of the first queued commit.
    int64_t pending_samples_ = 0;
    size_t pending_resets_ = 0;
    size_t pending_commits_ = 0;
    int64_t samples_before_commit_ = 0;
    bool closed_ = false;
    bool busy_ = false;                // worker is processing a batch
    std::thread worker_;
};

} // namespace starling::serve
