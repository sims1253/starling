// stream_pump.cpp — WS /stream worker: queue, drain, coalesce (issue #357).

#include "stream_pump.hpp"

#include <chrono>
#include <sstream>
#include <utility>

namespace starling::serve {

namespace {

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
        case '"':  out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n";  break;
        case '\r': out += "\\r";  break;
        case '\t': out += "\\t";  break;
        case '\b': out += "\\b";  break;
        case '\f': out += "\\f";  break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                out += buf;
            } else {
                out += c;
            }
        }
    }
    return out;
}

double steady_seconds() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

} // namespace

// Describe a refused binary audio frame as a structured WS error frame,
// mirroring the buffer-cap error style. A refused frame invalidates the take
// (or trips the buffer cap); the session then ignores audio until reset.
std::string ws_append_error(AppendOutcome outcome, const StreamSession& session,
                            double max_stream_seconds) {
    std::ostringstream ss;
    ss << "{\"type\":\"error\",\"message\":\"";
    switch (outcome) {
    case AppendOutcome::MalformedWav:
        ss << "malformed WAV frame rejected; audio ignored until reset";
        break;
    case AppendOutcome::RateMismatch:
        ss << "WAV sample rate mismatch: expected " << kSampleRate
           << "; audio ignored until reset";
        break;
    case AppendOutcome::OddPcmLength:
        ss << "odd-length PCM frame rejected (split sample);"
           << " audio ignored until reset";
        break;
    case AppendOutcome::Overflowed:
        ss << "stream buffer limit reached (" << max_stream_seconds
           << " s live buffer); audio ignored until reset";
        break;
    case AppendOutcome::TakeInvalid:
        // Only reached on a frame AFTER the invalidating one (whose own
        // outcome carried the reason); repeat that reason, not a generic.
        // invalid_reason_ is an internal [a-z_] code: safe to embed raw.
        ss << "take invalidated (" << session.invalid_reason()
           << "); audio ignored until reset";
        break;
    case AppendOutcome::Accepted:
        break;
    }
    ss << "\"}";
    return ss.str();
}

StreamPump::StreamPump(StreamSession& session, Options options, SendFn send)
    : session_(session), opt_(options), send_(std::move(send)) {
    worker_ = std::thread([this] { run(); });
}

StreamPump::~StreamPump() {
    close();
    if (worker_.joinable()) worker_.join();
}

void StreamPump::push(Event ev) {
    std::unique_lock<std::mutex> lk(mu_);
    if (ev.kind == Kind::Audio) {
        // Bounded queue, never dropping: wait for space (backpressure).
        space_cv_.wait(lk, [&] {
            return closed_ || pending_bytes_ == 0
                || pending_bytes_ + ev.bytes.size() <= opt_.max_pending_bytes;
        });
        if (closed_) return;
        pending_bytes_ += ev.bytes.size();
    }
    if (closed_) return;
    queue_.push_back(std::move(ev));
    cv_.notify_one();
}

void StreamPump::push_audio(std::string bytes) {
    push(Event{Kind::Audio, std::move(bytes)});
}

void StreamPump::push_commit() { push(Event{Kind::Commit, {}}); }

void StreamPump::push_reset() { push(Event{Kind::Reset, {}}); }

void StreamPump::push_ping() { push(Event{Kind::Ping, {}}); }

void StreamPump::close() {
    std::lock_guard<std::mutex> lk(mu_);
    closed_ = true;
    queue_.clear();
    pending_bytes_ = 0;
    cv_.notify_all();
    space_cv_.notify_all();
    idle_cv_.notify_all();
}

void StreamPump::drain() {
    std::unique_lock<std::mutex> lk(mu_);
    idle_cv_.wait(lk, [&] { return closed_ || (queue_.empty() && !busy_); });
}

bool StreamPump::is_closed() {
    std::lock_guard<std::mutex> lk(mu_);
    return closed_;
}

bool StreamPump::newer_pending() {
    // Any queued event makes a preview obsolete: newer audio would not be in
    // it, and a queued commit/reset ends or discards the take anyway. So
    // does a closed connection.
    std::lock_guard<std::mutex> lk(mu_);
    return closed_ || !queue_.empty();
}

void StreamPump::run() {
    for (;;) {
        std::deque<Event> batch;
        {
            std::unique_lock<std::mutex> lk(mu_);
            busy_ = false;
            idle_cv_.notify_all();
            cv_.wait(lk, [&] { return closed_ || !queue_.empty(); });
            if (closed_) return;
            batch.swap(queue_);
            pending_bytes_ = 0;
            busy_ = true;
            space_cv_.notify_all();
        }
        // Append every queued frame first; one step then covers all of it.
        bool need_step = false;
        for (Event& ev : batch) {
            if (is_closed()) return;  // the peer is gone: no more engine work
            switch (ev.kind) {
            case Kind::Audio:
                if (handle_audio(ev.bytes)) need_step = true;
                break;
            case Kind::Ping:
                if (need_step) step();
                need_step = false;
                send_("{\"type\":\"pong\"}");
                break;
            case Kind::Commit:
                // The flush covers everything appended so far; a preview
                // before it would be obsolete.
                need_step = false;
                handle_commit();
                break;
            case Kind::Reset:
                need_step = false;
                session_.reset();
                reject_error_sent_ = false;
                send_("{\"type\":\"reset_ack\"}");
                break;
            }
        }
        // A frame refused later in the batch (cap, invalid audio) ends
        // partials for this take, as it did frame by frame before.
        if (need_step && !session_.overflowed() && !session_.take_invalid()
            && !is_closed())
            step();
    }
}

bool StreamPump::handle_audio(const std::string& msg) {
    // Enforce the per-connection buffer cap (--max-stream-seconds) and the
    // frame-validity policy (issue #145): a refused frame is reported once
    // as an error frame, and the session stops accepting audio until reset.
    AppendOutcome outcome = AppendOutcome::Accepted;
    if (!session_.overflowed() && !session_.take_invalid()) {
        if (msg.size() >= 12 && msg.compare(0, 4, "RIFF") == 0
            && msg.compare(8, 4, "WAVE") == 0) {
            outcome = session_.append_wav(msg);
        } else {
            outcome = session_.append_pcm(msg);
        }
    } else if (session_.take_invalid()) {
        outcome = AppendOutcome::TakeInvalid;
    } else {
        outcome = AppendOutcome::Overflowed;
    }
    if (outcome != AppendOutcome::Accepted && !reject_error_sent_) {
        reject_error_sent_ = true;
        send_(ws_append_error(outcome, session_, opt_.max_stream_seconds));
    }
    return outcome == AppendOutcome::Accepted;
}

void StreamPump::step() {
    // stream_step runs once per drained batch with an accepted frame —
    // including audio-less no-ops (an empty payload is Accepted): those
    // duplicate snapshots of an unchanged buffer are answered by the
    // session's exact-tail reuse instead of re-running the engine.
    auto text_opt = session_.stream_step(steady_seconds(),
                                         [this] { return newer_pending(); });
    if (!session_.terminal_error().empty()) {
        if (!reject_error_sent_) {
            reject_error_sent_ = true;
            send_("{\"type\":\"error\",\"message\":\"request timed out\"}");
        }
        return;
    }
    if (!text_opt.has_value()) return;
    std::string safe_text = json_escape(*text_opt);
    double dur = session_.buffered_seconds();
    std::ostringstream ss;
    ss << "{\"type\":\"partial\",\"text\":\""
       << safe_text << "\",\"segments\":[{\"text\":\""
       << safe_text << "\",\"start_s\":0.0,\"end_s\":"
       << dur << "}],\"start_s\":0.0,\"end_s\":" << dur
       << ",\"stable_words\":" << session_.stable_words();
    if (opt_.trace) ss << ",\"trace\":" << session_.trace_partial_json();
    ss << "}";
    send_(ss.str());
}

void StreamPump::handle_commit() {
    // An invalidated take (a rejected binary frame) holds incomplete audio:
    // committing it as an ordinary successful final would silently miss
    // speech, so the commit is refused and the client falls back to its
    // authoritative local WAV after a reset (issue #145). The busy-retry
    // path below is untouched: it retains VALID audio, while this path
    // refuses INVALID audio.
    if (session_.take_invalid()) {
        std::ostringstream ss;
        ss << "{\"type\":\"error\",\"message\":\"take "
           << "invalidated (" << session_.invalid_reason()
           << "); reset and resend\"}";
        send_(ss.str());
        return;
    }
    double dur = session_.buffered_seconds();
    std::string text;
    if (dur > 0.0) {
        auto final = session_.stream_flush();
        if (!final.has_value()) {
            send_(session_.terminal_error().empty()
                ? "{\"type\":\"error\",\"message\":\"server busy\"}"
                : "{\"type\":\"error\",\"message\":\"request timed out\"}");
            return;
        }
        text = *final;
    }
    std::string safe_text = json_escape(text);
    std::ostringstream ss;
    ss << "{\"type\":\"final\",\"text\":\""
       << safe_text << "\",\"segments\":[{\"text\":\""
       << safe_text << "\",\"start_s\":0.0,\"end_s\":"
       << dur << "}],\"duration_s\":" << dur;
    if (opt_.trace) ss << ",\"trace\":" << session_.trace_final_json();
    ss << "}";
    send_(ss.str());
    session_.reset();
    // reset() re-enables audio (clears the buffer cap and any take
    // invalidation); re-arm the one-shot error frame with it.
    reject_error_sent_ = false;
}

} // namespace starling::serve
