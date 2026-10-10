// server.cpp — native starling-serve HTTP/WebSocket server implementation.
//
// Implements the StarlingServer class: model lifecycle, serial request queue,
// transcribe dispatch, and health introspection. The HTTP/WS transport layer
// (cpp-httplib) is wired in main.cpp; this file is transport-agnostic.

#include "server.hpp"

#include "lib/model_registry.hpp"
#include "lib/granite_job_internal.hpp"
#include "runtime/trace.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace starling::serve {

namespace trace = starling::ggml::trace;

// ---- JSON helpers ---------------------------------------------------------
namespace {

// Escape a string for JSON (handles quotes, backslash, control chars).
std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
        case '"':  out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\b': out += "\\b";  break;
        case '\f': out += "\\f";  break;
        case '\n': out += "\\n";  break;
        case '\r': out += "\\r";  break;
        case '\t': out += "\\t";  break;
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

const char* phase_str(Phase p) {
    switch (p) {
    case Phase::Unloaded: return "unloaded";
    case Phase::Loading:  return "loading";
    case Phase::Ready:    return "ready";
    case Phase::Busy:     return "busy";
    }
    return "unknown";
}

} // namespace

// ---- model slug ↔ enum ----------------------------------------------------
// Derived from the central model table (lib/model_registry.hpp): adding a
// model is one registry row, and the slug mapping, supported check, and
// --version list all follow. starling-serve links the library statically, so
// it can include the internal header; nothing is added to the public
// starling_ggml.h.
starling_ggml_model slug_to_model(const std::string& slug) {
    const starling::ggml::lib::ModelDescriptor* d =
        starling::ggml::lib::find_model_by_slug(slug);
    return d ? d->kind : (starling_ggml_model)0;
}

const char* model_to_slug(starling_ggml_model m) {
    const starling::ggml::lib::ModelDescriptor* d =
        starling::ggml::lib::find_model(m);
    return d ? d->slug : "unknown";
}

bool is_supported_model(const std::string& slug) {
    return slug_to_model(slug) != (starling_ggml_model)0;
}

// Build the supported-models list string for --version (registry order).
std::string supported_models_str() {
    size_t n = 0;
    const starling::ggml::lib::ModelDescriptor* regs =
        starling::ggml::lib::model_registry(&n);
    std::string s;
    for (size_t i = 0; i < n; ++i) {
        if (i) s += ' ';
        s += regs[i].slug;
    }
    return s;
}

// ---- StarlingServer -------------------------------------------------------
StarlingServer::StarlingServer(ServerConfig cfg)
    : cfg_(std::move(cfg)), backend_identity_(starling_ggml_backend_name()) {}

std::string StarlingServer::backend_identity() const {
    std::lock_guard<std::mutex> lk(status_mutex_);
    return backend_identity_;
}

std::string StarlingServer::load_error() const {
    std::lock_guard<std::mutex> lk(status_mutex_);
    return load_error_;
}

bool StarlingServer::warm() const {
    std::lock_guard<std::mutex> lk(warmup_mutex_);
    return warmup_done_;
}

StarlingServer::~StarlingServer() {
    if (model_) {
        starling_ggml_free(model_);
        model_ = nullptr;
    }
    starling_ggml_shutdown();
}

void StarlingServer::load() {
    std::lock_guard<std::mutex> lk(load_mutex_);
    if (loaded_.load()) return;
    phase_.store(Phase::Loading);

    auto t0 = std::chrono::steady_clock::now();
    std::fprintf(stderr, "[starling-serve] loading model '%s' from %s ...\n",
                 cfg_.model_slug.c_str(), cfg_.gguf_path.c_str());

    auto kind = slug_to_model(cfg_.model_slug);
    model_ = starling_ggml_load(kind, cfg_.gguf_path.c_str());
    if (!model_) {
        const char* err = starling_ggml_last_error(nullptr);
        std::fprintf(stderr, "[starling-serve] load FAILED: %s\n",
                     err ? err : "(no message)");
        {
            std::lock_guard<std::mutex> status(status_mutex_);
            load_error_ = (err && *err) ? err : "model load failed";
        }
        phase_.store(Phase::Unloaded);
        return;  // loaded_ stays false
    }

    // Loading selects the actual device. Capture its name before publishing
    // loaded_: later WS sessions can build their cache key without waiting on
    // the C API runtime mutex during an active inference chunk.
    {
        std::lock_guard<std::mutex> status(status_mutex_);
        backend_identity_ = starling_ggml_backend_name();
        load_error_.clear();
    }
    loaded_.store(true);
    auto dt = std::chrono::duration<double>(
                  std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "[starling-serve] model loaded in %.1fs\n", dt);

    phase_.store(Phase::Ready);
}

void StarlingServer::warmup() {
    if (!loaded_.load() || !model_) return;
    {
        std::lock_guard<std::mutex> lk(warmup_mutex_);
        if (warmup_done_ || warmup_in_progress_) return;
        warmup_in_progress_ = true;
    }
    std::string err;
    // Warmup runs one real inference — captures CUDA graphs etc. It takes a
    // serial-queue ticket (blocking) so it can't overlap a real request;
    // run_with_turn sets phase Busy→Ready internally. Text models warm up on
    // a probe transcript; audio models on a silent clip.
    if (is_text_model()) {
        std::fprintf(stderr, "[starling-serve] warming up on probe text ...\n");
        auto text = normalize_text(
            "um this is a warmup transcript with like forty two tokens in it",
            "", "", "", nullptr, &err);
        (void)text;
    } else {
        std::fprintf(stderr,
            "[starling-serve] warming up on %.1fs silent clip ...\n",
            kWarmupSeconds);
        int n = static_cast<int>(kWarmupSeconds * kSampleRate);
        std::vector<float> dummy(n, 0.0f);
        auto result = do_transcribe(dummy.data(), n, nullptr, &err, QueuePolicy::Block);
        (void)result;
    }
    if (!err.empty())
        std::fprintf(stderr, "[starling-serve] warmup error: %s\n", err.c_str());
    {
        std::lock_guard<std::mutex> lk(warmup_mutex_);
        warmup_in_progress_ = false;
        warmup_done_ = true;
    }
    std::fprintf(stderr, "[starling-serve] warmup complete\n");
}

RequestContext* StarlingServer::register_request(const std::string& id) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (requests_.count(id)) return nullptr;  // duplicate request ID
    auto ctx = std::make_unique<RequestContext>();
    ctx->id = id;
    auto* raw = ctx.get();
    requests_[id] = std::move(ctx);
    return raw;
}

void StarlingServer::finish_request(RequestContext* ctx) {
    if (!ctx) return;
    std::lock_guard<std::mutex> lk(mutex_);
    requests_.erase(ctx->id);
}

bool StarlingServer::cancel_request(const std::string& id) {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = requests_.find(id);
    if (it == requests_.end()) return false;
    if (it->second->done) return false;  // completion already claimed the result
    it->second->cancelled.store(true);
    return true;
}

TranscribeResult StarlingServer::transcribe_pcm(
    const float* samples, int64_t n, RequestContext* ctx, std::string* err,
    QueuePolicy policy) {
    if (!loaded_.load() || !model_) {
        load();
        if (!loaded_.load() || !model_) {
            if (err) *err = "model not loaded";
            return {};
        }
    }
    return do_transcribe(samples, n, ctx, err, policy);
}

TranscribeResult StarlingServer::do_transcribe(
    const float* samples, int64_t n, RequestContext* ctx, std::string* err,
    QueuePolicy policy) {
    std::string text;
    std::string req_id;
    std::optional<std::vector<TimedWord>> words;
    if (cfg_.granite_chunk_fairness && cfg_.model_slug == "granite") {
        using namespace starling::ggml::lib;
        std::unique_ptr<GraniteChunkJob, decltype(&free_granite_job)> job(
            create_granite_job(model_, samples, n), &free_granite_job);
        if (!job) {
            const char* emsg = starling_ggml_last_error(model_);
            if (err) *err = emsg ? emsg : "Granite job creation failed";
            return {};
        }
        bool continuing = false;
        for (;;) {
            const bool final_chunk = granite_job_last_chunk(job.get());
            // The job owns partial text until its final step. This one-byte
            // success token satisfies run_with_turn's malloc-string contract
            // without publishing a partial or false final to the transport.
            if (!run_with_turn(ctx, policy, [&] {
                    std::string final_text;
                    const int status = step_granite_job(model_, job.get(), &final_text);
                    if (status < 0) {
                        const char* emsg = starling_ggml_last_error(model_);
                        if (err) *err = emsg ? emsg : "Granite chunk failed";
                        return static_cast<char*>(nullptr);
                    }
                    if ((status == 1) != final_chunk) {
                        if (err) *err = "Granite job completion did not match chunk policy";
                        return static_cast<char*>(nullptr);
                    }
                    const std::string& emitted = status == 1 ? final_text : std::string();
                    char* out = static_cast<char*>(std::malloc(emitted.size() + 1));
                    if (!out) { if (err) *err = "malloc failed"; return out; }
                    std::memcpy(out, emitted.data(), emitted.size());
                    out[emitted.size()] = '\0';
                    return out;
                }, &text, err, &req_id, final_chunk, continuing))
                return {};
            if (final_chunk) break;
            continuing = true;
        }
    } else {
        if (!run_with_turn(ctx, policy, [&] {
                starling_ggml_word* raw = nullptr;
                int64_t count = -1;
                std::unique_ptr<char, decltype(&starling_ggml_free_string)> out(
                    starling_ggml_transcribe_pcm_words(model_, samples, n, kSampleRate,
                                                       &raw, &count),
                    &starling_ggml_free_string);
                std::unique_ptr<starling_ggml_word, decltype(&starling_ggml_free_words)> raw_guard(
                    raw, &starling_ggml_free_words);
                if (out && raw && count >= 0) {
                    const size_t len = std::strlen(out.get());
                    std::vector<TimedWord> timed;
                    timed.reserve(static_cast<size_t>(count));
                    for (int64_t i = 0; i < count; ++i) {
                        const auto& w = raw[i];
                        if (w.text_begin < 0 || w.text_end < w.text_begin
                            || static_cast<size_t>(w.text_end) > len)
                            break;
                        timed.push_back({std::string(out.get() + w.text_begin, out.get() + w.text_end),
                                         w.start_s, w.end_s});
                    }
                    // One bad offset drops the window's word times (the
                    // untimed stitch), never a partial list.
                    if (static_cast<int64_t>(timed.size()) == count) words = std::move(timed);
                }
                return out.release();
            }, &text, err, &req_id))
            return {};
    }

    // Response emission (result marshalling — the transport-level body build
    // and socket write stay outside the trace; see docs/native-serving.md).
    // RequestScope re-established with the ticket's id so the response record
    // correlates with the queue/request records above.
    trace::RequestScope trace_resp(req_id);
    const bool tr_on = trace::on();
    const auto t_resp0 = std::chrono::steady_clock::now();
    TranscribeResult result;
    result.text = std::move(text);
    result.words = std::move(words);
    if (tr_on) {
        trace::response_event(
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_resp0).count());
    }
    return result;
}

bool StarlingServer::run_with_turn(RequestContext* ctx, QueuePolicy policy,
                                   const std::function<char*()>& engine_call,
                                   std::string* out_text, std::string* err,
                                   std::string* effective_req_id,
                                   bool complete_request,
                                   bool continuing) {
    // Acquire the serial queue position. Every caller gets a ticket —
    // anonymous ones (warmup, WS streaming) get a synthesized id so they
    // queue like everyone else instead of racing the engine.
    std::string req_id = ctx ? ctx->id : "";
    {
        std::unique_lock<std::mutex> lk(mutex_);
        if (continuing) {
            // The preceding chunk reserved this request's admission slot.
            // Transfer the reservation to its new FIFO ticket atomically.
            // A continuation must wait for its turn even if the initial
            // request used SkipIfBusy.
            policy = QueuePolicy::Block;
            if (reserved_continuations_ <= 0) {
                if (err) *err = "Granite continuation lost its queue reservation";
                return false;
            }
            --reserved_continuations_;
        } else if (n_waiters_ + reserved_continuations_ >= kMaxWaiters) {
            if (err) *err = "server busy";
            return false;
        }
        if (req_id.empty()) req_id = "#anon-" + std::to_string(next_anon_id_++);
        if (effective_req_id) *effective_req_id = req_id;
        request_order_.push_back(req_id);
        n_waiters_++;
        if (trace::on()) {
            // Arrival point + admission policy + waiter depth after enqueue
            // (issue #180: queue entry/start/end with host-wait attribution).
            trace::queue_event(
                "queue_enter", req_id, n_waiters_,
                policy == QueuePolicy::SkipIfBusy ? "skip_if_busy" : "block");
        }

        // Wait for our turn (head of the queue), with a timeout. wait_start
        // also measures abandoned waits: leave_queue emits it so skip
        // refusals, timeouts and cancellations carry their host-blocked time.
        auto wait_start = std::chrono::steady_clock::now();

        // Leave the queue (waiter gone, ticket removed). Lock is held.
        // `reason` feeds the queue_exit trace record so early departures
        // (busy/cancel/timeout) stay visible and the record-derived depth
        // stays balanced — exactly the outcomes the trace exists to
        // diagnose (pullfrog review of #183).
        auto leave_queue = [&](const char* reason) {
            n_waiters_--;
            auto it = std::find(request_order_.begin(),
                                request_order_.end(), req_id);
            if (it != request_order_.end()) request_order_.erase(it);
            if (trace::on()) {
                trace::queue_wait_event(
                    req_id,
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - wait_start).count());
                trace::queue_exit_event(req_id, n_waiters_, reason);
            }
            queue_cv_.notify_all();
        };

        bool waited = false;
        for (;;) {
            if (ctx && ctx->cancelled.load()) {
                leave_queue("cancelled");
                if (err) *err = "cancelled";
                return false;
            }
            // Check the deadline before accepting a newly freed turn. A
            // notify at (or after) the deadline must not bypass the timeout.
            // A caller already at the head never waited and has no queue
            // deadline to enforce.
            double timeout = cfg_.request_timeout_seconds;
            double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - wait_start).count();
            if (waited && timeout > 0 && elapsed >= timeout) {
                leave_queue("timed_out");
                if (err) *err = "request timed out";
                return false;
            }
            if (request_order_.front() == req_id) break;
            if (policy == QueuePolicy::SkipIfBusy) {
                // Anonymous latency-sensitive caller (WS streaming chunk):
                // don't park on the queue — report busy and retry later.
                leave_queue("server_busy");
                if (err) *err = "server busy";
                return false;
            }
            waited = true;
            queue_cv_.wait_for(lk, std::chrono::duration<double>(
                timeout > 0 ? std::max(0.0, std::min(0.1, timeout - elapsed)) : 0.1));
        }
        if (trace::on()) {
            // Host time blocked waiting for the turn (0 when immediately
            // front-of-queue).
            trace::queue_wait_event(
                req_id,
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - wait_start).count());
        }
    }

    if (ctx && ctx->cancelled.load()) {
        std::lock_guard<std::mutex> lk(mutex_);
        // We're at the front of the queue (our turn arrived).
        n_waiters_--;
        request_order_.pop_front();
        queue_cv_.notify_all();
        if (trace::on()) trace::queue_exit_event(req_id, n_waiters_, "cancelled");
        if (err) *err = "cancelled";
        return false;
    }

    phase_.store(Phase::Busy);
    if (ctx) ctx->running.store(true);

    // Run the engine call (the C engine is synchronous). RequestScope
    // correlates every engine-layer trace record (chunks, stages, graph
    // replays, cache events) with this request id — the engine runs on this
    // thread, so the thread-local context carries through the C boundary.
    trace::RequestScope trace_req(req_id);
    const bool tr_on = trace::on();
    const auto t_engine0 = std::chrono::steady_clock::now();
    char* result_text = engine_call();
    if (tr_on) {
        trace::request_event(
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_engine0).count());
    }

    if (ctx) ctx->running.store(false);
    phase_.store(Phase::Ready);

    bool cancel_won = false;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        // We're at the front of the queue; release the turn to the next waiter.
        n_waiters_--;
        request_order_.pop_front();
        if (ctx) {
            // Claim completion under the same lock cancel_request uses: if
            // cancellation already won, discard the result; otherwise the
            // result stands and later cancels return false.
            ctx->done = complete_request || !result_text;
            cancel_won = ctx->cancelled.load();
        }
        // do_transcribe's next statement is run_with_turn(continuing=true),
        // which consumes this reservation under this mutex on entry, before
        // any wait, cancel or timeout exit. Failed or cancelled chunks never
        // reserve, so no exit path can strand a slot.
        if (!complete_request && result_text && !cancel_won)
            ++reserved_continuations_;
        queue_cv_.notify_all();
        if (tr_on) trace::queue_exit_event(req_id, n_waiters_,
            !result_text ? "failed" : (complete_request ? "completed" : "yielded"));
    }

    if (cancel_won) {
        if (result_text) starling_ggml_free_string(result_text);
        if (err) *err = "cancelled";
        return false;
    }

    if (!result_text) {
        const char* emsg = starling_ggml_last_error(model_);
        if (err && err->empty()) *err = emsg ? emsg : "engine call failed";
        return false;
    }

    if (out_text) *out_text = result_text;
    starling_ggml_free_string(result_text);
    return true;
}

std::string StarlingServer::normalize_text(
    const std::string& transcript, const std::string& styling,
    const std::string& structure, const std::string& context,
    RequestContext* ctx, std::string* err, QueuePolicy policy) {
    if (!loaded_.load() || !model_) {
        load();
        if (!loaded_.load() || !model_) {
            if (err) *err = "model not loaded";
            return {};
        }
    }
    if (!is_text_model()) {
        if (err) *err = "model '" + cfg_.model_slug + "' has no text path";
        return {};
    }
    std::string text;
    const char* s = styling.empty() ? nullptr : styling.c_str();
    const char* st = structure.empty() ? nullptr : structure.c_str();
    const char* cx = context.empty() ? nullptr : context.c_str();
    if (!run_with_turn(ctx, policy, [&] {
            return starling_ggml_normalize_text(model_, transcript.c_str(), s, st, cx);
        }, &text, err))
        return {};
    return text;
}

bool StarlingServer::is_text_model() const {
    const starling::ggml::lib::ModelDescriptor* d =
        starling::ggml::lib::find_model(slug_to_model(cfg_.model_slug));
    return d && d->normalize_fn != nullptr;
}

bool StarlingServer::loaded() const { return loaded_.load(); }
bool StarlingServer::busy() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return n_waiters_ > 0;
}
Phase StarlingServer::phase() const { return phase_.load(); }

int StarlingServer::queue_depth() const {
    std::lock_guard<std::mutex> lk(mutex_);
    // Count queued (not running) requests.
    int depth = 0;
    for (const auto& [id, ctx] : requests_) {
        if (!ctx->running.load()) depth++;
    }
    return depth;
}

std::string StarlingServer::health_json() const {
    // backend_identity and load_error are read together under one
    // status_mutex_ acquisition: two separate reads could pair a load
    // that just succeeded (new backend, error cleared) with the stale
    // error, or vice versa, reporting a torn state (#366).
    std::string backend;
    std::string failure;
    {
        std::lock_guard<std::mutex> lk(status_mutex_);
        backend = backend_identity_;
        failure = load_error_;
    }
    std::ostringstream ss;
    ss << "{"
       << "\"status\":\"ok\","
       << "\"model\":\"" << json_escape(cfg_.model_slug) << "\","
       << "\"loaded\":" << (loaded_.load() ? "true" : "false") << ","
       << "\"busy\":" << (busy() ? "true" : "false") << ","
       << "\"phase\":\"" << phase_str(phase_.load()) << "\","
       << "\"queue_depth\":" << queue_depth() << ","
       // Additive supervision fields (#362): the device the engine actually
       // runs on (the compile-time family until a load selects one), whether
       // warmup finished, and why the last load failed (null when it did not).
       << "\"backend\":\"" << json_escape(backend) << "\","
       << "\"warm\":" << (warm() ? "true" : "false") << ","
       << "\"load_error\":";
    if (failure.empty()) ss << "null";
    else ss << "\"" << json_escape(failure) << "\"";
    ss << "}";
    return ss.str();
}

} // namespace starling::serve
