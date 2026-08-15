// server.cpp — native starling-serve HTTP/WebSocket server implementation.
//
// Implements the StarlingServer class: model lifecycle, serial request queue,
// transcribe dispatch, and health introspection. The HTTP/WS transport layer
// (cpp-httplib) is wired in main.cpp; this file is transport-agnostic.

#include "server.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace starling::serve {

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

// Format a double with the same rounding as Python's round(x, 3).
std::string fmt_double(double v, int precision = 3) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*f", precision, v);
    return buf;
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
starling_ggml_model slug_to_model(const std::string& slug) {
    if (slug == "parakeet") return STARLING_GGML_PARAKEET_TDT;
    if (slug == "moss")     return STARLING_GGML_MOSS;
    if (slug == "ark")      return STARLING_GGML_ARK;
    if (slug == "higgs")    return STARLING_GGML_HIGGS;
    if (slug == "hojo")     return STARLING_GGML_HOJO;
    return (starling_ggml_model)0;
}

const char* model_to_slug(starling_ggml_model m) {
    switch (m) {
    case STARLING_GGML_PARAKEET_TDT: return "parakeet";
    case STARLING_GGML_MOSS:         return "moss";
    case STARLING_GGML_ARK:          return "ark";
    case STARLING_GGML_HIGGS:        return "higgs";
    case STARLING_GGML_HOJO:         return "hojo";
    }
    return "unknown";
}

bool is_supported_model(const std::string& slug) {
    return slug_to_model(slug) != (starling_ggml_model)0;
}

// Build the supported-models list string for --version.
std::string supported_models_str() {
    std::string s = "parakeet moss ark higgs hojo";
    return s;
}

// ---- TranscribeResult::to_json -------------------------------------------
std::string TranscribeResult::to_json() const {
    std::ostringstream ss;
    ss << "{\"text\":\"" << json_escape(text) << "\","
       << "\"segments\":[";
    for (size_t i = 0; i < segments.size(); ++i) {
        if (i) ss << ",";
        ss << "{\"text\":\"" << json_escape(segments[i].text) << "\","
           << "\"start_s\":" << fmt_double(segments[i].start_s) << ","
           << "\"end_s\":" << fmt_double(segments[i].end_s) << "}";
    }
    ss << "],\"duration_s\":" << fmt_double(duration_s) << "}";
    return ss.str();
}

// ---- StarlingServer -------------------------------------------------------
StarlingServer::StarlingServer(ServerConfig cfg) : cfg_(std::move(cfg)) {}

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
        phase_.store(Phase::Unloaded);
        return;  // loaded_ stays false
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
    std::fprintf(stderr,
        "[starling-serve] warming up on %.1fs silent clip ...\n",
        kWarmupSeconds);
    int n = static_cast<int>(kWarmupSeconds * kSampleRate);
    std::vector<float> dummy(n, 0.0f);
    std::string err;
    // Warmup is a transcribe of silence — captures CUDA graphs etc. It takes
    // a serial-queue ticket (blocking) so it can't overlap a real request.
    // do_transcribe sets phase Busy→Ready internally.
    auto result = do_transcribe(dummy.data(), n, nullptr, &err, QueuePolicy::Block);
    (void)result;
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
    // Acquire the serial queue position. Every caller gets a ticket —
    // anonymous ones (warmup, WS streaming) get a synthesized id so they
    // queue like everyone else instead of racing the engine.
    std::string req_id = ctx ? ctx->id : "";
    {
        std::unique_lock<std::mutex> lk(mutex_);
        if (n_waiters_ >= kMaxWaiters) {
            if (err) *err = "server busy";
            return {};
        }
        if (req_id.empty()) req_id = "#anon-" + std::to_string(next_anon_id_++);
        request_order_.push_back(req_id);
        n_waiters_++;

        // Leave the queue (waiter gone, ticket removed). Lock is held.
        auto leave_queue = [&]() {
            n_waiters_--;
            auto it = std::find(request_order_.begin(),
                                request_order_.end(), req_id);
            if (it != request_order_.end()) request_order_.erase(it);
            queue_cv_.notify_all();
        };

        // Wait for our turn (head of the queue), with a timeout.
        auto wait_start = std::chrono::steady_clock::now();
        while (request_order_.front() != req_id) {
            if (ctx && ctx->cancelled.load()) {
                leave_queue();
                if (err) *err = "cancelled";
                return {};
            }
            if (policy == QueuePolicy::SkipIfBusy) {
                // Anonymous latency-sensitive caller (WS streaming chunk):
                // don't park on the queue — report busy and retry later.
                leave_queue();
                if (err) *err = "server busy";
                return {};
            }
            double timeout = cfg_.request_timeout_seconds;
            if (timeout > 0) {
                auto elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - wait_start).count();
                if (elapsed >= timeout) {
                    leave_queue();
                    if (err) *err = "request timed out";
                    return {};
                }
            }
            queue_cv_.wait_for(lk, std::chrono::milliseconds(100));
        }
    }

    if (ctx && ctx->cancelled.load()) {
        std::lock_guard<std::mutex> lk(mutex_);
        // We're at the front of the queue (our turn arrived).
        n_waiters_--;
        request_order_.pop_front();
        queue_cv_.notify_all();
        if (err) *err = "cancelled";
        return {};
    }

    phase_.store(Phase::Busy);
    if (ctx) ctx->running.store(true);

    // Run the transcribe (the C engine is synchronous).
    char* result_text = starling_ggml_transcribe_pcm(
        model_, samples, n, kSampleRate);

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
            ctx->done = true;
            cancel_won = ctx->cancelled.load();
        }
        queue_cv_.notify_all();
    }

    if (cancel_won) {
        if (result_text) starling_ggml_free_string(result_text);
        if (err) *err = "cancelled";
        return {};
    }

    if (!result_text) {
        const char* emsg = starling_ggml_last_error(model_);
        if (err) *err = emsg ? emsg : "transcribe failed";
        return {};
    }

    TranscribeResult result;
    result.text = result_text;
    starling_ggml_free_string(result_text);
    result.duration_s = static_cast<double>(n) / kSampleRate;
    result.segments.push_back({result.text, 0.0, result.duration_s});
    return result;
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
    std::ostringstream ss;
    ss << "{"
       << "\"status\":\"ok\","
       << "\"model\":\"" << json_escape(cfg_.model_slug) << "\","
       << "\"loaded\":" << (loaded_.load() ? "true" : "false") << ","
       << "\"busy\":" << (busy() ? "true" : "false") << ","
       << "\"phase\":\"" << phase_str(phase_.load()) << "\","
       << "\"queue_depth\":" << queue_depth()
       << "}";
    return ss.str();
}

} // namespace starling::serve
