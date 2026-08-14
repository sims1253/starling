// server.hpp — native starling-serve HTTP/WebSocket server.
//
// Wraps libstarling_ggml (the flat C API in cpp/include/starling_ggml.h) behind
// the same HTTP/WS contract as the Python `starling.server`, so a supervisor
// that spawns the Python server can switch with a one-line change:
//
//   spawn("starling-serve", ["--model", slug, "--gguf", path, "--port", "8181"])
//
// Endpoints (mirroring the Python server):
//   GET    /health          → { model, loaded, phase, queue_depth, busy }
//   POST   /transcribe      → multipart/raw WAV → { text, segments, duration_s, request_id }
//   POST   /inference       → alias for /transcribe
//   POST   /warmup          → idempotent silent-clip warmup (202)
//   DELETE /inference/<id>  → cancel a queued/in-flight request by X-Request-Id
//   WS     /stream          → real-time streaming dictation
//
// One model is resident at a time (the supervisor enforces this). Inference is
// serialised through a request queue so only one transcribe is in-flight
// (matching the Python server's single-GPU-worker model).
#pragma once

#include "starling_ggml.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace starling::serve {

// ---- constants (mirror src/starling/server.py) ---------------------------
constexpr int    kSampleRate           = 16000;
constexpr double kDefaultMaxChunk      = 30.0;
constexpr double kDefaultMinChunk      = 5.0;
constexpr double kDefaultPartialInt    = 3.0;
constexpr double kDefaultStreamChunk   = 12.0;
constexpr double kDefaultStreamOverlap = 3.0;
constexpr double kWarmupSeconds        = 5.0;
constexpr int    kMaxWaiters           = 8;
constexpr int    kMaxUploadMB          = 256;
constexpr double kRequestTimeoutS      = 600.0;

// ---- model slug ↔ enum ----------------------------------------------------
// Maps the CLI slug to the C-API model enum. Returns 0 (invalid)
// if the slug is unknown.
starling_ggml_model slug_to_model(const std::string& slug);
const char* model_to_slug(starling_ggml_model m);
bool is_supported_model(const std::string& slug);

// Build the supported-models space-separated string for --version output.
std::string supported_models_str();

// ---- config ---------------------------------------------------------------
struct ServerConfig {
    std::string model_slug;                        // "parakeet", "moss", ...
    std::string gguf_path;                         // path to the .gguf file
    std::string host = "127.0.0.1";
    int  port        = 8181;
    bool warmup      = false;
    bool eager_load  = true;
    double max_chunk_seconds       = kDefaultMaxChunk;
    double min_chunk_seconds       = kDefaultMinChunk;
    double partial_interval        = kDefaultPartialInt;
    double stream_chunk_seconds    = kDefaultStreamChunk;
    double stream_overlap_seconds  = kDefaultStreamOverlap;
    double request_timeout_seconds = kRequestTimeoutS;
    int    max_upload_mb           = kMaxUploadMB;
};

// ---- lifecycle phase ------------------------------------------------------
//   unloaded → loading → ready → busy → ready (repeats)
// "busy" is transient (reported during inference); "ready" is the idle state.
enum class Phase { Unloaded, Loading, Ready, Busy };

// ---- transcribe result ----------------------------------------------------
struct TranscribeResult {
    std::string text;
    // one segment: { text, start_s, end_s }
    struct Segment { std::string text; double start_s; double end_s; };
    std::vector<Segment> segments;
    double duration_s = 0.0;
    std::string to_json() const;
};

// ---- request context (for cancellation) ----------------------------------
struct RequestContext {
    std::string id;
    std::atomic<bool> cancelled{false};
    std::atomic<bool> running{false};
};

// Queue behaviour for callers without a RequestContext (WS streaming chunks,
// warmup). Everyone — including anonymous callers — takes a ticket in the
// serial queue so only one transcribe is ever in-flight; the policy only
// decides whether an anonymous caller waits for its turn or bails out with
// "server busy" so it can retry later.
enum class QueuePolicy { Block, SkipIfBusy };

// ---- the server -----------------------------------------------------------
class StarlingServer {
public:
    explicit StarlingServer(ServerConfig cfg);
    ~StarlingServer();

    // Lifecycle: load the model into device memory. Idempotent + thread-safe.
    void load();
    void warmup();

    // Synchronous transcribe of raw float32 mono PCM samples. Enqueues onto the
    // serial worker, blocks until done. ctx is used for cancellation; pass
    // nullptr for fire-and-forget (no cancellation) — the call still queues
    // behind earlier requests instead of bypassing them.
    // Returns the result, or sets *err on failure.
    TranscribeResult transcribe_pcm(const float* samples, int64_t n,
                                     RequestContext* ctx, std::string* err,
                                     QueuePolicy policy = QueuePolicy::Block);

    // Cancel a queued/in-flight request by id. Returns true if found.
    bool cancel_request(const std::string& id);

    // Introspection for /health.
    std::string model_slug() const { return cfg_.model_slug; }
    bool loaded() const;
    bool busy() const;
    Phase phase() const;
    int  queue_depth() const;

    // Build a JSON health response body.
    std::string health_json() const;

    // Register a request for cancellation tracking; returns the context.
    // Removes it on scope exit via finish_request().
    RequestContext* register_request(const std::string& id);
    void finish_request(RequestContext* ctx);

    const ServerConfig& config() const { return cfg_; }

private:
    ServerConfig cfg_;
    starling_ggml_ctx* model_ = nullptr;

    mutable std::mutex mutex_;       // protects queue state + request registry
    std::mutex load_mutex_;           // serializes model loading (long operation)
    std::condition_variable queue_cv_;

    // Serial inference queue: requests wait in arrival order. Anonymous
    // callers (no RequestContext) get a synthesized "#anon-N" ticket so they
    // queue too; those tickets are never in the requests_ registry and can't
    // be cancelled.
    std::deque<std::string> request_order_;
    std::unordered_map<std::string, std::unique_ptr<RequestContext>> requests_;
    int n_waiters_ = 0;
    uint64_t next_anon_id_ = 0;

    // Lifecycle phase.
    std::atomic<Phase> phase_{Phase::Unloaded};
    std::atomic<bool> loaded_{false};

    // Warmup dedup.
    std::mutex warmup_mutex_;
    bool warmup_done_ = false;
    bool warmup_in_progress_ = false;

    // Internal: run transcribe under the serial lock.
    TranscribeResult do_transcribe(const float* samples, int64_t n,
                                    RequestContext* ctx, std::string* err,
                                    QueuePolicy policy);
};

} // namespace starling::serve
