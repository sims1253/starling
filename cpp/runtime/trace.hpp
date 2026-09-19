// trace.hpp — the opt-in structured timing trace (issue #180).
//
// STARLING_TRACE (set to any value) turns on one JSON record per line on
// stderr, prefixed "[trace] ", correlating one transcription request's timing
// across the layers it touches:
//
//   serving layer (cpp/serve/server.cpp)
//     queue_enter / queue_wait / queue_exit / request / response
//   engine layer  (cpp/granite/capi_granite.cpp; other engines emit the
//                  runtime records below, without chunk/stage spans yet)
//     chunk / stage
//   runtime layer (ReplayGraph in cpp/runtime/backend.cpp, LruCache)
//     graph_build / graph_replay / readback_sync / cache
//
// Correlation fields on every record: "req" (serving-layer request id, set
// via RequestScope around the engine call; "#anon-N" for anonymous callers),
// "chunk" (engine chunk index, set via ChunkScope; omitted when absent), and
// "stage" (model stage name, present on stage records). Graph and cache
// records inherit whatever req/chunk is active on the emitting thread, so a
// replay graph fired inside chunk 3 of request R carries both.
//
// Labeling rules (binding, from #170 — do not relax):
//   - Host enqueue, host blocked time, and device measurements are separate
//     kinds: graph_replay measures the async launch (host enqueue — returns
//     almost immediately on CUDA), readback_sync measures the single
//     trailing sync (host blocked; INCLUDES waiting for prior GPU work, so it
//     is NOT transfer time alone), and true per-graph device time is NOT
//     measured — device_us is reported as "unavailable", never fabricated.
//   - Nesting: wall times are CLOCK-NESTED, so parents contain children.
//     Aggregate by summing only sibling records of ONE kind (e.g. the three
//     stage records of a chunk); never add a child kind into its parent
//     (chunk totals already contain their stages; request totals contain
//     their chunks; graph_replay + readback_sync overlap the stage walls).
//   - No audio, transcript, prompt, or tensor contents in trace records:
//     request ids, chunk indices, shape dimensions, and cache occupancy only.
//   - The gate latches once (above); with it off nothing is measured,
//     formatted, or printed, and ReplayGraph::compute keeps its single-sync
//     fast path untouched.
#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <thread>

namespace starling::ggml {
namespace trace {

// Gate: STARLING_TRACE set (any value) = on. Latched once on the first call
// (magic static): the emit helpers sit next to per-replay hot paths, so the
// gate must be a load, not a getenv scan. Consequence: a process either
// traces or it doesn't — tests that need both modes use separate binaries
// (see trace_off_test).
inline bool on() {
    static const bool v = std::getenv("STARLING_TRACE") != nullptr;
    return v;
}

// ---- correlation context (thread-local) -----------------------------------

struct Correlation {
    std::string req;    // serving-layer request id ("" = not under a request)
    long long chunk = -1;  // engine chunk index (-1 = not inside a chunk)
};

inline thread_local Correlation t_correlation;

// RAII: set the request id for everything the engine does on this thread
// (the C engine call is synchronous, so the serving thread's context is the
// engine's context). Restores the previous value on exit.
class RequestScope {
public:
    explicit RequestScope(const std::string& id) : prev_(t_correlation.req) {
        t_correlation.req = id;
    }
    ~RequestScope() { t_correlation.req = std::move(prev_); }
    RequestScope(const RequestScope&) = delete;
    RequestScope& operator=(const RequestScope&) = delete;

private:
    std::string prev_;
};

// RAII: set the chunk index for the runtime records emitted inside one
// long-audio chunk (graph replays, cache events).
class ChunkScope {
public:
    explicit ChunkScope(long long idx) : prev_(t_correlation.chunk) {
        t_correlation.chunk = idx;
    }
    ~ChunkScope() { t_correlation.chunk = prev_; }
    ChunkScope(const ChunkScope&) = delete;
    ChunkScope& operator=(const ChunkScope&) = delete;

private:
    long long prev_;
};

// ---- emission --------------------------------------------------------------

// Microseconds since the first trace record on this process (monotonic).
long long now_us();

// Escape a string for JSON (same rules as the server's json_escape).
std::string json_escape(const std::string& s);

// ---- record kinds ------------------------------------------------------------
// All helpers are no-ops when the gate is off.

// queue_enter: arrival at the serial queue, with the admission policy and
// the waiter depth after enqueue. queue_exit: the ONE terminal record every
// queue_enter eventually gets, carrying why the ticket left — "completed"
// after the engine call, or "cancelled" / "server_busy" / "timed_out" for
// the early-departure paths. queue_wait fires whenever a ticket stops
// waiting, including abandoned waits (skip refusal, timeout, cancellation)
// — the duration is the host time blocked up to departure. `req` is passed
// explicitly: the waiting phase sits outside RequestScope, which only wraps
// the engine call.
void queue_event(const char* ev, const std::string& req, int depth,
                 const char* policy = nullptr);
void queue_exit_event(const std::string& req, int depth, const char* reason);

// queue_wait: host time blocked waiting for the serial-queue turn.
void queue_wait_event(const std::string& req, double dur_ms);

// request: wall time of the engine call (service time; contains the engine's
// chunks/stages and their graph work). response: wall time assembling the
// result for emission (result marshalling, not the socket write).
void request_event(double dur_ms);
void response_event(double dur_ms);

// chunk: one long-audio chunk (mel through detokenize). stage: one model
// stage of a chunk ("mel_enc_proj", "prompt_embeds", "generate") — the same
// clocks that feed STARLING_GRANITE_TIMING, so the two renderings cannot
// disagree.
void chunk_event(long long chunk, double dur_ms);
void stage_event(const char* stage, double dur_ms);

// graph_build: ReplayGraph construction + allocation (one-time cost per
// shape), with the device's free memory AFTER admission (the available
// allocation counter; "unavailable" when the device cannot report it).
// graph_replay: the async launch (host enqueue). readback_sync: the
// single trailing sync (host blocked, includes waiting for prior GPU work —
// NOT transfer time alone). out_ne is the output tensor shape.
void graph_event(const char* ev, double dur_ms, unsigned uid, int nodes,
                 const long long* out_ne, const char* device);
void graph_build_event(double dur_ms, unsigned uid, int nodes,
                       const long long* out_ne, const char* device,
                       long long mem_free);

// cache: replay-cache admission. op is "hit" or "miss"; evicted counts LRU
// victims of the inserting miss; mem_free < 0 renders as "unavailable"
// (never fabricated).
void cache_event(const char* cache, const char* op, size_t evicted,
                 size_t size, size_t capacity, long long mem_free);

// ---- inline implementation ---------------------------------------------------

namespace detail {

// One fprintf per record (glibc serializes stderr, so lines never interleave
// mid-line). "v" is the schema version; bump only for breaking changes.
inline void write_record(const std::string& fields) {
    std::fprintf(stderr, "[trace] {\"v\":1,\"ts\":%lld,\"tid\":%llu,%s}\n",
                 now_us(),
                 (unsigned long long)std::hash<std::thread::id>{}(
                     std::this_thread::get_id()),
                 fields.c_str());
}

// Common correlation suffix: req + chunk when active. Kept LAST in each
// record so readers can strip it uniformly.
inline std::string correlation_fields() {
    std::string f = ",\"req\":\"" + json_escape(t_correlation.req) + "\"";
    if (t_correlation.chunk >= 0)
        f += ",\"chunk\":" + std::to_string(t_correlation.chunk);
    return f;
}

} // namespace detail

inline long long now_us() {
    static const auto start = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now() - start)
        .count();
}

inline std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    char buf[8];
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
            if ((unsigned char)c < 0x20) {
                std::snprintf(buf, sizeof buf, "\\u%04x", (unsigned char)c);
                out += buf;
            } else {
                out += c;
            }
        }
    }
    return out;
}

inline void queue_event(const char* ev, const std::string& req, int depth,
                        const char* policy) {
    if (!on()) return;
    std::string f = std::string("\"ev\":\"") + ev +
                    "\",\"req\":\"" + json_escape(req) + "\"" +
                    (policy ? (",\"policy\":\"" + std::string(policy) + "\"") : "") +
                    ",\"depth\":" + std::to_string(depth);
    detail::write_record(f);
}

inline void queue_wait_event(const std::string& req, double dur_ms) {
    if (!on()) return;
    char buf[96];
    std::snprintf(buf, sizeof buf, "%.3f", dur_ms);
    detail::write_record(std::string("\"ev\":\"queue_wait\",\"req\":\"") +
                         json_escape(req) + "\",\"dur_ms\":" + buf);
}

inline void queue_exit_event(const std::string& req, int depth, const char* reason) {
    if (!on()) return;
    detail::write_record("\"ev\":\"queue_exit\",\"req\":\"" + json_escape(req) +
                         "\",\"reason\":\"" + reason +
                         "\",\"depth\":" + std::to_string(depth));
}

inline void request_event(double dur_ms) {
    if (!on()) return;
    char buf[96];
    std::snprintf(buf, sizeof buf, "%.3f", dur_ms);
    detail::write_record(std::string("\"ev\":\"request\",\"dur_ms\":") + buf +
                         detail::correlation_fields());
}

inline void response_event(double dur_ms) {
    if (!on()) return;
    char buf[96];
    std::snprintf(buf, sizeof buf, "%.3f", dur_ms);
    detail::write_record(std::string("\"ev\":\"response\",\"dur_ms\":") + buf +
                         detail::correlation_fields());
}

inline void chunk_event(long long chunk, double dur_ms) {
    if (!on()) return;
    char buf[96];
    std::snprintf(buf, sizeof buf, "%.3f", dur_ms);
    detail::write_record("\"ev\":\"chunk\",\"chunk\":" + std::to_string(chunk) +
                         ",\"dur_ms\":" + buf + detail::correlation_fields());
}

inline void stage_event(const char* stage, double dur_ms) {
    if (!on()) return;
    char buf[96];
    std::snprintf(buf, sizeof buf, "%.3f", dur_ms);
    detail::write_record(std::string("\"ev\":\"stage\",\"stage\":\"") + stage +
                         "\",\"dur_ms\":" + buf + detail::correlation_fields());
}

inline void graph_event(const char* ev, double dur_ms, unsigned uid, int nodes,
                        const long long* out_ne, const char* device) {
    if (!on()) return;
    char buf[96];
    std::snprintf(buf, sizeof buf, "%.3f", dur_ms);
    std::string f = std::string("\"ev\":\"") + ev + "\",\"dur_ms\":" + buf +
                    ",\"uid\":" + std::to_string(uid) +
                    ",\"nodes\":" + std::to_string(nodes) + ",\"out_ne\":[" +
                    std::to_string(out_ne[0]) + "," + std::to_string(out_ne[1]) +
                    "," + std::to_string(out_ne[2]) + "," + std::to_string(out_ne[3]) +
                    "],\"device\":\"" + json_escape(device) + "\"" +
                    detail::correlation_fields();
    detail::write_record(f);
}

inline void graph_build_event(double dur_ms, unsigned uid, int nodes,
                              const long long* out_ne, const char* device,
                              long long mem_free) {
    if (!on()) return;
    char buf[96];
    std::snprintf(buf, sizeof buf, "%.3f", dur_ms);
    std::string f = std::string("\"ev\":\"graph_build\",\"dur_ms\":") + buf +
                    ",\"uid\":" + std::to_string(uid) +
                    ",\"nodes\":" + std::to_string(nodes) + ",\"out_ne\":[" +
                    std::to_string(out_ne[0]) + "," + std::to_string(out_ne[1]) +
                    "," + std::to_string(out_ne[2]) + "," + std::to_string(out_ne[3]) +
                    "],\"device\":\"" + json_escape(device) + "\",\"mem_free\":";
    f += mem_free < 0 ? "\"unavailable\"" : std::to_string(mem_free);
    detail::write_record(f + detail::correlation_fields());
}

inline void cache_event(const char* cache, const char* op, size_t evicted,
                        size_t size, size_t capacity, long long mem_free) {
    if (!on()) return;
    std::string f = std::string("\"ev\":\"cache\",\"cache\":\"") + cache +
                    "\",\"op\":\"" + op +
                    "\",\"evicted\":" + std::to_string(evicted) +
                    ",\"size\":" + std::to_string(size) +
                    ",\"cap\":" + std::to_string(capacity) + ",\"mem_free\":";
    if (mem_free < 0)
        f += "\"unavailable\"";
    else
        f += std::to_string(mem_free);
    detail::write_record(f + detail::correlation_fields());
}

} // namespace trace
} // namespace starling::ggml
