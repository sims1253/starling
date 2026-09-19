// trace_schema_test.cpp — unit checks for the STARLING_TRACE record schema
// (issue #180). Runs with the gate latched ON from main() before any trace
// call: record shape, common fields, correlation scopes (RequestScope /
// ChunkScope set + restore), per-kind required fields, cache events from a
// labeled LruCache (and silence from an unlabeled one), the no-contents rule
// (cache keys never appear in records), JSON escaping, and ts monotonicity.
// The gate-off behavior lives in trace_off_test (separate binary — the gate
// latches once per process).
//
// Usage: ./trace_schema_test
#include "runtime/lru_cache.hpp"
#include "runtime/trace.hpp"
#include "trace_test_support.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace starling::ggml;
namespace trace = starling::ggml::trace;

int failures = 0;

void check(bool ok, const std::string& what, const std::string& detail = "") {
    std::printf("[%s] %s%s%s\n", ok ? "PASS" : "FAIL", what.c_str(),
                (!ok && !detail.empty()) ? " -- " : "", ok ? "" : detail.c_str());
    if (!ok) ++failures;
}

namespace {

std::vector<std::string> trace_lines(const std::string& log) {
    std::vector<std::string> out;
    size_t pos = 0;
    while (pos < log.size()) {
        const size_t nl = log.find('\n', pos);
        const std::string line = log.substr(pos, nl == std::string::npos ? nl : nl - pos);
        if (line.rfind("[trace] ", 0) == 0) out.push_back(line.substr(8));
        pos = (nl == std::string::npos) ? log.size() : nl + 1;
    }
    return out;
}

// Extract "key":"value" (no escape decoding — test values stay escape-free).
std::string json_str(const std::string& rec, const char* key) {
    const std::string want = std::string("\"") + key + "\":\"";
    const size_t at = rec.find(want);
    if (at == std::string::npos) return {};
    const size_t v0 = at + want.size();
    const size_t v1 = rec.find('"', v0);
    return v1 == std::string::npos ? std::string() : rec.substr(v0, v1 - v0);
}

// Extract "key":<number>.
double json_num(const std::string& rec, const char* key) {
    const std::string want = std::string("\"") + key + "\":";
    const size_t at = rec.find(want);
    if (at == std::string::npos) return NAN;
    return std::atof(rec.c_str() + at + want.size());
}

bool has_field(const std::string& rec, const char* key) {
    return rec.find(std::string("\"") + key + "\":") != std::string::npos;
}

// Minimal JSON object sanity: balanced braces/quotes, escapes consumed, no
// raw control chars outside strings.
bool json_shaped(const std::string& rec) {
    if (rec.front() != '{' || rec.back() != '}') return false;
    int depth = 0; bool in_str = false;
    for (size_t i = 0; i < rec.size(); ++i) {
        const char c = rec[i];
        if (in_str) {
            if (c == '\\') { ++i; continue; }  // consume the escaped char
            if (c == '"') in_str = false;
            else if ((unsigned char)c < 0x20) return false;
            continue;
        }
        if (c == '"') in_str = true;
        else if (c == '{') ++depth;
        else if (c == '}') --depth;
    }
    return depth == 0 && !in_str;
}

std::string ev_of(const std::string& rec) { return json_str(rec, "ev"); }

} // namespace

int main() {
    SETENV("STARLING_TRACE", "1");
    check(starling::ggml::trace::on(), "gate: STARLING_TRACE=1 latches on");

    // --- record shape + common fields ------------------------------------
    {
        const std::string log = capture_stderr([] {
            starling::ggml::trace::request_event(1.25);
        });
        const auto lines = trace_lines(log);
        check(lines.size() == 1, "emit: exactly one record per call",
              std::to_string(lines.size()));
        if (lines.size() == 1) {
            const std::string& rec = lines[0];
            check(json_shaped(rec), "emit: record is a balanced JSON object", rec);
            check(json_num(rec, "v") == 1.0, "emit: schema version v=1");
            check(has_field(rec, "ts") && json_num(rec, "ts") >= 0.0,
                  "emit: monotonic ts present");
            check(has_field(rec, "tid"), "emit: thread id present");
            check(ev_of(rec) == "request", "emit: ev field");
            check(json_num(rec, "dur_ms") == 1.25, "emit: dur_ms value");
        }
    }

    // --- correlation scopes set AND restore -------------------------------
    {
        const std::string log = capture_stderr([] {
            starling::ggml::trace::stage_event("generate", 2.0);            // no req, no chunk
            {
                starling::ggml::trace::RequestScope r("req-A");
                starling::ggml::trace::stage_event("generate", 2.0);        // req only
                {
                    starling::ggml::trace::ChunkScope c(3);
                    starling::ggml::trace::stage_event("generate", 2.0);    // req + chunk
                }
                starling::ggml::trace::stage_event("generate", 2.0);        // chunk restored
            }
            starling::ggml::trace::stage_event("generate", 2.0);            // both restored
        });
        const auto lines = trace_lines(log);
        check(lines.size() == 5, "scope: five records", std::to_string(lines.size()));
        if (lines.size() == 5) {
            check(json_str(lines[0], "req").empty() && !has_field(lines[0], "chunk"),
                  "scope: bare record has no req/chunk");
            check(json_str(lines[1], "req") == "req-A" && !has_field(lines[1], "chunk"),
                  "scope: RequestScope sets req only");
            check(json_str(lines[2], "req") == "req-A" && json_num(lines[2], "chunk") == 3.0,
                  "scope: ChunkScope adds chunk");
            check(json_str(lines[3], "req") == "req-A" && !has_field(lines[3], "chunk"),
                  "scope: ChunkScope restores");
            check(json_str(lines[4], "req").empty(),
                  "scope: RequestScope restores");
        }
    }

    // --- per-kind required fields -----------------------------------------
    {
        const std::string log = capture_stderr([] {
            starling::ggml::trace::queue_event("queue_enter", "req-Q", 2, "block");
            starling::ggml::trace::queue_event("queue_exit", "req-Q", 1);
            starling::ggml::trace::queue_wait_event("req-Q", 0.5);
            starling::ggml::trace::response_event(0.25);
            starling::ggml::trace::chunk_event(2, 12.5);
            starling::ggml::trace::graph_event("graph_replay", 0.75, 7, 42, (const long long[]){1, 2, 3, 4},
                               "CUDA0");
            starling::ggml::trace::cache_event("granite.encoder", "miss", 1, 2, 16, -1);
        });
        const auto lines = trace_lines(log);
        check(lines.size() == 7, "kinds: seven records", std::to_string(lines.size()));
        for (const auto& rec : lines) check(json_shaped(rec), "kinds: shaped " + rec);
        if (lines.size() == 7) {
            check(json_str(lines[0], "req") == "req-Q" && json_num(lines[0], "depth") == 2.0 &&
                      json_str(lines[0], "policy") == "block",
                  "kinds: queue_enter req+depth+policy");
            check(ev_of(lines[1]) == "queue_exit" && json_num(lines[1], "depth") == 1.0 &&
                      !has_field(lines[1], "policy"),
                  "kinds: queue_exit has no policy");
            check(ev_of(lines[2]) == "queue_wait" && json_num(lines[2], "dur_ms") == 0.5,
                  "kinds: queue_wait dur");
            check(ev_of(lines[3]) == "response", "kinds: response");
            check(ev_of(lines[4]) == "chunk" && json_num(lines[4], "chunk") == 2.0,
                  "kinds: chunk carries its index");
            const std::string& g = lines[5];
            check(ev_of(g) == "graph_replay" && json_num(g, "uid") == 7.0 &&
                      json_num(g, "nodes") == 42.0 && json_str(g, "device") == "CUDA0" &&
                      g.find("\"out_ne\":[1,2,3,4]") != std::string::npos,
                  "kinds: graph uid+nodes+out_ne+device");
            const std::string& c = lines[6];
            check(ev_of(c) == "cache" && json_str(c, "cache") == "granite.encoder" &&
                      json_str(c, "op") == "miss" && json_num(c, "evicted") == 1.0 &&
                      json_num(c, "size") == 2.0 && json_num(c, "cap") == 16.0 &&
                      json_str(c, "mem_free") == "unavailable",
                  "kinds: cache label+op+evicted+occupancy, mem_free unavailable");
        }
    }

    // --- LruCache: labeled events, unlabeled silence, no key contents -----
    {
        const std::string secret = "CANARY-KEY-CONTENTS";
        const std::string log = capture_stderr([&] {
            LruCache<std::string, int> labeled(2, "test.cache");
            labeled.get_or_init("k1", [](int& v) { v = 1; });   // miss (evict 0)
            labeled.get_or_init("k1", [](int& v) { v = 2; });   // hit
            labeled.get_or_init("k2", [](int& v) { v = 3; });   // miss
            labeled.get_or_init("k3", [](int& v) { v = 4; });   // miss, evicts k1
            LruCache<std::string, int> quiet(2);
            quiet.get_or_init(secret, [](int& v) { v = 5; });   // unlabeled: silent
        });
        const auto lines = trace_lines(log);
        check(lines.size() == 4, "cache: three labeled ops + eviction count",
              std::to_string(lines.size()));
        if (lines.size() == 4) {
            check(ev_of(lines[0]) == "cache" && json_str(lines[0], "op") == "miss" &&
                      json_num(lines[0], "evicted") == 0.0, "cache: first get_or_init misses");
            check(json_str(lines[1], "op") == "hit", "cache: same key hits");
            check(json_str(lines[2], "op") == "miss", "cache: new key misses");
            check(json_str(lines[3], "op") == "miss" && json_num(lines[3], "evicted") == 1.0 &&
                      json_num(lines[3], "size") == 2.0,
                  "cache: capacity-bounded insert evicts one LRU victim");
        }
        check(log.find(secret) == std::string::npos,
              "cache: key contents never appear in records (no-contents rule)");
    }

    // --- escaping + ts monotonicity ----------------------------------------
    {
        const std::string log = capture_stderr([] {
            starling::ggml::trace::RequestScope r("req-\"back\\slash\"\n");
            starling::ggml::trace::request_event(0.0);
            starling::ggml::trace::request_event(0.0);
        });
        const auto lines = trace_lines(log);
        check(lines.size() == 2, "escape: records emitted");
        if (lines.size() == 2) {
            check(lines[0].find("\\\"back\\\\slash\\\"\\n") != std::string::npos,
                  "escape: quote/backslash/newline escaped", lines[0]);
            check(json_shaped(lines[0]), "escape: escaped record stays balanced");
            check(json_num(lines[1], "ts") >= json_num(lines[0], "ts"),
                  "ts: monotonic across records");
        }
    }

    std::printf("%s\n", failures ? "TRACE SCHEMA FAILED" : "TRACE SCHEMA OK");
    return failures ? 1 : 0;
}
