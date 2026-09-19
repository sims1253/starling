// serve_trace_test.cpp — the serving-layer STARLING_TRACE checks (issue
// #180) over StarlingServer with the tiny synthesized granite GGUF
// (CPU-only, no downloads). Gate latched ON from main(). Verifies for one
// registered request:
//
//   - queue_enter (with admission policy + waiter depth) -> queue_wait ->
//     engine records (correlated by req) -> request -> queue_exit ->
//     response, in that order;
//   - engine-layer records (chunk/stage on CPU) carry the request id
//     through the C boundary (RequestScope) — the captured-graph cache
//     records are GPU-gated and covered by trace_schema_test;
//   - early queue departures (server_busy / timed_out / cancelled) emit
//     their terminal records, keeping the enter/exit ledger balanced;
//   - the anon-caller path (no RequestContext) synthesizes "#anon-N" ids.
//
// Usage: ./serve_trace_test
#include "server.hpp"
#include "runtime/trace.hpp"
#include "tiny_granite_fixture.hpp"
#include "trace_test_support.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <thread>
#include <utility>
#include <vector>

int failures = 0;

void check(bool ok, const std::string& what, const std::string& detail = "") {
    std::printf("[%s] %s%s%s\n", ok ? "PASS" : "FAIL", what.c_str(),
                (!ok && !detail.empty()) ? " -- " : "", ok ? "" : detail.c_str());
    if (!ok) ++failures;
}

namespace {

struct Record {
    std::string raw, ev, req, policy;
    double depth = NAN, dur_ms = NAN;
};

std::string json_str(const std::string& rec, const char* key) {
    const std::string want = std::string("\"") + key + "\":\"";
    const size_t at = rec.find(want);
    if (at == std::string::npos) return {};
    const size_t v0 = at + want.size();
    const size_t v1 = rec.find('"', v0);
    return v1 == std::string::npos ? std::string() : rec.substr(v0, v1 - v0);
}

double json_num(const std::string& rec, const char* key) {
    const std::string want = std::string("\"") + key + "\":";
    const size_t at = rec.find(want);
    if (at == std::string::npos) return NAN;
    return std::atof(rec.c_str() + at + want.size());
}

bool has_field(const std::string& rec, const char* key) {
    return rec.find(std::string("\"") + key + "\":") != std::string::npos;
}

std::vector<Record> parse(const std::string& log) {
    std::vector<Record> out;
    size_t pos = 0;
    while (pos < log.size()) {
        const size_t nl = log.find('\n', pos);
        std::string line = log.substr(pos, nl == std::string::npos ? nl : nl - pos);
        pos = (nl == std::string::npos) ? log.size() : nl + 1;
        if (line.rfind("[trace] ", 0) != 0) continue;
        line = line.substr(8);
        Record r;
        r.raw = line;
        r.ev = json_str(line, "ev");
        r.req = json_str(line, "req");
        r.policy = json_str(line, "policy");
        r.depth = json_num(line, "depth");
        r.dur_ms = json_num(line, "dur_ms");
        out.push_back(std::move(r));
    }
    return out;
}

// Position of the first record with ev==ev and req==req (NPOS-like = -1).
long long find_first(const std::vector<Record>& rs, const char* ev,
                     const std::string& req) {
    for (size_t i = 0; i < rs.size(); ++i)
        if (rs[i].ev == ev && rs[i].req == req) return (long long)i;
    return -1;
}

size_t count_req(const std::vector<Record>& rs, const std::string& req) {
    size_t n = 0;
    for (const auto& r : rs)
        if (r.req == req) ++n;
    return n;
}

} // namespace

#ifndef _WIN32
static void e2e_checks(starling::serve::StarlingServer& server) {
    const int64_t kSampleRate = 16000;
    std::vector<float> pcm((size_t)(2.5 * kSampleRate), 0.0f);
    std::string err;

    auto* ctx = server.register_request("req-42");
    check(ctx != nullptr, "e2e: request registered");
    const std::string log1 = capture_stderr([&] {
        err.clear();
        auto r1 = server.transcribe_pcm(pcm.data(), (int64_t)pcm.size(), ctx, &err);
        (void)r1;
    });
    server.finish_request(ctx);
    check(err.empty(), "e2e: first transcribe succeeded", "err=" + err);
    auto rs = parse(log1);

    // The full serving-layer chain for req-42, in order.
    const long long i_enter = find_first(rs, "queue_enter", "req-42");
    const long long i_wait = find_first(rs, "queue_wait", "req-42");
    const long long i_req = find_first(rs, "request", "req-42");
    const long long i_exit = find_first(rs, "queue_exit", "req-42");
    const long long i_resp = find_first(rs, "response", "req-42");
    check(i_enter >= 0, "e2e: queue_enter recorded");
    check(i_wait >= 0, "e2e: queue_wait recorded");
    check(i_req >= 0, "e2e: request recorded");
    check(i_exit >= 0, "e2e: queue_exit recorded");
    check(i_resp >= 0, "e2e: response recorded");
    if (i_enter >= 0 && i_wait >= 0 && i_req >= 0 && i_exit >= 0 && i_resp >= 0) {
        check(i_enter < i_wait && i_wait < i_req && i_req < i_exit && i_exit < i_resp,
              "e2e: queue_enter < queue_wait < request < queue_exit < response");
        check(rs[(size_t)i_enter].policy == "block",
              "e2e: queue_enter carries the block policy");
        check(rs[(size_t)i_enter].depth == 1.0,
              "e2e: queue_enter depth 1 for the sole waiter");
        check(rs[(size_t)i_exit].depth == 0.0, "e2e: queue_exit depth back to 0");
        check(rs[(size_t)i_wait].dur_ms >= 0.0, "e2e: queue_wait dur non-negative");
        check(rs[(size_t)i_req].dur_ms > 0.0, "e2e: request dur positive");
    }

    // Engine-layer correlation: chunk/stage/graph records carry req-42 via
    // RequestScope through the C boundary.
    size_t engine42 = 0;
    for (const auto& r : rs) {
        if (r.ev == "chunk" || r.ev == "stage" || r.ev == "graph_replay" ||
            r.ev == "graph_build" || r.ev == "readback_sync" || r.ev == "cache") {
            if (r.req == "req-42") ++engine42;
        }
    }
    check(engine42 >= 4, "e2e: engine records correlated with req-42",
          std::to_string(engine42));

    // Second request: full independent chain (the granite encoder's
    // captured-graph cache is GPU-gated, so CPU runs assert the chain, not
    // cache hits — the cache record schema is pinned by trace_schema_test).
    auto* ctx2 = server.register_request("req-43");
    std::string err2;
    const std::string log2 = capture_stderr([&] {
        auto r2 = server.transcribe_pcm(pcm.data(), (int64_t)pcm.size(), ctx2, &err2);
        (void)r2;
    });
    server.finish_request(ctx2);
    check(err2.empty(), "e2e: second transcribe succeeded", "err=" + err2);
    auto rs2 = parse(log2);
    const long long e2 = find_first(rs2, "queue_enter", "req-43");
    const long long w2 = find_first(rs2, "queue_wait", "req-43");
    const long long q2 = find_first(rs2, "request", "req-43");
    const long long x2 = find_first(rs2, "queue_exit", "req-43");
    const long long p2 = find_first(rs2, "response", "req-43");
    check(e2 >= 0 && w2 >= 0 && q2 >= 0 && x2 >= 0 && p2 >= 0 &&
              e2 < w2 && w2 < q2 && q2 < x2 && x2 < p2,
          "e2e: full ordered chain for req-43");

    // Anonymous caller: synthesized ticket id, skip_if_busy policy declared.
    std::string aerr;
    const std::string log3 = capture_stderr([&] {
        aerr.clear();
        auto ra = server.transcribe_pcm(pcm.data(), (int64_t)pcm.size(), nullptr,
                                        &aerr, starling::serve::QueuePolicy::SkipIfBusy);
        (void)ra;
    });
    auto rs3 = parse(log3);
    bool anon_enter = false, anon_policy = false;
    for (const auto& r : rs3) {
        if (r.ev == "queue_enter" && r.req.rfind("#anon-", 0) == 0) {
            anon_enter = true;
            anon_policy = (r.policy == "skip_if_busy");
        }
    }
    check(anon_enter, "e2e: anonymous caller gets a synthesized #anon-N ticket");
    check(anon_policy, "e2e: anonymous caller declares skip_if_busy policy");
}
// Early-departure coverage (pullfrog review of #183): every queue_enter
// must balance against exactly one terminal queue_exit carrying a reason,
// and abandoned waits must be measured. Contention is deterministic: a long
// transcription holds the turn (waited on via ctx->running) before each
// departing probe runs; retries cover a slow machine finishing the
// occupier early.
static void contention_checks(starling::serve::StarlingServer& server,
                              std::vector<std::string>& logs) {
    const int64_t kSampleRate = 16000;
    // 120 s -> ~120 one-second chunks on the tiny fixture: comfortably
    // longer than the 150 ms timeout and the probe delays.
    std::vector<float> long_pcm((size_t)(120.0 * kSampleRate), 0.0f);

    auto occupy = [&](const char* id)
        -> std::pair<starling::serve::RequestContext*, std::thread> {
        auto* occ = server.register_request(id);
        // The occupier thread outlives occupy()'s frame (occupy returns once
        // running latches, with the transcription still in flight), so it must
        // not capture a local for the error out-param — nullptr (every *err
        // write in the server is guarded) instead of a dangling reference.
        std::thread t([&server, occ, &long_pcm] {
            auto r = server.transcribe_pcm(long_pcm.data(),
                                           (int64_t)long_pcm.size(), occ, nullptr);
            (void)r;
        });
        for (int i = 0; i < 600 && !occ->running.load(); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        check(occ->running.load(), std::string("contention: ") + id + " reached the turn");
        return {occ, std::move(t)};
    };

    // --- skip_if_busy refusal while the turn is held ----------------------
    for (int attempt = 0; attempt < 4; ++attempt) {
        auto [occ, t] = occupy("req-occupy-busy");
        std::string berr;
        const std::string log = capture_stderr([&] {
            auto rb = server.transcribe_pcm(long_pcm.data(),
                                            (int64_t)long_pcm.size(), nullptr, &berr,
                                            starling::serve::QueuePolicy::SkipIfBusy);
            (void)rb;
        });
        t.join();
        server.finish_request(occ);
        if (berr == "server busy") {
            bool refused = false;
            for (const auto& r : parse(log)) {
                if (r.ev == "queue_exit" && r.req.rfind("#anon-", 0) == 0 &&
                    json_str(r.raw, "reason") == "server_busy")
                    refused = true;
            }
            check(refused,
                  "contention: skip_if_busy refusal emits queue_exit(reason=server_busy)");
            logs.push_back(log);
            break;
        }
        // Fail only after the LAST retry missed: check() latches a failure
        // immediately, so a mid-loop check would sink a miss-then-hit run.
        if (attempt == 3)
            check(false, "contention: busy scenario hit within the retries");
    }

    // --- timeout while parked behind the turn (server built with 0.15 s) --
    for (int attempt = 0; attempt < 4; ++attempt) {
        auto [occ, t] = occupy("req-occupy-timeout");
        auto* ctx = server.register_request("req-timeout");
        std::string terr;
        const std::string log = capture_stderr([&] {
            auto rt = server.transcribe_pcm(long_pcm.data(),
                                            (int64_t)long_pcm.size(), ctx, &terr);
            (void)rt;
        });
        t.join();
        server.finish_request(occ);
        server.finish_request(ctx);
        if (terr == "request timed out") {
            auto rs = parse(log);
            bool timed_out = false, waited = false;
            for (const auto& r : rs) {
                if (r.ev == "queue_exit" && r.req == "req-timeout" &&
                    json_str(r.raw, "reason") == "timed_out")
                    timed_out = true;
                if (r.ev == "queue_wait" && r.req == "req-timeout" && r.dur_ms > 0.0)
                    waited = true;
            }
            check(timed_out, "contention: timeout emits queue_exit(reason=timed_out)");
            check(waited, "contention: the abandoned wait is measured (queue_wait > 0)");
            logs.push_back(log);
            break;
        }
        if (attempt == 3)
            check(false, "contention: timeout scenario hit within the retries");
    }

    // --- cancellation while parked behind the turn -------------------------
    for (int attempt = 0; attempt < 4; ++attempt) {
        auto [occ, t] = occupy("req-occupy-cancel");
        auto* ctx = server.register_request("req-cancel");
        std::string cerr_;
        // The whole victim lifecycle (enter -> wait -> cancelled exit) runs
        // inside the capture so the ledger sees a balanced pair.
        const std::string log = capture_stderr([&] {
            std::thread blk([&server, ctx, &long_pcm, &cerr_] {
                auto rc = server.transcribe_pcm(long_pcm.data(),
                                                (int64_t)long_pcm.size(), ctx, &cerr_);
                (void)rc;
            });
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            server.cancel_request("req-cancel");
            blk.join();
        });
        t.join();
        server.finish_request(occ);
        server.finish_request(ctx);
        if (cerr_ == "cancelled") {
            bool cancelled = false;
            for (const auto& r : parse(log)) {
                if (r.ev == "queue_exit" && r.req == "req-cancel" &&
                    json_str(r.raw, "reason") == "cancelled")
                    cancelled = true;
            }
            check(cancelled, "contention: cancellation emits queue_exit(reason=cancelled)");
            logs.push_back(log);
            break;
        }
        if (attempt == 3)
            check(false, "contention: cancel scenario hit within the retries");
    }
}
#endif // !_WIN32

int main() {
    SETENV("STARLING_TRACE", "1");
    SETENV("STARLING_GGML_DEVICE", "cpu");
#ifdef _WIN32
    std::printf("[SKIP] e2e serve trace checks (POSIX stderr capture)\n");
#else
    TinyGraniteFixture fixture(std::filesystem::temp_directory_path() /
                               "serve_trace_test.gguf");
    check(fixture.wrote(), "e2e: synthesized tiny granite GGUF written");
    if (fixture.wrote()) {
        starling::serve::ServerConfig cfg;
        cfg.model_slug = "granite";
        cfg.gguf_path = fixture.path.string();
        starling::serve::StarlingServer server(cfg);
        e2e_checks(server);

        // A second, short-deadline server instance drives the early
        // departures (busy/timeout/cancel).
        starling::serve::ServerConfig cfg2 = cfg;
        cfg2.request_timeout_seconds = 0.15;
        starling::serve::StarlingServer contention_server(cfg2);
        std::vector<std::string> contention_logs;
        contention_checks(contention_server, contention_logs);

        // Ledger invariant across everything captured: every request whose
        // queue_enter is IN a captured window also has its terminal
        // queue_exit there (an occupier's records can straddle the capture
        // window — enter before, exit after — so enter-less tails are
        // expected and excluded).
        std::map<std::string, std::pair<int, int>> ledger;
        for (const auto& log : contention_logs) {
            for (const auto& r : parse(log)) {
                if (r.ev == "queue_enter") ledger[r.req].first++;
                if (r.ev == "queue_exit") ledger[r.req].second++;
            }
        }
        int fully_seen = 0;
        bool balanced = true;
        for (const auto& [req, cnt] : ledger) {
            if (cnt.first == 0) continue;  // straddling occupier tail
            ++fully_seen;
            if (cnt.first != cnt.second) balanced = false;
        }
        check(fully_seen >= 3 && balanced,
              "ledger: every queue_enter balances one queue_exit");
    }
#endif
    std::printf("%s\n", failures ? "SERVE TRACE FAILED" : "SERVE TRACE OK");
    return failures ? 1 : 0;
}
