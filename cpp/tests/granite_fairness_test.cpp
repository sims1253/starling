// CPU-only Granite chunk-fairness contract over a tiny synthesized GGUF.
#include "server.hpp"
#include "stream_session.hpp"
#include "tiny_granite_fixture.hpp"
#include "trace_test_support.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <thread>
#include <vector>

using namespace starling::serve;

static int failures = 0;
static void check(bool ok, const char* what) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}
static bool wait_until(const std::function<bool()>& pred) {
    for (int i = 0; i < 2000; ++i) {
        if (pred()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return pred();
}
static std::string field(const std::string& line, const char* key) {
    const std::string needle = std::string("\"") + key + "\":\"";
    const size_t i = line.find(needle);
    if (i == std::string::npos) return {};
    const size_t start = i + needle.size();
    const size_t end = line.find('"', start);
    return end == std::string::npos ? "" : line.substr(start, end - start);
}
static std::string baseline(starling_ggml_ctx* model, const std::vector<float>& pcm) {
    char* p = starling_ggml_transcribe_pcm(model, pcm.data(),
                                           (int64_t) pcm.size(), kSampleRate);
    if (!p) return {};
    std::string result(p);
    starling_ggml_free_string(p);
    return result;
}

int main() {
#ifdef _WIN32
    std::printf("[SKIP] POSIX trace-capture concurrency test\n");
    return 0;
#else
    SETENV("STARLING_GGML_DEVICE", "cpu");
    SETENV("STARLING_GGML_THREADS", "4");
    SETENV("OMP_NUM_THREADS", "4");
    SETENV("STARLING_TRACE", "1");
    TinyGraniteFixture fixture(std::filesystem::temp_directory_path() /
                               "granite_fairness_test.gguf");
    check(fixture.wrote(), "tiny Granite fixture exists");
    if (!fixture.wrote()) return 1;
    auto* model = starling_ggml_load(STARLING_GGML_GRANITE, fixture.path.c_str());
    check(model != nullptr, "baseline model loaded");
    if (!model) return 1;

    // Distinct PCM, including a non-full final chunk in both long jobs.
    std::vector<float> a((size_t)(60.25 * kSampleRate), 0.0f);
    std::vector<float> b((size_t)(61.5 * kSampleRate), 0.0f);
    std::vector<float> short_pcm((size_t)(0.75 * kSampleRate), 0.0f);
    for (size_t i = 0; i < b.size(); ++i)
        b[i] = 0.05f * std::sin((double)i * 0.019);
    for (size_t i = 0; i < short_pcm.size(); ++i)
        short_pcm[i] = 0.07f * std::sin((double)i * 0.011);
    const std::string base_a = baseline(model, a);
    const std::string base_b = baseline(model, b);
    const std::string base_short = baseline(model, short_pcm);
    check(!base_a.empty() && !base_b.empty() && !base_short.empty(),
          "serial baseline outputs are complete");
    starling_ggml_free(model);

    ServerConfig cfg;
    cfg.model_slug = "granite";
    cfg.gguf_path = fixture.path.string();
    cfg.granite_chunk_fairness = true;
    StarlingServer server(cfg);
    server.load();
    check(server.loaded(), "opt-in Granite server loaded");
    if (!server.loaded()) return 1;
    check(server.backend_identity() == starling_ggml_backend_name(),
          "server caches the selected device identity after load");
    auto* ca = server.register_request("long-a");
    auto* cb = server.register_request("long-b");
    auto* cs = server.register_request("short");
    std::string ea, eb, es, ra, rb, rs;
    const std::string log = capture_stderr([&] {
        std::thread ta([&] {
            ra = server.transcribe_pcm(a.data(), (int64_t)a.size(), ca, &ea,
                                       QueuePolicy::SkipIfBusy).text;
        });
        check(wait_until([&] { return ca->running.load(); }), "long A entered engine");
        std::thread tb([&] {
            rb = server.transcribe_pcm(b.data(), (int64_t)b.size(), cb, &eb).text;
        });
        std::thread ts([&] {
            rs = server.transcribe_pcm(short_pcm.data(),
                (int64_t)short_pcm.size(), cs, &es, QueuePolicy::Block).text;
        });
        ta.join(); tb.join(); ts.join();
    });
    server.finish_request(ca);
    server.finish_request(cb);
    server.finish_request(cs);
    check(ea.empty() && eb.empty() && es.empty(), "all three jobs succeeded");
    check(ra == base_a && rb == base_b && rs == base_short,
          "interleaved outputs equal serial baselines, including padded tails");
    int active = 0, max_active = 0, chunk_a = 0, chunk_b = 0, chunk_s = 0;
    int yielded_a = 0, yielded_b = 0;
    int first_a_skip = 0, continuation_a_block = 0;
    int next_chunk = 0, short_chunk = -1, last_a = -1, last_b = -1;
    size_t pos = 0;
    while (pos < log.size()) {
        size_t end = log.find('\n', pos);
        std::string line = log.substr(pos, end == std::string::npos ? end : end-pos);
        pos = end == std::string::npos ? log.size() : end+1;
        if (line.rfind("[trace] ", 0) != 0) continue;
        const std::string ev = field(line, "ev"), req = field(line, "req");
        if (req != "long-a" && req != "long-b" && req != "short") continue;
        if (ev == "queue_enter" && req == "long-a") {
            if (field(line, "policy") == "skip_if_busy") ++first_a_skip;
            if (field(line, "policy") == "block") ++continuation_a_block;
        }
        if (ev == "queue_wait") max_active = std::max(max_active, ++active);
        if (ev == "queue_exit") {
            --active;
            if (field(line, "reason") == "yielded") {
                if (req == "long-a") ++yielded_a;
                if (req == "long-b") ++yielded_b;
            }
        }
        if (ev == "chunk") {
            if (req == "long-a") { ++chunk_a; last_a = next_chunk; }
            if (req == "long-b") { ++chunk_b; last_b = next_chunk; }
            if (req == "short") { ++chunk_s; short_chunk = next_chunk; }
            ++next_chunk;
        }
    }
    check(active == 0 && max_active == 1,
          "at most one engine chunk active and all turns exited");
    check(chunk_a > 1 && chunk_b > 1 && chunk_s == 1 &&
          yielded_a == chunk_a-1 && yielded_b == chunk_b-1,
          "each long chunk yields once; short runs once");
    check(first_a_skip == 1 && continuation_a_block == chunk_a - 1,
          "SkipIfBusy first chunk continues with blocking reserved turns");
    check(short_chunk >= 0 && short_chunk < last_a && short_chunk < last_b,
          "short engine chunk ran before both long final chunks");

    // Cancellation after the first A chunk while B holds the next turn.
    auto* xa = server.register_request("cancel-between-a");
    auto* xb = server.register_request("cancel-between-b");
    std::string cancel_err_a, cancel_err_b, cancel_text_a;
    std::thread xat([&] {
        cancel_text_a = server.transcribe_pcm(a.data(), (int64_t)a.size(),
                                               xa, &cancel_err_a).text;
    });
    check(wait_until([&] { return xa->running.load(); }), "cancel A entered first chunk");
    std::thread xbt([&] {
        (void)server.transcribe_pcm(b.data(), (int64_t)b.size(), xb, &cancel_err_b);
    });
    check(wait_until([&] { return xb->running.load(); }),
          "cancel B entered after A yielded");
    check(server.cancel_request("cancel-between-a"),
          "A is cancellable between chunks");
    server.cancel_request("cancel-between-b");
    xat.join(); xbt.join();
    server.finish_request(xa); server.finish_request(xb);
    check(cancel_err_a == "cancelled" && cancel_text_a.empty(),
          "cancelled upload has no false final");

    ServerConfig timeout_cfg = cfg;
    timeout_cfg.request_timeout_seconds = 0.005;
    timeout_cfg.stream_chunk_seconds = 1.0;
    timeout_cfg.stream_overlap_seconds = 0.25;
    timeout_cfg.min_chunk_seconds = 0.5;
    timeout_cfg.partial_interval = 0.0;
    StarlingServer timeout_server(timeout_cfg);
    timeout_server.load();
    auto* hold = timeout_server.register_request("timeout-hold");
    auto* queued = timeout_server.register_request("timeout-queued");
    std::string queued_err;
    std::thread holder([&] {
        (void)timeout_server.transcribe_pcm(a.data(), (int64_t)a.size(), hold, nullptr);
    });
    check(wait_until([&] { return hold->running.load(); }), "timeout holder entered");
    (void)timeout_server.transcribe_pcm(short_pcm.data(),
        (int64_t)short_pcm.size(), queued, &queued_err);
    timeout_server.cancel_request("timeout-hold");
    holder.join();
    timeout_server.finish_request(hold); timeout_server.finish_request(queued);
    check(queued_err == "request timed out", "queued request meets deadline");

    // A timed-out blocking WS flush is terminal for this take. It must not
    // enter five fresh queue waits through ChunkStreamer::flush's busy retry.
    StreamSession timeout_session(&timeout_server);
    check(timeout_session.append_pcm(std::string(12000 * 2, '\0')) ==
              AppendOutcome::Accepted, "timeout test audio accepted");
    auto* hold2 = timeout_server.register_request("timeout-stream-hold");
    std::thread holder2([&] {
        (void)timeout_server.transcribe_pcm(a.data(), (int64_t)a.size(), hold2, nullptr);
    });
    check(wait_until([&] { return hold2->running.load(); }),
          "stream timeout holder entered");
    std::optional<std::string> timed_out_final;
    const std::string timeout_log = capture_stderr([&] {
        timed_out_final = timeout_session.stream_flush();
        (void)timeout_session.stream_flush();  // must not queue again
    });
    timeout_server.cancel_request("timeout-stream-hold");
    holder2.join();
    timeout_server.finish_request(hold2);
    int stream_queue_entries = 0;
    for (size_t at = 0; at < timeout_log.size();) {
        size_t end = timeout_log.find('\n', at);
        std::string line = timeout_log.substr(at,
            end == std::string::npos ? end : end - at);
        at = end == std::string::npos ? timeout_log.size() : end + 1;
        if (line.find("\"ev\":\"queue_enter\"") != std::string::npos &&
            line.find("\"req\":\"#anon-") != std::string::npos)
            ++stream_queue_entries;
    }
    check(!timed_out_final && timeout_session.terminal_error() == "request timed out",
          "stream flush reports a terminal queue timeout");
    check(timeout_session.take_invalid() && stream_queue_entries == 1,
          "timed-out stream take requires reset and gets only one queue ticket");
    timeout_session.reset();
    check(timeout_session.terminal_error().empty() && !timeout_session.take_invalid(),
          "reset clears terminal stream timeout");

    std::printf("%s\n", failures ? "GRANITE FAIRNESS FAILED" : "GRANITE FAIRNESS OK");
    return failures ? 1 : 0;
#endif
}
