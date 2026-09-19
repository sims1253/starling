// granite_stage_test.cpp — multi-chunk stage-attribution regression (issue
// #170). Two layers:
//
//   1. Unit checks of StageTiming (granite/stage_timing.hpp): per-chunk stage
//      durations ACCUMULATE across the long-audio loop (the old code kept
//      only the last chunk's array next to a whole-request total), the
//      one-chunk case stays exact, and the rendered log lines are pinned.
//
//   2. An end-to-end run through starling_ggml_granite_decode with a tiny
//      synthesized GGUF (zero weights, 1 encoder layer, 1 qformer layer, 1
//      LLM layer, chunk_seconds=1) — no model download, CPU-only. The test
//      captures stderr under STARLING_GRANITE_TIMING=1 and verifies the
//      emitted summaries: one per-chunk line per chunk, a whole-request line
//      whose stage aggregates cover EVERY chunk, and stage totals that
//      reconcile with the whole-request wall time within the bookkeeping
//      remainder. A tracing-off run emits no GRANITE_STAGE lines at all.
//
// Usage: ./granite_stage_test
#include "granite/stage_timing.hpp"
#include "tiny_granite_fixture.hpp"
#include "trace_test_support.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

extern "C" {
void* starling_ggml_granite_load(const char* gguf_path, const char** err_out);
void starling_ggml_granite_free(void* handle);
char* starling_ggml_granite_decode(void* handle, const float* pcm, int64_t n,
                                   const char** err_out);
}

namespace {

int failures = 0;

void check(bool ok, const std::string& what, const std::string& detail = "") {
    std::printf("[%s] %s%s%s\n", ok ? "PASS" : "FAIL", what.c_str(),
                (!ok && !detail.empty()) ? " -- " : "", ok ? "" : detail.c_str());
    if (!ok) ++failures;
}

constexpr double kRoundingMs = 1.0;  // %.1f rounding across summed lines

// --------------------------------------------------------------------------- //
// StageTiming unit checks.
// --------------------------------------------------------------------------- //
void unit_checks() {
    using starling::ggml::granite::StageTiming;
    using starling::ggml::granite::format_stage_chunk_line;
    using starling::ggml::granite::format_stage_request_line;

    const double c1[3] = {100.0, 10.0, 50.0};  // piece 160.0
    const double c2[3] = {20.0, 30.0, 40.0};   // piece 90.0
    const double c3[3] = {5.0, 15.0, 80.0};    // piece 100.0

    StageTiming st;
    st.add_chunk(c1);
    st.add_chunk(c2);
    st.add_chunk(c3);
    check(st.chunks == 3, "unit: three chunks recorded");
    check(st.total_ms[0] == 125.0 && st.total_ms[1] == 55.0 && st.total_ms[2] == 170.0,
          "unit: stage totals sum EVERY chunk (not the last one)");
    check(st.total_ms[0] != c3[0] && st.total_ms[2] != c3[2],
          "unit: totals are not the last chunk's values");
    check(st.chunk_ms[0] == c3[0] && st.chunk_ms[1] == c3[1] && st.chunk_ms[2] == c3[2],
          "unit: per-chunk slots latch the last chunk");
    check(st.chunk_total_ms() == 100.0, "unit: chunk total sums the last chunk");
    check(st.stages_total_ms() == 350.0, "unit: stages total sums all chunks");
    check(st.bookkeeping_ms(400.0) == 50.0, "unit: bookkeeping = total - stages");
    check(st.bookkeeping_ms(350.0) == 0.0, "unit: zero bookkeeping reconciles");

    StageTiming one;
    one.add_chunk(c1);
    check(one.chunks == 1 && one.total_ms[0] == 100.0 && one.total_ms[1] == 10.0 &&
              one.total_ms[2] == 50.0 && one.stages_total_ms() == 160.0,
          "unit: one-chunk behavior unchanged");

    // Pin the rendered lines (the emission in capi_granite.cpp uses these
    // exact helpers, indexing with StageTiming::chunks after add_chunk — so
    // the line renders the latched last chunk at its 1-based index).
    check(format_stage_chunk_line(st, st.chunks) ==
              "GRANITE_STAGE chunk=3 mel+enc+proj=5.0ms prompt+embeds=15.0ms "
              "gen=80.0ms piece=100.0ms",
          "unit: per-chunk line format");
    check(format_stage_request_line(st, 7.5, 400.0) ==
              "GRANITE_STAGE request chunks=3 audio=7.50s mel+enc+proj=125.0ms "
              "prompt+embeds=55.0ms gen=170.0ms stages=350.0ms bookkeeping=50.0ms "
              "total=400.0ms",
          "unit: whole-request line format");
}

// The fixture lives in tiny_granite_fixture.hpp (shared with the trace tests).

// capture_stderr lives in trace_test_support.hpp (shared with the trace tests).

std::vector<std::string> stage_lines(const std::string& log, const char* kind) {
    // kind: "chunk" -> per-chunk lines; "request" -> the request summary.
    std::vector<std::string> out;
    const std::string want = std::string("GRANITE_STAGE ") + kind;
    size_t pos = 0;
    while (pos <= log.size()) {
        size_t eol = log.find('\n', pos);
        if (eol == std::string::npos) eol = log.size();
        if (log.compare(pos, want.size(), want) == 0)
            out.push_back(log.substr(pos, eol - pos));
        if (eol == log.size()) break;
        pos = eol + 1;
    }
    return out;
}

// Parse "key=<number>" from a line (ms values carry a trailing "ms").
double field(const std::string& line, const char* key) {
    const std::string want = std::string(key) + "=";
    const size_t at = line.find(want);
    if (at == std::string::npos) return NAN;
    return std::atof(line.c_str() + at + want.size());
}

// --------------------------------------------------------------------------- //
// End-to-end stage-attribution checks through the real decode path.
// --------------------------------------------------------------------------- //
void e2e_checks(void* handle) {
    // 2.5 s of audio at chunk_seconds=1 -> chunks of 1 s, 1 s, 0.5 s (the
    // last zero-padded to a full chunk).
    const int64_t multi_n = 40000, single_n = 8000;
    // 2*pi*220 Hz test tone, spelled out (M_PI is not portable to the MSVC
    // test targets, which do not get _USE_MATH_DEFINES). The exact frequency
    // is irrelevant to the zero-weight model.
    constexpr double kToneW = 6.283185307179586 * 220.0;
    std::vector<float> pcm(multi_n);
    for (int64_t i = 0; i < multi_n; ++i)
        // std::sin's float overload, not std::sinf: the C99 float math
        // functions are not reliably in namespace std on older libstdc++
        // (CI's GCC 11 rejects std::sinf).
        pcm[i] = 0.2f * std::sin((float) (kToneW * i / 16000.0));

    char* text = nullptr;
    const char* err = nullptr;
    std::string log = capture_stderr([&] {
        err = nullptr;
        text = starling_ggml_granite_decode(handle, pcm.data(), multi_n, &err);
    });
    check(text != nullptr, "e2e: multi-chunk decode succeeded",
          err ? err : "");
    if (!text) return;
    check(std::string(text).find('a') != std::string::npos,
          "e2e: multi-chunk decode returned text");
    std::free(text);

    const auto chunk_lines = stage_lines(log, "chunk");
    const auto request_lines = stage_lines(log, "request");
    check(chunk_lines.size() == 3, "e2e: one summary line per chunk (3)",
          "got " + std::to_string(chunk_lines.size()));
    check(request_lines.size() == 1, "e2e: exactly one whole-request summary");
    if (chunk_lines.size() != 3 || request_lines.size() != 1) {
        std::printf("---- captured ----\n%s------------------\n", log.c_str());
        return;
    }
    for (size_t i = 0; i < 3; ++i)
        check(field(chunk_lines[i], "chunk") == (double) (i + 1),
              "e2e: chunk lines are indexed in order");

    // Every per-chunk line's stages sum to its own piece total.
    for (const std::string& l : chunk_lines)
        check(std::fabs(field(l, "piece") - (field(l, "mel+enc+proj") +
                        field(l, "prompt+embeds") + field(l, "gen"))) < 0.3,
              "e2e: chunk line piece == its three stages");

    // The request aggregates cover EVERY chunk: each stage aggregate matches
    // the sum of that stage over the per-chunk lines (the pre-fix code
    // printed only the LAST chunk's durations here).
    const std::string& req = request_lines[0];
    for (const char* stage : {"mel+enc+proj", "prompt+embeds", "gen"}) {
        double sum = 0;
        for (const std::string& l : chunk_lines) sum += field(l, stage);
        check(std::fabs(sum - field(req, stage)) < kRoundingMs,
              std::string("e2e: request ") + stage + " covers every chunk",
              "sum=" + std::to_string(sum) + " line=" +
                  std::to_string(field(req, stage)));
    }
    check(field(req, "chunks") == 3.0, "e2e: request line reports 3 chunks");

    // Reconciliation: stages == the three aggregates, bookkeeping ==
    // total - stages, and the stages fit inside the whole-request wall time.
    const double stages = field(req, "stages");
    const double total = field(req, "total");
    const double book = field(req, "bookkeeping");
    check(std::fabs(stages - (field(req, "mel+enc+proj") + field(req, "prompt+embeds") +
                              field(req, "gen"))) < 0.5,
          "e2e: request stages == sum of stage aggregates");
    check(std::fabs(book - (total - stages)) < 0.3,
          "e2e: bookkeeping == total - stages");
    check(total > 0.0 && stages <= total + 0.5,
          "e2e: stage totals fit in the whole-request wall time");
    check(book >= -0.5, "e2e: bookkeeping is non-negative",
          "bookkeeping=" + std::to_string(book));

    // One chunk stays correct: a single chunk line and aggregates equal to
    // that chunk's stages.
    std::string log1 = capture_stderr([&] {
        err = nullptr;
        text = starling_ggml_granite_decode(handle, pcm.data(), single_n, &err);
    });
    check(text != nullptr, "e2e: one-chunk decode succeeded", err ? err : "");
    if (text) std::free(text);
    const auto chunk1 = stage_lines(log1, "chunk");
    const auto req1 = stage_lines(log1, "request");
    check(chunk1.size() == 1 && req1.size() == 1,
          "e2e: one-chunk request emits one chunk line + one request line");
    if (chunk1.size() == 1 && req1.size() == 1) {
        check(field(req1[0], "chunks") == 1.0, "e2e: one-chunk request reports 1 chunk");
        for (const char* stage : {"mel+enc+proj", "prompt+embeds", "gen"})
            check(std::fabs(field(req1[0], stage) - field(chunk1[0], stage)) < kRoundingMs,
                  std::string("e2e: one-chunk ") + stage + " aggregate == the chunk");
    }

    // Tracing off: no GRANITE_STAGE output at all.
    UNSETENV("STARLING_GRANITE_TIMING");
    const std::string quiet = capture_stderr([&] {
        err = nullptr;
        text = starling_ggml_granite_decode(handle, pcm.data(), single_n, &err);
    });
    check(text != nullptr, "e2e: tracing-off decode succeeded", err ? err : "");
    if (text) std::free(text);
    check(quiet.find("GRANITE_STAGE") == std::string::npos,
          "e2e: tracing off emits no GRANITE_STAGE lines");
}

} // namespace

int main() {
    unit_checks();
#ifdef _WIN32
    std::printf("[SKIP] e2e stage-attribution checks (POSIX stderr capture)\n");
#else
    // CPU backend + timing gate, before the first load creates the backend.
    SETENV("STARLING_GGML_DEVICE", "cpu");
    SETENV("STARLING_GRANITE_TIMING", "1");
    TinyGraniteFixture fixture(std::filesystem::temp_directory_path() /
                               "granite_stage_test.gguf");
    check(fixture.wrote(), "e2e: synthesized tiny granite GGUF written");
    if (fixture.wrote()) {
        const char* err = nullptr;
        void* handle = starling_ggml_granite_load(fixture.path.string().c_str(), &err);
        check(handle != nullptr, "e2e: tiny granite model loaded", err ? err : "");
        if (handle) {
            e2e_checks(handle);
            starling_ggml_granite_free(handle);
        }
    }
#endif
    std::printf("%s\n", failures ? "GRANITE STAGE FAILED" : "GRANITE STAGE OK");
    return failures ? 1 : 0;
}
