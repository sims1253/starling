// trace_off_test.cpp — the STARLING_TRACE gate-off contract (issue #180):
// with the env var unset (the default everywhere), a full engine + cache
// workload emits ZERO [trace] records — nothing measured, formatted, or
// printed — and the transcript is unaffected. Separate binary from the
// on-tests because the gate latches once per process.
//
// Usage: ./trace_off_test
#include "runtime/lru_cache.hpp"
#include "runtime/trace.hpp"
#include "tiny_granite_fixture.hpp"
#include "trace_test_support.hpp"

#include <cstdio>
#include <string>
#include <vector>

extern "C" {
void* starling_ggml_granite_load(const char* gguf_path, const char** err_out);
void starling_ggml_granite_free(void* handle);
char* starling_ggml_granite_decode(void* handle, const float* pcm, int64_t n,
                                   const char** err_out);
}

using namespace starling::ggml;
namespace trace = starling::ggml::trace;

int failures = 0;

void check(bool ok, const std::string& what, const std::string& detail = "") {
    std::printf("[%s] %s%s%s\n", ok ? "PASS" : "FAIL", what.c_str(),
                (!ok && !detail.empty()) ? " -- " : "", ok ? "" : detail.c_str());
    if (!ok) ++failures;
}

int main() {
    // Deliberately NOT setting STARLING_TRACE: this process must stay silent.
    UNSETENV("STARLING_TRACE");
    UNSETENV("STARLING_GGML_DEVICE");
    check(!starling::ggml::trace::on(), "off: gate latched off without STARLING_TRACE");

    // Unit level: a labeled cache stays silent with the gate off.
    {
        const std::string log = capture_stderr([] {
            LruCache<int, int> labeled(2, "off.cache");
            labeled.get_or_init(1, [](int& v) { v = 1; });
            labeled.get(1);
            starling::ggml::trace::request_event(1.0);  // direct emit attempt
        });
        check(log.find("[trace]") == std::string::npos,
              "off: no records from cache ops or direct emits");
    }

#ifdef _WIN32
    std::printf("[SKIP] e2e gate-off checks (POSIX stderr capture)\n");
#else
    SETENV("STARLING_GGML_DEVICE", "cpu");
    TinyGraniteFixture fixture(std::filesystem::temp_directory_path() /
                               "trace_off_test.gguf");
    check(fixture.wrote(), "off: synthesized tiny granite GGUF written");
    if (fixture.wrote()) {
        const char* err = nullptr;
        void* handle = starling_ggml_granite_load(fixture.path.string().c_str(), &err);
        check(handle != nullptr, "off: tiny granite model loaded", err ? err : "");
        if (handle) {
            const int64_t kSampleRate = 16000;
            std::vector<float> pcm((size_t)(2.5 * kSampleRate), 0.0f);
            char* text = nullptr;
            const std::string log = capture_stderr([&] {
                err = nullptr;
                text = starling_ggml_granite_decode(handle, pcm.data(),
                                                    (int64_t)pcm.size(), &err);
            });
            check(text != nullptr, "off: decode succeeded", err ? err : "");
            if (text) std::free(text);
            check(log.find("[trace]") == std::string::npos,
                  "off: a full multi-chunk decode emits no [trace] lines");
            starling_ggml_granite_free(handle);
        }
    }
#endif
    std::printf("%s\n", failures ? "TRACE OFF FAILED" : "TRACE OFF OK");
    return failures ? 1 : 0;
}
