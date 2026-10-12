// starling_bench.cpp — transcribe WAV files through the public C API and
// report latency, for comparing engines on the same build:
//
//   STARLING_ENGINE=ggml starling-bench --model parakeet --gguf m.gguf a.wav
//   STARLING_ENGINE=fast starling-bench --model parakeet --gguf m.gguf a.wav
//
// Prints one line per file and run: wall time, real-time factor and the
// transcript (the first run of each file includes one-time graph setup;
// use --warmup to exclude it).

#include "starling_ggml.h"
#include "lib/model_registry.hpp"
#include "runtime/audio_io.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#endif
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// SIGTERM/SIGINT → cooperative stop (#325 crash circumvention): the engine
// finishes its current unit of work (a decode round for MOSS, the current
// call otherwise) and the process exits through normal destructors — the
// VkDevice is destroyed instead of abandoned for the driver to reap (a
// leading wedge correlate: neither SIGKILL nor default SIGTERM runs C++
// destructors). One-shot: the handler restores the default disposition, so
// a second signal terminates a process that does not wind down.
volatile std::sig_atomic_t g_stop_sig = 0;

void on_terminate(int sig) {
    g_stop_sig = sig;
    starling_ggml_request_stop();      // lock-free atomic store
    std::signal(sig, SIG_DFL);         // async-signal-safe (POSIX)
}

bool stop_now() {
    if (!starling_ggml_stop_requested()) return false;
    static bool said = false;
    if (!said) std::fprintf(stderr, "[stop] SIGTERM/SIGINT: exiting cleanly\n");
    said = true;
    return true;
}

void usage() {
    std::fprintf(stderr,
        "usage: starling-bench --model <parakeet|moss|...> --gguf <file> [--runs N]\n"
        "                      [--cycles N] [--warmup] [--quiet] file.wav [file.wav ...]\n"
        "  --cycles N  load, run and free the model N times in this process\n"
        "  --cold      drop the GGUF from the page cache before each load\n");
}

starling_ggml_model model_kind(const std::string& s) {
    // Any registry slug resolves through the model table (model_registry.cpp),
    // so every native model (parakeet, moss, ark, ark06, qwen3, granite,
    // s1, voxtral, ...) is benchable; the historical
    // hardcoded names are registry slugs and keep working unchanged (#176).
    const starling::ggml::lib::ModelDescriptor* d =
        starling::ggml::lib::find_model_by_slug(s);
    return d ? d->kind : (starling_ggml_model)0;
}

// Resident memory from /proc/self/status (Linux/Android; -1 elsewhere), in
// MiB. VmHWM is the peak since process start: printed right after the first
// load it is that load's peak (the GGUF, repacked weights, and the device
// memory the driver maps into the process).
double status_mib(const char* key) {
    FILE* f = std::fopen("/proc/self/status", "r");
    if (!f) return -1.0;
    char line[256];
    double v = -1.0;
    const size_t n = std::strlen(key);
    while (std::fgets(line, sizeof line, f))
        if (!std::strncmp(line, key, n) && line[n] == ':') { v = std::atof(line + n + 1) / 1024.0; break; }
    std::fclose(f);
    return v;
}

// --cold: drop `path` from the page cache (clean pages only, no root
// needed) so the next load reads it from storage.
void evict_file(const std::string& path) {
#if defined(__linux__)
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return;
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    close(fd);
#else
    (void)path;
#endif
}

// System-wide memory drop during a load (MemAvailable sampled every 10 ms):
// covers what VmHWM misses — device allocations the driver keeps outside
// the process's mappings, page cache pressure.
class MemSampler {
public:
    MemSampler() : base_(mem_available_mib()), min_(base_) {
        th_ = std::thread([this] {
            while (!stop_) {
                const double v = mem_available_mib();
                if (v >= 0 && v < min_) min_ = v;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });
    }
    // Peak drop in MiB (-1 when /proc/meminfo is unreadable).
    double finish() {
        stop_ = true;
        th_.join();
        return base_ < 0 ? -1.0 : base_ - min_;
    }

private:
    static double mem_available_mib() {
        FILE* f = std::fopen("/proc/meminfo", "r");
        if (!f) return -1.0;
        char line[256];
        double v = -1.0;
        while (std::fgets(line, sizeof line, f))
            if (!std::strncmp(line, "MemAvailable:", 13)) { v = std::atof(line + 13) / 1024.0; break; }
        std::fclose(f);
        return v;
    }
    double base_, min_;
    std::atomic<bool> stop_{false};
    std::thread th_;
};

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

} // namespace

int main(int argc, char** argv) {
    std::string model, gguf;
    int runs = 1, cycles = 1;
    bool warmup = false, quiet = false, cold = false;
    std::vector<std::string> wavs;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) model = argv[++i];
        else if (a == "--gguf" && i + 1 < argc) gguf = argv[++i];
        else if (a == "--runs" && i + 1 < argc) runs = std::max(1, std::atoi(argv[++i]));
        else if (a == "--cycles" && i + 1 < argc) cycles = std::max(1, std::atoi(argv[++i]));
        else if (a == "--warmup") warmup = true;
        else if (a == "--quiet") quiet = true;
        else if (a == "--cold") cold = true;
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else wavs.push_back(a);
    }
    const starling_ggml_model kind = model_kind(model);
    if (!kind || gguf.empty() || wavs.empty()) { usage(); return 2; }

    for (int sig : {SIGTERM, SIGINT})
        if (std::signal(sig, on_terminate) == SIG_ERR)
            std::fprintf(stderr, "warning: cannot install the signal %d handler; "
                                 "it will kill without teardown\n", sig);

    int rc = 0;
    for (int cycle = 0; cycle < cycles && !stop_now(); ++cycle) {
        if (cycles > 1) std::printf("cycle %d/%d\n", cycle + 1, cycles);
        if (cold) evict_file(gguf);
        MemSampler sampler;
        const double t0 = now_ms();
        starling_ggml_ctx* ctx = starling_ggml_load(kind, gguf.c_str());
        const double sys_drop = sampler.finish();
        if (!ctx) {
            std::fprintf(stderr, "load failed: %s\n", starling_ggml_last_error(nullptr));
            rc = 1;
            break;
        }
        const double t_load = now_ms() - t0;
        std::printf("load %.1f ms%s  backend=%s  rss=%.0f MiB (anon %.0f, file %.0f) peak=%.0f MiB "
                    "sys_drop=%.0f MiB\n",
                    t_load, cold ? " (cold)" : "", starling_ggml_backend_name(), status_mib("VmRSS"),
                    status_mib("RssAnon"), status_mib("RssFile"), status_mib("VmHWM"), sys_drop);

        for (const std::string& path : wavs) {
            if (stop_now()) break;
            std::vector<float> pcm;
            int sr = 0;
            std::string err;
            if (!starling::ggml::read_wav(path.c_str(), pcm, sr, err)) {
                std::fprintf(stderr, "%s: %s\n", path.c_str(), err.c_str());
                rc = 1;
                continue;
            }
            if (sr != 16000) {
                std::vector<float> rs;
                starling::ggml::resample_pcm(pcm.data(), pcm.size(), sr, 16000, rs);
                pcm.swap(rs);
            }
            if (pcm.empty()) {
                std::fprintf(stderr, "%s: no samples\n", path.c_str());
                rc = 1;
                continue;
            }
            const double dur = pcm.size() / 16000.0;
            if (warmup) {
                char* w = starling_ggml_transcribe_pcm(ctx, pcm.data(), (int64_t)pcm.size(), 16000);
                starling_ggml_free_string(w);
            }
            for (int r = 0; r < runs; ++r) {
                if (stop_now()) break;
                const double a = now_ms();
                char* text = starling_ggml_transcribe_pcm(ctx, pcm.data(), (int64_t)pcm.size(), 16000);
                const double ms = now_ms() - a;
                if (!text) {
                    std::fprintf(stderr, "%s: transcribe failed: %s\n", path.c_str(), starling_ggml_last_error(ctx));
                    rc = 1;
                    break;
                }
                // A stop mid-call may have cut the transcript short (MOSS returns
                // the prefix): label it so it is never read as a full result.
                const bool cut = starling_ggml_stop_requested();
                std::printf("%s run=%d audio=%.2fs time=%.1fms rtf=%.4f%s%s%s\n", path.c_str(), r, dur, ms,
                            ms / 1000.0 / dur, cut ? " [stopped: may be truncated]" : "",
                            quiet ? "" : "\n  ", quiet ? "" : text);
                std::fflush(stdout);
                starling_ggml_free_string(text);
            }
        }
        starling_ggml_free(ctx);
    }
    starling_ggml_shutdown();
    // Stopped by a signal: exit like one (128 + signo) so callers never read
    // a stopped — possibly truncated or incomplete — bench as a full result.
    if (g_stop_sig) return 128 + (int)g_stop_sig;
    return rc;
}
