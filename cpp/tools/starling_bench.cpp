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
#include "runtime/audio_io.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

void usage() {
    std::fprintf(stderr,
        "usage: starling-bench --model <parakeet|moss|...> --gguf <file> [--runs N]\n"
        "                      [--warmup] [--quiet] file.wav [file.wav ...]\n");
}

starling_ggml_model model_kind(const std::string& s) {
    if (s == "parakeet") return STARLING_GGML_PARAKEET_TDT;
    if (s == "moss") return STARLING_GGML_MOSS;
    if (s == "ark") return STARLING_GGML_ARK;
    if (s == "higgs") return STARLING_GGML_HIGGS;
    if (s == "hojo") return STARLING_GGML_HOJO;
    if (s == "granite") return STARLING_GGML_GRANITE;
    if (s == "qwen3") return STARLING_GGML_QWEN3;
    if (s == "audex") return STARLING_GGML_AUDEX;
    return (starling_ggml_model)0;
}

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

} // namespace

int main(int argc, char** argv) {
    std::string model, gguf;
    int runs = 1;
    bool warmup = false, quiet = false;
    std::vector<std::string> wavs;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) model = argv[++i];
        else if (a == "--gguf" && i + 1 < argc) gguf = argv[++i];
        else if (a == "--runs" && i + 1 < argc) runs = std::max(1, std::atoi(argv[++i]));
        else if (a == "--warmup") warmup = true;
        else if (a == "--quiet") quiet = true;
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else wavs.push_back(a);
    }
    const starling_ggml_model kind = model_kind(model);
    if (!kind || gguf.empty() || wavs.empty()) { usage(); return 2; }

    const double t0 = now_ms();
    starling_ggml_ctx* ctx = starling_ggml_load(kind, gguf.c_str());
    if (!ctx) {
        std::fprintf(stderr, "load failed: %s\n", starling_ggml_last_error(nullptr));
        return 1;
    }
    std::printf("load %.1f ms  backend=%s\n", now_ms() - t0, starling_ggml_backend_name());

    int rc = 0;
    for (const std::string& path : wavs) {
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
        const double dur = pcm.size() / 16000.0;
        if (warmup) {
            char* w = starling_ggml_transcribe_pcm(ctx, pcm.data(), (int64_t)pcm.size(), 16000);
            starling_ggml_free_string(w);
        }
        for (int r = 0; r < runs; ++r) {
            const double a = now_ms();
            char* text = starling_ggml_transcribe_pcm(ctx, pcm.data(), (int64_t)pcm.size(), 16000);
            const double ms = now_ms() - a;
            if (!text) {
                std::fprintf(stderr, "%s: transcribe failed: %s\n", path.c_str(), starling_ggml_last_error(ctx));
                rc = 1;
                break;
            }
            std::printf("%s run=%d audio=%.2fs time=%.1fms rtf=%.4f%s%s\n", path.c_str(), r, dur, ms,
                        ms / 1000.0 / dur, quiet ? "" : "\n  ", quiet ? "" : text);
            std::fflush(stdout);
            starling_ggml_free_string(text);
        }
    }
    starling_ggml_free(ctx);
    starling_ggml_shutdown();
    return rc;
}
