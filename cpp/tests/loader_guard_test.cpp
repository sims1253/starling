// loader_guard_test.cpp — CPU-only loader smoke checks for the higgs/hojo
// GGML metadata guards. Loading parses GGUF metadata + tensor names only (no
// backend/GPU is touched), so this runs even while the GPU is busy. Exits 0
// with SKIP when the model files are absent (they are gitignored).
//
// Usage: ./loader_guard_test [higgs.gguf [hojo.gguf]]
#include "higgs/loader.hpp"
#include "hojo/loader.hpp"
#include "ggml.h"
#include <cstdio>
#include <string>

static int failures = 0;

static void check(bool ok, const char* what, const std::string& err = "") {
    std::printf("[%s] %s%s%s\n", ok ? "PASS" : "FAIL", what,
                (!ok && !err.empty()) ? " -- " : "", ok ? "" : err.c_str());
    if (!ok) failures++;
}

int main(int argc, char** argv) {
    const char* higgs_path = argc > 1 ? argv[1] : "models/higgs-audio-v3-bf16-exact.gguf";
    const char* hojo_path = argc > 2 ? argv[2] : "models/hojo-asr-v1.gguf";

    {
        starling::ggml::higgs::HiggsModel m;
        std::string err;
        const bool ok = m.load(higgs_path, err);
        if (ok) check(true, "higgs GGUF loads (metadata guards accept)");
        else if (err.find("failed to open") != std::string::npos ||
                 err.find("cannot open") != std::string::npos ||
                 err.find("open ") != std::string::npos)
            std::printf("[SKIP] higgs GGUF absent (%s)\n", higgs_path);
        else check(false, "higgs GGUF loads", err);
    }
    {
        starling::ggml::hojo::HojoModel m;
        std::string err;
        const bool ok = m.load(hojo_path, err);
        if (ok) {
            check(true, "hojo GGUF loads (metadata guards accept)");
            // The tower's conv width check must agree with conv_out.weight's
            // input dim on the real model (480 * (128>>3) == ne[0] == 7680).
            const auto& tc = m.config.tower;
            ggml_tensor* w = m.loader.tensor("audio.conv_out.weight");
            const int64_t conv_width = w ? (int64_t) w->ne[0] : -1;
            const int64_t meta_width = (int64_t) tc.downsample_hidden_size *
                                       ((int64_t) tc.num_mel_bins >> 3);
            check(conv_width > 0 && conv_width == meta_width,
                  "hojo conv width matches conv_out.weight",
                  "weight=" + std::to_string(conv_width) +
                  " meta=" + std::to_string(meta_width));
        } else if (err.find("open") != std::string::npos) {
            std::printf("[SKIP] hojo GGUF absent (%s)\n", hojo_path);
        } else {
            check(false, "hojo GGUF loads", err);
        }
    }
    std::printf("%s\n", failures ? "LOADER GUARD FAILED" : "LOADER GUARD OK");
    return failures ? 1 : 0;
}
