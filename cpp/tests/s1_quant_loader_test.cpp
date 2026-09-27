// S1's quantized profile is opt-in; unrelated profiles remain rejected.
#include "s1/loader.hpp"
#include "gguf.h"
#include <cstdio>
#include <string>

static int failures = 0;

static void check(bool ok, const std::string& what, const std::string& err = "") {
    std::printf("[%s] %s%s%s\n", ok ? "PASS" : "FAIL", what.c_str(),
                ok || err.empty() ? "" : " -- ", ok ? "" : err.c_str());
    if (!ok) ++failures;
}

static bool write_profile(const char* path, const char* profile) {
    std::remove(path);
    gguf_context* gf = gguf_init_empty();
    gguf_set_val_str(gf, "general.architecture", "s1");
    gguf_set_val_u32(gf, "starling.format_version", 1);
    if (profile) gguf_set_val_str(gf, "starling.numeric_profile", profile);
    const bool ok = gguf_write_to_file(gf, path, true);
    gguf_free(gf);
    return ok;
}

int main(int argc, char** argv) {
    const char* path = "/tmp/s1_quant_loader_test.gguf";
    for (const char* profile : {"bf16_exact", "quantized", "f16", "garbage"}) {
        if (!write_profile(path, profile)) { check(false, "write metadata GGUF"); continue; }
        starling::ggml::s1::S1Model model;
        std::string err;
        const bool loaded = model.load(path, err);
        const bool accepted = std::string(profile) == "bf16_exact" ||
                              std::string(profile) == "quantized";
        check(!loaded && (accepted ? err.find("contains no tensors") != std::string::npos
                                   : err.find("numeric profile") != std::string::npos),
              std::string("profile '") + profile + (accepted ? "' accepted" : "' rejected"), err);
    }
    std::remove(path);

    // A real artifact exercises the tensor-type guard and entire loader.
    if (argc > 1) {
        starling::ggml::s1::S1Model model;
        std::string err;
        check(model.load(argv[1], err), "quantized S1 GGUF loads", err);
    }
    return failures ? 1 : 0;
}
