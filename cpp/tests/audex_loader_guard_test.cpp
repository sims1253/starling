// audex_loader_guard_test.cpp — the audex GGUF numeric-profile gate must
// accept bf16_exact AND quantized (the quantization-port allowlist) while
// still rejecting values the loader does not promise (f16,
// mixed_f32_bf16_exact, garbage), and treat an absent profile as accepted
// (the gate validates the field only when present). Synthesizes
// metadata-only GGUFs so the gate checks run anywhere (no model files,
// CPU-only, no skip): the profile gate runs inside check_gguf_header before
// its trailing no-tensors check, so an ACCEPTED profile surfaces as
// "contains no tensors", a REJECTED one as "numeric profile".
//
// Usage: ./audex_loader_guard_test [quantized_audex.gguf]
//   Optional real-file smoke (default models/audex-2b-q4_k_m.gguf): a full
//   load of a quantized audex GGUF when present, SKIP when absent (they are
//   gitignored) — mirrors loader_guard_test's convention.
#include "audex/loader.hpp"
#include "gguf.h"
#include <cstdio>
#include <string>

static int failures = 0;

static void check(bool ok, const char* what, const std::string& err = "") {
    std::printf("[%s] %s%s%s\n", ok ? "PASS" : "FAIL", what,
                (!ok && !err.empty()) ? " -- " : "", ok ? "" : err.c_str());
    if (!ok) failures++;
}

// Metadata-only GGUF with the given starling.numeric_profile (nullptr =
// omit the field). No tensors: AudexModel::load gates the profile before
// any config/tensor check, which is exactly what this test probes.
static bool write_profile_gguf(const char* path, const char* profile) {
    std::remove(path);
    gguf_context* gf = gguf_init_empty();
    gguf_set_val_str(gf, "general.architecture", "audex");
    gguf_set_val_u32(gf, "starling.format_version", 1);
    if (profile) gguf_set_val_str(gf, "starling.numeric_profile", profile);
    const bool wrote = gguf_write_to_file(gf, path, /*only_meta=*/true);
    gguf_free(gf);
    return wrote;
}

// load() must FAIL (missing config/tensors) with an error that got PAST the
// profile gate — i.e. never mentions it.
static void expect_accepted(const char* profile) {
    const std::string label = profile ? "profile '" + std::string(profile) + "'"
                                      : "absent profile";
    const char* path = "/tmp/audex_loader_guard_test.gguf";
    if (!write_profile_gguf(path, profile)) {
        check(false, ("synthesized GGUF written (" + label + ")").c_str());
        return;
    }
    starling::ggml::audex::AudexModel m;
    std::string err;
    const bool loaded = m.load(path, err);
    // Accepted profile => check_gguf_header proceeds past the profile gate
    // to its trailing no-tensors check (the metadata-only file has none).
    check(!loaded && err.find("contains no tensors") != std::string::npos,
          (label + " passes the gate").c_str(), err);
}

static void expect_rejected(const char* profile) {
    const std::string label = "profile '" + std::string(profile) + "'";
    const char* path = "/tmp/audex_loader_guard_test.gguf";
    if (!write_profile_gguf(path, profile)) {
        check(false, ("synthesized GGUF written (" + label + ")").c_str());
        return;
    }
    starling::ggml::audex::AudexModel m;
    std::string err;
    const bool loaded = m.load(path, err);
    check(!loaded && err.find("numeric profile") != std::string::npos,
          (label + " rejected loudly").c_str(), err);
}

int main(int argc, char** argv) {
    expect_accepted("bf16_exact");            // pre-existing acceptance
    expect_accepted("quantized");             // the allowlist addition
    expect_accepted(nullptr);                 // field validated only when present
    expect_rejected("f16");                   // moss/ark/higgs value, not audex's
    expect_rejected("mixed_f32_bf16_exact");  // hojo value, not audex's
    expect_rejected("garbage");
    std::remove("/tmp/audex_loader_guard_test.gguf");

    // Optional real-file smoke: a full quantized-file load (metadata gate +
    // config + tensor map) where a quantized audex GGUF is present.
    const char* qpath = argc > 1 ? argv[1] : "models/audex-2b-q4_k_m.gguf";
    {
        starling::ggml::audex::AudexModel m;
        std::string err;
        const bool ok = m.load(qpath, err);
        if (ok)
            check(true, "quantized audex GGUF loads end-to-end (gate accepts)");
        else if (err.find("failed to open") != std::string::npos ||
                 err.find("open ") != std::string::npos)
            std::printf("[SKIP] quantized audex GGUF absent (%s)\n", qpath);
        else
            check(false, "quantized audex GGUF loads", err);
    }

    std::printf("%s\n", failures ? "AUDEX LOADER GUARD FAILED" : "AUDEX LOADER GUARD OK");
    return failures ? 1 : 0;
}
