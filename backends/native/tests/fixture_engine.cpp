#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"
// Contract-test engine only. This file is never linked into starling-serve.
#include "starling_ggml.h"
#include "lib/model_registry.hpp"
#include <cstdlib>
#include <cstring>

struct starling_ggml_ctx {};
extern "C" {
int starling_ggml_abi_version() { return STARLING_GGML_ABI_VERSION; }
const char* starling_ggml_backend_name() { return "contract-fixture"; }
starling_ggml_ctx* starling_ggml_load(starling_ggml_model, const char*) { return new starling_ggml_ctx; }
void starling_ggml_free(starling_ggml_ctx* ctx) { delete ctx; }
void starling_ggml_shutdown() {}
const char* starling_ggml_last_error(starling_ggml_ctx*) { return "fixture error"; }
void starling_ggml_free_string(char* text) { std::free(text); }
static char* echo(const char* text) {
    auto* copy = static_cast<char*>(std::malloc(std::strlen(text) + 1));
    std::strcpy(copy, text);
    return copy;
}

char* starling_ggml_transcribe_pcm(starling_ggml_ctx*, const float*, int64_t, int) {
    return echo(u8"5. Keep auth.\n6. I'd prefer to never merge this.\n7. I like orange, err, yellow.\n8. A. Agreed. caf\u00e9 \U0001F399");
}
// Echo the transcript back: the /normalize contract tests pin the JSON
// transport decoding (raw UTF-8 vs \u escapes) byte-for-byte against the
// engine input, without a real text model (issue #123).
char* starling_ggml_normalize_text(starling_ggml_ctx*, const char* transcript, const char*, const char*, const char*) {
    return echo(transcript);
}
}
namespace starling::ggml::lib {
// s1 registers a normalize entry point so the fixture can serve /normalize.
char* fixture_normalize(void*, const char* transcript, const char*, const char*, const char*, const char**) {
    return echo(transcript);
}
const ModelDescriptor entries[] = {
    {STARLING_GGML_PARAKEET_TDT, "parakeet", nullptr, nullptr, nullptr, "", false, "", nullptr},
    {STARLING_GGML_S1, "s1", nullptr, nullptr, nullptr, "", false, "", fixture_normalize},
};
const ModelDescriptor* model_registry(size_t* count) { if (count) *count = 2; return entries; }
const ModelDescriptor* find_model(starling_ggml_model kind) {
    for (const auto& e : entries) if (e.kind == kind) return &e;
    return nullptr;
}
const ModelDescriptor* find_model_by_slug(const std::string& slug) {
    for (const auto& e : entries) if (slug == e.slug) return &e;
    return nullptr;
}
}
