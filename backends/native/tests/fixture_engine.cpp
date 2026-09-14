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
char* starling_ggml_transcribe_pcm(starling_ggml_ctx*, const float*, int64_t, int) {
    const char* raw = u8"5. Keep auth.\n6. I'd prefer to never merge this.\n7. I like orange, err, yellow.\n8. A. Agreed. caf\u00e9 \U0001F399";
    auto* text = static_cast<char*>(std::malloc(std::strlen(raw) + 1));
    std::strcpy(text, raw);
    return text;
}
char* starling_ggml_normalize_text(starling_ggml_ctx*, const char*, const char*, const char*, const char*) { return nullptr; }
}
namespace starling::ggml::lib {
const ModelDescriptor entry = {STARLING_GGML_PARAKEET_TDT, "parakeet", nullptr, nullptr, nullptr, "", false, "", nullptr};
const ModelDescriptor* model_registry(size_t* count) { if (count) *count = 1; return &entry; }
const ModelDescriptor* find_model(starling_ggml_model kind) { return kind == entry.kind ? &entry : nullptr; }
const ModelDescriptor* find_model_by_slug(const std::string& slug) { return slug == entry.slug ? &entry : nullptr; }
}
