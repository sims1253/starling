// model_registry.cpp — the one central model table (see model_registry.hpp).
//
// Adding a model = its cpp/<model>/ implementation + ONE row here. The rows
// reference the per-model entry points; everything else in the tree
// (capi.cpp dispatch, serve slug mapping, --version's model list) derives
// from this table.

#include "model_registry.hpp"

#include <cstring>

namespace starling::ggml::lib {
namespace {

// Per-model internal entry points (defined in cpp/<model>/capi_<model>.cpp).
// Parakeet's extra debug passthroughs (mel/encode/decode_ids) stay private to
// capi.cpp, which is their only caller.
extern "C" {
void * starling_ggml_parakeet_load(const char * gguf_path, const char ** err_out);
void   starling_ggml_parakeet_free(void * handle);
char * starling_ggml_parakeet_decode(void * handle, const float * pcm, int64_t n,
                                     const char ** err_out);
void * starling_ggml_moss_load(const char * gguf_path, const char ** err_out);
void   starling_ggml_moss_free(void * handle);
char * starling_ggml_moss_decode(void * handle, const float * pcm, int64_t n,
                                 const char ** err_out);
void * starling_ggml_ark_load(const char * gguf_path, const char ** err_out);
void   starling_ggml_ark_free(void * handle);
char * starling_ggml_ark_decode(void * handle, const float * pcm, int64_t n,
                                const char ** err_out);
void * starling_ggml_higgs_load(const char * gguf_path, const char ** err_out);
void   starling_ggml_higgs_free(void * handle);
char * starling_ggml_higgs_decode(void * handle, const float * pcm, int64_t n,
                                  const char ** err_out);
void * starling_ggml_hojo_load(const char * gguf_path, const char ** err_out);
void   starling_ggml_hojo_free(void * handle);
char * starling_ggml_hojo_decode(void * handle, const float * pcm, int64_t n,
                                 const char ** err_out);
}

// One row per engine, in starling_ggml_model enum order (serve's
// supported-models string is printed in table order). The message fields
// preserve each engine's historical error strings exactly; see
// ModelDescriptor in model_registry.hpp.
//
// constexpr so the most likely one-row-add mistakes — a copy-pasted kind, a
// row appended out of enum order, a missing renumber — fail the static_assert
// below at compile time instead of surfacing later as an unreachable engine
// (find_model returns the first matching row) while --version still
// advertises it.
constexpr ModelDescriptor kRegistry[] = {
    { STARLING_GGML_PARAKEET_TDT, "parakeet",
      starling_ggml_parakeet_load, starling_ggml_parakeet_free,
      starling_ggml_parakeet_decode,
      "starling_ggml_transcribe_pcm: parakeet expects 16 kHz, got %d",
      /*rate_error_in_ctx=*/false,
      "transcribe failed" },
    { STARLING_GGML_MOSS, "moss",
      starling_ggml_moss_load, starling_ggml_moss_free,
      starling_ggml_moss_decode,
      "starling_ggml_transcribe_pcm: MOSS expects 16 kHz",
      /*rate_error_in_ctx=*/true,
      "MOSS transcribe failed" },
    { STARLING_GGML_ARK, "ark",
      starling_ggml_ark_load, starling_ggml_ark_free,
      starling_ggml_ark_decode,
      "starling_ggml_transcribe_pcm: ARK expects 16 kHz",
      /*rate_error_in_ctx=*/true,
      "ARK transcribe failed" },
    { STARLING_GGML_HIGGS, "higgs",
      starling_ggml_higgs_load, starling_ggml_higgs_free,
      starling_ggml_higgs_decode,
      "starling_ggml_transcribe_pcm: HIGGS expects 16 kHz",
      /*rate_error_in_ctx=*/true,
      "HIGGS transcribe failed" },
    { STARLING_GGML_HOJO, "hojo",
      starling_ggml_hojo_load, starling_ggml_hojo_free,
      starling_ggml_hojo_decode,
      "starling_ggml_transcribe_pcm: HOJO expects 16 kHz",
      /*rate_error_in_ctx=*/true,
      "HOJO transcribe failed" },
};

static_assert([] {
    for (size_t i = 0; i < sizeof(kRegistry) / sizeof(kRegistry[0]); ++i)
        if (kRegistry[i].kind != (starling_ggml_model)(i + 1)) return false;
    return true;
}(), "registry rows must stay in starling_ggml_model enum order (1..N, no duplicate or skipped kinds)");

} // namespace

const ModelDescriptor * model_registry(size_t * out_n) {
    if (out_n) *out_n = sizeof(kRegistry) / sizeof(kRegistry[0]);
    return kRegistry;
}

const ModelDescriptor * find_model(starling_ggml_model kind) {
    size_t n = 0;
    const ModelDescriptor * regs = model_registry(&n);
    for (size_t i = 0; i < n; ++i)
        if (regs[i].kind == kind) return &regs[i];
    return nullptr;
}

const ModelDescriptor * find_model_by_slug(const std::string & slug) {
    size_t n = 0;
    const ModelDescriptor * regs = model_registry(&n);
    for (size_t i = 0; i < n; ++i)
        if (slug == regs[i].slug) return &regs[i];
    return nullptr;
}

} // namespace starling::ggml::lib
