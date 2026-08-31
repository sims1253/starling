// model_registry.hpp — the one central table of engines built into
// libstarling_ggml (INTERNAL; not part of the public starling_ggml.h).
//
// Before this table existed, adding a model meant touching four or five
// places: the load/free/transcribe if-chains in capi.cpp plus the slug
// mapping and supported-list string in serve/server.cpp. Now a model brings
// its own implementation (cpp/<model>/, with capi_<model>.cpp exposing the
// three entry points below) plus ONE contiguous addition in
// model_registry.cpp — the three entry-point declarations and the kRegistry
// row, side by side in that file. The public enum kind is added to
// starling_ggml.h with the usual ABI bump (the Python binding's expected
// version follows). Everything else — C-API dispatch, serve slug mapping,
// --version's model list — is derived from the table.
//
// The registry lives in the library's internal namespace. serve links
// starling_ggml statically and includes this header directly; nothing here
// leaks into the public ABI (starling_ggml.h stays byte-identical, and the
// registry itself never touches STARLING_GGML_ABI_VERSION — model additions
// bump it in starling_ggml.h per the usual convention).

#pragma once

#include "starling_ggml.h"

#include <cstddef>
#include <string>

namespace starling::ggml::lib {

// The per-model C entry points implemented in each cpp/<model>/capi_<model>.cpp.
// Handles are opaque to the registry; each engine owns its concrete ctx type.
using ModelLoadFn   = void * (*)(const char * gguf_path, const char ** err_out);
using ModelFreeFn   = void   (*)(void * handle);
using ModelDecodeFn = char * (*)(void * handle, const float * pcm, int64_t n,
                                 const char ** err_out);
// Text-in/text-out models (s1): normalize one raw transcript under the given
// styling/structure/context controls. Null for the audio engines.
using ModelNormalizeFn = char * (*)(void * handle, const char * transcript,
                                    const char * styling, const char * structure,
                                    const char * context, const char ** err_out);

struct ModelDescriptor {
    starling_ggml_model kind;  // public enum value (starling_ggml.h)
    const char * slug;         // serve CLI/HTTP name ("parakeet", ...)

    ModelLoadFn   load_fn;
    ModelFreeFn   free_fn;
    ModelDecodeFn decode_fn;

    // Shape of the error messages the shared 16 kHz guard and the decode
    // fallback in capi.cpp emit, kept per row so the shared paths reproduce
    // each engine's historical message byte-for-byte:
    //   * rate_error_fmt — printf format for a wrong-sample-rate rejection.
    //     Parakeet's historically includes ", got %d" (sample_rate); the LLM
    //     engines' is a fixed string with no conversion.
    //   * rate_error_in_ctx — parakeet historically stored the rejection only
    //     in the global last-error, not the per-context one; the LLM engines
    //     set both.
    //   * decode_fallback — message used when decode fails without an
    //     engine-provided error.
    const char * rate_error_fmt;
    bool rate_error_in_ctx;
    const char * decode_fallback;

    // Text-in/text-out path (s1). Null for the audio engines; appended after
    // the historical fields so existing rows' positional init is unchanged.
    ModelNormalizeFn normalize_fn = nullptr;
};

// The table, one row per model in enum order. |out_n| receives the row count.
const ModelDescriptor * model_registry(size_t * out_n);

// Row for a public model kind, or nullptr for an unknown value.
const ModelDescriptor * find_model(starling_ggml_model kind);

// Row for a serve slug ("parakeet", "moss", ...), or nullptr if unknown.
const ModelDescriptor * find_model_by_slug(const std::string & slug);

} // namespace starling::ggml::lib
