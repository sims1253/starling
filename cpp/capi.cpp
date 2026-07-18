// capi.cpp — shared C API shell for libstarling_ggml.
//
// Implements the build-introspection + lifecycle + last-error surface defined
// in cpp/include/starling_ggml.h. The model-specific load/transcribe entry
// points are stubs at Phase 0 (they return nullptr with an error) and are
// filled in once the parakeet (Phase 1) and moss (Phase 2) implementations
// land. Every entry is exception-fenced: a C++ exception never crosses the
// boundary; it's caught and stored in last_error.

#include "starling_ggml.h"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

namespace {

// Thread-local-ish last error. Starling's engine serialises calls (the ggml
// backend is process-global), so a single string + mutex is sufficient. The
// empty string means "no error".
std::mutex g_err_mutex;
std::string g_last_error;

void set_error(const char * ctx_tag, const std::string & msg) {
    std::lock_guard<std::mutex> lk(g_err_mutex);
    g_last_error = msg;
}

const char * backend_name_for_build() {
#if defined(GGML_USE_CUDA)
    return "cuda";
#elif defined(GGML_USE_METAL)
    return "metal";
#elif defined(GGML_USE_VULKAN)
    return "vulkan";
#elif defined(GGML_USE_HIP)
    return "hip";
#else
    return "cpu";
#endif
}

} // namespace

extern "C" {

// --------------------------------------------------------------------------- //
// ABI / build introspection
// --------------------------------------------------------------------------- //
int starling_ggml_abi_version(void) {
    return STARLING_GGML_ABI_VERSION;
}

const char * starling_ggml_backend_name(void) {
    return backend_name_for_build();
}

// --------------------------------------------------------------------------- //
// Lifecycle (stubs at Phase 0)
// --------------------------------------------------------------------------- //
starling_ggml_ctx * starling_ggml_load(starling_ggml_model /*model*/,
                                       const char * /*gguf_path*/) {
    // Phase 0 stub. The parakeet (Phase 1) and moss (Phase 2) implementations
    // replace this with real model loading.
    set_error("load", "starling_ggml: no model implementation linked yet (Phase 0 stub)");
    return nullptr;
}

void starling_ggml_free(starling_ggml_ctx * /*ctx*/) {
    // No-op until Phase 1 wires a real context.
}

void starling_ggml_shutdown(void) {
    // Phase 0: the global backend doesn't exist yet (no model loads). The real
    // teardown (clear_decode_caches -> backend.reset -> flag, registered via
    // std::atexit) lands with the runtime in Phase 0c.
}

const char * starling_ggml_last_error(starling_ggml_ctx * /*ctx*/) {
    std::lock_guard<std::mutex> lk(g_err_mutex);
    return g_last_error.c_str();
}

// --------------------------------------------------------------------------- //
// Inference (stubs at Phase 0)
// --------------------------------------------------------------------------- //
char * starling_ggml_transcribe_pcm(starling_ggml_ctx * /*ctx*/,
                                    const float * /*samples*/, int64_t /*n*/,
                                    int /*sample_rate*/) {
    set_error("transcribe", "starling_ggml: no model implementation linked yet (Phase 0 stub)");
    return nullptr;
}

void starling_ggml_free_string(char * s) {
    if (s) std::free(s);
}

} // extern "C"
