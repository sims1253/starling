// capi.cpp — shared C API shell for libstarling_ggml.
//
// Implements the C API declared in cpp/include/starling_ggml.h. Dispatches
// load/free/transcribe on a per-model basis (parakeet-tdt now; moss in Phase 2).
// Every entry is exception-fenced: a C++ exception never crosses the boundary;
// it's caught and stored in last_error.

#include "starling_ggml.h"

#include "runtime/graph.hpp"  // global_backend, shutdown_backend, shutting_down

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

// The opaque context: a tagged variant over the per-model handles. Only one
// model kind is active per context (selected at load). Defined at global scope
// so it IS the header's incomplete `struct starling_ggml_ctx` made complete.
struct starling_ggml_ctx {
    starling_ggml_model kind = (starling_ggml_model)0;
    void* model = nullptr;   // the per-model handle (ParakeetCtx*, ...)
    std::string last_error;
};

namespace {

// ---- per-model internal entry points (defined in capi_parakeet.cpp etc.) ----
extern "C" {
void * starling_ggml_parakeet_load(const char * gguf_path, const char ** err_out);
void   starling_ggml_parakeet_free(void * handle);
float * starling_ggml_parakeet_mel(void * handle, const float * pcm, int64_t n,
                                   int * out_T, const char ** err_out);
// (moss entries declared in capi_moss.cpp, Phase 2)
}

std::mutex g_err_mutex;
std::string g_last_error;

void set_global_error(const std::string & msg) {
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
int starling_ggml_abi_version(void) { return STARLING_GGML_ABI_VERSION; }
const char * starling_ggml_backend_name(void) { return backend_name_for_build(); }

// --------------------------------------------------------------------------- //
// Lifecycle
// --------------------------------------------------------------------------- //
starling_ggml_ctx * starling_ggml_load(starling_ggml_model model,
                                       const char * gguf_path) {
    const char* err = nullptr;
    void* handle = nullptr;
    if (model == STARLING_GGML_PARAKEET_TDT) {
        handle = starling_ggml_parakeet_load(gguf_path, &err);
    } else {
        set_global_error("starling_ggml_load: unsupported model kind");
        return nullptr;
    }
    if (!handle) {
        set_global_error(err ? err : "starling_ggml_load: failed (no message)");
        return nullptr;
    }
    auto* ctx = new starling_ggml_ctx;
    ctx->kind = model;
    ctx->model = handle;
    return ctx;
}

void starling_ggml_free(starling_ggml_ctx * ctx) {
    if (!ctx) return;
    if (ctx->model) {
        if (ctx->kind == STARLING_GGML_PARAKEET_TDT)
            starling_ggml_parakeet_free(ctx->model);
    }
    delete ctx;
}

void starling_ggml_shutdown(void) {
    starling::ggml::shutdown_backend();
}

const char * starling_ggml_last_error(starling_ggml_ctx * ctx) {
    if (ctx) return ctx->last_error.c_str();
    std::lock_guard<std::mutex> lk(g_err_mutex);
    return g_last_error.c_str();
}

// --------------------------------------------------------------------------- //
// Inference
// --------------------------------------------------------------------------- //
char * starling_ggml_transcribe_pcm(starling_ggml_ctx * ctx,
                                    const float * samples, int64_t n,
                                    int sample_rate) {
    if (!ctx || !ctx->model) {
        set_global_error("starling_ggml_transcribe_pcm: null context");
        return nullptr;
    }
    if (ctx->kind == STARLING_GGML_PARAKEET_TDT) {
        // Full transcribe lands in Phase 1d (encoder 1b + decode 1c). For now
        // it's a stub so the engine wiring can be validated end-to-end.
        (void)samples; (void)n; (void)sample_rate;
        set_global_error("starling_ggml_transcribe_pcm: parakeet decode not yet linked (Phase 1a: mel only)");
        return nullptr;
    }
    set_global_error("starling_ggml_transcribe_pcm: unsupported model kind");
    return nullptr;
}

void starling_ggml_free_string(char * s) {
    if (s) std::free(s);
}

// --------------------------------------------------------------------------- //
// Internal mel-test passthrough (not in the public header; used by the Phase 1a
// validation script). Exposed so the Python binding can call it via ctypes.
// --------------------------------------------------------------------------- //
float * starling_ggml_parakeet_mel_pub(starling_ggml_ctx * ctx,
                                       const float * pcm, int64_t n,
                                       int * out_T) {
    if (!ctx || ctx->kind != STARLING_GGML_PARAKEET_TDT) {
        set_global_error("starling_ggml_parakeet_mel_pub: not a parakeet context");
        return nullptr;
    }
    const char* err = nullptr;
    float* r = starling_ggml_parakeet_mel(ctx->model, pcm, n, out_T, &err);
    if (!r) set_global_error(err ? err : "mel failed");
    return r;
}

} // extern "C"
