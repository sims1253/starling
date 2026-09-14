// capi.cpp — shared C API shell for libstarling_ggml.
//
// Implements the C API declared in cpp/include/starling_ggml.h. Dispatches
// load/free/transcribe through the central model table
// (lib/model_registry.hpp); adding a model is one registry row, not another
// if-chain here. Every entry is exception-fenced: a C++ exception never
// crosses the boundary; it's caught and stored in last_error.

#include "starling_ggml.h"

#include "lib/model_registry.hpp"
#include "runtime/graph.hpp"  // global_backend, shutdown_backend, shutting_down
#include "runtime/imatrix.hpp"  // ImatrixCollector (imatrix_flush_pub)

#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <mutex>
#include <string>
#include <memory>
#include <type_traits>
#include <stdexcept>

// The internal model table (cpp/lib/model_registry.hpp), aliased for brevity.
namespace lib = starling::ggml::lib;

// The opaque context: a tagged variant over the per-model handles. Only one
// model kind is active per context (selected at load). Defined at global scope
// so it IS the header's incomplete `struct starling_ggml_ctx` made complete.
struct starling_ggml_ctx {
    starling_ggml_model kind = (starling_ggml_model)0;
    void* model = nullptr;   // the per-model handle (ParakeetCtx*, ...)
    std::string last_error;
};

namespace {

// ---- parakeet debug-passthrough entry points (capi_parakeet.cpp) ----
// Only the mel/encode/decode/decode_ids introspection entries used by the
// _pub wrappers below; every model's load/free/decode trio (parakeet's
// included) is declared once, next to the registry table in
// lib/model_registry.cpp.
extern "C" {
float * starling_ggml_parakeet_mel(void * handle, const float * pcm, int64_t n,
                                   int * out_T, const char ** err_out);
float * starling_ggml_parakeet_encode(void * handle, const float * pcm, int64_t n,
                                      int * out_T, const char ** err_out);
char * starling_ggml_parakeet_decode(void * handle, const float * pcm, int64_t n,
                                     const char ** err_out);
int64_t * starling_ggml_parakeet_decode_ids(void * handle, const float * pcm, int64_t n,
                                            int64_t * out_n, const char ** err_out);
}

thread_local char g_last_error[2048] = {};

void set_global_error(const std::string& msg) {
    std::snprintf(g_last_error, sizeof(g_last_error), "%s", msg.c_str());
}

template<class F>
auto api_call(starling_ggml_ctx* ctx, F&& fn) -> decltype(fn()) {
    using Result = decltype(fn());
    try {
        std::lock_guard<std::recursive_mutex> lock(starling::ggml::runtime_mutex());
        try {
            return fn();
        } catch (const std::exception& e) {
            std::snprintf(g_last_error, sizeof(g_last_error), "%s", e.what());
        } catch (...) {
            std::snprintf(g_last_error, sizeof(g_last_error), "%s", "unknown native exception");
        }
        if (ctx) {
            // Keep the context protected through error reporting, including OOM.
            try { ctx->last_error = g_last_error; } catch (...) {}
        }
    } catch (...) {
        std::snprintf(g_last_error, sizeof(g_last_error), "%s", "native runtime lock failed");
    }
    if constexpr (!std::is_void_v<Result>) return Result{};
}

void require_running() {
    if (starling::ggml::shutting_down())
        throw std::runtime_error("Starling backend has been shut down");
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
const char * starling_ggml_backend_name(void) {
    return api_call(nullptr, [&]() -> const char * {
        // Report the runtime-selected device (STARLING_GGML_DEVICE / auto-pick)
        // once a model load has created the global Backend; before that, the
        // compile-time backend family. The device name is latched into a static
        // so the returned pointer outlives the Backend (the header promises a
        // static string, and a caller may hold it across shutdown).
        // try_global_backend_device_name() returns the name BY VALUE (copied
        // under the Backend lock), so a concurrent shutdown cannot free the
        // string mid-copy.
        if (auto dev = starling::ggml::try_global_backend_device_name()) {
            static const std::string runtime_name = *dev;
            return runtime_name.c_str();
        }
        return backend_name_for_build();
    });
}

// --------------------------------------------------------------------------- //
// Lifecycle
// --------------------------------------------------------------------------- //
starling_ggml_ctx * starling_ggml_load(starling_ggml_model model,
                                       const char * gguf_path) {
    return api_call(nullptr, [&]() -> starling_ggml_ctx * {
        require_running();
        const lib::ModelDescriptor* d = lib::find_model(model);
        if (!d) {
            set_global_error("starling_ggml_load: unsupported model kind");
            return nullptr;
        }
        auto ctx = std::make_unique<starling_ggml_ctx>();
        const char* err = nullptr;
        void* handle = d->load_fn(gguf_path, &err);
        if (!handle) {
            set_global_error(err ? err : "starling_ggml_load: failed (no message)");
            return nullptr;
        }
        ctx->kind = model;
        ctx->model = handle;
        return ctx.release();
    });
}

void starling_ggml_free(starling_ggml_ctx * ctx) {
    return api_call(nullptr, [&]() -> void {
        if (!ctx) return;
        if (ctx->model) {
            if (const lib::ModelDescriptor* d = lib::find_model(ctx->kind))
                d->free_fn(ctx->model);
        }
        delete ctx;
    });
}

void starling_ggml_shutdown(void) {
    return api_call(nullptr, [&]() -> void {
        starling::ggml::shutdown_backend();
    });
}

const char * starling_ggml_last_error(starling_ggml_ctx * ctx) {
    return api_call(ctx, [&]() -> const char * {
        if (ctx) return ctx->last_error.c_str();
        return g_last_error;
    });
}

// --------------------------------------------------------------------------- //
// Inference
// --------------------------------------------------------------------------- //
char * starling_ggml_transcribe_pcm(starling_ggml_ctx * ctx,
                                    const float * samples, int64_t n,
                                    int sample_rate) {
    return api_call(ctx, [&]() -> char * {
        require_running();
        if (!ctx || !ctx->model) {
            set_global_error("starling_ggml_transcribe_pcm: null context");
            return nullptr;
        }
        const lib::ModelDescriptor* d = lib::find_model(ctx->kind);
        if (!d) {
            ctx->last_error = "starling_ggml_transcribe_pcm: unsupported model kind";
            set_global_error(ctx->last_error);
            return nullptr;
        }
        // Every engine consumes 16 kHz mono; resample upstream if needed. We only
        // reject (not resample) on a mismatch, and tolerate sample_rate=0 so a
        // 16k caller that doesn't know the rate still works.
        if (sample_rate != 0 && sample_rate != 16000) {
            char msg[128];
            std::snprintf(msg, sizeof(msg), d->rate_error_fmt, sample_rate);
            if (d->rate_error_in_ctx) ctx->last_error = msg;
            set_global_error(msg);
            return nullptr;
        }
        const char* err = nullptr;
        char* r = d->decode_fn(ctx->model, samples, n, &err);
        if (!r) {
            ctx->last_error = err ? err : d->decode_fallback;
            set_global_error(ctx->last_error);
        }
        return r;
    });
}

void starling_ggml_free_string(char * s) {
    if (s) std::free(s);
}

char * starling_ggml_normalize_text(starling_ggml_ctx * ctx,
                                    const char * transcript,
                                    const char * styling,
                                    const char * structure,
                                    const char * context) {
    return api_call(ctx, [&]() -> char * {
        require_running();
        if (!ctx || !ctx->model) {
            set_global_error("starling_ggml_normalize_text: null context");
            return nullptr;
        }
        const lib::ModelDescriptor* d = lib::find_model(ctx->kind);
        if (!d) {
            ctx->last_error = "starling_ggml_normalize_text: unsupported model kind";
            set_global_error(ctx->last_error);
            return nullptr;
        }
        if (!d->normalize_fn) {
            ctx->last_error = std::string("starling_ggml_normalize_text: model '")
                              + d->slug + "' has no text path";
            set_global_error(ctx->last_error);
            return nullptr;
        }
        const char* err = nullptr;
        char* r = d->normalize_fn(ctx->model, transcript, styling, structure, context, &err);
        if (!r) {
            ctx->last_error = err ? err : "S1 normalize failed";
            set_global_error(ctx->last_error);
        }
        return r;
    });
}

// --------------------------------------------------------------------------- //
// Internal mel-test passthrough (not in the public header; used by the Phase 1a
// validation script). Exposed so the Python binding can call it via ctypes.
// --------------------------------------------------------------------------- //
float * starling_ggml_parakeet_mel_pub(starling_ggml_ctx * ctx,
                                       const float * pcm, int64_t n,
                                       int * out_T) {
    return api_call(ctx, [&]() -> float * {
        require_running();
        if (!ctx || ctx->kind != STARLING_GGML_PARAKEET_TDT) {
            set_global_error("starling_ggml_parakeet_mel_pub: not a parakeet context");
            return nullptr;
        }
        const char* err = nullptr;
        float* r = starling_ggml_parakeet_mel(ctx->model, pcm, n, out_T, &err);
        if (!r) set_global_error(err ? err : "mel failed");
        return r;
    });
}

// Internal encoder-test passthrough (Phase 1b validation). Runs mel + encoder
// + joint.enc projection and returns the [640, T'] feat-major f32 buffer. The
// caller frees the buffer with starling_ggml_free.
float * starling_ggml_parakeet_encode_pub(starling_ggml_ctx * ctx,
                                          const float * pcm, int64_t n,
                                          int * out_T) {
    return api_call(ctx, [&]() -> float * {
        require_running();
        if (!ctx || ctx->kind != STARLING_GGML_PARAKEET_TDT) {
            set_global_error("starling_ggml_parakeet_encode_pub: not a parakeet context");
            return nullptr;
        }
        const char* err = nullptr;
        float* r = starling_ggml_parakeet_encode(ctx->model, pcm, n, out_T, &err);
        if (!r) set_global_error(err ? err : "encode failed");
        return r;
    });
}

// Internal decode passthrough (Phase 1c validation + the transcribe path). Runs
// the FULL pipeline (mel + encoder + decode + detokenize) and returns a
// malloc'd UTF-8 text string the caller frees with starling_ggml_free_string.
char * starling_ggml_parakeet_decode_pub(starling_ggml_ctx * ctx,
                                         const float * pcm, int64_t n) {
    return api_call(ctx, [&]() -> char * {
        require_running();
        if (!ctx || ctx->kind != STARLING_GGML_PARAKEET_TDT) {
            set_global_error("starling_ggml_parakeet_decode_pub: not a parakeet context");
            return nullptr;
        }
        const char* err = nullptr;
        char* r = starling_ggml_parakeet_decode(ctx->model, pcm, n, &err);
        if (!r) set_global_error(err ? err : "decode failed");
        return r;
    });
}

// Internal decode-ids passthrough (Phase 1c validation). Runs the FULL pipeline
// and returns the emitted id stream (INCLUDING blanks, matching golden
// parakeet_tdt_*_ids.pt) as a malloc'd int64 array the caller frees with
// starling_ggml_free. Writes the count to *out_n.
int64_t * starling_ggml_parakeet_decode_ids_pub(starling_ggml_ctx * ctx,
                                                const float * pcm, int64_t n,
                                                int64_t * out_n) {
    return api_call(ctx, [&]() -> int64_t * {
        require_running();
        if (!ctx || ctx->kind != STARLING_GGML_PARAKEET_TDT) {
            set_global_error("starling_ggml_parakeet_decode_ids_pub: not a parakeet context");
            return nullptr;
        }
        const char* err = nullptr;
        int64_t* r = starling_ggml_parakeet_decode_ids(ctx->model, pcm, n, out_n, &err);
        if (!r) set_global_error(err ? err : "decode_ids failed");
        return r;
    });
}

// Flush the activation-importance collector (STARLING_IMATRIX mode) to disk
// NOW instead of at process exit. The ctypes drivers call this before freeing
// the model so a teardown crash can never eat the collection.
void starling_ggml_imatrix_flush_pub(void) {
    return api_call(nullptr, [&]() -> void {
        starling::ggml::ImatrixCollector::instance().flush();
    });
}

} // extern "C"
