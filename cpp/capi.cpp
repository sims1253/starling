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
#include <cstdio>
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
float * starling_ggml_parakeet_encode(void * handle, const float * pcm, int64_t n,
                                      int * out_T, const char ** err_out);
char * starling_ggml_parakeet_decode(void * handle, const float * pcm, int64_t n,
                                     const char ** err_out);
int64_t * starling_ggml_parakeet_decode_ids(void * handle, const float * pcm, int64_t n,
                                            int64_t * out_n, const char ** err_out);
void * starling_ggml_moss_load(const char * gguf_path, const char ** err_out);
void   starling_ggml_moss_free(void * handle);
char * starling_ggml_moss_decode(void * handle, const float * pcm, int64_t n,
                                 const char ** err_out);
void * starling_ggml_ark_load(const char * gguf_path, const char ** err_out);
void   starling_ggml_ark_free(void * handle);
char * starling_ggml_ark_decode(void * handle, const float * pcm, int64_t n,
                                const char ** err_out);
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
    } else if (model == STARLING_GGML_MOSS) {
        handle = starling_ggml_moss_load(gguf_path, &err);
    } else if (model == STARLING_GGML_ARK) {
        handle = starling_ggml_ark_load(gguf_path, &err);
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
        else if (ctx->kind == STARLING_GGML_MOSS)
            starling_ggml_moss_free(ctx->model);
        else if (ctx->kind == STARLING_GGML_ARK)
            starling_ggml_ark_free(ctx->model);
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
        // parakeet-tdt is a 16 kHz model; resample upstream if needed. We only
        // warn (not fail) on a mismatch so a 16k caller passing sample_rate=0
        // still works.
        if (sample_rate != 0 && sample_rate != 16000) {
            char msg[128];
            std::snprintf(msg, sizeof(msg),
                "starling_ggml_transcribe_pcm: parakeet expects 16 kHz, got %d", sample_rate);
            set_global_error(msg);
            return nullptr;
        }
        const char* err = nullptr;
        char* r = starling_ggml_parakeet_decode(ctx->model, samples, n, &err);
        if (!r) { ctx->last_error = err ? err : "transcribe failed"; set_global_error(ctx->last_error); }
        return r;
    }
    if (ctx->kind == STARLING_GGML_MOSS) {
        if (sample_rate != 0 && sample_rate != 16000) {
            ctx->last_error = "starling_ggml_transcribe_pcm: MOSS expects 16 kHz";
            set_global_error(ctx->last_error);
            return nullptr;
        }
        const char* err = nullptr;
        char* r = starling_ggml_moss_decode(ctx->model, samples, n, &err);
        if (!r) { ctx->last_error = err ? err : "MOSS transcribe failed"; set_global_error(ctx->last_error); }
        return r;
    }
    if (ctx->kind == STARLING_GGML_ARK) {
        if (sample_rate != 0 && sample_rate != 16000) {
            ctx->last_error = "starling_ggml_transcribe_pcm: ARK expects 16 kHz";
            set_global_error(ctx->last_error);
            return nullptr;
        }
        const char* err = nullptr;
        char* r = starling_ggml_ark_decode(ctx->model, samples, n, &err);
        if (!r) { ctx->last_error = err ? err : "ARK transcribe failed"; set_global_error(ctx->last_error); }
        return r;
    }
    ctx->last_error = "starling_ggml_transcribe_pcm: unsupported model kind";
    set_global_error(ctx->last_error);
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

// Internal encoder-test passthrough (Phase 1b validation). Runs mel + encoder
// + joint.enc projection and returns the [640, T'] feat-major f32 buffer. The
// caller frees the buffer with starling_ggml_free.
float * starling_ggml_parakeet_encode_pub(starling_ggml_ctx * ctx,
                                          const float * pcm, int64_t n,
                                          int * out_T) {
    if (!ctx || ctx->kind != STARLING_GGML_PARAKEET_TDT) {
        set_global_error("starling_ggml_parakeet_encode_pub: not a parakeet context");
        return nullptr;
    }
    const char* err = nullptr;
    float* r = starling_ggml_parakeet_encode(ctx->model, pcm, n, out_T, &err);
    if (!r) set_global_error(err ? err : "encode failed");
    return r;
}

// Internal decode passthrough (Phase 1c validation + the transcribe path). Runs
// the FULL pipeline (mel + encoder + decode + detokenize) and returns a
// malloc'd UTF-8 text string the caller frees with starling_ggml_free_string.
char * starling_ggml_parakeet_decode_pub(starling_ggml_ctx * ctx,
                                         const float * pcm, int64_t n) {
    if (!ctx || ctx->kind != STARLING_GGML_PARAKEET_TDT) {
        set_global_error("starling_ggml_parakeet_decode_pub: not a parakeet context");
        return nullptr;
    }
    const char* err = nullptr;
    char* r = starling_ggml_parakeet_decode(ctx->model, pcm, n, &err);
    if (!r) set_global_error(err ? err : "decode failed");
    return r;
}

// Internal decode-ids passthrough (Phase 1c validation). Runs the FULL pipeline
// and returns the emitted id stream (INCLUDING blanks, matching golden
// parakeet_tdt_*_ids.pt) as a malloc'd int64 array the caller frees with
// starling_ggml_free. Writes the count to *out_n.
int64_t * starling_ggml_parakeet_decode_ids_pub(starling_ggml_ctx * ctx,
                                                const float * pcm, int64_t n,
                                                int64_t * out_n) {
    if (!ctx || ctx->kind != STARLING_GGML_PARAKEET_TDT) {
        set_global_error("starling_ggml_parakeet_decode_ids_pub: not a parakeet context");
        return nullptr;
    }
    const char* err = nullptr;
    int64_t* r = starling_ggml_parakeet_decode_ids(ctx->model, pcm, n, out_n, &err);
    if (!r) set_global_error(err ? err : "decode_ids failed");
    return r;
}

} // extern "C"
