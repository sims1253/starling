// starling_ggml.h — the flat C API surface for Starling's ggml engine.
//
// Driven from Python via ctypes (src/starling/_ggml/_native.py). One shared
// library (libstarling_ggml) serves both the parakeet-tdt and moss models via
// model-tagged entry points. Opaque `starling_ggml_ctx` holds a loaded model.
//
// ABI version: bumped whenever the signature/layout of these entry points
// changes. _native.py checks starling_ggml_abi_version() against the expected
// value and refuses to load on mismatch.
//
// All entry points are exception-fenced: a C++ exception never crosses the
// boundary. On error, the function returns a null/empty result and writes a
// UTF-8 message retrievable via starling_ggml_last_error().

#ifndef STARLING_GGML_H
#define STARLING_GGML_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Bumped on any breaking change to this API. The Python binding refuses to
// load on mismatch. History:
//   1 — initial (Phase 0): abi + version + shutdown only.
//   2 — added STARLING_GGML_HIGGS (bosonai/higgs-audio-v3-stt).
//   3 — added STARLING_GGML_HOJO (HojoAI/Hojo-ASR-V1).
//   4 — added STARLING_GGML_GRANITE (ibm-granite/granite-speech-4.1-2b).
//   5 — added STARLING_GGML_QWEN3 (Qwen/Qwen3-ASR-1.7B-hf).
#define STARLING_GGML_ABI_VERSION 5

// ABI / build introspection --------------------------------------------------

// Returns STARLING_GGML_ABI_VERSION.
int starling_ggml_abi_version(void);

// Returns a static UTF-8 string naming the active ggml backend at build time
// (e.g. "cuda", "metal", "vulkan", "cpu"). For diagnostics.
const char * starling_ggml_backend_name(void);

// Opaque model context. One per loaded model.
typedef struct starling_ggml_ctx starling_ggml_ctx;

// The model kind selects which implementation backs the context.
typedef enum {
    STARLING_GGML_PARAKEET_TDT = 1,  // nvidia/parakeet-tdt-0.6b-v3
    STARLING_GGML_MOSS         = 2,  // MOSS-Transcribe-preview-2B
    STARLING_GGML_ARK          = 3,  // AutoArk-AI/ARK-ASR-3B
    STARLING_GGML_HIGGS        = 4,  // bosonai/higgs-audio-v3-stt
    STARLING_GGML_HOJO         = 5,  // HojoAI/Hojo-ASR-V1
    STARLING_GGML_GRANITE      = 6,  // ibm-granite/granite-speech-4.1-2b
    STARLING_GGML_QWEN3        = 7,  // Qwen/Qwen3-ASR-1.7B-hf
} starling_ggml_model;

// Lifecycle ------------------------------------------------------------------

// Load a model from a GGUF file. Returns a new context, or NULL on error
// (call starling_ggml_last_error(NULL_or_ctx) for the message). `model` selects
// the implementation; the GGUF must match. The caller owns the context and must
// free it with starling_ggml_free.
starling_ggml_ctx * starling_ggml_load(starling_ggml_model model,
                                       const char * gguf_path);

// Free a context (idempotent; safe on NULL). Releases the model + its device
// buffers. The global ggml backend itself is freed by starling_ggml_shutdown.
void starling_ggml_free(starling_ggml_ctx * ctx);

// Tear down the process-global ggml backend (frees device buffers + captured
// CUDA graphs) so they're released while the CUDA driver is still alive, rather
// than by static destruction at process exit (which runs after the driver's
// own atexit handler and aborts with "driver shutting down"). Idempotent.
//
// OPTIONAL: starling_ggml also registers an internal std::atexit handler on
// first backend creation that performs the same teardown automatically at
// process exit, so a caller that never calls this still exits cleanly. Safe to
// call multiple times, and safe alongside the atexit handler.
void starling_ggml_shutdown(void);

// Retrieve the last error message for a context (or the global last-error if
// ctx is NULL). Returns "" if no error. The pointer is owned by the library
// and valid until the next call into the library on the same context.
const char * starling_ggml_last_error(starling_ggml_ctx * ctx);

// Inference ------------------------------------------------------------------

// Transcribe mono float32 PCM. Returns a malloc'd UTF-8 string the caller must
// free with starling_ggml_free_string, or NULL on error.
//
// `samples` is `n` interleaved float32 in [-1, 1]; `sample_rate` must be 16000
// (resample upstream if needed — see audio_io). `ctx` selects the model.
char * starling_ggml_transcribe_pcm(starling_ggml_ctx * ctx,
                                    const float * samples, int64_t n,
                                    int sample_rate);

// Free a string returned by starling_ggml_transcribe_pcm (no-op on NULL).
void starling_ggml_free_string(char * s);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // STARLING_GGML_H
