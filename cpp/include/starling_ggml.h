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
#include <stdbool.h>

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
//   6 — added STARLING_GGML_S1 (superwhisper/s1-mini) + the text-in entry
//       point starling_ggml_normalize_text, and STARLING_GGML_AUDEX
//       (nvidia/Nemotron-Labs-Audex-2B).
//   7 — added STARLING_GGML_ARK06 (Audio8/ARK-ASR-0.6B; served by the ARK
//       engine — same architecture, dims come from GGUF metadata).
//   8 — added STARLING_GGML_VOXTRAL (mistralai/Voxtral-Mini-4B-Realtime-2602;
//       Phase 1: GGUF load + metadata/tokenizer validation only; decode
//       returns the Phase-2 error until the encoder/decoder graph lands).
//   9 — added STARLING_GGML_QWEN3_06 (Qwen/Qwen3-ASR-0.6B-hf; served by the
//       QWEN3 engine — same architecture, dims come from GGUF metadata).
// Additions that leave every existing entry point unchanged keep the version
// (starling_ggml_transcribe_pcm_words, issue #357): a caller loading the
// library at run time probes for such a symbol (dlsym) instead of checking
// the version.
#define STARLING_GGML_ABI_VERSION 9

// ABI / build introspection --------------------------------------------------

// Returns STARLING_GGML_ABI_VERSION.
int starling_ggml_abi_version(void);

// Returns a static UTF-8 string naming the ggml device used at RUNTIME
// (e.g. "CPU", "Vulkan0", "CUDA0" — as selected by STARLING_GGML_DEVICE or
// auto-pick) once a model load has created the global backend; before the
// first load it names the compile-time backend family ("cuda", "metal",
// "vulkan", "cpu"). The pointer stays valid across calls. For diagnostics.
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
    STARLING_GGML_S1           = 8,  // superwhisper/s1-mini (text normalizer)
    STARLING_GGML_AUDEX        = 9,  // nvidia/Nemotron-Labs-Audex-2B
    STARLING_GGML_ARK06        = 10, // Audio8/ARK-ASR-0.6B (ARK engine, 0.6B GGUF)
    STARLING_GGML_VOXTRAL      = 11, // mistralai/Voxtral-Mini-4B-Realtime-2602
    STARLING_GGML_QWEN3_06     = 12, // Qwen/Qwen3-ASR-0.6B-hf (QWEN3 engine, 0.6B GGUF)
} starling_ggml_model;

// Lifecycle ------------------------------------------------------------------

// Calls through the common starling_ggml_ctx API that load, free or run
// models are serialized inside the library. Keep each context alive until
// its pending calls finish. Model-specific raw handles have a separate
// caller-serialization contract below.
//
// Load a model from a GGUF file. Returns a new context, or NULL on error
// (call starling_ggml_last_error(NULL_or_ctx) for the message). `model` selects
// the implementation; the GGUF must match. The caller owns the context and must
// free it with starling_ggml_free.
starling_ggml_ctx * starling_ggml_load(starling_ggml_model model,
                                       const char * gguf_path);

// Free a context (safe on NULL; a non-NULL pointer must be freed once). Releases the model + its device
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
// call multiple times, and safe alongside the atexit handler. Shutdown is
// terminal: subsequent model loads and inference return an error.
void starling_ggml_shutdown(void);

// Cooperative stop (#325 crash circumvention): request_stop() asks the
// process to wind down so a SIGTERM'd process can exit through normal
// destructors — destroying the Vulkan device instead of leaving ~1.6 GB of
// live GPU state for the driver to reap asynchronously. Correlate: the Pixel
// wedges (#325, RESEARCH_LOG P2-7) follow processes that die with a live
// VkDevice (SIGKILL/SIGTERM run no destructors).
// Granularity: the fast MOSS engine checks the flag between decode rounds
// and returns the valid prefix decoded so far (logged to stderr; the result
// is otherwise indistinguishable from a normal one). Other engines finish
// the current transcribe call; the caller checks stop_requested() between
// calls. Async-signal-safe (lock-free atomic). stop_requested() is the
// polling side; the flag stays set until clear_stop() (deliberate reuse of
// the process).
void starling_ggml_request_stop(void);
int  starling_ggml_stop_requested(void);
void starling_ggml_clear_stop(void);

// Flush the STARLING_IMATRIX activation-importance collector to its output
// file immediately (idempotent; the collector also flushes at process exit).
// Call this before teardown in collection runs so the data is on disk no
// matter how the process ends.
void starling_ggml_imatrix_flush_pub(void);

// Retrieve the last error message for a context (or this thread's last error
// if ctx is NULL). Returns "" if no error. The pointer is owned by the library
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

// One word of a transcript: bytes [text_begin, text_end) of the returned
// text, heard from start_s to end_s seconds into the transcribed audio.
typedef struct starling_ggml_word {
    int32_t text_begin;
    int32_t text_end;
    float start_s;
    float end_s;
} starling_ggml_word;

// starling_ggml_transcribe_pcm with word timestamps. On success also writes
// a malloc'd array of the text's whitespace-separated words, in order, to
// *words (free it with starling_ggml_free_words) and their count to *n_words.
// A model without word timestamps (every engine but Parakeet) returns the
// plain transcript with *words = NULL and *n_words = -1. On error returns
// NULL like starling_ggml_transcribe_pcm and leaves *words NULL.
char * starling_ggml_transcribe_pcm_words(starling_ggml_ctx * ctx,
                                          const float * samples, int64_t n,
                                          int sample_rate,
                                          starling_ggml_word ** words,
                                          int64_t * n_words);

// Free a word array returned by starling_ggml_transcribe_pcm_words (no-op on
// NULL).
void starling_ggml_free_words(starling_ggml_word * words);

// Raw Granite research handles (from starling_ggml_granite_load) do not use
// the common API's whole-call lock. Callers must serialize raw load, decode,
// draft and free operations across all raw handles, and must not overlap them
// with common-API load, inference, free or starling_ggml_shutdown. Free all
// raw handles before global shutdown. These calls share backend state;
// graph-level locking alone does not protect a raw handle's KV cache or error.
// An error string returned through err_out belongs to the handle (or is a
// static string); do not free it, and read it before the next call or free.
void * starling_ggml_granite_load(const char * gguf_path, const char ** err_out);
void starling_ggml_granite_free(void * handle);
char * starling_ggml_granite_decode(void * handle, const float * pcm, int64_t n,
                                   const char ** err_out);
//
// Research-only Granite CTC draft probe. `handle` is returned by the
// model-specific starling_ggml_granite_load symbol. A successful call writes
// token IDs and count. On a too-small buffer it writes the required capacity
// to count, leaves token_ids untouched, and returns false: count does not
// describe valid output on failure. Invalid arguments clear count when it is
// non-null. A zero-capacity sizing call runs the full encoder and head.
bool starling_ggml_granite_ctc_draft(void * handle, const float * pcm, int64_t n,
                                    int32_t * token_ids, int32_t capacity,
                                    int32_t * count, const char ** err_out);

// Research-only opt-in Granite transcription with native CTC drafts and the
// greedy batched verifier. `handle` is returned by the model-specific
// starling_ggml_granite_load symbol; max_k must be 1..16. Uses the same
// padded chunk/budget policy as ordinary Granite decode. The returned string
// is malloc'd and freed with starling_ggml_free_string. Default transcription
// continues through the ordinary greedy entry point. On failure returns
// NULL and sets *err_out when err_out is non-NULL. The raw-handle
// serialization requirement above applies to this call.
char * starling_ggml_granite_decode_ctc(void * handle, const float * pcm, int64_t n,
                                        int32_t max_k, const char ** err_out);

// Free a malloc'd string returned by a Starling transcription or
// normalization entry point (no-op on NULL).
void starling_ggml_free_string(char * s);

// Normalize one raw ASR transcript with a text model (s1). Returns a
// malloc'd UTF-8 string the caller frees with starling_ggml_free_string, or
// NULL on error. The control arguments accept NULL for their defaults
// (styling="semi-formal", structure="prose", context="general"); unknown
// values are rejected — the model was only trained on the shipped sets.
char * starling_ggml_normalize_text(starling_ggml_ctx * ctx,
                                    const char * transcript,
                                    const char * styling,
                                    const char * structure,
                                    const char * context);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // STARLING_GGML_H
