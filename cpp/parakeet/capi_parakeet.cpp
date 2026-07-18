// capi_parakeet.cpp — the parakeet-tdt C API entry points (load/transcribe).
//
// Wires the parakeet model (loader + mel + encoder + decode) behind the shared
// C API declared in cpp/include/starling_ggml.h. Phase 1a implements load +
// the mel-test entry used to validate mel parity against the golden; the full
// transcribe path lands with the encoder (1b) and decode (1c).

#include "loader.hpp"
#include "mel.hpp"
#include "encoder.hpp"
#include "config.hpp"

#include "runtime/graph.hpp"
#include "runtime/audio_io.hpp"

#include "starling_ggml.h"

#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <memory>
#include <new>
#include <string>
#include <vector>

namespace {

// The parakeet context: a loaded model + its mel frontend. One per
// starling_ggml_load(STARLING_GGML_PARAKEET_TDT). Held in the opaque
// starling_ggml_ctx via a tagged variant (see capi.cpp).
struct ParakeetCtx {
    std::unique_ptr<starling::ggml::parakeet::ParakeetModel> model;
    starling::ggml::parakeet::MelConstants mel_const;
    // Persistent GPU mel (kept warm across utterances); null on CPU.
    std::unique_ptr<starling::ggml::parakeet::GpuMel> gmel;
    std::string err;
};

} // namespace

extern "C" {

// Internal entry called by capi.cpp's starling_ggml_load dispatcher. Loads the
// GGUF, reads config + mel constants, realizes weights to the device. Returns
// an opaque handle (caller wraps in starling_ggml_ctx) or nullptr on error.
void * starling_ggml_parakeet_load(const char * gguf_path, const char ** err_out) {
    auto ctx = std::make_unique<ParakeetCtx>();
    ctx->model = std::make_unique<starling::ggml::parakeet::ParakeetModel>();
    if (!ctx->model->load(gguf_path, ctx->err)) {
        if (err_out) *err_out = ctx->err.c_str();
        return nullptr;
    }
    ctx->mel_const.read_from(ctx->model->loader, ctx->model->config);
    // Realize weights to the process-global backend (zero-copy on CPU / upload
    // on GPU). Also forces global_backend() creation so the atexit handler is
    // registered before any compute.
    ctx->model->loader.realize_weights(starling::ggml::global_backend());
    if (std::getenv("STARLING_MEL_DEBUG"))
        std::fprintf(stderr, "[MEL_DEBUG] load: STARLING_GGML_DEVICE=%s dev=%s is_gpu=%d\n",
            std::getenv("STARLING_GGML_DEVICE") ? std::getenv("STARLING_GGML_DEVICE") : "(auto)",
            starling::ggml::global_backend().device_name(),
            starling::ggml::global_backend().is_gpu() ? 1 : 0);
    if (starling::ggml::global_backend().is_gpu()) {
        ctx->gmel = std::make_unique<starling::ggml::parakeet::GpuMel>(
            starling::ggml::global_backend(), ctx->mel_const);
    }
    if (err_out) *err_out = nullptr;
    return ctx.release();
}

void starling_ggml_parakeet_free(void * handle) {
    delete static_cast<ParakeetCtx*>(handle);
}

// Mel test entry: run the mel frontend on `n` mono float32 PCM samples and write
// the feat-major [n_mels, T] float32 result into a malloc'd buffer the caller
// frees with starling_ggml_free. Writes the frame count to *out_T. Returns the
// buffer or nullptr on error.
float * starling_ggml_parakeet_mel(void * handle, const float * pcm, int64_t n,
                                   int * out_T, const char ** err_out) {
    auto* c = static_cast<ParakeetCtx*>(handle);
    if (!c) { if (err_out) *err_out = "null parakeet handle"; return nullptr; }
    std::vector<float> feats;
    int T = 0;
    try {
        if (c->gmel) c->gmel->compute(pcm, (size_t)n, feats, T);
        else {
            starling::ggml::parakeet::MelFrontend cpu(c->mel_const);
            cpu.compute(pcm, (size_t)n, feats, T);
        }
    } catch (const std::exception& e) {
        if (err_out) *err_out = e.what();
        return nullptr;
    }
    if (out_T) *out_T = T;
    float* out = (float*)std::malloc(feats.size() * sizeof(float));
    if (!out) { if (err_out) *err_out = "malloc failed"; return nullptr; }
    std::memcpy(out, feats.data(), feats.size() * sizeof(float));
    return out;
}

// Encoder entry: run the mel frontend + Conformer encoder + joint.enc projection
// on `n` mono float32 PCM samples and return the projected encoder output as a
// malloc'd [640, T'] feat-major float32 buffer the caller frees with
// starling_ggml_free. Writes T' (the encoder length) to *out_T. Returns the
// buffer or nullptr on error.
//
// Validation entry: forces the CPU path (the byte-identical reference). The
// output layout matches the golden parakeet_tdt_*_enc.pt (T_enc rows x 640 cols)
// reinterpreted feat-major (out[c*T' + t]).
float * starling_ggml_parakeet_encode(void * handle, const float * pcm, int64_t n,
                                      int * out_T, const char ** err_out) {
    auto* c = static_cast<ParakeetCtx*>(handle);
    if (!c) { if (err_out) *err_out = "null parakeet handle"; return nullptr; }
    // 1. mel frontend -> feat-major [n_mels, T].
    std::vector<float> feats;
    int T_mel = 0;
    try {
        if (c->gmel) c->gmel->compute(pcm, (size_t)n, feats, T_mel);
        else {
            starling::ggml::parakeet::MelFrontend cpu(c->mel_const);
            cpu.compute(pcm, (size_t)n, feats, T_mel);
        }
    } catch (const std::exception& e) {
        if (err_out) *err_out = e.what();
        return nullptr;
    }
    // 2. encoder + joint.enc projection -> feat-major [640, T'].
    starling::ggml::parakeet::Encoder enc(*c->model);
    std::vector<float> enc_out;
    int Tp = 0;
    try {
        if (!enc.encode(feats, (int)c->mel_const.n_mels, T_mel, enc_out, Tp)) {
            if (err_out) *err_out = "encoder graph failed";
            return nullptr;
        }
    } catch (const std::exception& e) {
        if (err_out) *err_out = e.what();
        return nullptr;
    }
    if (out_T) *out_T = Tp;
    float* out = (float*)std::malloc(enc_out.size() * sizeof(float));
    if (!out) { if (err_out) *err_out = "malloc failed"; return nullptr; }
    std::memcpy(out, enc_out.data(), enc_out.size() * sizeof(float));
    return out;
}

} // extern "C"
