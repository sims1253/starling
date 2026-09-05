// capi_parakeet.cpp — the parakeet-tdt C API entry points (load/transcribe).
//
// Wires the parakeet model (loader + mel + encoder + decode) behind the shared
// C API declared in cpp/include/starling_ggml.h. Phase 1a implements load +
// the mel-test entry used to validate mel parity against the golden; the full
// transcribe path lands with the encoder (1b) and decode (1c).

#include "loader.hpp"
#include "mel.hpp"
#include "encoder.hpp"
#include "prediction.hpp"
#include "joint.hpp"
#include "tdt.hpp"
#include "tokenizer.hpp"
#include "config.hpp"

#include "runtime/graph.hpp"
#include "runtime/audio_io.hpp"

#include "starling_ggml.h"

#include <chrono>
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
    // Model-bound compute objects persist across transcription calls so their
    // lazy host weights and per-shape replay graphs reach steady state.
    std::unique_ptr<starling::ggml::parakeet::Encoder> encoder;
    std::unique_ptr<starling::ggml::parakeet::PredictionNet> prediction;
    std::unique_ptr<starling::ggml::parakeet::Joint> joint;
    std::string err;
};

// *err_out must remain readable after this TU returns (cpp/capi.cpp copies it
// into the global error after the call). Every message that lives in
// call-scoped storage goes through here: ctx->err outlives its unique_ptr on
// the load-failure path and e.what() dangles once its catch block exits. Same
// pattern as the g_load_error/report_load_error pairs in the other capi TUs.
thread_local std::string g_capi_error;
void report_error(const char** out, const char* msg) {
    g_capi_error = msg;
    if (out) *out = g_capi_error.c_str();
}

} // namespace

extern "C" {

// Internal entry called by capi.cpp's starling_ggml_load dispatcher. Loads the
// GGUF, reads config + mel constants, realizes weights to the device. Returns
// an opaque handle (caller wraps in starling_ggml_ctx) or nullptr on error.
void * starling_ggml_parakeet_load(const char * gguf_path, const char ** err_out) {
    // Exception-fenced like every sibling capi TU (capi.cpp's contract: a
    // C++ exception never crosses the boundary): the ctx/model construction,
    // the ~1.2 GB weight realize and the persistent compute objects can all
    // throw (bad_alloc, GpuMel/Encoder construction).
    try {
        auto ctx = std::make_unique<ParakeetCtx>();
        ctx->model = std::make_unique<starling::ggml::parakeet::ParakeetModel>();
        if (!ctx->model->load(gguf_path, ctx->err)) {
            report_error(err_out, ctx->err.c_str());
            return nullptr;
        }
        ctx->mel_const.read_from(ctx->model->loader, ctx->model->config);
        // Realize weights to the process-global backend (zero-copy on CPU /
        // upload on GPU). Also forces global_backend() creation so the
        // atexit handler is registered before any compute.
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
        ctx->encoder = std::make_unique<starling::ggml::parakeet::Encoder>(*ctx->model);
        ctx->prediction = std::make_unique<starling::ggml::parakeet::PredictionNet>(
            ctx->model->loader, ctx->model->config);
        ctx->joint = std::make_unique<starling::ggml::parakeet::Joint>(
            ctx->model->loader, ctx->model->config);
        if (err_out) *err_out = nullptr;
        return ctx.release();
    } catch (const std::exception& e) {
        report_error(err_out, e.what());
        return nullptr;
    } catch (...) {
        report_error(err_out, "unknown exception loading parakeet model");
        return nullptr;
    }
}

void starling_ggml_parakeet_free(void * handle) {
    try {
        delete static_cast<ParakeetCtx*>(handle);
    } catch (...) {
        // C ABI: never allow an exception to escape.
    }
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
        report_error(err_out, e.what());
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
        report_error(err_out, e.what());
        return nullptr;
    }
    // 2. encoder + joint.enc projection -> feat-major [640, T'].
    std::vector<float> enc_out;
    int Tp = 0;
    try {
        if (!c->encoder->encode(feats, (int)c->mel_const.n_mels, T_mel, enc_out, Tp)) {
            if (err_out) *err_out = "encoder graph failed";
            return nullptr;
        }
    } catch (const std::exception& e) {
        report_error(err_out, e.what());
        return nullptr;
    }
    if (out_T) *out_T = Tp;
    float* out = (float*)std::malloc(enc_out.size() * sizeof(float));
    if (!out) { if (err_out) *err_out = "malloc failed"; return nullptr; }
    std::memcpy(out, enc_out.data(), enc_out.size() * sizeof(float));
    return out;
}

// Internal helper: run the FULL pipeline (mel + encoder + decode) on `n` mono
// float32 PCM samples and return the emitted token id stream (INCLUDING blanks,
// matching the golden parakeet_tdt_*_ids.pt). Used by both _decode (text) and
// _decode_ids (id stream) entry points so they share one code path. Returns
// true on success; sets *err_out on failure.
//
// The encoder Phase 1b output is feat-major [H, T'] (out[h*T' + t]); tdt_greedy
// wants row-major [T', H] (enc_proj[t*H + h]), so we transpose here.
static bool parakeet_full_decode(ParakeetCtx* c,
                                 const float* pcm, int64_t n,
                                 std::vector<int32_t>& ids,
                                 const char** err_out) {
    using Clock = std::chrono::steady_clock;
    const char* timing_env = std::getenv("STARLING_PARAKEET_TIMING");
    const bool timing = timing_env && std::strcmp(timing_env, "1") == 0;
    const auto t0 = Clock::now();

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
        report_error(err_out, e.what());
        return false;
    }

    // 2. encoder + joint.enc projection -> feat-major [640, T'].
    const auto t_mel = Clock::now();
    std::vector<float> enc_feat;
    int Tp = 0;
    try {
        if (!c->encoder->encode(feats, (int)c->mel_const.n_mels, T_mel, enc_feat, Tp)) {
            if (err_out) *err_out = "encoder graph failed";
            return false;
        }
    } catch (const std::exception& e) {
        report_error(err_out, e.what());
        return false;
    }

    const auto t_encoder = Clock::now();
    const int H = c->encoder->proj_dim();  // 640
    // 3. The encoder Phase 1b output is row-major [T', H] (out[t*H + h]) — frame
    //    t's projected vector is contiguous, exactly what tdt_greedy expects.
    //    (encoder.hpp's "feat-major" comment is misleading; the buffer is
    //    emitted row-major, verified byte-exact vs golden _enc.pt.)
    std::vector<float>& enc_proj = enc_feat;
    (void)H;

    // 4. serial TDT greedy decode -> id stream (incl. blanks).
    try {
        ids = starling::ggml::parakeet::tdt_greedy(
            *c->prediction, *c->joint, enc_proj, Tp, H,
            c->model->config.tdt_durations,
            (int)c->model->config.blank_id,
            (int)c->model->config.max_symbols);
    } catch (const std::exception& e) {
        report_error(err_out, e.what());
        return false;
    }
    if (timing) {
        const auto t_decode = Clock::now();
        auto ms = [](Clock::duration d) {
            return std::chrono::duration<double, std::milli>(d).count();
        };
        std::fprintf(stderr,
            "[PARAKEET_TIMING] mel=%.3f ms encoder=%.3f ms decode=%.3f ms total=%.3f ms\n",
            ms(t_mel - t0), ms(t_encoder - t_mel),
            ms(t_decode - t_encoder), ms(t_decode - t0));
    }
    return true;
}

// Decode entry: run the full pipeline and return the detokenized UTF-8 text as
// a malloc'd string the caller frees with starling_ggml_free_string. This IS
// the transcribe path (Phase 1d wiring calls this).
char * starling_ggml_parakeet_decode(void * handle, const float * pcm, int64_t n,
                                     const char ** err_out) {
    auto* c = static_cast<ParakeetCtx*>(handle);
    if (!c) { if (err_out) *err_out = "null parakeet handle"; return nullptr; }
    std::vector<int32_t> ids;
    if (!parakeet_full_decode(c, pcm, n, ids, err_out)) return nullptr;
    std::string text = starling::ggml::parakeet::detokenize(
        c->model->config.tokenizer_pieces, ids);
    char* out = (char*)std::malloc(text.size() + 1);
    if (!out) { if (err_out) *err_out = "malloc failed"; return nullptr; }
    std::memcpy(out, text.data(), text.size());
    out[text.size()] = '\0';
    return out;
}

// Decode-ids entry: run the full pipeline and return the emitted id stream
// (INCLUDING blanks, matching golden parakeet_tdt_*_ids.pt) as a malloc'd int64
// array the caller frees with starling_ggml_free. Writes the count to *out_n.
//
// The stream is prefixed with the blank id (decoder_start_token_id) to match
// the format of HF `model.generate`'s `sequences` output (which the goldens
// were saved from): the first element is the prepended start token, followed by
// the greedy-decode emissions (blanks included). The detokenize path naturally
// drops blanks (id 8192 is out of the [0, vocab_size) piece range), so this
// leading blank contributes nothing to the text.
int64_t * starling_ggml_parakeet_decode_ids(void * handle, const float * pcm, int64_t n,
                                            int64_t * out_n, const char ** err_out) {
    auto* c = static_cast<ParakeetCtx*>(handle);
    if (!c) { if (err_out) *err_out = "null parakeet handle"; return nullptr; }
    std::vector<int32_t> ids;
    if (!parakeet_full_decode(c, pcm, n, ids, err_out)) return nullptr;
    // Prepend the decoder_start_token_id (= blank_id) to match the golden
    // stream format (HF model.generate's sequences[0] includes it).
    const int64_t blank = (int64_t)c->model->config.blank_id;
    if (out_n) *out_n = (int64_t)ids.size() + 1;
    int64_t* out = (int64_t*)std::malloc(((size_t)ids.size() + 1) * sizeof(int64_t));
    if (!out) { if (err_out) *err_out = "malloc failed"; return nullptr; }
    out[0] = blank;
    for (size_t i = 0; i < ids.size(); ++i) out[i + 1] = (int64_t)ids[i];
    return out;
}

} // extern "C"
