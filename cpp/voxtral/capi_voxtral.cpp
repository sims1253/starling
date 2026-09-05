// capi_voxtral.cpp — Voxtral-Mini-4B-Realtime C API entry points (Phase 1).
//
// Phase 1 loads and validates the GGUF (metadata guards + tensor presence via
// VoxtralModel::load) and reads the tokenizer table. The encoder/decoder graph
// is Phase 2, so decode returns the Phase-2 error instead of transcribing.
#include "loader.hpp"
#include "prompt.hpp"
#include "tokenizer.hpp"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <string>

namespace {

thread_local std::string g_load_error;

struct VoxtralCtx {
    std::unique_ptr<starling::ggml::voxtral::VoxtralModel> model;
    starling::ggml::voxtral::Tokenizer tokenizer;
    std::string err;
};

void report(const char** out, const std::string& message) {
    if (out) *out = message.c_str();
}

void report_load_error(const char** out, const std::string& message) {
    g_load_error = message;
    if (out) *out = g_load_error.c_str();
}

} // namespace

extern "C" {

void* starling_ggml_voxtral_load(const char* gguf_path, const char** err_out) {
    try {
        if (!gguf_path || !*gguf_path) {
            if (err_out) *err_out = "null or empty VOXTRAL GGUF path";
            return nullptr;
        }
        auto ctx = std::make_unique<VoxtralCtx>();
        ctx->model = std::make_unique<starling::ggml::voxtral::VoxtralModel>();
        if (!ctx->model->load(gguf_path, ctx->err)) {
            report_load_error(err_out, ctx->err);
            return nullptr;
        }
        if (!ctx->tokenizer.load(ctx->model->loader, ctx->model->config, ctx->err)) {
            report_load_error(err_out, ctx->err);
            return nullptr;
        }
        if (err_out) *err_out = nullptr;
        return ctx.release();
    } catch (const std::exception& e) {
        report_load_error(err_out, e.what());
    } catch (...) {
        report_load_error(err_out, "unknown exception loading VOXTRAL model");
    }
    return nullptr;
}

void starling_ggml_voxtral_free(void* handle) {
    try {
        delete static_cast<VoxtralCtx*>(handle);
    } catch (...) {
        // C ABI: never allow an exception to escape.
    }
}

char* starling_ggml_voxtral_decode(void* handle, const float* pcm, int64_t n,
                                   const char** err_out) {
    auto* c = static_cast<VoxtralCtx*>(handle);
    if (!c) { if (err_out) *err_out = "null VOXTRAL handle"; return nullptr; }
    if (n < 0 || (n > 0 && !pcm)) {
        if (err_out) *err_out = "invalid VOXTRAL PCM buffer";
        return nullptr;
    }
    try {
        // Phase 2 owns the encoder/decoder graph (causal left-pad convs,
        // rotate-half RoPE tables, sliding-window band-causal masks, AdaRMSNorm,
        // additive injection, per-T_enc encoder ReplayGraph). Copy into the
        // context's owned error string: a temporary's c_str() would dangle
        // after return, and cpp/capi.cpp reads *err_out after we return.
        c->err = "VOXTRAL engine not implemented (Phase 2)";
        report(err_out, c->err);
    } catch (const std::exception& e) {
        c->err = e.what();
        report(err_out, c->err);
    } catch (...) {
        report(err_out, "unknown exception transcribing VOXTRAL audio");
    }
    return nullptr;
}

} // extern "C"
