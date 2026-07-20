// capi_moss.cpp — MOSS-Transcribe C API entry points behind the shared shell.
#include "loader.hpp"
#include "mel.hpp"
#include "audio_encoder.hpp"
#include "adapter.hpp"
#include "prompt.hpp"
#include "llm.hpp"
#include "tokenizer.hpp"
#include "runtime/graph.hpp"

#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <vector>

namespace {

thread_local std::string g_load_error;

struct MossCtx {
    std::unique_ptr<starling::ggml::moss::MossModel> model;
    starling::ggml::moss::Tokenizer tokenizer;
    std::string err;
};

void report(const char ** out, const std::string & message) {
    if (out) *out = message.c_str();
}

void report_load_error(const char ** out, const std::string & message) {
    g_load_error = message;
    if (out) *out = g_load_error.c_str();
}

} // namespace

extern "C" {

void * starling_ggml_moss_load(const char * gguf_path, const char ** err_out) {
    try {
        if (!gguf_path || !*gguf_path) {
            if (err_out) *err_out = "null or empty MOSS GGUF path";
            return nullptr;
        }
        auto ctx = std::make_unique<MossCtx>();
        ctx->model = std::make_unique<starling::ggml::moss::MossModel>();
        if (!ctx->model->load(gguf_path, ctx->err)) {
            report_load_error(err_out, ctx->err);
            return nullptr;
        }
        if (!ctx->tokenizer.load(ctx->model->loader, ctx->model->config, ctx->err)) {
            report_load_error(err_out, ctx->err);
            return nullptr;
        }
        // Persist model weights (and force backend creation/its orderly atexit
        // shutdown registration) across all transcription calls.
        ctx->model->loader.realize_weights(starling::ggml::global_backend());
        // The current one-shot LLM state owns no replay cache, but register a
        // shutdown hook with the shared cache-clearer protocol so future MOSS
        // decode caches are released before CUDA backend teardown.
        starling::ggml::register_decode_cache_clearer([]() {});
        if (err_out) *err_out = nullptr;
        return ctx.release();
    } catch (const std::exception & e) {
        report_load_error(err_out, e.what());
    } catch (...) {
        report_load_error(err_out, "unknown exception loading MOSS model");
    }
    return nullptr;
}

void starling_ggml_moss_free(void * handle) {
    try {
        delete static_cast<MossCtx *>(handle);
    } catch (...) {
        // C ABI: never allow an exception to escape.
    }
}

char * starling_ggml_moss_decode(void * handle, const float * pcm, int64_t n,
                                 const char ** err_out) {
    auto * c = static_cast<MossCtx *>(handle);
    if (!c) { if (err_out) *err_out = "null MOSS handle"; return nullptr; }
    if (n < 0 || (n > 0 && !pcm)) {
        if (err_out) *err_out = "invalid MOSS PCM buffer";
        return nullptr;
    }
    try {
        using namespace starling::ggml::moss;
        const bool timing = std::getenv("STARLING_MOSS_TIMING") != nullptr;
        auto now = [&]() { return std::chrono::steady_clock::now(); };
        auto ms = [&](auto t0, auto t1) {
            return std::chrono::duration<double, std::milli>(t1 - t0).count(); };

        auto a0 = now();
        MelFeatures mel;
        if (!compute_log_mel(c->model->config, c->model->loader, pcm,
                             static_cast<size_t>(n), mel, c->err)) {
            report(err_out, c->err); return nullptr;
        }
        auto a1 = now();
        AudioEncoding adapted;
        if (!encode_audio_and_adapt(*c->model, mel, adapted, c->err)) {
            report(err_out, c->err); return nullptr;
        }
        auto a2 = now();
        Prompt prompt = build_transcribe_prompt(c->model->config, mel.n_frames);
        InputsEmbeds inputs;
        if (!build_inputs_embeds(*c->model, prompt, adapted, inputs, c->err)) {
            report(err_out, c->err); return nullptr;
        }
        auto a3 = now();
        if (std::getenv("STARLING_MOSS_DEBUG")) {
            double mx = 0; size_t bad = 0;
            for (float v : inputs.data) { if (!std::isfinite(v)) ++bad; mx = std::max(mx, std::abs((double)v)); }
            std::fprintf(stderr, "MOSS_DEBUG inputs_embeds n=%lld frames=%lld prompt=%zu audio=%lld nonfinite=%zu max_abs=%.6g\n",
                         (long long)inputs.n_tokens, (long long)mel.n_frames, prompt.ids.size(),
                         (long long)adapted.n_tokens, bad, mx);
        }
        GenerateOptions options;
        options.max_new_tokens = static_cast<int32_t>(c->model->config.max_new_tokens);
        options.max_cache_len = static_cast<int32_t>(c->model->config.llm.max_cache);
        options.eos_token_id = c->model->config.eos_token_id;
        GenerateResult generated;
        if (!greedy_generate(*c->model, inputs, options, generated, c->err)) {
            report(err_out, c->err); return nullptr;
        }
        auto a4 = now();
        if (timing) {
            std::fprintf(stderr, "MOSS_STAGE frames=%lld mel=%.1fms enc+adapt=%.1fms prompt+embeds=%.1fms gen=%.1fms audio_tokens=%lld prompt_tokens=%lld gen_tokens=%zu\n",
                         (long long)mel.n_frames, ms(a0, a1), ms(a1, a2), ms(a2, a3),
                         ms(a3, a4), (long long)adapted.n_tokens, (long long)inputs.n_tokens,
                         generated.ids.size());
        }
        const std::string text = c->tokenizer.decode(generated.ids, true);
        char * out = static_cast<char *>(std::malloc(text.size() + 1));
        if (!out) { if (err_out) *err_out = "malloc failed"; return nullptr; }
        std::memcpy(out, text.data(), text.size());
        out[text.size()] = '\0';
        if (err_out) *err_out = nullptr;
        return out;
    } catch (const std::exception & e) {
        // Copy into the context's owned error string: e.what() dangles once
        // the catch exits, and cpp/capi.cpp reads *err_out after we return.
        c->err = e.what();
        report(err_out, c->err);
    } catch (...) {
        report(err_out, "unknown exception transcribing MOSS audio");
    }
    return nullptr;
}

} // extern "C"
