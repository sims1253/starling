// capi_voxtral.cpp — Voxtral-Mini-4B-Realtime C API entry points.
//
// C API flow: load -> mel (offline pad) -> encode+project -> prompt/embeds
// (additive injection) -> offline greedy decode -> detokenize, mirroring
// capi_ark.cpp's ownership/error discipline. The serve layer wraps the
// returned text in its JSON envelope; the C API itself returns plain text.
#include "loader.hpp"
#include "mel.hpp"
#include "encoder.hpp"
#include "prompt.hpp"
#include "llm.hpp"
#include "tokenizer.hpp"
#include "runtime/graph.hpp"

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
        // Persist weights + force backend creation (and its orderly atexit
        // shutdown registration) across all transcription calls.
        if (!ctx->model->loader.realize_weights(starling::ggml::global_backend())) {
            report_load_error(err_out, ctx->model->loader.last_error());
            return nullptr;
        }
        starling::ggml::register_decode_cache_clearer([]() {});
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
        using namespace starling::ggml::voxtral;
        const bool timing = std::getenv("STARLING_VOXTRAL_TIMING") != nullptr;
        auto now = [&]() { return std::chrono::steady_clock::now(); };
        auto ms = [&](auto t0, auto t1) {
            return std::chrono::duration<double, std::milli>(t1 - t0).count();
        };

        auto a0 = now();
        MelFeatures mel;
        if (!compute_log_mel(c->model->config, c->model->loader, pcm,
                             static_cast<size_t>(n), mel, c->err)) {
            report(err_out, c->err);
            return nullptr;
        }
        auto a1 = now();
        AudioEncoding audio;
        if (!encode_audio_and_project(*c->model, mel, audio, c->err)) {
            report(err_out, c->err);
            return nullptr;
        }
        auto a2 = now();
        // The prompt is the baked 39-id prefix; the total-length cap derives
        // from the mel-frame count (stock ceil(mel/8) bound).
        const std::vector<int32_t> prompt = build_transcribe_prompt(c->model->config);
        InputsEmbeds inputs;
        if (!build_inputs_embeds(*c->model, prompt, audio, inputs, c->err)) {
            report(err_out, c->err);
            return nullptr;
        }
        auto a3 = now();
        GenerateOptions options;
        options.max_cache_len = static_cast<int32_t>(c->model->config.llm.max_cache);
        options.eos_token_id = c->model->config.eos_token_id;
        GenerateResult generated;
        if (!greedy_generate(*c->model, inputs, audio, mel.n_frames, options,
                             generated, c->err)) {
            report(err_out, c->err);
            return nullptr;
        }
        auto a4 = now();
        if (timing) {
            std::fprintf(stderr,
                "VOXTRAL_STAGE frames=%lld mel=%.1fms enc+proj=%.1fms prompt+embeds=%.1fms "
                "gen=%.1fms audio_tokens=%lld prompt_tokens=%lld gen_tokens=%zu\n",
                (long long) mel.n_frames, ms(a0, a1), ms(a1, a2), ms(a2, a3),
                ms(a3, a4), (long long) audio.n_tokens, (long long) inputs.n_tokens,
                generated.ids.size());
        }
        const std::string text = c->tokenizer.decode(generated.ids);
        char* out = static_cast<char*>(std::malloc(text.size() + 1));
        if (!out) { if (err_out) *err_out = "malloc failed"; return nullptr; }
        std::memcpy(out, text.data(), text.size());
        out[text.size()] = '\0';
        if (err_out) *err_out = nullptr;
        return out;
    } catch (const std::exception& e) {
        // Copy into the context's owned error string: e.what() dangles after the
        // catch exits, and cpp/capi.cpp reads *err_out after we return.
        c->err = e.what();
        report(err_out, c->err);
    } catch (...) {
        c->err = "unknown exception transcribing VOXTRAL audio";
        report(err_out, c->err);
    }
    return nullptr;
}

} // extern "C"
