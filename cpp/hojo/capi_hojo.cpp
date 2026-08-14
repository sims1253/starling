// capi_hojo.cpp — Hojo-ASR-V1 C API entry points behind the shared shell.
// C API flow: load -> mel -> audio_tower -> bottleneck -> ln_speech
// (in prompt) -> prompt -> beam-4 decode -> detokenize.
#include "loader.hpp"
#include "mel.hpp"
#include "audio_tower.hpp"
#include "conformer.hpp"
#include "prompt.hpp"
#include "llm.hpp"
#include "tokenizer.hpp"
#include "runtime/graph.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <vector>

namespace {

thread_local std::string g_load_error;

struct HojoCtx {
    std::unique_ptr<starling::ggml::hojo::HojoModel> model;
    starling::ggml::hojo::Tokenizer tokenizer;
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

void* starling_ggml_hojo_load(const char* gguf_path, const char** err_out) {
    try {
        if (!gguf_path || !*gguf_path) {
            if (err_out) *err_out = "null or empty Hojo GGUF path";
            return nullptr;
        }
        auto ctx = std::make_unique<HojoCtx>();
        ctx->model = std::make_unique<starling::ggml::hojo::HojoModel>();
        if (!ctx->model->load(gguf_path, ctx->err)) {
            report_load_error(err_out, ctx->err);
            return nullptr;
        }
        if (!ctx->tokenizer.load(ctx->model->loader, ctx->model->config, ctx->err)) {
            report_load_error(err_out, ctx->err);
            return nullptr;
        }
        // Persist weights + force backend creation across all transcription calls.
        ctx->model->loader.realize_weights(starling::ggml::global_backend());
        starling::ggml::register_decode_cache_clearer([]() {});
        if (err_out) *err_out = nullptr;
        return ctx.release();
    } catch (const std::exception& e) {
        report_load_error(err_out, e.what());
    } catch (...) {
        report_load_error(err_out, "unknown exception loading Hojo model");
    }
    return nullptr;
}

void starling_ggml_hojo_free(void* handle) {
    try {
        delete static_cast<HojoCtx*>(handle);
    } catch (...) {
        // C ABI: never allow an exception to escape.
    }
}

char* starling_ggml_hojo_decode(void* handle, const float* pcm, int64_t n,
                                const char** err_out) {
    auto* c = static_cast<HojoCtx*>(handle);
    if (!c) { if (err_out) *err_out = "null Hojo handle"; return nullptr; }
    if (n < 0 || (n > 0 && !pcm)) {
        if (err_out) *err_out = "invalid Hojo PCM buffer";
        return nullptr;
    }
    try {
        using namespace starling::ggml::hojo;
        const bool timing = std::getenv("STARLING_HOJO_TIMING") != nullptr;
        auto now = [&]() { return std::chrono::steady_clock::now(); };
        auto ms = [&](auto t0, auto t1) {
            return std::chrono::duration<double, std::milli>(t1 - t0).count();
        };

        auto a0 = now();
        // 1. Mel.
        MelFeatures mel;
        if (!compute_log_mel(c->model->config, c->model->loader, pcm,
                             static_cast<size_t>(n), mel, c->err)) {
            report(err_out, c->err);
            return nullptr;
        }
        // 2. Audio tower (Qwen3-Omni).
        TowerOutput tower;
        if (!encode_audio_tower(*c->model, mel, tower, c->err)) {
            report(err_out, c->err);
            return nullptr;
        }
        // 3. Bottleneck (WeNet Conformer).
        BottleneckOutput bn;
        if (!encode_bottleneck(*c->model, tower, bn, c->err)) {
            report(err_out, c->err);
            return nullptr;
        }
        // 4. Prompt (ln_speech + bos prepend).
        InputsEmbeds inputs;
        if (!build_inputs_embeds(*c->model, bn, inputs, c->err)) {
            report(err_out, c->err);
            return nullptr;
        }
        auto a1 = now();
        // 5. Beam-4 decode.
        GenerateOptions options;
        options.num_beams = c->model->config.decode.num_beams;
        options.repetition_penalty = c->model->config.decode.repetition_penalty;
        options.length_penalty = c->model->config.decode.length_penalty;
        options.min_length = c->model->config.decode.min_length;
        options.eos_token_id = c->model->config.eos_token_id;
        options.pad_token_id = c->model->config.pad_token_id;
        // max_new_tokens = min(200, feat_len*2 + 10), at least 10.
        int32_t feat_len = (int32_t) inputs.n_tokens - 1;
        int32_t mnt = (int32_t) c->model->config.decode.max_new_tokens;
        int32_t derived = std::min(mnt, feat_len * 2 + 10);
        options.max_new_tokens = std::max(derived, 10);
        options.max_cache_len = (int32_t) c->model->config.llm.max_cache;
        GenerateResult generated;
        if (!beam_generate(*c->model, inputs, options, generated, c->err)) {
            report(err_out, c->err);
            return nullptr;
        }
        auto a2 = now();
        if (timing) {
            std::fprintf(stderr,
                "HOJO_STAGE mel+enc+bottleneck+prompt=%.1fms beam_decode=%.1fms "
                "speech=%lld prompt=%lld gen=%zu\n",
                ms(a0, a1), ms(a1, a2), (long long) tower.n_speech,
                (long long) inputs.n_tokens, generated.ids.size());
        }
        // 6. Detokenize. The decode output mirrors HF: skip special tokens
        // (<|im_end|>, <|endoftext|>) and strip whitespace.
        const std::string text = c->tokenizer.decode(generated.ids, true);
        char* out = static_cast<char*>(std::malloc(text.size() + 1));
        if (!out) { if (err_out) *err_out = "malloc failed"; return nullptr; }
        std::memcpy(out, text.data(), text.size());
        out[text.size()] = '\0';
        if (err_out) *err_out = nullptr;
        return out;
    } catch (const std::exception& e) {
        c->err = e.what();
        report(err_out, c->err);
    } catch (...) {
        report(err_out, "unknown exception transcribing Hojo audio");
    }
    return nullptr;
}

} // extern "C"
