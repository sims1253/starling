// capi_ark.cpp — ARK-ASR-3B C API entry points behind the shared shell.
// C API flow: load -> mel -> encode+adapt -> prompt/embeds -> greedy
// decode -> detokenize, with a STARLING_ARK_TIMING phase-timing gate.
#include "loader.hpp"
#include "lib/capi_helpers.hpp"
#include "mel.hpp"
#include "audio_encoder.hpp"
#include "prompt.hpp"
#include "llm.hpp"
#include "tokenizer.hpp"
#include "runtime/graph.hpp"
#include "runtime/backend.hpp"

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

using ArkCtx = starling::ggml::lib::EngineContext<starling::ggml::ark::ArkModel, starling::ggml::ark::Tokenizer>;
using starling::ggml::lib::report;

} // namespace

extern "C" {

void* starling_ggml_ark_load(const char* gguf_path, const char** err_out) {
    return starling::ggml::lib::load_engine<ArkCtx>(gguf_path, "ARK", err_out);
}

void starling_ggml_ark_free(void* handle) {
    try {
        delete static_cast<ArkCtx*>(handle);
    } catch (...) {
        // C ABI: never allow an exception to escape.
    }
}

char* starling_ggml_ark_decode(void* handle, const float* pcm, int64_t n,
                               const char** err_out) {
    auto* c = static_cast<ArkCtx*>(handle);
    if (!c) { if (err_out) *err_out = "null ARK handle"; return nullptr; }
    if (n < 0 || (n > 0 && !pcm)) {
        if (err_out) *err_out = "invalid ARK PCM buffer";
        return nullptr;
    }
    try {
        using namespace starling::ggml::ark;
        const bool timing = std::getenv("STARLING_ARK_TIMING") != nullptr;
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
        AudioEncoding adapted;
        if (!encode_audio_and_adapt(*c->model, mel, adapted, c->err)) {
            report(err_out, c->err);
            return nullptr;
        }
        auto a2 = now();
        // The prompt's audio-token count derives from the UNCAPPED mel-frame
        // count (len(pcm)//hop), not the extractor's capped n_frames (3000 for
        // long audio). compute_log_mel returns the capped count; recompute the
        // uncapped count for the prompt so it matches the eager reference.
        const int64_t uncapped_frames = c->model->config.frontend.hop_length > 0
            ? (int64_t) n / (int64_t) c->model->config.frontend.hop_length
            : mel.n_frames;
        Prompt prompt = build_transcribe_prompt(c->model->config, uncapped_frames);
        InputsEmbeds inputs;
        if (!build_inputs_embeds(*c->model, prompt, adapted, inputs, c->err)) {
            report(err_out, c->err);
            return nullptr;
        }
        auto a3 = now();
        GenerateOptions options;
        options.max_new_tokens = static_cast<int32_t>(c->model->config.max_new_tokens);
        options.max_cache_len = static_cast<int32_t>(c->model->config.llm.max_cache);
        options.eos_token_id = c->model->config.eos_token_id;
        GenerateResult generated;
        if (!greedy_generate(*c->model, inputs, options, generated, c->err)) {
            report(err_out, c->err);
            return nullptr;
        }
        auto a4 = now();
        if (timing) {
            std::fprintf(stderr,
                "ARK_STAGE frames=%lld mel=%.1fms enc+adapt=%.1fms prompt+embeds=%.1fms "
                "gen=%.1fms audio_tokens=%lld prompt_tokens=%lld gen_tokens=%zu\n",
                (long long) uncapped_frames, ms(a0, a1), ms(a1, a2), ms(a2, a3),
                ms(a3, a4), (long long) adapted.n_tokens, (long long) inputs.n_tokens,
                generated.ids.size());
        }
        const std::string text = c->tokenizer.decode(generated.ids, true);
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
        report(err_out, "unknown exception transcribing ARK audio");
    }
    return nullptr;
}

} // extern "C"
