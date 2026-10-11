// capi_voxtral.cpp — Voxtral-Mini-4B-Realtime C API entry points.
//
// C API flow: load -> mel (offline pad) -> encode+project -> prompt/embeds
// (additive injection) -> offline greedy decode -> detokenize, mirroring
// capi_ark.cpp's ownership/error discipline. The serve layer wraps the
// returned text in its JSON envelope; the C API itself returns plain text.
//
// With the fast engine (cpp/fast, STARLING_ENGINE=auto|fast) everything
// after the CPU log-mel runs on the Vulkan GPU: AudioLlmEngine takes the
// mel and the prompt ids and runs the causal encoder, projector, prefill
// and the on-device greedy loop with the per-position audio add.
// STARLING_FAST_VOXTRAL_HOST_ENCODER=1 keeps the encoder+projector on ggml
// and hands the engine their rows (decoder only). The loader's weights the
// engine owns are released from host memory (release_host_weights).
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

#if defined(STARLING_HAVE_FAST)
#include "engine_select.hpp"
#include "audio_llm_engine.hpp"
#endif

namespace {

thread_local std::string g_load_error;

struct VoxtralCtx {
    std::unique_ptr<starling::ggml::voxtral::VoxtralModel> model;
    starling::ggml::voxtral::Tokenizer tokenizer;
    std::string err;
#if defined(STARLING_HAVE_FAST)
    // Model-specialized Vulkan engine (cpp/fast). When set it owns the
    // decoder (and, unless it says otherwise, the encoder + projector).
    std::unique_ptr<starling::fast::AudioLlmEngine> fast;
#endif
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
#if defined(STARLING_HAVE_FAST)
        const auto choice = starling::fast::engine_choice();
        if (choice != starling::fast::EngineChoice::Ggml) {
            std::string ferr;
            ctx->fast = starling::fast::AudioLlmEngine::create(*ctx->model, ferr);
            const bool verbose = std::getenv("STARLING_FAST_VERBOSE") != nullptr;
            if (ctx->fast) {
                // With the encoder on the device the loader keeps only the
                // mel constants (and the tokenizer's metadata) in use.
                std::vector<std::string> owned{"llm."};
                if (ctx->fast->encodes_audio()) owned.insert(owned.end(), {"enc.", "proj."});
                const size_t freed = starling::fast::release_host_weights(ctx->model->loader, owned);
                if (verbose)
                    std::fprintf(stderr, "[fast] voxtral engine (%s): %s (%.0f MiB of host weights released)\n",
                                 ctx->fast->encodes_audio() ? "encoder+decoder" : "decoder",
                                 ctx->fast->describe().c_str(), freed / 1048576.0);
            } else if (choice == starling::fast::EngineChoice::Fast) {
                report_load_error(err_out, "fast engine unavailable: " + ferr);
                return nullptr;
            } else if (verbose) {
                std::fprintf(stderr, "[fast] voxtral falls back to ggml: %s\n", ferr.c_str());
            }
        }
#endif
        // Persist weights + force backend creation (and its orderly atexit
        // shutdown registration) across all transcription calls.
        if (!ctx->model->loader.realize_weights(starling::ggml::global_backend())) {
            report_load_error(err_out, ctx->model->loader.last_error());
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
        // The prompt is the baked 39-id prefix; the total-length cap derives
        // from the mel-frame count (stock ceil(mel/8) bound).
        const std::vector<int32_t> prompt = build_transcribe_prompt(c->model->config);
#if defined(STARLING_HAVE_FAST)
        if (c->fast) {
            // Fast engine: the stock cap (sequence length == audio rows) is
            // the engine's default budget.
            std::vector<int32_t> ids;
            bool eos = false;
            bool ok;
            int64_t audio_rows = audio_token_count(mel.n_frames);
            if (c->fast->encodes_audio()) {
                static_assert(sizeof(ggml_bf16_t) == sizeof(uint16_t), "bf16 bits");
                ok = c->fast->generate_additive_from_mel(prompt, reinterpret_cast<const uint16_t*>(mel.data.data()),
                                                         mel.n_frames, 0, ids, eos, c->err);
            } else {
                AudioEncoding audio;
                ok = encode_audio_and_project(*c->model, mel, audio, c->err);
                if (ok && audio.width != (int64_t) c->model->config.llm.hidden) {
                    c->err = "VOXTRAL projector width does not match the decoder";
                    ok = false;
                }
                if (ok) {
                    audio_rows = audio.n_tokens;
                    ok = c->fast->generate_additive(prompt, audio.data.data(), audio.n_tokens, 0, ids, eos, c->err);
                }
            }
            if (!ok) {
                report(err_out, c->err);
                return nullptr;
            }
            if (timing)
                std::fprintf(stderr,
                    "VOXTRAL_STAGE frames=%lld mel=%.1fms fast_%s=%.1fms audio_tokens=%lld prompt_tokens=%zu gen_tokens=%zu\n",
                    (long long) mel.n_frames, ms(a0, a1), c->fast->encodes_audio() ? "enc+gen" : "hostenc+gen",
                    ms(a1, now()), (long long) audio_rows, prompt.size(), ids.size());
            if (const char* fp = std::getenv("STARLING_VOXTRAL_DUMP_IDS")) {
                if (FILE* f = std::fopen(fp, "wb")) {
                    std::fwrite(ids.data(), sizeof(int32_t), ids.size(), f);
                    std::fclose(f);
                }
            }
            const std::string text = c->tokenizer.decode(ids);
            char* out = static_cast<char*>(std::malloc(text.size() + 1));
            if (!out) { if (err_out) *err_out = "malloc failed"; return nullptr; }
            std::memcpy(out, text.data(), text.size());
            out[text.size()] = '\0';
            if (err_out) *err_out = nullptr;
            return out;
        }
#endif
        AudioEncoding audio;
        if (!encode_audio_and_project(*c->model, mel, audio, c->err)) {
            report(err_out, c->err);
            return nullptr;
        }
        auto a2 = now();
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
