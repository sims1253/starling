// capi_higgs.cpp — higgs-audio-v3-stt C API entry points behind the shared shell.
// Mirrors capi_ark.cpp: load -> mel -> encode+project -> prompt+embeds -> greedy
// decode -> detokenize, with a STARLING_HIGGS_TIMING phase-timing gate.
#include "loader.hpp"
#include "mel.hpp"
#include "audio_encoder.hpp"
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

struct HiggsCtx {
    std::unique_ptr<starling::ggml::higgs::HiggsModel> model;
    starling::ggml::higgs::Tokenizer tokenizer;
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

void* starling_ggml_higgs_load(const char* gguf_path, const char** err_out) {
    try {
        if (!gguf_path || !*gguf_path) {
            if (err_out) *err_out = "null or empty Higgs GGUF path";
            return nullptr;
        }
        auto ctx = std::make_unique<HiggsCtx>();
        ctx->model = std::make_unique<starling::ggml::higgs::HiggsModel>();
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
        ctx->model->loader.realize_weights(starling::ggml::global_backend());
        starling::ggml::register_decode_cache_clearer([]() {});
        if (err_out) *err_out = nullptr;
        return ctx.release();
    } catch (const std::exception& e) {
        report_load_error(err_out, e.what());
    } catch (...) {
        report_load_error(err_out, "unknown exception loading Higgs model");
    }
    return nullptr;
}

void starling_ggml_higgs_free(void* handle) {
    try {
        delete static_cast<HiggsCtx*>(handle);
    } catch (...) {
        // C ABI: never allow an exception to escape.
    }
}

char* starling_ggml_higgs_decode(void* handle, const float* pcm, int64_t n,
                                 const char** err_out) {
    auto* c = static_cast<HiggsCtx*>(handle);
    if (!c) { if (err_out) *err_out = "null Higgs handle"; return nullptr; }
    if (n < 0 || (n > 0 && !pcm)) {
        if (err_out) *err_out = "invalid Higgs PCM buffer";
        return nullptr;
    }
    try {
        using namespace starling::ggml::higgs;
        const bool timing = std::getenv("STARLING_HIGGS_TIMING") != nullptr;
        auto now = [&]() { return std::chrono::steady_clock::now(); };
        auto ms = [&](auto t0, auto t1) {
            return std::chrono::duration<double, std::milli>(t1 - t0).count();
        };

        auto a0 = now();
        // Higgs chunks each clip into ceil(n / chunk_size_samples) <= 4 s chunks
        // (config.chunk_size_seconds), runs the audio tower + projector PER chunk,
        // and emits one <|audio_bos|>...<|audio_eos|> segment per chunk in the
        // prompt. Replicate that here: mel + encode + project each chunk, collect
        // the per-chunk token counts + the concatenated feature stream. The eager
        // collator pads each chunk's mel to nb_max_frames (3000) and masks the
        // padding in self-attention; running the encoder on each chunk's UNPADDED
        // valid mel (no mask) is bit-identical for the valid outputs (the masked
        // keys contribute zero attention weight), so no mask is needed.
        const auto& fc = c->model->config.frontend;
        const int64_t sr = fc.sample_rate > 0 ? (int64_t) fc.sample_rate : 16000;
        const int64_t chunk_samples = c->model->config.frontend.chunk_size_seconds > 0.0f
            ? (int64_t)(c->model->config.frontend.chunk_size_seconds * (float) sr)
            : (int64_t) c->model->config.frontend.n_samples;
        const int64_t num_chunks = (n + chunk_samples - 1) / chunk_samples;
        AudioEncoding projected;
        std::vector<int64_t> chunk_tokens;
        chunk_tokens.reserve((size_t) num_chunks);
        int64_t total_audio_tokens = 0;
        auto a1 = now();
        for (int64_t ci = 0; ci < num_chunks; ++ci) {
            const int64_t off = ci * chunk_samples;
            const int64_t len = std::min(chunk_samples, n - off);
            MelFeatures mel;
            if (!compute_log_mel(c->model->config, c->model->loader, pcm + off,
                                 static_cast<size_t>(len), mel, c->err)) {
                report(err_out, c->err);
                return nullptr;
            }
            AudioEncoding chunk_enc;
            if (!encode_audio_and_project(*c->model, mel, chunk_enc, c->err)) {
                report(err_out, c->err);
                return nullptr;
            }
            chunk_tokens.push_back(chunk_enc.n_tokens);
            total_audio_tokens += chunk_enc.n_tokens;
            // Concatenate this chunk's projector features onto the stream that
            // gets scattered into the prompt's <|AUDIO|> slots (in chunk order,
            // matching merge_input_ids_with_audio_features).
            projected.data.insert(projected.data.end(),
                                  chunk_enc.data.begin(), chunk_enc.data.end());
        }
        projected.n_tokens = total_audio_tokens;
        projected.width = c->model->config.projector.output_size;
        auto a2 = now();
        Prompt prompt = build_transcribe_prompt(c->model->config, chunk_tokens);
        InputsEmbeds inputs;
        if (!build_inputs_embeds(*c->model, prompt, projected, inputs, c->err)) {
            report(err_out, c->err);
            return nullptr;
        }
        auto a3 = now();
        GenerateOptions options;
        options.max_new_tokens = static_cast<int32_t>(c->model->config.max_new_tokens);
        options.max_cache_len = static_cast<int32_t>(c->model->config.llm.max_cache);
        options.eos_token_id = c->model->config.eos_token_id;
        options.im_end_id = c->model->config.im_end_id;
        GenerateResult generated;
        if (!greedy_generate(*c->model, inputs, options, generated, c->err)) {
            report(err_out, c->err);
            return nullptr;
        }
        auto a4 = now();
        if (timing) {
            std::fprintf(stderr,
                "HIGGS_STAGE chunks=%lld mel+enc+proj=%.1fms prompt+embeds=%.1fms "
                "gen=%.1fms audio_tokens=%lld prompt_tokens=%lld gen_tokens=%zu\n",
                (long long) num_chunks, ms(a0, a2), ms(a2, a3),
                ms(a3, a4), (long long) projected.n_tokens, (long long) inputs.n_tokens,
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
        // Copy into the context's owned error string: report() stores
        // message.c_str() into *err_out, and a string literal passed by const
        // reference materializes a temporary std::string whose buffer dangles
        // once the call returns (unlike report_load_error, report does not copy).
        c->err = "unknown exception transcribing Higgs audio";
        report(err_out, c->err);
    }
    return nullptr;
}

} // extern "C"
