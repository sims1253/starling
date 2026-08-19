// capi_audex.cpp — Nemotron-Labs-Audex-2B C API entry points behind the
// shared shell. C API flow: load -> (chunk policy) -> per chunk {mel ->
// encode + project -> prompt/embeds -> greedy decode -> detokenize +
// extract} -> join, with a STARLING_AUDEX_TIMING phase-timing gate.
//
// The chunk policy mirrors the Python server path (AudexBackend.transcribe
// via ModelBackend._transcribe_chunked) exactly: max_chunk = min(30 s,
// (max_new_tokens-32)/5) (= 30 s at 200 tokens — a chunk is ALWAYS one
// 30 s clip, since the pipeline's clip size equals the chunk size); audio
// up to max_chunk goes through single shot with budget min(max_new_tokens,
// ceil(dur*5)+32); longer audio is cut into chunk_seconds waveform chunks —
// each padded to a full 30 s clip at the mel level (padding="max_length")
// and decoded with budget max(1, min(budget(dur), max_cache_len -
// prompt_len - 1)); the per-chunk texts joined with whitespace collapsed.
// Empty input produces no chunks at all (the server's range(0, 0, size)).
//
// The per-chunk text replicates MegaPipeline._decode_response: decode with
// special tokens KEPT, strip after the last "</think>" (guard), cut at the
// first "<|im_end|>", strip, then extract the transcript between the FIRST
// and LAST single quote (the model wraps ASR output as "The content of the
// input audio is '<transcript>'." and re.search(r"'(.+)'", DOTALL) is greedy
// to the last quote); without quotes the stripped raw text is the result.
#include "loader.hpp"
#include "mel.hpp"
#include "encoder.hpp"
#include "prompt.hpp"
#include "llm.hpp"
#include "tokenizer.hpp"
#include "runtime/graph.hpp"

#include <algorithm>
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

struct AudexCtx {
    std::unique_ptr<starling::ggml::audex::AudexModel> model;
    starling::ggml::audex::Tokenizer tokenizer;
    std::string err;
};

void report(const char** out, const std::string& message) {
    if (out) *out = message.c_str();
}

void report_load_error(const char** out, const std::string& message) {
    g_load_error = message;
    if (out) *out = g_load_error.c_str();
}

constexpr double kSampleRate = 16000.0;

// Mirror ModelBackend._decode_budget: scale the decode cap to the clip length.
int32_t decode_budget(const starling::ggml::audex::Config& c, double duration_s) {
    int64_t estimated = (int64_t) std::ceil(duration_s * 5.0) + 32;
    if (estimated < 1) estimated = 1;
    int64_t cap = c.max_new_tokens > 0 ? c.max_new_tokens : 1;
    return (int32_t) std::min(cap, estimated);
}

// ---- MegaPipeline._decode_response port ------------------------------------

std::string strip_ws(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace((unsigned char) s[b])) ++b;
    while (e > b && std::isspace((unsigned char) s[e - 1])) --e;
    return s.substr(b, e - b);
}

// Everything between the first and last single quote (the greedy DOTALL
// re.search match; `.+` needs at least one char), or the stripped raw text.
std::string extract_transcription(std::string raw) {
    if (const size_t p = raw.rfind("</think>"); p != std::string::npos)
        raw = raw.substr(p + strlen("</think>"));
    if (const size_t p = raw.find("<|im_end|>"); p != std::string::npos)
        raw = raw.substr(0, p);
    raw = strip_ws(raw);
    const size_t q1 = raw.find('\'');
    const size_t q2 = raw.rfind('\'');
    if (q1 != std::string::npos && q2 > q1)
        return strip_ws(raw.substr(q1 + 1, q2 - q1 - 1));
    return raw;
}

// One chunk through mel -> encoder+projector -> prompt -> greedy -> text.
bool transcribe_piece(AudexCtx& ctx, const float* pcm, int64_t n, int32_t budget,
                      std::string& text, double* stage_ms = nullptr) {
    using namespace starling::ggml::audex;
    const AudexModel& m = *ctx.model;
    auto t0 = std::chrono::steady_clock::now();
    MelFeatures mel;
    if (!compute_log_mel(m.config, m.loader, pcm, (size_t) n, mel, ctx.err))
        return false;
    AudioEmbeds audio;
    if (!encode_audio_and_project(m, mel, audio, ctx.err))
        return false;
    auto t1 = std::chrono::steady_clock::now();
    Prompt prompt = build_transcribe_prompt(m.config);
    InputsEmbeds inputs;
    if (!build_inputs_embeds(m, prompt, audio, inputs, ctx.err))
        return false;
    auto t2 = std::chrono::steady_clock::now();
    GenerateOptions options;
    options.max_new_tokens = budget;
    options.max_cache_len = (int32_t) m.config.llm.max_cache;
    options.eos_token_id = m.config.eos_token_id;
    GenerateResult generated;
    if (!greedy_generate(m, inputs, options, generated, ctx.err))
        return false;
    auto t3 = std::chrono::steady_clock::now();
    text = extract_transcription(ctx.tokenizer.decode(generated.ids, false));
    if (stage_ms) {
        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        stage_ms[0] = ms(t0, t1);  // mel + encode + project
        stage_ms[1] = ms(t1, t2);  // prompt + embeds
        stage_ms[2] = ms(t2, t3);  // generate
    }
    return true;
}

// Mirror ModelBackend._transcribe_chunked's join: join the texts, then
// collapse whitespace runs to single spaces.
std::string join_texts(const std::vector<std::string>& texts) {
    std::string joined;
    for (const std::string& t : texts) {
        if (t.empty()) continue;
        if (!joined.empty()) joined += ' ';
        joined += t;
    }
    std::string out;
    out.reserve(joined.size());
    bool space = false;
    for (char ch : joined) {
        if (std::isspace((unsigned char) ch)) {
            space = true;
            continue;
        }
        if (space && !out.empty()) out += ' ';
        space = false;
        out += ch;
    }
    return out;
}

} // namespace

extern "C" {

void* starling_ggml_audex_load(const char* gguf_path, const char** err_out) {
    try {
        if (!gguf_path || !*gguf_path) {
            if (err_out) *err_out = "null or empty AUDEX GGUF path";
            return nullptr;
        }
        auto ctx = std::make_unique<AudexCtx>();
        ctx->model = std::make_unique<starling::ggml::audex::AudexModel>();
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
        report_load_error(err_out, "unknown exception loading AUDEX model");
    }
    return nullptr;
}

void starling_ggml_audex_free(void* handle) {
    try {
        delete static_cast<AudexCtx*>(handle);
    } catch (...) {
        // C ABI: never allow an exception to escape.
    }
}

char* starling_ggml_audex_decode(void* handle, const float* pcm, int64_t n,
                                 const char** err_out) {
    auto* c = static_cast<AudexCtx*>(handle);
    if (!c) { if (err_out) *err_out = "null AUDEX handle"; return nullptr; }
    if (n < 0 || (n > 0 && !pcm)) {
        if (err_out) *err_out = "invalid AUDEX PCM buffer";
        return nullptr;
    }
    try {
        using namespace starling::ggml::audex;
        const Config& cfg = c->model->config;
        const bool timing = std::getenv("STARLING_AUDEX_TIMING") != nullptr;
        auto now = [] { return std::chrono::steady_clock::now(); };
        auto t_start = now();

        // max_chunk = min(chunk_seconds, (max_new_tokens - 32) / 5) — the
        // server's _effective_chunk_seconds(DEFAULT_MAX_CHUNK_SECONDS). At
        // the baked defaults this is exactly the 30 s clip size, so every
        // chunk is one clip (750 audio tokens).
        const double token_limited =
            std::max(0.1, ((double) (int) cfg.max_new_tokens - 32.0) / 5.0);
        const double max_chunk_s = std::min(cfg.chunk_seconds, token_limited);
        const int64_t chunk_samples =
            (int64_t) std::llround(max_chunk_s * kSampleRate);
        const double duration_s = (double) n / kSampleRate;

        std::vector<std::string> texts;
        double stage_ms[3] = {0, 0, 0};
        if (n <= 0) {
            // The server's chunk loop over range(0, 0, chunk) is empty.
        } else if (chunk_samples <= 0 || n <= chunk_samples) {
            std::string text;
            if (!transcribe_piece(*c, pcm, n, decode_budget(cfg, duration_s), text,
                                  stage_ms))
            {
                report(err_out, c->err);
                return nullptr;
            }
            texts.push_back(std::move(text));
        } else {
            for (int64_t start = 0; start < n; start += chunk_samples) {
                const int64_t len = std::min(chunk_samples, n - start);
                const double piece_s = (double) len / kSampleRate;
                // The budget cap keeps the greedy loop inside the static KV
                // cache (prompt + new tokens <= max_cache_len) exactly like
                // the Python LLMMega's max_new_tokens guard. The prompt is
                // fixed-length: prefix + 750 audio slots + suffix.
                const int64_t prompt_len = (int64_t) cfg.prompt_prefix.size() +
                                           (int64_t) cfg.sound_embedding_size +
                                           (int64_t) cfg.prompt_suffix.size();
                int32_t budget = decode_budget(cfg, piece_s);
                const int64_t headroom =
                    (int64_t) cfg.llm.max_cache - prompt_len - 1;
                if ((int64_t) budget > headroom) budget = (int32_t) std::max<int64_t>(1, headroom);
                std::string text;
                if (!transcribe_piece(*c, pcm + start, len, budget, text,
                                      stage_ms))
                {
                    report(err_out, c->err);
                    return nullptr;
                }
                texts.push_back(std::move(text));
            }
        }
        auto t_end = now();
        if (timing) {
            std::fprintf(stderr,
                "AUDEX_STAGE chunks=%zu audio=%.2fs mel+enc+proj=%.1fms "
                "prompt+embeds=%.1fms gen=%.1fms total=%.1fms\n",
                texts.size(), duration_s, stage_ms[0], stage_ms[1], stage_ms[2],
                std::chrono::duration<double, std::milli>(t_end - t_start).count());
        }

        const std::string text = join_texts(texts);
        char* out = static_cast<char*>(std::malloc(text.size() + 1));
        if (!out) { if (err_out) *err_out = "malloc failed"; return nullptr; }
        std::memcpy(out, text.data(), text.size());
        out[text.size()] = '\0';
        if (err_out) *err_out = nullptr;
        return out;
    } catch (const std::exception& e) {
        // Copy into the context's owned error string: e.what() dangles after
        // the catch exits, and cpp/capi.cpp reads *err_out after we return.
        c->err = e.what();
        report(err_out, c->err);
    } catch (...) {
        // Same ownership rule as the std::exception branch: report() would
        // hand *err_out a pointer into a temporary std::string that dies at
        // the end of the statement, and capi.cpp reads it after we return.
        c->err = "unknown exception transcribing AUDEX audio";
        report(err_out, c->err);
    }
    return nullptr;
}

} // extern "C"
