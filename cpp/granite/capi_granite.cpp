// capi_granite.cpp — granite-speech-4.1-2b C API entry points behind the
// shared shell. C API flow: load -> (chunk policy) -> per chunk {mel -> encode
// +project -> prompt/embeds -> greedy decode -> detokenize} -> join, with a
// STARLING_GRANITE_TIMING phase-timing gate.
//
// The chunk policy mirrors the Python server path (GraniteBackend.transcribe)
// exactly: audio up to min(30 s, (max_new_tokens-32)/5) goes through single
// shot with budget min(max_new_tokens, ceil(dur*5)+32); longer audio is cut
// into chunk_seconds waveform chunks — the LAST chunk zero-padded to the full
// chunk length, exactly what chunk_audio(pad_last=True) feeds the processor —
// each decoded with budget max(1, min(budget(dur), max_cache_len - prompt_len
// - 1)), and the per-chunk texts joined with whitespace collapsed.
#include "loader.hpp"
#include "lib/capi_helpers.hpp"
#include "mel.hpp"
#include "encoder.hpp"
#include "prompt.hpp"
#include "llm.hpp"
#include "tokenizer.hpp"
#include "runtime/graph.hpp"
#include "runtime/backend.hpp"

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

using GraniteCtx = starling::ggml::lib::EngineContext<starling::ggml::granite::GraniteModel, starling::ggml::granite::Tokenizer>;
using starling::ggml::lib::report;

constexpr double kSampleRate = 16000.0;

// Mirror ModelBackend._decode_budget: scale the decode cap to the clip length.
int32_t decode_budget(const starling::ggml::granite::Config& c, double duration_s) {
    int64_t estimated = (int64_t) std::ceil(duration_s * 5.0) + 32;
    if (estimated < 1) estimated = 1;
    int64_t cap = c.max_new_tokens > 0 ? c.max_new_tokens : 1;
    return (int32_t) std::min(cap, estimated);
}

// One chunk through mel -> encoder+projector -> prompt -> greedy -> text.
bool transcribe_piece(GraniteCtx& ctx, const float* pcm, int64_t n, int32_t budget,
                      std::string& text, double* stage_ms = nullptr) {
    using namespace starling::ggml::granite;
    const GraniteModel& m = *ctx.model;
    auto t0 = std::chrono::steady_clock::now();
    MelFeatures mel;
    if (!compute_log_mel(m.config, m.loader, pcm, (size_t) n, mel, ctx.err))
        return false;
    AudioEmbeds audio;
    if (!encode_audio_and_project(m, mel, audio, ctx.err))
        return false;
    auto t1 = std::chrono::steady_clock::now();
    Prompt prompt = build_transcribe_prompt(m.config, n);
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
    text = ctx.tokenizer.decode(generated.ids, true);
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

// Mirror granite.long_audio._join_chunk_texts with zero overlap: join the
// stripped non-empty texts, then collapse whitespace runs to single spaces.
std::string join_texts(const std::vector<std::string>& texts) {
    std::string joined;
    for (const std::string& t : texts) {
        const char* b = t.c_str();
        const char* e = b + t.size();
        while (b < e && std::isspace((unsigned char) *b)) ++b;
        while (e > b && std::isspace((unsigned char) e[-1])) --e;
        if (e <= b) continue;
        if (!joined.empty()) joined += ' ';
        joined.append(b, (size_t) (e - b));
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

void* starling_ggml_granite_load(const char* gguf_path, const char** err_out) {
    return starling::ggml::lib::load_engine<GraniteCtx>(gguf_path, "GRANITE", err_out);
}

void starling_ggml_granite_free(void* handle) {
    try {
        delete static_cast<GraniteCtx*>(handle);
    } catch (...) {
        // C ABI: never allow an exception to escape.
    }
}

char* starling_ggml_granite_decode(void* handle, const float* pcm, int64_t n,
                                   const char** err_out) {
    auto* c = static_cast<GraniteCtx*>(handle);
    if (!c) { if (err_out) *err_out = "null GRANITE handle"; return nullptr; }
    if (n < 0 || (n > 0 && !pcm)) {
        if (err_out) *err_out = "invalid GRANITE PCM buffer";
        return nullptr;
    }
    try {
        using namespace starling::ggml::granite;
        const Config& cfg = c->model->config;
        const bool timing = std::getenv("STARLING_GRANITE_TIMING") != nullptr;
        auto now = [] { return std::chrono::steady_clock::now(); };
        auto t_start = now();

        // max_chunk = min(chunk_seconds, (max_new_tokens - 32) / 5) — the
        // server's _effective_chunk_seconds(DEFAULT_CHUNK_SECONDS).
        const double token_limited =
            std::max(0.1, ((double) (int) cfg.max_new_tokens - 32.0) / 5.0);
        const double max_chunk_s = std::min(cfg.chunk_seconds, token_limited);
        const int64_t chunk_samples =
            (int64_t) std::llround(max_chunk_s * kSampleRate);
        const double duration_s = (double) n / kSampleRate;

        std::vector<std::string> texts;
        double stage_ms[3] = {0, 0, 0};
        if (chunk_samples <= 0 || n <= chunk_samples) {
            std::string text;
            if (!transcribe_piece(*c, pcm, n, decode_budget(cfg, duration_s), text,
                                  stage_ms))
            {
                report(err_out, c->err);
                return nullptr;
            }
            texts.push_back(std::move(text));
        } else {
            std::vector<float> padded((size_t) chunk_samples, 0.0f);
            for (int64_t start = 0; start < n; start += chunk_samples) {
                const int64_t len = std::min(chunk_samples, n - start);
                // pad_last: the trailing chunk rides on a zero-padded full
                // chunk buffer (identical mel shape; the reference's CUDA-graph
                // encoder required it and the serve path kept it).
                std::memcpy(padded.data(), pcm + start, (size_t) len * sizeof(float));
                if (len < chunk_samples)
                    std::memset(padded.data() + len, 0,
                                (size_t) (chunk_samples - len) * sizeof(float));
                const double piece_s = (double) len / kSampleRate;
                // prompt_len reflects the PADDED chunk (the mel/projector see
                // chunk_samples); the budget cap uses the unpadded duration —
                // exactly the server's ids.shape[1] vs (end - start).
                const int64_t prompt_len = (int64_t) cfg.prompt_prefix.size() +
                                           audio_token_count(chunk_samples, cfg) +
                                           (int64_t) cfg.prompt_suffix.size();
                int32_t budget = decode_budget(cfg, piece_s);
                const int64_t headroom =
                    (int64_t) cfg.llm.max_cache - prompt_len - 1;
                if ((int64_t) budget > headroom) budget = (int32_t) std::max<int64_t>(1, headroom);
                std::string text;
                if (!transcribe_piece(*c, padded.data(), chunk_samples, budget, text,
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
                "GRANITE_STAGE chunks=%zu audio=%.2fs mel+enc+proj=%.1fms "
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
        c->err = "unknown exception transcribing GRANITE audio";
        report(err_out, c->err);
    }
    return nullptr;
}

} // extern "C"
