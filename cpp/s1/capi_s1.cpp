// capi_s1.cpp — S1-mini C API entry points behind the shared shell.
//
// S1-mini is text-in/text-out: there is no PCM path. The engine exposes the
// standard load/free trio plus a normalize entry point (registered as the
// registry row's normalize_fn; the PCM decode entry is a stub that explains
// the text-only contract). Flow:
//
//   normalize(transcript, styling, structure, context)
//     -> validate controls against the trained value sets
//     -> build user content "[Styling: s] [Structure: t] [Context: c]\n..."
//     -> BPE encode (lib::BpeTokenizer::encode)
//     -> prefix + content + suffix (chat template baked in the GGUF,
//        including the enable_thinking=False assistant prefix)
//     -> embed lookup (llm.embed.weight) -> shared Qwen-trunk greedy decode
//        stopping on <|im_end|> OR <|endoftext|>
//     -> detokenize with special tokens skipped.
//
// The greedy budget mirrors the model card + the Python pipeline:
// min(1.3 * prompt_len + 32, max_cache_len - prompt_len - 1).
#include "loader.hpp"
#include "llm.hpp"
#include "lib/bpe_tokenizer.hpp"
#include "lib/embed_scatter.hpp"
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

struct S1Ctx {
    std::unique_ptr<starling::ggml::s1::S1Model> model;
    starling::ggml::lib::BpeTokenizer tokenizer;
    std::string err;
};

void report(const char** out, const std::string& message) {
    if (out) *out = message.c_str();
}

void report_load_error(const char** out, const std::string& message) {
    g_load_error = message;
    if (out) *out = g_load_error.c_str();
}

bool contains(const std::vector<std::string>& values, const char* v) {
    return std::find(values.begin(), values.end(), std::string(v ? v : "")) != values.end();
}

} // namespace

extern "C" {

void* starling_ggml_s1_load(const char* gguf_path, const char** err_out) {
    try {
        if (!gguf_path || !*gguf_path) {
            if (err_out) *err_out = "null or empty S1 GGUF path";
            return nullptr;
        }
        auto ctx = std::make_unique<S1Ctx>();
        ctx->model = std::make_unique<starling::ggml::s1::S1Model>();
        if (!ctx->model->load(gguf_path, ctx->err)) {
            report_load_error(err_out, ctx->err);
            return nullptr;
        }
        if (!ctx->tokenizer.load(ctx->model->loader, ctx->model->config, ctx->err)) {
            report_load_error(err_out, ctx->err);
            return nullptr;
        }
        // Encoder self-test: the tokenizer must round-trip an ASCII probe
        // (a GGUF written without merges disables encode — fail at load,
        // not mid-request).
        {
            std::vector<int32_t> probe_ids;
            std::string probe_err;
            if (!ctx->tokenizer.encode("normalizer self test 123", probe_ids, probe_err)) {
                report_load_error(err_out, "S1 tokenizer cannot encode: " + probe_err);
                return nullptr;
            }
            const std::string back = ctx->tokenizer.decode(probe_ids, true);
            if (back != "normalizer self test 123") {
                report_load_error(err_out, "S1 tokenizer round-trip mismatch: " + back);
                return nullptr;
            }
        }
        // Persist weights + force backend creation (and its orderly atexit
        // shutdown registration) across all normalization calls.
        ctx->model->loader.realize_weights(starling::ggml::global_backend());
        starling::ggml::register_decode_cache_clearer([]() {});
        if (err_out) *err_out = nullptr;
        return ctx.release();
    } catch (const std::exception& e) {
        report_load_error(err_out, e.what());
    } catch (...) {
        report_load_error(err_out, "unknown exception loading S1 model");
    }
    return nullptr;
}

void starling_ggml_s1_free(void* handle) {
    try {
        delete static_cast<S1Ctx*>(handle);
    } catch (...) {
        // C ABI: never allow an exception to escape.
    }
}

char* starling_ggml_s1_decode(void* handle, const float* /*pcm*/, int64_t /*n*/,
                              const char** err_out) {
    (void) handle;
    // report() stores a c_str(); keep the message in a static storage
    // duration string so the pointer outlives the call.
    static const std::string kTextOnly =
        "S1 is a text normalizer (no audio path); use "
        "starling_ggml_normalize_text / POST /normalize";
    if (err_out) *err_out = kTextOnly.c_str();
    return nullptr;
}

char* starling_ggml_s1_normalize(void* handle, const char* transcript,
                                 const char* styling, const char* structure,
                                 const char* context, const char** err_out) {
    auto* c = static_cast<S1Ctx*>(handle);
    if (!c) { if (err_out) *err_out = "null S1 handle"; return nullptr; }
    if (!transcript) { if (err_out) *err_out = "null S1 transcript"; return nullptr; }
    try {
        using namespace starling::ggml::s1;
        const Config& cfg = c->model->config;
        const bool timing = std::getenv("STARLING_S1_TIMING") != nullptr;
        auto t_start = std::chrono::steady_clock::now();

        // (1) control line: validated against the trained value sets. Every
        // report() below passes c->err (ctx-owned) — a temporary would
        // dangle through the const char** out.
        if (styling && !contains(cfg.styling_values, styling)) {
            c->err = std::string("S1 unknown styling '") + styling + "'";
            report(err_out, c->err);
            return nullptr;
        }
        if (structure && !contains(cfg.structure_values, structure)) {
            c->err = std::string("S1 unknown structure '") + structure + "'";
            report(err_out, c->err);
            return nullptr;
        }
        if (context && !contains(cfg.context_values, context)) {
            c->err = std::string("S1 unknown context '") + context + "'";
            report(err_out, c->err);
            return nullptr;
        }
        const std::string s = styling ? styling : "semi-formal";
        const std::string st = structure ? structure : "prose";
        const std::string cx = context ? context : "general";
        const std::string content =
            "[Styling: " + s + "] [Structure: " + st + "] [Context: " + cx + "]\n" +
            transcript;

        // (2) BPE encode + chat template.
        std::vector<int32_t> content_ids;
        if (!c->tokenizer.encode(content, content_ids, c->err)) {
            report(err_out, c->err);
            return nullptr;
        }
        std::vector<int32_t> ids = cfg.prompt_prefix;
        ids.insert(ids.end(), content_ids.begin(), content_ids.end());
        ids.insert(ids.end(), cfg.prompt_suffix.begin(), cfg.prompt_suffix.end());
        const int64_t T = (int64_t) ids.size();
        if (T > cfg.max_input_tokens) {
            char msg[160];
            std::snprintf(msg, sizeof msg,
                          "S1 prompt is %lld tokens > trained max %d; chunk the transcript",
                          (long long) T, (int) cfg.max_input_tokens);
            c->err = msg;
            report(err_out, c->err);
            return nullptr;
        }

        // (3) embedding lookup (pure llm.embed.weight rows).
        std::vector<float> emb;
        {
            std::vector<uint8_t> no_mask(ids.size(), 0);
            if (!starling::ggml::lib::embed_and_scatter_audio(
                    c->model->loader, cfg.llm.hidden, ids, no_mask,
                    nullptr, 0, cfg.llm.hidden, 0, emb, "S1", c->err)) {
                report(err_out, c->err);
                return nullptr;
            }
        }

        // (4) greedy decode: budget 1.3*T + 32, capped by the static cache.
        int64_t budget = (int64_t) std::llround(
                             cfg.max_new_tokens_input_factor * (double) T) +
                         (int64_t) cfg.max_new_tokens_fixed;
        const int64_t headroom = (int64_t) cfg.llm.max_cache - T - 1;
        if (budget > headroom) budget = headroom;
        if (budget < 1) budget = 1;
        starling::ggml::lib::InputsEmbeds in;
        in.data = std::move(emb);
        in.n_tokens = T;
        in.width = cfg.llm.hidden;
        GenerateOptions op;
        op.max_new_tokens = (int32_t) budget;
        op.max_cache_len = (int32_t) cfg.llm.max_cache;
        op.eos_token_id = cfg.eos_token_id;
        op.eos2_token_id = cfg.eos2_token_id;
        starling::ggml::lib::GenerateResult res;
        if (!greedy_generate(*c->model, in, op, res, c->err)) {
            report(err_out, c->err);
            return nullptr;
        }

        // (5) detokenize (skip specials: drops the stop token).
        const std::string text = c->tokenizer.decode(res.ids, true);
        if (timing) {
            auto t_end = std::chrono::steady_clock::now();
            std::fprintf(stderr,
                         "S1_TIMING prompt=%lldtok gen=%zutok total=%.1fms\n",
                         (long long) T, res.ids.size(),
                         std::chrono::duration<double, std::milli>(t_end - t_start).count());
        }

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
        c->err = "unknown exception in S1 normalize";
        report(err_out, c->err);
    }
    return nullptr;
}

} // extern "C"
