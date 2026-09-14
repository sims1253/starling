// capi_qwen3.cpp — Qwen3-ASR-1.7B C API entry points behind the shared
// shell. C API flow: load -> (chunk policy) -> per chunk {mel -> encode
// +project -> prompt/embeds -> greedy decode -> detokenize + extract} ->
// join, with a STARLING_QWEN3_TIMING phase-timing gate.
//
// The chunk policy mirrors the Python server path (Qwen3Backend.transcribe
// via ModelBackend._transcribe_chunked) exactly: audio up to
// min(30 s, (max_new_tokens-32)/5) goes through single shot with budget
// min(max_new_tokens, ceil(dur*5)+32); longer audio is cut into
// chunk_seconds waveform chunks — the LAST chunk is passed through SHORT
// (qwen3, unlike granite, does not zero-pad the tail: the mel-level chunk
// padding handles it) — each decoded with budget max(1, min(budget(dur),
// max_cache_len - prompt_len - 1)), and the per-chunk texts joined with
// whitespace collapsed.
//
// The per-chunk text replicates processor.decode(ids,
// return_format="transcription_only"): decode with special tokens skipped,
// then _parse_single_output — strip, cut at "assistant\n" (defensive; only
// present when the model echoes the prompt), the repetition fix from the
// original Qwen3-ASR library, then everything after the first "<asr_text>"
// marker (or the whole text) stripped.
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

using Qwen3Ctx = starling::ggml::lib::EngineContext<starling::ggml::qwen3::Qwen3Model, starling::ggml::qwen3::Tokenizer>;
using starling::ggml::lib::report;

constexpr double kSampleRate = 16000.0;

// Mirror ModelBackend._decode_budget: scale the decode cap to the clip length.
int32_t decode_budget(const starling::ggml::qwen3::Config& c, double duration_s) {
    int64_t estimated = (int64_t) std::ceil(duration_s * 5.0) + 32;
    if (estimated < 1) estimated = 1;
    int64_t cap = c.max_new_tokens > 0 ? c.max_new_tokens : 1;
    return (int32_t) std::min(cap, estimated);
}

// ---- _parse_single_output port (processing_qwen3_asr.py) ------------------

// fix_char_repeats: collapse runs of ONE identical character longer than
// `thresh` to a single copy (byte-level; a >20 run of identical bytes is a
// repeated character in any realistic transcript).
std::string fix_char_repeats(const std::string& s, size_t thresh) {
    std::string res;
    res.reserve(s.size());
    size_t i = 0;
    const size_t n = s.size();
    while (i < n) {
        size_t count = 1;
        while (i + count < n && s[i + count] == s[i]) count++;
        if (count > thresh) {
            res += s[i];
            i += count;
        } else {
            res.append(s, i, count);
            i += count;
        }
    }
    return res;
}

// fix_pattern_repeats: collapse any pattern (1..max_len BYTES vs the Python
// oracle's 20 characters — degenerate multibyte repeats longer than 20 bytes
// would collapse differently; unreachable on realistic transcripts) repeated
// `thresh` or more times consecutively to a single copy, recursing on the
// remainder. Literal port of the Qwen3-ASR library post-processing.
std::string fix_pattern_repeats(const std::string& s, size_t thresh, size_t max_len = 20) {
    const size_t n = s.size();
    const size_t min_repeat_chars = thresh * 2;
    if (n < min_repeat_chars) return s;
    std::string result;
    size_t i = 0;
    bool found = false;
    while (i <= n - min_repeat_chars) {
        found = false;
        for (size_t k = 1; k <= max_len; ++k) {
            if (i + k * thresh > n) break;
            const std::string pattern = s.substr(i, k);
            bool valid = true;
            for (size_t rep = 1; rep < thresh; ++rep) {
                if (s.compare(i + rep * k, k, pattern) != 0) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                size_t end = i + thresh * k;
                while (end + k <= n && s.compare(end, k, pattern) == 0) end += k;
                result += pattern;
                result += fix_pattern_repeats(s.substr(end), thresh, max_len);
                i = n;
                found = true;
                break;
            }
        }
        if (found) break;
        result += s[i];
        i += 1;
    }
    if (!found) result.append(s, i, std::string::npos);
    return result;
}

std::string strip_ws(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace((unsigned char) s[b])) ++b;
    while (e > b && std::isspace((unsigned char) s[e - 1])) --e;
    return s.substr(b, e - b);
}

// extract_transcription: everything after the first "<asr_text>" marker (or
// the whole text when absent), with the reference's strip / assistant-cut /
// repetition-fix preprocessing.
std::string extract_transcription(std::string text) {
    text = strip_ws(text);
    const size_t cut = text.find("assistant\n");
    if (cut != std::string::npos)
        text = text.substr(cut + strlen("assistant\n"));
    text = fix_char_repeats(text, 20);
    text = fix_pattern_repeats(text, 20);
    const size_t marker = text.find("<asr_text>");
    if (marker != std::string::npos)
        return strip_ws(text.substr(marker + strlen("<asr_text>")));
    return strip_ws(text);
}

// One chunk through mel -> encoder+projector -> prompt -> greedy -> text.
bool transcribe_piece(Qwen3Ctx& ctx, const float* pcm, int64_t n, int32_t budget,
                      std::string& text, double* stage_ms = nullptr) {
    using namespace starling::ggml::qwen3;
    const Qwen3Model& m = *ctx.model;
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
    text = extract_transcription(ctx.tokenizer.decode(generated.ids, true));
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

void* starling_ggml_qwen3_load(const char* gguf_path, const char** err_out) {
    return starling::ggml::lib::load_engine<Qwen3Ctx>(gguf_path, "QWEN3", err_out);
}

void starling_ggml_qwen3_free(void* handle) {
    try {
        delete static_cast<Qwen3Ctx*>(handle);
    } catch (...) {
        // C ABI: never allow an exception to escape.
    }
}

char* starling_ggml_qwen3_decode(void* handle, const float* pcm, int64_t n,
                                 const char** err_out) {
    auto* c = static_cast<Qwen3Ctx*>(handle);
    if (!c) { if (err_out) *err_out = "null QWEN3 handle"; return nullptr; }
    if (n < 0 || (n > 0 && !pcm)) {
        if (err_out) *err_out = "invalid QWEN3 PCM buffer";
        return nullptr;
    }
    try {
        using namespace starling::ggml::qwen3;
        const Config& cfg = c->model->config;
        const bool timing = std::getenv("STARLING_QWEN3_TIMING") != nullptr;
        auto now = [] { return std::chrono::steady_clock::now(); };
        auto t_start = now();

        // max_chunk = min(chunk_seconds, (max_new_tokens - 32) / 5) — the
        // server's _effective_chunk_seconds(DEFAULT_MAX_CHUNK_SECONDS).
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
            for (int64_t start = 0; start < n; start += chunk_samples) {
                const int64_t len = std::min(chunk_samples, n - start);
                const double piece_s = (double) len / kSampleRate;
                // The budget cap keeps the greedy loop inside the static KV
                // cache (prompt + new tokens <= max_cache_len) exactly like
                // the Python LLMMega's max_new_tokens guard.
                const int64_t prompt_len = (int64_t) cfg.prompt_prefix.size() +
                                           audio_token_count(len, cfg) +
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
                "QWEN3_STAGE chunks=%zu audio=%.2fs mel+enc+proj=%.1fms "
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
        c->err = "unknown exception transcribing QWEN3 audio";
        report(err_out, c->err);
    }
    return nullptr;
}

} // extern "C"
