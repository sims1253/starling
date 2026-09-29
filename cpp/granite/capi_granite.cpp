// capi_granite.cpp — granite-speech-4.1-2b C API entry points behind the
// shared shell. C API flow: load -> (chunk policy) -> per chunk {mel -> encode
// +project -> prompt/embeds -> greedy decode -> detokenize} -> join, with a
// STARLING_GRANITE_TIMING phase-timing gate (per-chunk summaries plus a
// whole-request aggregate; see stage_timing.hpp).
//
// The chunk policy mirrors the Python server path (GraniteBackend.transcribe)
// exactly: audio up to min(30 s, (max_new_tokens-32)/5) goes through single
// shot with budget min(max_new_tokens, ceil(dur*5)+32); longer audio is cut
// into chunk_seconds waveform chunks — the LAST chunk zero-padded to the full
// chunk length, exactly what chunk_audio(pad_last=True) feeds the processor —
// each decoded with budget max(1, min(budget(dur), max_cache_len - prompt_len
// - 1)), and the per-chunk texts joined with whitespace collapsed.
#include "loader.hpp"
#include "chunk_policy.hpp"
#include "starling_ggml.h"
#include "lib/capi_helpers.hpp"
#include "lib/granite_job_internal.hpp"
#include "mel.hpp"
#include "encoder.hpp"
#include "prompt.hpp"
#include "llm.hpp"
#include "speculative.hpp"
#include "stage_timing.hpp"
#include "tokenizer.hpp"
#include "runtime/graph.hpp"
#include "runtime/backend.hpp"
#include "runtime/trace.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <climits>
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

constexpr double kSampleRate = starling::ggml::granite::kChunkSampleRate;
using starling::ggml::granite::decode_budget;

// One chunk through mel -> encoder/projector -> prompt -> decode -> text.
// stage_ms (optional) receives THIS chunk's three stage durations — the
// caller accumulates them across chunks via StageTiming (stage_timing.hpp).
bool transcribe_piece(GraniteCtx& ctx, const float* pcm, int64_t n, int32_t budget,
                      std::string& text, double* stage_ms = nullptr,
                      int ctc_max_k = 0, int chunk_index = 0) {
    using namespace starling::ggml::granite;
    const GraniteModel& m = *ctx.model;
    auto t0 = std::chrono::steady_clock::now();
    MelFeatures mel;
    if (!compute_log_mel(m.config, m.loader, pcm, (size_t) n, mel, ctx.err))
        return false;
    auto t_mel = std::chrono::steady_clock::now();
    GenerateOptions options;
    options.max_new_tokens = budget;
    options.max_cache_len = (int32_t) m.config.llm.max_cache;
    options.eos_token_id = m.config.eos_token_id;
    GenerateResult generated;
    if (ctc_max_k > 0) {
        CtcSpeculativeStats spec;
        if (!ctc_speculative_generate(m, mel, n, options, ctc_max_k, {},
                                      generated, spec, ctx.err)) return false;
        const auto t3 = std::chrono::steady_clock::now();
        text = ctx.tokenizer.decode(generated.ids, true);
        const auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        const double mel_ms = ms(t0, t_mel);
        const double enc_ctc_ms = spec.encoder_project_ms + spec.ctc_head_ms;
        if (stage_ms) {
            stage_ms[0] = mel_ms + enc_ctc_ms;
            stage_ms[1] = spec.embed_ms;
            stage_ms[2] = spec.verifier.total_ms;
        }
        if (std::getenv("STARLING_GRANITE_TIMING")) {
            std::fprintf(stderr,
                "GRANITE_CTC chunk=%d k=%d draft=%zu accepted=%d proposed=%d "
                "verify_calls=%d fallback_steps=%d mel=%.3fms enc_proj=%.3fms "
                "ctc_head=%.3fms embed=%.3fms proposal=%.3fms prefill=%.3fms "
                "verify=%.3fms fallback=%.3fms piece=%.3fms\n",
                chunk_index, ctc_max_k, spec.draft_count, spec.verifier.accepted,
                spec.verifier.proposed, spec.verifier.verify_calls,
                spec.verifier.fallback_steps, mel_ms, spec.encoder_project_ms,
                spec.ctc_head_ms, spec.embed_ms, spec.verifier.proposal_ms,
                spec.verifier.prefill_ms, spec.verifier.verify_ms,
                spec.verifier.fallback_ms, ms(t0, t3));
        }
        if (starling::ggml::trace::on()) {
            starling::ggml::trace::stage_event("mel_enc_proj_ctc", mel_ms + enc_ctc_ms);
            starling::ggml::trace::stage_event("prompt_embeds", spec.embed_ms);
            starling::ggml::trace::stage_event("generate", spec.verifier.total_ms);
        }
        return true;
    }
    AudioEmbeds audio;
    if (!encode_audio_and_project(m, mel, audio, ctx.err))
        return false;
    auto t1 = std::chrono::steady_clock::now();
    Prompt prompt = build_transcribe_prompt(m.config, n);
    InputsEmbeds inputs;
    if (!build_inputs_embeds(m, prompt, audio, inputs, ctx.err))
        return false;
    auto t2 = std::chrono::steady_clock::now();
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
    if (starling::ggml::trace::on()) {
        // Same clocks as stage_ms (single source: the trace and the
        // STARLING_GRANITE_TIMING lines cannot disagree). Chunk index comes
        // from the enclosing ChunkScope's thread-local correlation.
        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        starling::ggml::trace::stage_event("mel_enc_proj", ms(t0, t1));
        starling::ggml::trace::stage_event("prompt_embeds", ms(t1, t2));
        starling::ggml::trace::stage_event("generate", ms(t2, t3));
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

namespace starling::ggml::lib {

// Borrows PCM and model for one synchronous request; the caller keeps both
// alive and steps only while holding the global runtime lock.
struct GraniteChunkJob {
    GraniteCtx* ctx;
    const granite::Config policy;
    const float* pcm;
    int64_t n;
    int64_t chunk_samples;
    int64_t offset = 0;
    bool multi;
    bool finished = false;
    bool timing;
    std::vector<float> padded;
    std::vector<std::string> texts;
    granite::StageTiming stages;
    double active_ms = 0.0;

    int ctc_max_k;  // 0 = greedy; 1..16 = opt-in CTC speculative decode

    GraniteChunkJob(GraniteCtx* c, const float* p, int64_t count, int ctc_k = 0)
        : ctx(c), policy(c->model->config), pcm(p), n(count),
          chunk_samples(granite::effective_chunk_samples(policy)),
          multi(chunk_samples > 0 && count > chunk_samples),
          timing(std::getenv("STARLING_GRANITE_TIMING") != nullptr),
          ctc_max_k(ctc_k) {
        if (multi) padded.resize((size_t) chunk_samples);
    }

    bool last_chunk() const { return !multi || n - offset <= chunk_samples; }

    int step(std::string* final_text, const char** err) {
        if (finished) { ctx->err = "GRANITE job already completed"; report(err, ctx->err); return -1; }
        const auto step_start = std::chrono::steady_clock::now();
        const auto& cfg = policy;
        const float* piece = pcm;
        int64_t piece_n = n;
        int64_t len = n;
        int32_t budget = multi ? 0 : decode_budget(cfg, (double) n / kSampleRate);
        if (multi) {
            len = std::min(chunk_samples, n - offset);
            std::memcpy(padded.data(), pcm + offset, (size_t) len * sizeof(float));
            if (len < chunk_samples)
                std::memset(padded.data() + len, 0,
                            (size_t) (chunk_samples - len) * sizeof(float));
            piece = padded.data();
            piece_n = chunk_samples;
            const int64_t prompt_len = (int64_t) cfg.prompt_prefix.size() +
                granite::audio_token_count(chunk_samples, cfg) +
                (int64_t) cfg.prompt_suffix.size();
            budget = decode_budget(cfg, (double) len / kSampleRate, prompt_len);
        }
        double piece_ms[granite::kStageCount] = {};
        std::string text;
        const bool tr_on = starling::ggml::trace::on();
        const auto t0 = std::chrono::steady_clock::now();
        bool ok;
        {
            starling::ggml::trace::ChunkScope scope(stages.chunks + 1);
            ok = transcribe_piece(*ctx, piece, piece_n, budget, text, piece_ms,
                                  ctc_max_k, stages.chunks + 1);
        }
        if (tr_on && ok)
            starling::ggml::trace::chunk_event(stages.chunks + 1,
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count());
        if (!ok) {
            finished = true;  // A failed job cannot be resumed safely.
            texts.clear();
            report(err, ctx->err);
            return -1;
        }
        texts.push_back(std::move(text));
        stages.add_chunk(piece_ms);
        if (timing)
            std::fprintf(stderr, "%s\n",
                granite::format_stage_chunk_line(stages, stages.chunks).c_str());
        offset += len;
        if (multi && offset < n) {
            active_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - step_start).count();
            return 0;
        }
        finished = true;
        if (final_text) *final_text = join_texts(texts);
        active_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - step_start).count();
        if (timing)
            std::fprintf(stderr, "%s\n",
                granite::format_stage_request_line(stages, (double) n / kSampleRate,
                    active_ms).c_str());
        if (err) *err = nullptr;
        return 1;
    }
};

GraniteChunkJob* granite_job_create_impl(void* model, const float* pcm,
                                          int64_t n, const char** err,
                                          int ctc_max_k) {
    // Creation runs before the caller holds the runtime lock, so it must not
    // write the shared ctx->err. The message stays valid on this thread until
    // the next failed creation; callers copy it immediately.
    static thread_local char create_error[2048];
    auto* ctx = static_cast<GraniteCtx*>(model);
    if (!ctx) { report(err, "null GRANITE handle"); return nullptr; }
    if (n < 0 || (n > 0 && !pcm)) { report(err, "invalid GRANITE PCM buffer"); return nullptr; }
    try { return new GraniteChunkJob(ctx, pcm, n, ctc_max_k); }
    catch (const std::exception& e) {
        std::snprintf(create_error, sizeof(create_error), "%s", e.what());
    }
    catch (...) {
        std::snprintf(create_error, sizeof(create_error), "%s",
                      "unknown exception creating GRANITE job");
    }
    report(err, create_error);
    return nullptr;
}

int granite_job_step_impl(GraniteChunkJob* job, std::string* text,
                          const char** err) {
    if (!job) { report(err, "null GRANITE job"); return -1; }
    try { return job->step(text, err); }
    catch (const std::exception& e) { job->ctx->err = e.what(); }
    catch (...) { job->ctx->err = "unknown exception stepping GRANITE job"; }
    job->finished = true;
    job->texts.clear();
    report(err, job->ctx->err);
    return -1;
}

bool granite_job_last_chunk_impl(const GraniteChunkJob* job) {
    return job && job->last_chunk();
}

void granite_job_free_impl(GraniteChunkJob* job) { delete job; }

} // namespace starling::ggml::lib

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

// Opt-in parity probe, separate from the ordinary greedy decode entry point.
// Caller supplies a token buffer; count receives the required length when it
// is too small. This C symbol is intentionally model-specific research API.
bool starling_ggml_granite_ctc_draft(void* handle, const float* pcm, int64_t n,
                                    int32_t* token_ids, int32_t capacity,
                                    int32_t* count, const char** err_out) {
    auto* c = static_cast<GraniteCtx*>(handle);
    if (!c || !count || n <= 0 || !pcm || capacity < 0 ||
        (capacity > 0 && !token_ids)) {
        if (count) *count = 0;
        if (c) {
            c->err = "invalid GRANITE CTC draft arguments";
            report(err_out, c->err);
        } else if (err_out) {
            *err_out = "invalid GRANITE CTC draft arguments";
        }
        return false;
    }
    *count = 0;
    try {
        using namespace starling::ggml::granite;
        MelFeatures mel;
        if (!compute_log_mel(c->model->config, c->model->loader, pcm,
                             (size_t)n, mel, c->err)) {
            report(err_out, c->err);
            return false;
        }
        std::vector<int32_t> ids;
        if (!extract_ctc_draft(*c->model, mel, ids, c->err)) {
            report(err_out, c->err);
            return false;
        }
        if (ids.size() > (size_t)INT32_MAX) {
            c->err = "GRANITE CTC draft token count exceeds INT32_MAX";
            report(err_out, c->err);
            return false;
        }
        *count = (int32_t)ids.size();
        if ((size_t)capacity < ids.size()) {
            c->err = "GRANITE CTC draft token buffer is too small";
            report(err_out, c->err);
            return false;
        }
        std::copy(ids.begin(), ids.end(), token_ids);
        if (err_out) *err_out = nullptr;
        return true;
    } catch (const std::exception& e) {
        c->err = e.what();
        report(err_out, c->err);
    } catch (...) {
        c->err = "unknown GRANITE CTC draft failure";
        report(err_out, c->err);
    }
    return false;
}

static char* granite_decode_impl(void* handle, const float* pcm, int64_t n,
                                 int ctc_max_k, const char** err_out) {
    using namespace starling::ggml::lib;
    std::unique_ptr<GraniteChunkJob, decltype(&granite_job_free_impl)> job(
        granite_job_create_impl(handle, pcm, n, err_out, ctc_max_k),
        &granite_job_free_impl);
    if (!job) return nullptr;
    std::string text;
    int status;
    do {
        status = granite_job_step_impl(job.get(), &text, err_out);
    } while (status == 0);
    if (status < 0) return nullptr;
    char* out = static_cast<char*>(std::malloc(text.size() + 1));
    if (!out) { if (err_out) *err_out = "malloc failed"; return nullptr; }
    std::memcpy(out, text.data(), text.size());
    out[text.size()] = '\0';
    if (err_out) *err_out = nullptr;
    return out;
}

char* starling_ggml_granite_decode(void* handle, const float* pcm, int64_t n,
                                   const char** err_out) {
    return granite_decode_impl(handle, pcm, n, 0, err_out);
}

char* starling_ggml_granite_decode_ctc(void* handle, const float* pcm, int64_t n,
                                       int32_t max_k, const char** err_out) {
    if (max_k < 1 || max_k > 16) {
        if (err_out) *err_out = "GRANITE CTC maximum proposal length must be 1..16";
        return nullptr;
    }
    return granite_decode_impl(handle, pcm, n, max_k, err_out);
}

} // extern "C"
