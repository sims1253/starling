// Real optional integration check and full-cost CPU pilot:
//   granite_ctc_verify_test ctc.gguf public.wav [paired_repeats]
// Warm greedy and K=1,2,4 once, then pair K=2 and K=4 against greedy in
// both orders (odd repetition greedy first; even repetition candidate first).
#include "granite/speculative.hpp"
#include "granite/prompt.hpp"
#include "runtime/audio_io.hpp"
#include "runtime/backend.hpp"
#include "runtime/graph.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
using namespace starling::ggml;
using namespace starling::ggml::granite;
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
struct Run {
    GenerateResult output;
    CtcSpeculativeStats spec;
    double mel_ms = 0, encoder_ms = 0, embed_ms = 0, decode_ms = 0, total_ms = 0;
};
bool execute(const GraniteModel& model, const std::vector<float>& pcm,
             int k, Run& run, std::string& err) {
    auto t0 = Clock::now();
    MelFeatures mel;
    if (!compute_log_mel(model.config, model.loader, pcm.data(), pcm.size(), mel, err)) return false;
    auto t1 = Clock::now();
    const double seconds = (double)pcm.size() / 16000.0;
    const int32_t budget = std::min<int32_t>(model.config.max_new_tokens,
                                            (int32_t)std::ceil(seconds * 5.0) + 32);
    GenerateOptions op;
    op.max_new_tokens = budget;
    op.max_cache_len = model.config.llm.max_cache;
    op.eos_token_id = model.config.eos_token_id;
    if (k > 0) {
        if (!ctc_speculative_generate(model, mel, (int64_t)pcm.size(), op, k, {},
                                      run.output, run.spec, err)) return false;
        run.mel_ms = ms(t0, t1);
        run.encoder_ms = run.spec.encoder_project_ms;
        run.embed_ms = run.spec.embed_ms;
        run.decode_ms = run.spec.verifier.total_ms;
        run.total_ms = ms(t0, Clock::now());
        return true;
    }
    AudioEmbeds audio;
    if (!encode_audio_and_project(model, mel, audio, err)) return false;
    auto t2 = Clock::now();
    const Prompt prompt = build_transcribe_prompt(model.config, (int64_t)pcm.size());
    InputsEmbeds inputs;
    if (!build_inputs_embeds(model, prompt, audio, inputs, err)) return false;
    auto t3 = Clock::now();
    const bool ok = greedy_generate(model, inputs, op, run.output, err);
    auto t4 = Clock::now();
    if (!ok) return false;
    run.mel_ms = ms(t0, t1);
    run.encoder_ms = ms(t1, t2);
    run.embed_ms = ms(t2, t3);
    run.decode_ms = ms(t3, t4);
    run.total_ms = ms(t0, t4);
    return true;
}
void report(const char* phase, int k, int rep, const Run& r, bool parity) {
    const auto& s = r.spec.verifier;
    std::printf("RUN phase=%s k=%d rep=%d parity=%d tokens=%zu stop=%d "
                "mel_ms=%.3f encoder_project_ms=%.3f embed_ms=%.3f "
                "ctc_head_ms=%.3f proposal_ms=%.3f prefill_ms=%.3f "
                "verify_ms=%.3f fallback_ms=%.3f decode_ms=%.3f full_ms=%.3f "
                "draft=%zu proposed=%d accepted=%d verify_calls=%d fallback_steps=%d\n",
                phase, k, rep, parity, r.output.ids.size(), (int)r.output.stop_reason,
                r.mel_ms, r.encoder_ms, r.embed_ms, r.spec.ctc_head_ms,
                s.proposal_ms, s.prefill_ms, s.verify_ms, s.fallback_ms,
                r.decode_ms, r.total_ms, r.spec.draft_count,
                s.proposed, s.accepted, s.verify_calls, s.fallback_steps);
    std::fflush(stdout);
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        std::fprintf(stderr, "usage: %s ctc.gguf public.wav [paired_repeats=2]\n", argv[0]);
        return 2;
    }
    const int repeats = argc == 4 ? std::atoi(argv[3]) : 2;
    if (repeats < 0 || repeats > 10) return 2;
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    GraniteModel model;
    std::string err;
    if (!model.load(argv[1], err)) {
        std::fprintf(stderr, "model: %s\n", err.c_str());
        return 2;
    }
    std::vector<float> pcm;
    int sample_rate = 0;
    if (!read_wav(argv[2], pcm, sample_rate, err) || sample_rate != 16000) {
        std::fprintf(stderr, "wav: %s (sample rate=%d)\n", err.c_str(), sample_rate);
        return 2;
    }
    std::printf("CASE backend=%s pcm_samples=%zu seconds=%.3f repeats=%d\n",
                global_backend().device_name(), pcm.size(), pcm.size()/16000.0, repeats);
    Run reference;
    if (!execute(model, pcm, 0, reference, err)) {
        std::fprintf(stderr, "warm greedy: %s\n", err.c_str());
        return 2;
    }
    report("warm", 0, 0, reference, true);
    std::printf("GREEDY_IDS");
    for (int32_t id : reference.output.ids) std::printf(" %d", id);
    std::printf("\n");
    bool all_match = true;
    const auto one = [&](const char* phase, int k, int rep) {
        Run r;
        if (!execute(model, pcm, k, r, err)) {
            std::fprintf(stderr, "%s k=%d rep=%d: %s\n", phase, k, rep, err.c_str());
            return false;
        }
        const bool parity = r.output.ids == reference.output.ids &&
                            r.output.stop_reason == reference.output.stop_reason;
        report(phase, k, rep, r, parity);
        if (!parity) {
            std::printf("MISMATCH_IDS");
            for (int32_t id : r.output.ids) std::printf(" %d", id);
            std::printf("\n");
        }
        all_match &= parity;
        return true;
    };
    for (int k : {1, 2, 4}) if (!one("warm", k, 0)) return 2;
    for (int rep = 1; rep <= repeats; ++rep) {
        for (int k : {2, 4}) {
            if (rep % 2 == 1) {
                if (!one("paired", 0, rep) || !one("paired", k, rep)) return 2;
            } else {
                if (!one("paired", k, rep) || !one("paired", 0, rep)) return 2;
            }
        }
    }
    return all_match ? 0 : 1;
}
