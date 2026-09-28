// Optional real long-audio parity: apply the same padded chunk/budget policy
// as capi_granite.cpp, then compare each fused CTC verification with greedy.
#include "granite/speculative.hpp"
#include "granite/prompt.hpp"
#include "runtime/audio_io.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s ctc.gguf long-public.wav\n", argv[0]);
        return 2;
    }
    using namespace starling::ggml;
    using namespace starling::ggml::granite;
    using Clock = std::chrono::steady_clock;
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
    const auto& cfg = model.config;
    const double limited = std::max(0.1, ((double)(int)cfg.max_new_tokens - 32.0) / 5.0);
    const int64_t chunk_samples = (int64_t)std::llround(
        std::min(cfg.chunk_seconds, limited) * 16000.0);
    if ((int64_t)pcm.size() <= chunk_samples) {
        std::fprintf(stderr, "audio must exceed one policy chunk (%lld samples)\n",
                     (long long)chunk_samples);
        return 2;
    }
    std::vector<float> padded((size_t)chunk_samples);
    bool all_equal = true;
    int chunk = 0;
    for (int64_t start = 0; start < (int64_t)pcm.size(); start += chunk_samples) {
        ++chunk;
        const int64_t len = std::min(chunk_samples, (int64_t)pcm.size() - start);
        std::memcpy(padded.data(), pcm.data() + start, (size_t)len * sizeof(float));
        std::fill(padded.begin() + len, padded.end(), 0.0f);
        const double seconds = (double)len / 16000.0;
        const int64_t prompt_len = (int64_t)cfg.prompt_prefix.size() +
            audio_token_count(chunk_samples, cfg) + (int64_t)cfg.prompt_suffix.size();
        int32_t budget = std::min<int32_t>(cfg.max_new_tokens,
                                          (int32_t)std::ceil(seconds * 5.0) + 32);
        const int64_t headroom = (int64_t)cfg.llm.max_cache - prompt_len - 1;
        budget = (int32_t)std::min<int64_t>(budget, std::max<int64_t>(1, headroom));
        const auto t0 = Clock::now();
        MelFeatures mel;
        if (!compute_log_mel(cfg, model.loader, padded.data(), padded.size(), mel, err)) {
            std::fprintf(stderr, "chunk %d mel: %s\n", chunk, err.c_str());
            return 2;
        }
        AudioEmbeds audio;
        if (!encode_audio_and_project(model, mel, audio, err)) {
            std::fprintf(stderr, "chunk %d audio: %s\n", chunk, err.c_str());
            return 2;
        }
        InputsEmbeds inputs;
        if (!build_inputs_embeds(model, build_transcribe_prompt(cfg, chunk_samples),
                                 audio, inputs, err)) {
            std::fprintf(stderr, "chunk %d embed: %s\n", chunk, err.c_str());
            return 2;
        }
        GenerateOptions op;
        op.max_new_tokens = budget;
        op.max_cache_len = cfg.llm.max_cache;
        op.eos_token_id = cfg.eos_token_id;
        GenerateResult greedy, speculative;
        if (!greedy_generate(model, inputs, op, greedy, err)) {
            std::fprintf(stderr, "chunk %d greedy: %s\n", chunk, err.c_str());
            return 2;
        }
        const auto t1 = Clock::now();
        CtcSpeculativeStats stats;
        if (!ctc_speculative_generate(model, mel, chunk_samples, op, 4, {},
                                      speculative, stats, err)) {
            std::fprintf(stderr, "chunk %d CTC: %s\n", chunk, err.c_str());
            return 2;
        }
        const auto t2 = Clock::now();
        const bool equal = greedy.ids == speculative.ids &&
                           greedy.stop_reason == speculative.stop_reason;
        all_equal &= equal;
        const auto elapsed = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        // Greedy includes mel from t0. CTC reuses that same mel after t1;
        // these diagnostic clocks are not a paired full-request comparison.
        std::printf("CHUNK index=%d real_samples=%lld padded_samples=%lld "
                    "budget=%d parity=%d tokens=%zu stop=%d draft=%zu "
                    "accepted=%d proposed=%d greedy_from_pcm_ms=%.3f "
                    "ctc_enc_project_ms=%.3f ctc_head_ms=%.3f "
                    "ctc_verify_ms=%.3f ctc_from_reused_mel_ms=%.3f\n",
                    chunk, (long long)len, (long long)chunk_samples, budget,
                    equal, greedy.ids.size(), (int)greedy.stop_reason,
                    stats.draft_count, stats.verifier.accepted,
                    stats.verifier.proposed, elapsed(t0, t1),
                    stats.encoder_project_ms, stats.ctc_head_ms,
                    stats.verifier.total_ms, elapsed(t1, t2));
        std::fflush(stdout);
    }
    return all_equal ? 0 : 1;
}
