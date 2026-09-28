// Real optional parity check for the shared encoder output. The old separate
// encoder/projector and CTC extraction are the independently verified paths.
#include "granite/encoder.hpp"
#include "granite/speculative.hpp"
#include "runtime/audio_io.hpp"

#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s ctc.gguf public.wav\n", argv[0]);
        return 2;
    }
    using namespace starling::ggml;
    using namespace starling::ggml::granite;
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
    MelFeatures mel;
    if (!compute_log_mel(model.config, model.loader, pcm.data(), pcm.size(), mel, err)) {
        std::fprintf(stderr, "mel: %s\n", err.c_str());
        return 2;
    }
    AudioEmbeds separate_audio, fused_audio;
    std::vector<int32_t> separate_ids, fused_ids;
    if (!encode_audio_and_project(model, mel, separate_audio, err) ||
        !extract_ctc_draft(model, mel, separate_ids, err) ||
        !encode_audio_project_and_extract_ctc(model, mel, fused_audio, fused_ids, err)) {
        std::fprintf(stderr, "encoder/CTC: %s\n", err.c_str());
        return 2;
    }
    const bool audio_equal = separate_audio.n_tokens == fused_audio.n_tokens &&
                             separate_audio.width == fused_audio.width &&
                             separate_audio.data == fused_audio.data;
    const bool ids_equal = separate_ids == fused_ids;
    std::printf("FUSED_PARITY audio_bitwise=%d draft_exact=%d audio_values=%zu "
                "draft_tokens=%zu\n", audio_equal, ids_equal, fused_audio.data.size(),
                fused_ids.size());
    if (!ids_equal) {
        std::printf("SEPARATE_IDS");
        for (int32_t id : separate_ids) std::printf(" %d", id);
        std::printf("\nFUSED_IDS");
        for (int32_t id : fused_ids) std::printf(" %d", id);
        std::printf("\n");
    }
    GenerateOptions options;
    options.max_new_tokens = 10;
    options.max_cache_len = model.config.llm.max_cache;
    options.eos_token_id = model.config.eos_token_id;
    GenerateResult cancelled;
    CtcSpeculativeStats cancel_stats;
    int cancel_checks = 0;
    const bool cancel_ok = ctc_speculative_generate(
        model, mel, (int64_t)pcm.size(), options, 2,
        [&] { return ++cancel_checks >= 2; },
        cancelled, cancel_stats, err);
    const bool clean_cancel = cancel_ok && cancel_checks == 2 &&
        cancelled.ids.empty() &&
        cancelled.stop_reason == lib::GenStopReason::kCancelled &&
        cancel_stats.draft_count == fused_ids.size();
    std::printf("FUSED_CANCEL after_draft=%d checks=%d\n", clean_cancel, cancel_checks);
    return audio_equal && ids_equal && clean_cancel ? 0 : 1;
}
