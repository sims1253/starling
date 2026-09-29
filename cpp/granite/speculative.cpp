#include "speculative.hpp"
#include "ctc_proposer.hpp"
#include "prompt.hpp"

#include <chrono>

namespace starling::ggml::granite {
namespace {
bool verify_draft(const GraniteModel& model, const InputsEmbeds& inputs,
                  const GenerateOptions& op, int max_k,
                  const lib::CancelCheck& cancelled, std::vector<int32_t> draft,
                  GenerateResult& output, CtcSpeculativeStats& stats,
                  std::string& err) {
    stats.draft_count = draft.size();
    CtcProposer proposer(std::move(draft), max_k);
    const auto callback = [&proposer](const std::vector<int32_t>& prefix, int cap) {
        return proposer.propose(prefix, cap);
    };
    return speculative_generate(model, inputs, op, max_k, callback,
                                cancelled, output, stats.verifier, err);
}
} // namespace

bool ctc_speculative_generate(const GraniteModel& model, const MelFeatures& mel,
                              const InputsEmbeds& inputs, const GenerateOptions& op,
                              int max_k, const lib::CancelCheck& cancelled,
                              GenerateResult& output, CtcSpeculativeStats& stats,
                              std::string& err) {
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    stats = {};
    output = {};
    if (max_k < 1 || max_k > 16) {
        err = "Granite CTC maximum proposal length must be 1..16";
        return false;
    }
    if (cancelled && cancelled()) {
        output.stop_reason = lib::GenStopReason::kCancelled;
        return true;
    }
    std::vector<int32_t> draft;
    if (!extract_ctc_draft(model, mel, draft, err)) return false;
    const auto t1 = Clock::now();
    stats.draft_count = draft.size();
    stats.extraction_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (cancelled && cancelled()) {
        output.stop_reason = lib::GenStopReason::kCancelled;
        stats.total_ms = stats.extraction_ms;
        return true;
    }
    const bool ok = verify_draft(model, inputs, op, max_k, cancelled,
                                 std::move(draft), output, stats, err);
    stats.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    return ok;
}

bool ctc_speculative_generate(const GraniteModel& model, const MelFeatures& mel,
                              int64_t n_samples, const GenerateOptions& op,
                              int max_k, const lib::CancelCheck& cancelled,
                              GenerateResult& output, CtcSpeculativeStats& stats,
                              std::string& err) {
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    stats = {};
    output = {};
    if (max_k < 1 || max_k > 16 || n_samples <= 0) {
        err = "Granite CTC requires positive samples and maximum proposal length 1..16";
        return false;
    }
    if (cancelled && cancelled()) {
        output.stop_reason = lib::GenStopReason::kCancelled;
        return true;
    }
    AudioEmbeds audio;
    std::vector<int32_t> draft;
    double stages[2] = {};
    if (!encode_audio_project_and_extract_ctc(model, mel, audio, draft, err, stages))
        return false;
    stats.draft_count = draft.size();
    stats.encoder_project_ms = stages[0];
    stats.ctc_head_ms = stages[1];
    if (cancelled && cancelled()) {
        output.stop_reason = lib::GenStopReason::kCancelled;
        stats.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        return true;
    }
    const auto t_embed = Clock::now();
    const Prompt prompt = build_transcribe_prompt(model.config, n_samples);
    InputsEmbeds inputs;
    if (!build_inputs_embeds(model, prompt, audio, inputs, err)) return false;
    stats.embed_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_embed).count();
    if (cancelled && cancelled()) {
        output.stop_reason = lib::GenStopReason::kCancelled;
        stats.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        return true;
    }
    const bool ok = verify_draft(model, inputs, op, max_k, cancelled,
                                 std::move(draft), output, stats, err);
    stats.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    return ok;
}

} // namespace starling::ggml::granite
