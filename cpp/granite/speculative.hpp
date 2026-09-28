#pragma once

#include "encoder.hpp"
#include "llm.hpp"

namespace starling::ggml::granite {

struct CtcSpeculativeStats {
    size_t draft_count = 0;
    double extraction_ms = 0.0; // split path: extra encoder + head
    double encoder_project_ms = 0.0; // fused path: shared graph
    double ctc_head_ms = 0.0; // fused path: pooling + optional head
    double embed_ms = 0.0; // fused path: prompt and merged inputs
    double total_ms = 0.0; // excludes caller's mel frontend
    lib::SpeculativeStats verifier;
};

// Opt-in split path retained as a cost baseline. `inputs` must describe the
// same mel chunk; its encoder pass is the caller's responsibility. This call
// runs an extra encoder pass for CTC extraction.
bool ctc_speculative_generate(const GraniteModel& model, const MelFeatures& mel,
                              const InputsEmbeds& inputs, const GenerateOptions& op,
                              int max_k, const lib::CancelCheck& cancelled,
                              GenerateResult& output, CtcSpeculativeStats& stats,
                              std::string& err);

// Opt-in production candidate: shared encoder + projector + CTC extraction,
// prompt/embedding merge, then native batched verification. `n_samples` is the
// chunk's exact PCM length for the Granite prompt template. The default C API
// continues to use greedy_generate.
bool ctc_speculative_generate(const GraniteModel& model, const MelFeatures& mel,
                              int64_t n_samples, const GenerateOptions& op,
                              int max_k, const lib::CancelCheck& cancelled,
                              GenerateResult& output, CtcSpeculativeStats& stats,
                              std::string& err);

} // namespace starling::ggml::granite
