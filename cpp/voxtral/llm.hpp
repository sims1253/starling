// llm.hpp — Voxtral Ministral-3-class text decoder on the shared qwen_decode
// stack + the offline additive-injection greedy loop.
//
// Binding: bias-free full-name projections, no q_norm/k_norm, tied lm_head,
// SwiGLU, two-round RMSNorm (the stack defaults), plus the voxtral-only
// AdaRMSNorm MLP-branch modulation and the decode-step audio-row add.
//
// The loop mirrors the Python oracle (_transcribe_fast): prefill over the
// prompt with audio rows 0..P-1 added host-side, then one decode step per
// token adding row P+t-1, stopping at EOS or the stock total-length cap
// (mel_T//8). It drives llm_prefill + llm_decode_step directly: the shared
// greedy_generate cannot serve per-step audio rows (its K-step graph chains
// argmax ids in-graph with no injection point).
#pragma once

#include "loader.hpp"
#include "encoder.hpp"
#include "prompt.hpp"
#include "lib/qwen_decode.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace starling::ggml::voxtral {

using LlmState = lib::LlmState;
using InputsEmbeds = lib::InputsEmbeds;
using PrefillResult = lib::PrefillResult;
using GenerateResult = lib::GenerateResult;

// Defaults are model-specific (max_cache_len 4096, EOS 2).
struct GenerateOptions {
    // -1 uses the stock audio-derived cap; nonnegative values limit output.
    int32_t max_new_tokens = -1, max_cache_len = 4096, eos_token_id = 2;
};

// Token-embedding lookup over `ids` with audio rows 0..P-1 added (bf16
// add = f32 add of bf16 values + one round; the host f32 sum of exact
// bf16 values rounds identically). Requires audio.n_tokens >= ids.size().
bool build_inputs_embeds(const VoxtralModel& m, const std::vector<int32_t>& ids,
                         const AudioEncoding& audio, InputsEmbeds& out,
                         std::string& err);

// Offline greedy loop over the merged prefill embeds + per-step audio rows.
// `mel_T` sets the stock total-length cap; `audio` supplies the decode rows.
bool greedy_generate(const VoxtralModel& m, const InputsEmbeds& prefill,
                     const AudioEncoding& audio, int64_t mel_T,
                     const GenerateOptions& op, GenerateResult& out,
                     std::string& err);

// Current number of cached per-S prefill graphs (diagnostic). Zero on CPU /
// before first GPU prefill.
size_t prefill_replay_cache_size();

} // namespace starling::ggml::voxtral
