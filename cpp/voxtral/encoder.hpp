// encoder.hpp — Voxtral causal audio encoder + downsample-4 projector.
//
// The audio path: causal Conv1d front-end (conv1 k3/s1 LEFT-pad 2, conv2
// k3/s2 LEFT-pad 1, halving mel_T) -> GELU -> transpose -> 32 pre-norm layers
// (RMSNorm, MHA with q bias / k NO bias / v+o bias, rotate-half RoPE,
// sliding-window band-causal mask, residual, RMSNorm, SwiGLU with down_proj
// bias, residual) -> final RMSNorm -> projector (group-by-4 reshape,
// Linear(input->output, no bias) GELU Linear(output->output, no bias)).
//
// Two paths share the layer/projector graph builders:
//   - host path (CPU, or STARLING_VOXTRAL_DEBUG=1): scalar left-pad convs on
//     the host (the portable reference) + a one-shot layers/projector graph.
//   - replay path (GPU, or STARLING_VOXTRAL_FORCE_REPLAY=1): ONE captured
//     ReplayGraph per (mel_T, debug-tag), LRU-bounded, running convs (over a
//     host-left-padded mel input) + layers + projector end-to-end.
//
// Phase 2b (decoder + capi) needs only encode_audio_and_project: mel in,
// token-major projected rows out.
#pragma once

#include "loader.hpp"
#include "mel.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace starling::ggml::voxtral {

// Projected audio embeddings: token-major f32, token n occupies
// data[n*width, (n+1)*width). Values are exactly BF16-representable (the
// graph readback rounds once at the bf16 boundary).
struct AudioEncoding {
    std::vector<float> data;
    int64_t n_tokens = 0;
    int64_t width = 0;
};

// Optional per-layer debug captures (tests only). embedder is the conv
// front-end output, (T_enc, d_model) row-major; layer_outs[i] is the
// post-residual hidden state after layer layer_idx[i], same layout. Null
// embedder/layer_outs (or an empty layer_idx) disables the capture.
//
// Layer-0 per-stage bisect captures (temporary; null disables each). All
// [AW, T]-flat in raw ggml order (element (aw,t) at aw+t*AW), which coincides
// with the torch [T, AW] row-major flattening; l0_prob is the raw [T, T, H]
// softmax flat (element (k,q,h) at k+T*q+T*T*h, so head 0's flat compares
// directly against a row-major [T, T] reference).
struct EncoderDebug {
    std::vector<float>* embedder = nullptr;
    std::vector<int64_t> layer_idx;
    std::vector<std::vector<float>>* layer_outs = nullptr;
    std::vector<float>* l0_n = nullptr;    // post attn_norm
    std::vector<float>* l0_q = nullptr;    // post q proj, pre-rope
    std::vector<float>* l0_k = nullptr;    // post k proj, pre-rope
    std::vector<float>* l0_v = nullptr;    // post v proj
    std::vector<float>* l0_qr = nullptr;   // post-rope q
    std::vector<float>* l0_kr = nullptr;   // post-rope k
    std::vector<float>* l0_prob = nullptr; // softmax probs [T, T, H]
    std::vector<float>* l0_ctx = nullptr;  // attention out, pre-o_proj
    std::vector<float>* l0_a = nullptr;    // post-o_proj, pre-residual
    std::vector<float>* l0_ffn = nullptr;  // post-ffn-down, pre-residual
};

// Band-mask budget: the [T_enc, T_enc] f32 additive mask is the O(T^2) term.
// A T_enc whose mask exceeds this is rejected with a shorter-audio hint.
constexpr int64_t kVoxtralMaxMaskBytes = 1LL << 30;

// Full encoder: mel -> convs -> layers -> final norm -> projector. Returns
// false with a load/encode error (bad mel shape, odd/unguarded lengths, mask
// over budget, graph failure).
bool encode_audio_and_project(const VoxtralModel& model, const MelFeatures& mel,
                              AudioEncoding& out, std::string& err,
                              EncoderDebug* dbg = nullptr);

// Current number of cached encoder ReplayGraphs (diagnostic). Zero on CPU /
// before first replay-path encode.
size_t encoder_replay_cache_size();

} // namespace starling::ggml::voxtral
