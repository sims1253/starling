#pragma once
#include "loader.hpp"
#include "mel.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace starling::ggml::hojo {
// The Qwen3-Omni audio tower output: [n_speech, output_dim=2048] f32 (the
// speech embeddings fed to the bottleneck). The tower also returns the
// per-window frame counts (cu_seqlens) the bottleneck consumes.
struct TowerOutput {
    std::vector<float> data;      // [n_speech, output_dim] f32, token-major
    int64_t n_speech = 0;         // total packed frames
    int64_t width = 0;            // output_dim (2048)
    // Per-window conv-output frame counts (the split_sizes encode_speech uses
    // to re-segment the tower output into per-utterance speech embeds). For a
    // single utterance this is one entry.
    std::vector<int64_t> per_window_frames;
};

// Host-side conv2d stack weights (conv2d1..3 weight + bias, f32). Read once
// per tower call and shared across windows; exposed so the CPU-only
// parity/perf test can drive host_conv2d_stack directly without a backend.
struct ConvStackWeights {
    std::vector<float> w[3], b[3];  // conv2d1..3: weight [OC,IC,3,3], bias [OC]
};

// 3 stride-2 conv2d + exact-erf GELU between them, host-side (f32 math, double
// accumulation, multithreaded over output channels — each output element keeps
// the serial accumulation order, so it is bit-identical to the single-threaded
// form). mel_win is one window's mel [win, n_mels] time-major; returns
// [out_T, conv_width] c-outer, f-inner per time step.
std::vector<float> host_conv2d_stack(const ConvStackWeights& cw,
                                     const std::vector<float>& mel_win,
                                     int64_t win, int64_t n_mels);

// Run the Qwen3-Omni audio tower over a single utterance's mel:
//   mel reshape -> 3x Conv2d downsample (GELU between) over windows of
//   tc.n_window*2 = 3000 frames (conv_chunksize controls the conv batching, not
//   the window length) -> flatten freq -> conv_out Linear -> add computed
//   SinusoidsPositionEmbedding -> 32 pre-norm LayerNorm transformer layers
//   (bidirectional, block-diagonal mask over the packed windows) -> ln_post ->
//   proj1 GELU proj2. Output [n_speech, 2048].
bool encode_audio_tower(const HojoModel& model, const MelFeatures& mel,
                        TowerOutput& out, std::string& err);
} // namespace starling::ggml::hojo
