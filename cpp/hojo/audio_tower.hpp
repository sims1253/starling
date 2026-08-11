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

// Run the Qwen3-Omni audio tower over a single utterance's mel:
//   mel reshape -> 3x Conv2d downsample (GELU between) over 200-frame windows
//   (conv_chunksize batched) -> flatten freq -> conv_out Linear -> add computed
//   SinusoidsPositionEmbedding -> 32 pre-norm LayerNorm transformer layers
//   (bidirectional, block-diagonal mask over the packed windows) -> ln_post ->
//   proj1 GELU proj2. Output [n_speech, 2048].
bool encode_audio_tower(const HojoModel& model, const MelFeatures& mel,
                        TowerOutput& out, std::string& err);
} // namespace starling::ggml::hojo
