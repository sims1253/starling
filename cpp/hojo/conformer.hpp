#pragma once
#include "audio_tower.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace starling::ggml::hojo {
// The WeNet Conformer bottleneck output: [B=1, T, 2560] f32 (post after_norm,
// pre ln_speech). For single-utterance parity, B=1 and T = n_speech.
struct BottleneckOutput {
    std::vector<float> data;   // [T, output_size] f32 (token-major, B=1)
    int64_t n_tokens = 0;
    int64_t width = 0;
};

// Run the WeNet Conformer bottleneck over the tower output:
//   LinearNoSubsampling (Linear 2048->2560 + LayerNorm) + RelPositionalEncoding
//   (x scaled by sqrt(2560); pos_emb from the baked pe buffer) ->
//   2 ConformerEncoderLayer (macaron FFN 0.5 -> rel-pos MHA with pos_bias_u/v,
//   NO rel_shift -> conv module with BatchNorm1d-inference-fold + depthwise k15
//   -> FFN 0.5 -> norm_final) -> after_norm.
bool encode_bottleneck(const HojoModel& model, const TowerOutput& tower,
                       BottleneckOutput& out, std::string& err);
} // namespace starling::ggml::hojo
