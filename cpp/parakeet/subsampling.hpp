// subsampling.hpp — FastConformer ConvSubsampling stage (mel -> [d_model, T']).
//
// Mirrors NeMo's ConvSubsampling (dw_striding variant) as Starling-authored
// first-party code: 3 stride-2 conv stages (full -> dw+pw -> dw+pw) on the mel
// spectrogram, then a 1x1 -> d_model Linear over the flattened channels.
//
// Layout convention:
//   mel   feat-major [n_mels, T] (mel[m*T + t]) from MelFrontend
//   out   row-major [T', d_model] (d_model fastest, ne=[d_model, T'])
//
// T' is the subsampled length: T' = subsample_len(T) where each of the 3 stages
// applies (x + 2 - 3) / 2 + 1 (non-causal symmetric pad). For T={756, 2244, 7444}
// (the golden mel lengths) T' = {93, 279, 930}.

#pragma once

#include "config.hpp"
#include "runtime/graph_builder.hpp"
#include "runtime/model_loader.hpp"

#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace starling::ggml::parakeet {

// Starling-authored port of NeMo's ConvSubsampling. Stateless except for a
// reference to the loader (weights are referenced as graph leaves via
// clone_weight — zero-copy on CPU).
class Subsampling {
public:
    Subsampling(const ModelLoader& ml, const Config& cfg)
        : ml_(ml),
          conv_channels_(cfg.subsampling_conv_channels),
          d_model_(cfg.d_model) {}

    // Subsampled length T' from the input mel length T (pure arithmetic; no
    // graph build). T' = apply (x+2-3)/2+1 three times.
    int subsample_len(int T) const {
        int x = T;
        for (int s = 0; s < 3; ++s) x = (x + 2 - 3) / 2 + 1;
        return x;
    }

    // Valid (non-pad) output length given the input valid length. NeMo's
    // offline convention is valid_in = T-1 (center-padding adds a trailing
    // frame); each stage halves via the same recurrence as subsample_len.
    int valid_out_len(int T, int in_valid_frames) const {
        int valid = (in_valid_frames >= 0) ? in_valid_frames : (T - 1);
        for (int st = 0; st < 3; ++st)
            valid = (valid + 2 - 3) / 2 + 1;
        return valid;
    }

    // GRAPH-BUILDER: append the subsampling sub-graph to `ctx`. Transposes the
    // feat-major mel to time-major, runs the 3 conv stages + Linear, and returns
    // a [d_model, T'] tensor (ne0=d_model fastest). Host inputs (the transposed
    // mel, the optional length mask) are registered into `pool` (must outlive
    // the compute). Writes T' to out_Tp and the valid output length to
    // out_valid.
    ggml_tensor* build_graph(ggml_context* ctx,
                             const std::vector<float>& mel,
                             int n_mels, int T,
                             GraphInputPool& pool,
                             int& out_Tp, int& out_valid,
                             int in_valid_frames = -1) const;

private:
    const ModelLoader& ml_;
    int conv_channels_;  // 256 (subsampling_conv_channels)
    int d_model_;        // 1024
};

} // namespace starling::ggml::parakeet
