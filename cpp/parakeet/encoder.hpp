// encoder.hpp — parakeet-tdt Conformer encoder (Phase 1b).
//
// Starling-authored port of NeMo's FastConformer encoder + the joint.enc
// projection used as the validation gate. Builds the WHOLE encoder as ONE ggml
// graph and runs it via run_graph on the CPU backend (the byte-identical
// reference path). The cached-per-shape ReplayGraph variant lands with the GPU
// perf phase.
//
//   mel (feat-major [n_mels, T])
//     -> Subsampling.build_graph            # [d_model=1024, T']
//     -> (xscaling OFF for parakeet-tdt-v3)
//     -> rel_pos_encoding(T', d_model)      # host-side sinusoid [2T'-1, d_model]
//     -> 24 x ConformerLayer.build_graph    # [1024, T']
//     -> cont(transpose(x))                 # [T', 1024] row-major
//     -> joint.enc projection (Linear 1024->640)  # [640, T']  (validation only)
//
// The joint.enc projection is included in the validation graph because the
// golden parakeet_tdt_{short,medium,long}_enc.pt tensors are joint.enc(encoder
// output), NOT the raw 1024-dim encoder output. Production callers of encode()
// without projection should be added when the joint net is wired.

#pragma once

#include "conformer.hpp"
#include "config.hpp"
#include "loader.hpp"
#include "pos_enc.hpp"
#include "subsampling.hpp"

#include "runtime/graph.hpp"
#include "runtime/graph_builder.hpp"
#include "runtime/model_loader.hpp"

#include <vector>

namespace starling::ggml::parakeet {

class Encoder {
public:
    Encoder(const ParakeetModel& model)
        : model_(model),
          sub_(model.loader, model.config),
          config_(model.config) {}

    // Run mel (feat-major [n_mels, T]) -> encoder -> joint.enc projection, on
    // the CPU backend via run_graph. Writes the [640, T'] feat-major f32 buffer
    // into `out` (640 fastest, T' columns — i.e. out[c*T' + t]). Returns true on
    // success. T' is written to out_Tp.
    //
    // This is the byte-identical CPU reference path (forces run_graph, not the
    // ReplayGraph) so the validation matches parakeet.cpp's CPU reference.
    bool encode(const std::vector<float>& mel, int n_mels, int T,
                std::vector<float>& out, int& out_Tp) const;

    // Projected output dim (= joint_hidden = 640).
    int proj_dim() const { return (int)config_.joint_hidden; }

private:
    const ParakeetModel& model_;
    Subsampling sub_;
    const Config& config_;
};

} // namespace starling::ggml::parakeet
