// encoder.cpp — parakeet-tdt Conformer encoder (Phase 1b) graph build + run.
//
// Starling-authored port. Mirrors the proven parakeet.cpp CPU reference path
// (encoder.cpp:130-172) bit-for-bit: subsampling -> (xscaling OFF) -> host-built
// rel_pos_encoding pe -> 24 ConformerLayer.build_graph (shared pe) -> final
// cont(transpose). For the validation gate we append the joint.enc projection
// (Linear 1024->640) so the output matches the golden parakeet_tdt_*_enc.pt.
//
// The graph is built and run via run_graph on the CPU backend — the byte-
// identical reference. (The cached-per-shape ReplayGraph variant is a future
// perf phase; correctness first.)

#include "encoder.hpp"

#include "runtime/backend.hpp"  // clone_weight, graph_input_tensor, run_graph

#include "ggml.h"

#include <cstring>
#include <vector>

namespace starling::ggml::parakeet {

bool Encoder::encode(const std::vector<float>& mel, int n_mels, int T,
                     std::vector<float>& out, int& out_Tp) const {
    GraphInputPool pool;
    const ModelLoader& ml = model_.loader;
    const int d_model = (int)config_.d_model;
    const int n_layers = (int)config_.n_layers;

    int Tp = 0, valid_len = 0;
    bool ok = run_graph([&](ggml_context* ctx) -> ggml_tensor* {
        // 1. Subsampling: mel [n_mels, T] -> x [d_model, T'] + valid_len.
        ggml_tensor* x = sub_.build_graph(ctx, mel, n_mels, T, pool,
                                          Tp, valid_len);
        // xscaling is OFF for parakeet-tdt-0.6b-v3 — skip ggml_scale(sqrt(D)).

        // 2. Positional encoding: host-built sinusoid table [2T'-1, d_model].
        const int pos_len = 2 * Tp - 1;
        std::vector<float> pe_vec;
        rel_pos_encoding(Tp, d_model, pe_vec);
        float* pe_host = pool.alloc_f32(pe_vec.size());
        std::memcpy(pe_host, pe_vec.data(), pe_vec.size() * sizeof(float));
        int64_t pe_ne[2] = {d_model, pos_len};
        ggml_tensor* pe = graph_input_tensor(ctx, GGML_TYPE_F32, 2, pe_ne,
                              pe_host, pe_vec.size() * sizeof(float));

        // 3. Conformer stack: 24 layers, shared pe.
        for (int i = 0; i < n_layers; ++i) {
            ConformerLayer layer(ml, config_, i);
            x = layer.build_graph(ctx, x, Tp, pe, pos_len, valid_len, pool);
        }

        // 4. joint.enc projection (Linear 1024->640), validation-only.
        (void)d_model;
        ggml_tensor* jw = clone_weight(ctx, ml, "joint.enc.weight");
        ggml_tensor* jb = clone_weight(ctx, ml, "joint.enc.bias");
        ggml_tensor* proj = ggml_mul_mat(ctx, jw, x);  // [640, T']
        proj = ggml_add(ctx, proj, jb);                // broadcast [640] over T'
        return proj;
    }, out);

    if (!ok) return false;
    out_Tp = Tp;
    return true;
}

} // namespace starling::ggml::parakeet
