// prediction.cpp — parakeet-tdt RNNT prediction network (Phase 1c).
//
// Starling-authored port of parakeet.cpp's prediction.cpp:73-149 (the CPU
// run_graph path), bit-for-bit. Builds the whole stacked LSTM as ONE ggml graph
// and runs it via run_graph on the CPU backend (the byte-identical reference).
// Each layer's new (h', c') is captured for out_state; the top layer's h' is
// returned as g (NO decoder_projector — joint.pred consumes h' directly).
//
// The embedding table is fetched to the host once (lazily, on the first step)
// so the SOS/lookup row can seed the layer-0 input via graph_input_tensor. The
// LSTM weights are referenced as zero-copy loader leaves via clone_weight.

#include "prediction.hpp"

#include "runtime/backend.hpp"  // clone_weight, graph_input_tensor, capture_graph_output
#include "runtime/graph.hpp"    // run_graph, ensure_weights_realized
#include "runtime/model_loader.hpp"

#include "ggml.h"
#include "ggml-backend.h"  // ggml_backend_tensor_get (D2H, works for host + device)

#include <cassert>
#include <cstring>
#include <string>
#include <vector>

namespace starling::ggml::parakeet {

PredState PredictionNet::zero_state() const {
    PredState s;
    s.h.assign((size_t)n_layers_, std::vector<float>((size_t)H_, 0.0f));
    s.c.assign((size_t)n_layers_, std::vector<float>((size_t)H_, 0.0f));
    return s;
}

void PredictionNet::ensure_embed_host_() const {
    if (!embed_host_.empty()) return;
    // Make sure the loader's weights have a backend buffer (idempotent). On CPU
    // the tensor's ->data is the GGUF mmap; on GPU it's a device pointer, so we
    // MUST use ggml_backend_tensor_get (D2H) rather than a raw ->data memcpy.
    ensure_weights_realized(ml_);
    ggml_tensor* emb = ml_.tensor("decoder.prediction.embed.weight");
    assert(emb && "missing decoder.prediction.embed.weight");
    embed_host_.resize((size_t)vocab_p1_ * H_);
    // Read the (F32) embedding table row-major: ggml row i == embedding of id i,
    // ne[0]=H fastest.
    ggml_backend_tensor_get(emb, embed_host_.data(), 0,
                            (size_t)vocab_p1_ * H_ * sizeof(float));
}

void PredictionNet::step(int32_t token_id, bool is_sos,
                         const PredState& in,
                         std::vector<float>& g,
                         PredState& out_state) const {
    const int H = H_;
    const int L = n_layers_;

    ensure_embed_host_();

    // Layer-0 input: zeros for SOS, else the embedding row for token_id.
    std::vector<float> x0((size_t)H, 0.0f);
    if (!is_sos) {
        assert(token_id >= 0 && token_id < vocab_p1_ && "embedding id out of range");
        std::memcpy(x0.data(), &embed_host_[(size_t)token_id * H],
                    (size_t)H * sizeof(float));
    }

    out_state.h.assign((size_t)L, std::vector<float>((size_t)H));
    out_state.c.assign((size_t)L, std::vector<float>((size_t)H));

    bool ok = run_graph([&](ggml_context* ctx) -> ggml_tensor* {
        int64_t ne1[1] = { H };
        ggml_tensor* layer_in = graph_input_tensor(ctx, GGML_TYPE_F32, 1, ne1,
                                    x0.data(), (size_t)H * sizeof(float));
        ggml_tensor* top_h = nullptr;
        for (int l = 0; l < L; ++l) {
            const std::string s = "_l" + std::to_string(l);
            ggml_tensor* Wih = clone_weight(ctx, ml_,
                ("decoder.prediction.dec_rnn.lstm.weight_ih" + s).c_str());
            ggml_tensor* Whh = clone_weight(ctx, ml_,
                ("decoder.prediction.dec_rnn.lstm.weight_hh" + s).c_str());
            ggml_tensor* bih = clone_weight(ctx, ml_,
                ("decoder.prediction.dec_rnn.lstm.bias_ih" + s).c_str());
            ggml_tensor* bhh = clone_weight(ctx, ml_,
                ("decoder.prediction.dec_rnn.lstm.bias_hh" + s).c_str());
            ggml_tensor* h_in = graph_input_tensor(ctx, GGML_TYPE_F32, 1, ne1,
                                    in.h[l].data(), (size_t)H * sizeof(float));
            ggml_tensor* c_in = graph_input_tensor(ctx, GGML_TYPE_F32, 1, ne1,
                                    in.c[l].data(), (size_t)H * sizeof(float));
            // z = (Wih·x + bih) + (Whh·h + bhh)   -> [4H]
            ggml_tensor* z = ggml_add(ctx,
                ggml_add(ctx, ggml_mul_mat(ctx, Wih, layer_in), bih),
                ggml_add(ctx, ggml_mul_mat(ctx, Whh, h_in),     bhh));
            // PyTorch gate order [i, f, g, o] stacked in the 4H dim.
            ggml_tensor* i  = ggml_sigmoid(ctx, ggml_cont(ctx, ggml_view_1d(ctx, z, H, 0)));
            ggml_tensor* f  = ggml_sigmoid(ctx, ggml_cont(ctx, ggml_view_1d(ctx, z, H, (size_t)H * sizeof(float))));
            ggml_tensor* gg = ggml_tanh   (ctx, ggml_cont(ctx, ggml_view_1d(ctx, z, H, (size_t)2 * H * sizeof(float))));
            ggml_tensor* o  = ggml_sigmoid(ctx, ggml_cont(ctx, ggml_view_1d(ctx, z, H, (size_t)3 * H * sizeof(float))));
            // c' = f*c_in + i*g ;  h' = o*tanh(c')
            ggml_tensor* c_out = ggml_add(ctx, ggml_mul(ctx, f, c_in), ggml_mul(ctx, i, gg));
            ggml_tensor* h_out = ggml_mul(ctx, o, ggml_tanh(ctx, c_out));
            capture_graph_output(c_out, &out_state.c[l]);
            capture_graph_output(h_out, &out_state.h[l]);
            layer_in = h_out;
            top_h    = h_out;
        }
        return top_h;  // top layer's h' == prediction output g
    }, g);
    assert(ok && "pred-net step graph failed");
    (void)ok;
}

} // namespace starling::ggml::parakeet
