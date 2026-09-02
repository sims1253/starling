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

#include "runtime/backend.hpp"  // clone_weight, graph_input_tensor, capture_graph_output, ReplayGraph
#include "runtime/graph.hpp"    // run_graph, ensure_weights_realized, global_backend
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

// Replayable per-step LSTM graph: built once, replayed each step. Inputs
// (registration order): one coalesced buffer [x0 | h0 | c0 | h1 | c1 ...]
// sliced into x0 and per-layer h_in/c_in views; captures per layer [c_out, h_out]
// land in the stable cap_* buffers. Mirrors parakeet.cpp's PredictionNet::StepReplay.
struct PredictionNet::StepReplay {
    std::unique_ptr<ReplayGraph> rg;
    ggml_tensor* x0 = nullptr;               // view into in_buf (layer-0 input)
    std::vector<ggml_tensor*> h_in;          // views into in_buf (input 1 + 2*l)
    std::vector<ggml_tensor*> c_in;          // views into in_buf (input 2 + 2*l)
    std::vector<float> in_buf;               // coalesced input [x0 | h0 | c0 | h1 | c1 ...]
    std::vector<std::vector<float>> cap_c;   // stable capture dsts
    std::vector<std::vector<float>> cap_h;
};

// Defined here so unique_ptr<StepReplay>'s ctor/deleter see the complete type
// (the members replay_ is constructed/destroyed in these).
PredictionNet::PredictionNet(const ModelLoader& ml, const Config& cfg)
    : ml_(ml),
      H_((int)cfg.pred_hidden),
      vocab_p1_((int)cfg.vocab_size + 1),
      n_layers_((int)cfg.pred_rnn_layers > 0 ? (int)cfg.pred_rnn_layers : 1) {}
PredictionNet::~PredictionNet() = default;

void PredictionNet::ensure_embed_host_() const {
    if (!embed_host_.empty()) return;
    // Make sure the loader's weights have a backend buffer (idempotent). On CPU
    // the tensor's ->data is the GGUF mmap; on GPU it's a device pointer, so we
    // MUST use ggml_backend_tensor_get (D2H) rather than a raw ->data memcpy.
    ensure_weights_realized(ml_);
    ggml_tensor* emb = ml_.tensor("decoder.prediction.embed.weight");
    assert(emb && "missing decoder.prediction.embed.weight");
    // The table may carry vocab_size rows (the checkpoint's layout: blank IS
    // the last SentencePiece piece) or vocab_size + 1 (converter-padded).
    // Fetch what exists row-major and zero-fill the remainder so a phantom
    // last-token lookup stays in bounds.
    embed_host_.assign((size_t)vocab_p1_ * H_, 0.0f);
    const int64_t rows = emb->ne[1] < (int64_t)vocab_p1_ ? emb->ne[1] : (int64_t)vocab_p1_;
    ggml_backend_tensor_get(emb, embed_host_.data(), 0,
                            (size_t)rows * H_ * sizeof(float));
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

    // CPU path (and the byte-identical reference): one-shot run_graph.
    if (!global_backend().is_gpu()) {
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
        return;
    }

    // GPU path: replay one captured LSTM graph instead of launching each op
    // directly (the per-step LSTM is launch-overhead bound). One coalesced
    // host->device upload per step; captures land in stable internal buffers
    // (out_state is .assign()'d every step and can move) and are copied out
    // after compute. Byte-identical to the CPU path (same math, same ops).
    if (!replay_) {
        replay_ = std::unique_ptr<StepReplay>(new StepReplay());
        replay_->h_in.assign(L, nullptr);
        replay_->c_in.assign(L, nullptr);
        replay_->cap_c.assign(L, std::vector<float>((size_t)H));
        replay_->cap_h.assign(L, std::vector<float>((size_t)H));
        StepReplay* r = replay_.get();  // captures must read this stable addr
        const int n_in_blocks = 1 + 2 * L;     // x0 + per-layer h,c
        r->in_buf.assign((size_t)n_in_blocks * H, 0.0f);
        r->rg = std::unique_ptr<ReplayGraph>(new ReplayGraph(
            global_backend(),
            [&](ggml_context* ctx) -> ggml_tensor* {
                // One input tensor holding [x0 | h0 | c0 | h1 | c1 ...].
                int64_t in_ne[1] = { (int64_t)n_in_blocks * H };
                ggml_tensor* in_all = graph_input_tensor(ctx, GGML_TYPE_F32, 1, in_ne,
                            r->in_buf.data(), (size_t)n_in_blocks * H * sizeof(float));
                r->x0 = ggml_view_1d(ctx, in_all, H, 0);
                ggml_tensor* layer_in = r->x0;
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
                    size_t h_off = (size_t)(1 + 2 * l)     * H * sizeof(float);
                    size_t c_off = (size_t)(1 + 2 * l + 1) * H * sizeof(float);
                    r->h_in[l] = ggml_view_1d(ctx, in_all, H, h_off);
                    r->c_in[l] = ggml_view_1d(ctx, in_all, H, c_off);
                    ggml_tensor* z = ggml_add(ctx,
                        ggml_add(ctx, ggml_mul_mat(ctx, Wih, layer_in), bih),
                        ggml_add(ctx, ggml_mul_mat(ctx, Whh, r->h_in[l]), bhh));
                    ggml_tensor* i  = ggml_sigmoid(ctx, ggml_cont(ctx, ggml_view_1d(ctx, z, H, 0)));
                    ggml_tensor* f  = ggml_sigmoid(ctx, ggml_cont(ctx, ggml_view_1d(ctx, z, H, (size_t)H * sizeof(float))));
                    ggml_tensor* gg = ggml_tanh   (ctx, ggml_cont(ctx, ggml_view_1d(ctx, z, H, (size_t)2 * H * sizeof(float))));
                    ggml_tensor* o  = ggml_sigmoid(ctx, ggml_cont(ctx, ggml_view_1d(ctx, z, H, (size_t)3 * H * sizeof(float))));
                    ggml_tensor* c_out = ggml_add(ctx, ggml_mul(ctx, f, r->c_in[l]),
                                                  ggml_mul(ctx, i, gg));
                    ggml_tensor* h_out = ggml_mul(ctx, o, ggml_tanh(ctx, c_out));
                    capture_graph_output(c_out, &r->cap_c[l]);
                    capture_graph_output(h_out, &r->cap_h[l]);
                    layer_in = h_out;
                    top_h    = h_out;
                }
                return top_h;
            }));
        assert(r->rg->n_inputs() == 1 && "pred step graph must have 1 coalesced input");
    }

    // Pack this step's inputs into the single coalesced buffer (host memcpy, no
    // sync) and upload once. Layout: [x0 | h0 | c0 | h1 | c1].
    std::vector<float>& inb = replay_->in_buf;
    std::memcpy(inb.data(), x0.data(), (size_t)H * sizeof(float));
    for (int l = 0; l < L; ++l) {
        std::memcpy(inb.data() + (size_t)(1 + 2*l)     * H, in.h[l].data(), (size_t)H * sizeof(float));
        std::memcpy(inb.data() + (size_t)(1 + 2*l + 1) * H, in.c[l].data(), (size_t)H * sizeof(float));
    }
    replay_->rg->set_input(0, inb.data(), (size_t)(1 + 2 * L) * H * sizeof(float));
    bool ok = replay_->rg->compute_with_captures(g);
    assert(ok && "pred-net step replay failed");
    (void)ok;
    for (int l = 0; l < L; ++l) {
        std::memcpy(out_state.h[l].data(), replay_->cap_h[l].data(), (size_t)H * sizeof(float));
        std::memcpy(out_state.c[l].data(), replay_->cap_c[l].data(), (size_t)H * sizeof(float));
    }
}

} // namespace starling::ggml::parakeet
