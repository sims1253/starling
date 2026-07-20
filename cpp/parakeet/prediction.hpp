// prediction.hpp — parakeet-tdt RNNT prediction network (Phase 1c).
//
// Starling-authored port. Mirrors the proven parakeet.cpp prediction.cpp
// bit-for-bit (the CPU run_graph path), used by the serial TDT greedy decode
// loop. The prediction net is a stacked LSTM (pred_rnn_layers = 2) preceded by
// an Embedding(vocab+1, pred_hidden, padding_idx=blank) lookup; the top LSTM
// layer's h' IS the prediction output g (NO decoder_projector — the parakeet-tdt
// anchor removes it, so the joint.pred projection consumes h' directly).
//
// LSTM math (per step, input x [H], prev h,c [H]; h0=c0=0), PyTorch gate order
// [input, forget, cell, output] stacked in the 4H dim:
//   z = W_ih · x + b_ih + W_hh · h + b_hh        # [4H]
//   i = sigmoid(z[0:H]);   f = sigmoid(z[H:2H]);
//   g = tanh(z[2H:3H]);    o = sigmoid(z[3H:4H]);
//   c' = f * c + i * g
//   h' = o * tanh(c')
//
// State across steps lives in a PredState (one (h, c) pair per layer). The
// greedy loop keeps a "committed" PredState that only advances on an emit (k !=
// blank); the blank-skip reuse means a non-emit step's prediction output g is
// re-used instead of re-running the LSTM.
//
// Implementation: step() builds the whole stacked LSTM as ONE ggml graph and
// runs it via run_graph on the CPU backend (the byte-identical reference path).
// Each layer's new (h', c') is captured for out_state; the top layer's h' is
// returned as g.

#pragma once

#include "config.hpp"
#include "runtime/model_loader.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace starling::ggml::parakeet {

// Carries the LSTM hidden + cell state for stateful single-step decoding. One
// (h, c) pair PER stacked LSTM layer (PyTorch nn.LSTM with num_layers>1).
struct PredState {
    std::vector<std::vector<float>> h;  // h[layer] = [hidden]
    std::vector<std::vector<float>> c;  // c[layer] = [hidden]
};

class PredictionNet {
public:
    PredictionNet(const ModelLoader& ml, const Config& cfg);
    ~PredictionNet();  // out-of-line: StepReplay is only complete in the .cpp

    // Returns a zero-initialised LSTM state (h and c each [hidden] zeros).
    PredState zero_state() const;

    // Advance the LSTM by one token (one run_graph call on the CPU backend).
    // token_id:  embedding index (ignored when is_sos=true).
    // is_sos:    use the zero SOS input vector instead of the embedding.
    // in:        previous (h, c) state (use zero_state() for the first call).
    // g:         OUT — the new top-layer h' vector [hidden].
    // out_state: OUT — the new (h', c') state to carry to the next step.
    void step(int32_t token_id, bool is_sos,
              const PredState& in,
              std::vector<float>& g,
              PredState& out_state) const;

    int hidden_size() const { return H_; }
    int num_layers() const  { return n_layers_; }
    int vocab_p1() const    { return vocab_p1_; }

    // Access used by the fused TDT decode step (Joint::step_fused_argmax) and
    // the K-step multistep graph: the loader (to clone the LSTM weights as
    // zero-copy graph leaves) and the host embedding table (to pack the
    // layer-0 input exactly like step()). embed_host() is lazily populated on
    // the first step() call (see ensure_embed_host_).
    const ModelLoader& model_loader() const { return ml_; }
    const std::vector<float>& embed_host() const { return embed_host_; }

private:
    const ModelLoader& ml_;
    int H_;          // pred_hidden
    int vocab_p1_;   // vocab + 1 (embedding rows, incl. blank)
    int n_layers_;   // pred_rnn_layers (stacked LSTM layers)

    // GPU-only: replayable per-step LSTM graph, lazily built on first step().
    struct StepReplay;
    mutable std::unique_ptr<StepReplay> replay_;

    // Host-side copy of the embedding table, lazily fetched on the first step()
    // via the loader tensor's ->data (the loader backs the tensor with the GGUF
    // mmap on CPU). [vocab_p1_ * H_], row-major: embed_host_[id*H_ + h].
    mutable std::vector<float> embed_host_;

    void ensure_embed_host_() const;
};

} // namespace starling::ggml::parakeet
