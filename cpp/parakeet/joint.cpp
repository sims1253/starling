// joint.cpp — parakeet-tdt RNNT joint network (Phase 1c).
//
// Starling-authored port of parakeet.cpp's joint.cpp:197-213 (the CPU run_graph
// path for step_argmax, which reuses step_logits + host argmax), bit-for-bit.
// Builds the joint as ONE ggml graph per step and runs it via run_graph on the
// CPU backend (the byte-identical reference path).
//
// joint.enc projection is ALREADY in enc_proj_t (the encoder Phase 1b output is
// the [T, H] joint-projected buffer); do NOT re-apply joint.enc here. The
// prediction output g IS the top LSTM layer's h' (NO decoder_projector).

#include "joint.hpp"

#include "runtime/backend.hpp"  // clone_weight, graph_input_tensor, ReplayGraph
#include "runtime/graph.hpp"    // run_graph, global_backend

#include "ggml.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <vector>

namespace starling::ggml::parakeet {

// Replayable FUSED prediction-LSTM + joint + argmax graph (one ReplayGraph, one
// host<-device sync per step). The prediction output g flows pred -> joint on
// the device (never read back). Mirrors parakeet.cpp's Joint::FusedReplay.
//
// Coalesced input layout (host-packed, one upload per step):
//   [ enc_proj_t (H_joint) | token_emb (H_pred) | h0,c0,h1,c1,... (2L*H_pred) ]
// Captures (read back in the single sync, batched as async copies):
//   cap_h[l], cap_c[l] (f32 [H_pred])  -- new LSTM state per layer
//   cap_dur_amax_f (1 float)           -- 4-byte i32 duration argmax
// Output: i32 [1] = token argmax.
struct Joint::FusedReplay {
    std::unique_ptr<ReplayGraph> rg;
    std::vector<float> in_buf;            // coalesced [enc_proj_t | emb | h0,c0,...]
    std::vector<std::vector<float>> cap_h;// [L][H_pred] new hidden state per layer
    std::vector<std::vector<float>> cap_c;// [L][H_pred] new cell state per layer
    std::vector<float> cap_dur_amax_f;    // 1-float bucket holding the 4-byte i32 duration argmax
    int H_pred = 0, L_pred = 0, H_joint = 0;
    size_t emb_off = 0;     // float offset of token_emb in in_buf
    size_t state_off = 0;   // float offset of LSTM state block in in_buf
    size_t in_nbytes = 0;   // total coalesced input size in bytes
};

Joint::Joint(const ModelLoader& ml, const Config& cfg) : ml_(ml) {
    // Read joint_hidden from the enc weight shape: ne[1] (ggml) = joint_hidden.
    ggml_tensor* ew = ml.tensor("joint.enc.weight");
    assert(ew && "missing joint.enc.weight");
    joint_hidden_ = (int)ew->ne[1];
    enc_hidden_   = (int)ew->ne[0];

    ggml_tensor* pw = ml.tensor("joint.pred.weight");
    assert(pw && "missing joint.pred.weight");
    pred_hidden_ = (int)pw->ne[0];
    assert((int)pw->ne[1] == joint_hidden_ && "pred/enc joint_hidden mismatch");

    vocab_size_    = (int)cfg.vocab_size;
    num_durations_ = (int)cfg.tdt_durations.size();
    V_plus_        = vocab_size_ + 1 + num_durations_;

    // Sanity-check the output projection shape (joint_net.2 is f32; enc/pred
    // projections are quantization-allowlisted and may be f16 — ggml_mul_mat
    // dequantizes those on the fly in the graph below).
    ggml_tensor* wout = ml.tensor("joint.joint_net.2.weight");
    assert(wout && "missing joint.joint_net.2.weight");
    assert((int)wout->ne[0] == joint_hidden_ && (int)wout->ne[1] == V_plus_ &&
           "joint_net.2 weight shape mismatch");
    (void)wout;
}

// Out-of-line so the header's unique_ptr<FusedReplay> can stay incomplete there
// (FusedReplay is defined above, complete at this point).
Joint::~Joint() = default;

void Joint::step_logits(const float* enc_proj_t,
                        const float* g, int pred_hidden,
                        std::vector<float>& logits) const {
    assert(pred_hidden == pred_hidden_ && "pred_hidden mismatch");
    const int H = joint_hidden_;

    bool ok = run_graph([&](ggml_context* ctx) -> ggml_tensor* {
        int64_t ep_ne[1] = { H };
        ggml_tensor* ep = graph_input_tensor(ctx, GGML_TYPE_F32, 1, ep_ne,
                              enc_proj_t, (size_t)H * sizeof(float));
        int64_t g_ne[1] = { pred_hidden_ };
        ggml_tensor* gv = graph_input_tensor(ctx, GGML_TYPE_F32, 1, g_ne,
                              g, (size_t)pred_hidden_ * sizeof(float));
        ggml_tensor* Wp = clone_weight(ctx, ml_, "joint.pred.weight");
        ggml_tensor* pp = ggml_mul_mat(ctx, Wp, gv);            // [H]
        ggml_tensor* bp = clone_weight(ctx, ml_, "joint.pred.bias");
        pp = ggml_add(ctx, pp, bp);
        ggml_tensor* f = ggml_relu(ctx, ggml_add(ctx, ep, pp)); // [H]
        ggml_tensor* Wo = clone_weight(ctx, ml_, "joint.joint_net.2.weight");
        ggml_tensor* y  = ggml_mul_mat(ctx, Wo, f);             // [V]
        ggml_tensor* bo = clone_weight(ctx, ml_, "joint.joint_net.2.bias");
        y = ggml_add(ctx, y, bo);
        return y;                                               // [V_plus]
    }, logits);
    assert(ok && "step_logits graph failed");
    (void)ok;
}

void Joint::step_argmax(const float* enc_proj_t, int token_count,
                        const float* g, int pred_hidden,
                        int& k_out, int& d_k_out) const {
    assert(pred_hidden == pred_hidden_ && "pred_hidden mismatch");
    assert(token_count == vocab_size_ + 1 && "token_count mismatch");
    const int num_dur = num_durations_;

    // CPU path: run step_logits and argmax on host (cheap vs the per-step
    // matmul, byte-identical to the GPU device-argmax path).
    std::vector<float> logits;
    step_logits(enc_proj_t, g, pred_hidden, logits);

    // token slice [0, token_count), duration slice [token_count, V_plus).
    auto amax = [](const float* p, int n) -> int {
        int best = 0;
        float bv = p[0];
        for (int i = 1; i < n; ++i) {
            if (p[i] > bv) { bv = p[i]; best = i; }
        }
        return best;
    };
    k_out   = amax(logits.data(), token_count);
    d_k_out = amax(logits.data() + token_count, num_dur);
    (void)amax;
}

void Joint::step_fused_argmax(const PredictionNet& pred,
                              const float* enc_proj_t, int token_count,
                              int32_t token_id, bool is_sos,
                              const PredState& in_state,
                              PredState& out_state,
                              int& k_out, int& d_k_out) const {
    assert(token_count == vocab_size_ + 1 && "token_count mismatch");
    const int Hj = joint_hidden_;       // joint hidden (= enc/pred projection dim)
    const int Hp = pred.hidden_size();  // prediction LSTM hidden
    const int L  = pred.num_layers();
    const int num_dur = num_durations_;

    // CPU / forced-serial fallback: no replay benefit, so run the unfused
    // pred.step + joint logits + host argmax (byte-identical to the GPU path).
    if (!global_backend().is_gpu()) {
        std::vector<float> g;
        pred.step(token_id, is_sos, in_state, g, out_state);
        std::vector<float> logits;
        step_logits(enc_proj_t, g.data(), Hp, logits);
        auto amax = [](const float* p, int n) -> int {
            int best = 0; float bv = p[0];
            for (int i = 1; i < n; ++i) if (p[i] > bv) { bv = p[i]; best = i; }
            return best;
        };
        k_out   = amax(logits.data(), token_count);
        d_k_out = amax(logits.data() + token_count, num_dur);
        return;
    }

    // Lazily build the fused graph once; replay every step thereafter. The build
    // mirrors prediction.cpp's per-step LSTM graph fused with the joint + argmax.
    // The prediction output (top layer's h') feeds the joint's pred projection
    // DIRECTLY on the device (never read back).
    auto& fused_replay_ = ml_.cache<FusedReplay>();
    if (!fused_replay_) {
        auto pending = std::unique_ptr<FusedReplay>(new FusedReplay());
        FusedReplay* r = pending.get();
        r->H_pred = Hp; r->L_pred = L; r->H_joint = Hj;
        r->cap_h.assign(L, std::vector<float>((size_t)Hp));
        r->cap_c.assign(L, std::vector<float>((size_t)Hp));
        r->cap_dur_amax_f.assign(1, 0.0f);

        // Coalesced input: [ enc_proj_t (Hj) | token_emb (Hp) | h0,c0,h1,c1,... ]
        const size_t n_in_floats = (size_t)Hj + (size_t)Hp + (size_t)(2 * L) * Hp;
        r->emb_off = (size_t)Hj;
        r->state_off = (size_t)Hj + (size_t)Hp;
        r->in_buf.assign(n_in_floats, 0.0f);
        r->in_nbytes = n_in_floats * sizeof(float);

        const ModelLoader& pml = pred.model_loader();
        r->rg = std::unique_ptr<ReplayGraph>(new ReplayGraph(
            global_backend(),
            [&](ggml_context* ctx) -> ggml_tensor* {
                int64_t in_ne[1] = { (int64_t)n_in_floats };
                ggml_tensor* in_all = graph_input_tensor(ctx, GGML_TYPE_F32, 1, in_ne,
                                      r->in_buf.data(), n_in_floats * sizeof(float));

                // ---- Prediction LSTM (mirrors prediction.cpp step()). ----
                // Layer-0 input is the token embedding (host-packed row at
                // emb_off); layer l>0 takes the previous layer's h'.
                ggml_tensor* layer_in = ggml_view_1d(ctx, in_all, Hp,
                                        r->emb_off * sizeof(float));
                ggml_tensor* top_h = nullptr;
                for (int l = 0; l < L; ++l) {
                    const std::string s = "_l" + std::to_string(l);
                    ggml_tensor* Wih = clone_weight(ctx, pml,
                        ("decoder.prediction.dec_rnn.lstm.weight_ih" + s).c_str());
                    ggml_tensor* Whh = clone_weight(ctx, pml,
                        ("decoder.prediction.dec_rnn.lstm.weight_hh" + s).c_str());
                    ggml_tensor* bih = clone_weight(ctx, pml,
                        ("decoder.prediction.dec_rnn.lstm.bias_ih" + s).c_str());
                    ggml_tensor* bhh = clone_weight(ctx, pml,
                        ("decoder.prediction.dec_rnn.lstm.bias_hh" + s).c_str());
                    size_t h_off = r->state_off + (size_t)(2 * l)     * Hp;
                    size_t c_off = r->state_off + (size_t)(2 * l + 1) * Hp;
                    ggml_tensor* h_in = ggml_view_1d(ctx, in_all, Hp, h_off * sizeof(float));
                    ggml_tensor* c_in = ggml_view_1d(ctx, in_all, Hp, c_off * sizeof(float));
                    ggml_tensor* z = ggml_add(ctx,
                        ggml_add(ctx, ggml_mul_mat(ctx, Wih, layer_in), bih),
                        ggml_add(ctx, ggml_mul_mat(ctx, Whh, h_in),     bhh));
                    ggml_tensor* i  = ggml_sigmoid(ctx, ggml_cont(ctx, ggml_view_1d(ctx, z, Hp, 0)));
                    ggml_tensor* f  = ggml_sigmoid(ctx, ggml_cont(ctx, ggml_view_1d(ctx, z, Hp, (size_t)Hp * sizeof(float))));
                    ggml_tensor* gg = ggml_tanh   (ctx, ggml_cont(ctx, ggml_view_1d(ctx, z, Hp, (size_t)2 * Hp * sizeof(float))));
                    ggml_tensor* o  = ggml_sigmoid(ctx, ggml_cont(ctx, ggml_view_1d(ctx, z, Hp, (size_t)3 * Hp * sizeof(float))));
                    ggml_tensor* c_out = ggml_add(ctx, ggml_mul(ctx, f, c_in), ggml_mul(ctx, i, gg));
                    ggml_tensor* h_out = ggml_mul(ctx, o, ggml_tanh(ctx, c_out));
                    capture_graph_output(c_out, &r->cap_c[l]);
                    capture_graph_output(h_out, &r->cap_h[l]);
                    layer_in = h_out;
                    top_h    = h_out;
                }
                // top_h is the prediction output g [Hp] -- feeds the joint on-device.

                // ---- Joint (mirrors step_logits / step_argmax). ----
                ggml_tensor* ep = ggml_view_1d(ctx, in_all, Hj, 0);
                ggml_tensor* Wp = clone_weight(ctx, ml_, "joint.pred.weight");
                ggml_tensor* pp = ggml_mul_mat(ctx, Wp, top_h);          // [Hj]
                ggml_tensor* bp = clone_weight(ctx, ml_, "joint.pred.bias");
                pp = ggml_add(ctx, pp, bp);
                ggml_tensor* f = ggml_relu(ctx, ggml_add(ctx, ep, pp));  // [Hj]
                ggml_tensor* Wo = clone_weight(ctx, ml_, "joint.joint_net.2.weight");
                ggml_tensor* y  = ggml_mul_mat(ctx, Wo, f);              // [V]
                ggml_tensor* bo = clone_weight(ctx, ml_, "joint.joint_net.2.bias");
                y = ggml_add(ctx, y, bo);                                // [V_plus]

                // ---- Argmax (token slice + duration slice) ON DEVICE. ----
                ggml_tensor* tok_view = ggml_view_1d(ctx, y, token_count, 0);
                ggml_tensor* dur_view = ggml_view_1d(ctx, y, num_dur,
                                        (size_t)token_count * sizeof(float));
                ggml_tensor* tok_amax = ggml_argmax(ctx, tok_view);  // i32 [1] (output)
                ggml_tensor* dur_amax = ggml_argmax(ctx, dur_view);  // i32 [1] (capture)
                capture_graph_output(dur_amax, &r->cap_dur_amax_f);
                return tok_amax;
            }));
        assert(r->rg->n_inputs() == 1 && "fused step graph must have 1 coalesced input");
        fused_replay_ = std::move(pending);
    }

    // Host-pack the coalesced input (no syncs): enc_proj_t, then the looked-up
    // embedding row (zeros for SOS), then the committed LSTM state per layer.
    FusedReplay* r = fused_replay_.get();
    const std::vector<float>& emb = pred.embed_host();
    // The embedding table must be resident on the host (warmed by pred.step on
    // the first unfused step, or a prior utterance). Guard defensively: if it is
    // still empty, run the unfused path this once (it populates embed_host_ for
    // subsequent fused steps).
    if (emb.empty()) {
        std::vector<float> g;
        pred.step(token_id, is_sos, in_state, g, out_state);
        std::vector<float> logits;
        step_logits(enc_proj_t, g.data(), (int)g.size(), logits);
        auto amax = [](const float* p, int n) -> int {
            int best = 0; float bv = p[0];
            for (int i = 1; i < n; ++i) if (p[i] > bv) { bv = p[i]; best = i; }
            return best;
        };
        k_out   = amax(logits.data(), token_count);
        d_k_out = amax(logits.data() + token_count, num_dur);
        return;
    }
    std::memcpy(r->in_buf.data(), enc_proj_t, (size_t)r->H_joint * sizeof(float));
    if (is_sos) {
        std::memset(r->in_buf.data() + r->emb_off, 0, (size_t)r->H_pred * sizeof(float));
    } else {
        assert(token_id >= 0 && token_id < pred.vocab_p1() && "embedding id out of range");
        std::memcpy(r->in_buf.data() + r->emb_off,
                    &emb[(size_t)token_id * r->H_pred],
                    (size_t)r->H_pred * sizeof(float));
    }
    for (int l = 0; l < r->L_pred; ++l) {
        std::memcpy(r->in_buf.data() + r->state_off + (size_t)(2 * l)     * r->H_pred,
                    in_state.h[l].data(), (size_t)r->H_pred * sizeof(float));
        std::memcpy(r->in_buf.data() + r->state_off + (size_t)(2 * l + 1) * r->H_pred,
                    in_state.c[l].data(), (size_t)r->H_pred * sizeof(float));
    }
    r->rg->set_input(0, r->in_buf.data(), r->in_nbytes);

    // ONE replay + ONE sync: reads back the token argmax (output), the duration
    // argmax, and the new LSTM (h', c') per layer -- all as batched async copies.
    out_state.h.assign((size_t)r->L_pred, std::vector<float>((size_t)r->H_pred));
    out_state.c.assign((size_t)r->L_pred, std::vector<float>((size_t)r->H_pred));
    std::vector<float> out_f;
    bool ok = r->rg->compute_with_captures(out_f);
    assert(ok && "fused step replay failed");
    (void)ok;
    static_assert(sizeof(int32_t) == sizeof(float), "int32 and float must be 4 bytes");
    std::memcpy(&k_out,   out_f.data(), sizeof(int32_t));
    std::memcpy(&d_k_out, r->cap_dur_amax_f.data(), sizeof(int32_t));
    for (int l = 0; l < r->L_pred; ++l) {
        std::memcpy(out_state.h[l].data(), r->cap_h[l].data(), (size_t)r->H_pred * sizeof(float));
        std::memcpy(out_state.c[l].data(), r->cap_c[l].data(), (size_t)r->H_pred * sizeof(float));
    }
}

} // namespace starling::ggml::parakeet
