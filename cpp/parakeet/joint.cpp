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

#include "runtime/backend.hpp"  // clone_weight, graph_input_tensor
#include "runtime/graph.hpp"    // run_graph

#include "ggml.h"

#include <algorithm>
#include <cassert>
#include <vector>

namespace starling::ggml::parakeet {

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

} // namespace starling::ggml::parakeet
