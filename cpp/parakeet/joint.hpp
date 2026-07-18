// joint.hpp — parakeet-tdt RNNT joint network (Phase 1c).
//
// Starling-authored port of parakeet.cpp's joint.cpp (the CPU run_graph path
// for step_logits/step_argmax), bit-for-bit. The joint combines an encoder
// projection row (enc_proj_t) and a prediction output (g) into the TDT logits
// vector [V_plus = vocab+1 + num_durations]:
//
//   pp = joint.pred.weight · g + joint.pred.bias                       # [H]
//   f  = relu(enc_proj_t + pp)                                         # [H]
//   logits = joint.joint_net.2.weight · f + joint.joint_net.2.bias     # [V_plus]
//
// (joint.enc projection is ALREADY applied by the encoder Phase 1b output
// `enc_proj` — do NOT re-apply joint.enc here. The decoder has no projector
// either; g IS the top LSTM layer's h'. See prediction.hpp.)
//
// argmax split: logits[0 : vocab+1] -> token id k; logits[vocab+1 : V_plus]
// -> duration index d_k (-> durations[d_k] skip). The host argmax is taken over
// the logits vector read back from the step_logits graph (CPU path; simplest
// and byte-identical, the validation reference).

#pragma once

#include "config.hpp"
#include "runtime/model_loader.hpp"

#include <vector>

namespace starling::ggml::parakeet {

class Joint {
public:
    Joint(const ModelLoader& ml, const Config& cfg);

    // Run the joint graph for one (enc_proj_t, g) step and read back the full
    // [V_plus] logits vector. One run_graph call (CPU reference path).
    //   enc_proj_t: row t of the precomputed encoder projection [H] (frame-major
    //               [T, H] buffer from the encoder Phase 1b output).
    //   g:          prediction output [pred_hidden] (top LSTM layer's h').
    //   logits:     OUT — [V_plus] = [vocab+1 + num_durations].
    void step_logits(const float* enc_proj_t,
                     const float* g, int pred_hidden,
                     std::vector<float>& logits) const;

    // step_logits + host argmax over the token slice [0, vocab+1) and the
    // duration slice [vocab+1, V_plus). Writes the token argmax to k_out and
    // the duration argmax to d_k_out. (CPU reference path.)
    void step_argmax(const float* enc_proj_t, int token_count,
                     const float* g, int pred_hidden,
                     int& k_out, int& d_k_out) const;

    int joint_hidden() const { return joint_hidden_; }
    int pred_hidden() const  { return pred_hidden_; }
    int enc_hidden() const   { return enc_hidden_; }
    int vocab_size() const   { return vocab_size_; }
    int num_durations() const { return num_durations_; }
    int V_plus() const       { return V_plus_; }

private:
    const ModelLoader& ml_;
    int joint_hidden_  = 0;  // joint.enc.weight ne[1] (= 640)
    int enc_hidden_    = 0;  // joint.enc.weight ne[0] (= 1024, unused here)
    int pred_hidden_   = 0;  // joint.pred.weight ne[0] (= 640)
    int vocab_size_    = 0;  // config.vocab_size (= 8192)
    int num_durations_ = 0;  // config.tdt_durations.size() (= 5)
    int V_plus_        = 0;  // vocab + 1 + num_durations (= 8198)
};

} // namespace starling::ggml::parakeet
