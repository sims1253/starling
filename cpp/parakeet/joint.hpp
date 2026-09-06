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
#include "prediction.hpp"  // PredictionNet, PredState (for step_fused_argmax)
#include "runtime/model_loader.hpp"

#include <memory>
#include <vector>

namespace starling::ggml::parakeet {

class Joint {
public:
    Joint(const ModelLoader& ml, const Config& cfg);
    ~Joint();  // out-of-line: FusedReplay is only complete in the .cpp

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

    // GPU fast path for the TDT serial decode loop: ONE ReplayGraph spanning
    // the prediction LSTM + joint + token/duration argmax, so the prediction
    // output g flows pred -> joint ENTIRELY on the device (no host round-trip)
    // and only ONE host<-device sync happens per step (vs two for pred.step +
    // step_argmax). Byte-identical to pred.step + step_argmax (same math, same
    // ggml ops, same argmax tie-break); g simply never leaves the device. The
    // K-step multistep path also uses this for its eager step-0 init.
    //
    //   pred:        the prediction net (supplies the LSTM weights + embedding).
    //   enc_proj_t:  joint enc-projection row for frame t ([joint_hidden]).
    //   token_count: vocab + 1 (the token-slice width).
    //   token_id/is_sos: embedding lookup for the prediction-net input.
    //   in_state:    committed LSTM (h, c) per layer.
    //   out_state:   OUT new LSTM (h', c') per layer (carried to the next emit).
    //   k_out:       OUT argmax over logits[0:token_count).
    //   d_k_out:     OUT argmax over logits[token_count:V_plus).
    void step_fused_argmax(const PredictionNet& pred,
                           const float* enc_proj_t, int token_count,
                           int32_t token_id, bool is_sos,
                           const PredState& in_state,
                           PredState& out_state,
                           int& k_out, int& d_k_out) const;

    int joint_hidden() const { return joint_hidden_; }
    int pred_hidden() const  { return pred_hidden_; }
    int enc_hidden() const   { return enc_hidden_; }
    int vocab_size() const   { return vocab_size_; }
    int num_durations() const { return num_durations_; }
    int V_plus() const       { return V_plus_; }

    // Access used by the fused TDT decode step (ONE graph spanning the
    // prediction LSTM + joint + argmax to keep g on-device between them) and
    // the K-step multistep graph (to clone the joint weights as graph leaves).
    const ModelLoader& model_loader() const { return ml_; }

private:
    const ModelLoader& ml_;
    int joint_hidden_  = 0;  // joint.enc.weight ne[1] (= 640)
    int enc_hidden_    = 0;  // joint.enc.weight ne[0] (= 1024, unused here)
    int pred_hidden_   = 0;  // joint.pred.weight ne[0] (= 640)
    int vocab_size_    = 0;  // config.vocab_size (= 8192)
    int num_durations_ = 0;  // config.tdt_durations.size() (= 5)
    int V_plus_        = 0;  // vocab + 1 + num_durations (= 8198)

    // GPU-only: replayable FUSED prediction-LSTM + joint + argmax graph (one
    // sync per step). Lazily built on the first step_fused_argmax. Incomplete in
    // the header (ReplayGraph stays in the .cpp).
    struct FusedReplay;
};

} // namespace starling::ggml::parakeet
