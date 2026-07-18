// tdt.hpp — parakeet-tdt serial greedy TDT decode loop (Phase 1c).
//
// Starling-authored port of parakeet.cpp's tdt.cpp:16-194 serial path. The TDT
// (Token-and-Duration Transducer) greedy decoder advances the encoder frame `t`
// by a per-step duration skip (the duration-class argmax of the joint's logits)
// and emits ONE token per step, looping while the duration is 0 (a "blank +
// dur-0" frame re-runs the prediction/joint without advancing t, up to
// max_symbols).
//
// IMPORTANT (differs from parakeet.cpp's hyp, which drops blanks): the golden
// parakeet_tdt_{short,medium,long}_ids.pt tensors INCLUDE the blank emissions
// (blank_id = 8192) — short has 5 blanks among 49 tokens. To match byte-exact,
// tdt_greedy() appends EVERY emitted token id (k, including blank) to the
// output stream. (The blank tokens are then naturally skipped at detokenize
// time: piece index 8192 is out of the [0, vocab_size=8192) range, so it
// contributes nothing to the text.)
//
// State carried across the loop:
//   - frame_idx t (encoder frame; init 0).
//   - committed LSTM state (h[L], c[L]; all zeros at start).
//   - last_token (init: blank; but tracked via emitted_any for SOS handling).
//   - emitted_any: false until the first non-blank emit (SOS = !emitted_any).
//   - g_valid: cached prediction output g is reused across non-emit steps
//     (blank-skip reuse; g depends only on (last_token, committed_state), which
//     only change on an emit).
//
// Loop structure mirrors parakeet.cpp's tdt_greedy serial path exactly:
//   while t < valid_len:
//     symbols_added = 0; need_loop = true; skip = 0
//     while need_loop and symbols_added < max_symbols:
//       enc_proj_t = enc_proj + t*H
//       if not g_valid: prediction.step(last_label, is_sos, committed) -> g
//       joint.step_argmax(enc_proj_t, g) -> k, d_k
//       skip = durations[d_k]
//       emit k  (EVERY step, including blank)
//       if k != blank_id: last_token=k; committed=out_state; emitted_any=true;
//                         g_valid=false
//       else: discard out_state (committed/last_token unchanged, g stays valid)
//       symbols_added += 1; t += skip; need_loop = (skip == 0)
//     if skip == 0: skip = 1   # blank+dur0 -> force step 1
//     if symbols_added == max_symbols: t += 1
// Stop when t >= valid_len.

#pragma once

#include "joint.hpp"
#include "prediction.hpp"

#include <vector>

namespace starling::ggml::parakeet {

// Serial TDT greedy decode.
//   pred:        prediction net (LSTM).
//   joint:       joint net (enc/pred projection + joint_net.2).
//   enc_proj:    precomputed joint.enc projection over all T frames — i.e. the
//                Phase 1b encoder output, row-major [T * H] (enc_proj[t*H + h],
//                frame t's projected vector contiguous).
//   T:           number of encoder frames (valid_len == T for the goldens).
//   H:           joint_hidden (frame-major stride for enc_proj).
//   durations:   TDT duration classes (e.g. [0,1,2,3,4]).
//   blank_id:    blank token id (= vocab_size = 8192).
//   max_symbols: cap on consecutive dur-0 emits at a single frame.
//
// Returns the emitted token id stream INCLUDING blanks (matches the golden
// parakeet_tdt_*_ids.pt byte-for-byte).
std::vector<int32_t> tdt_greedy(const PredictionNet& pred, const Joint& joint,
                                const std::vector<float>& enc_proj,
                                int T, int H,
                                const std::vector<int32_t>& durations,
                                int blank_id, int max_symbols);

} // namespace starling::ggml::parakeet
