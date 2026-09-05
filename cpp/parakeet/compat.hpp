// compat.hpp — community-GGUF dialect normalization for parakeet.
//
// parakeet GGUFs in the wild use (at least) three tensor/KV contracts:
//
//   1. starling native  — this repo's converter output
//   2. parakeet.cpp / CrispASR style (e.g. cstr/parakeet-tdt-0.6b-v3-GGUF):
//      decoder.lstm.N.w_ih / encoder.layers.N.attn.q.weight /
//      encoder.pre.* / joint.out.* names, flat parakeet.* KV, and embedded
//      preprocessor.fb / preprocessor.window tensors
//   3. transcribe.cpp style (e.g. handy-computer/parakeet-tdt-0.6b-v3-gguf):
//      enc.blocks.N.* / pred.lstm.N.{Wx,Wh,bias} names with a FUSED LSTM
//      bias, stt.* KV, and NO filterbank tensor (the runtime computes mel
//      from stt.frontend.* metadata)
//
// apply_community_dialect_compat() detects 2/3 after load and normalizes
// them onto the native contract WITHOUT touching tensor data: name aliases
// (both spellings resolve to the same tensor), synthesized KV, and — for
// dialect 3 — a synthesized slaney mel filterbank + hann window and the
// fused-bias split (bias_ih = fused, bias_hh = 0; mathematically identical
// since the two biases only ever sum).

#pragma once

#include <string>

namespace starling::ggml {
class ModelLoader;
}

namespace starling::ggml::parakeet {

// Detect + normalize. Returns true when the model is native or successfully
// normalized; false + err when a recognized dialect is malformed.
bool apply_community_dialect_compat(ModelLoader& ml, std::string& err);

} // namespace starling::ggml::parakeet
