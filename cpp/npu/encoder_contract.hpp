// Copyright 2026 The Starling Authors. Apache-2.0.
//
// What an NPU encoder artifact promises, shared by the engine (which looks for
// and checks artifacts) and npu_tool (which writes them):
//
//  * key: the graph — one Qwen3-ASR audio-encoder attention window
//    (n_window_infer mel frames -> tokens rows of the projector output), with
//    inputs [1, n_mels, frames] f32 mel and [1, 1, 1, tokens] f32 additive key
//    mask (0 valid, -1e4 padding) and output [1, tokens, out_dim] f32;
//  * model_hash: the model it was built for — FNV-1a over the raw bytes of a
//    few F32 tensors of the GGUF the engine loads (biases and norms are F32
//    in every Starling Qwen3-ASR GGUF, independent of the weight quant);
//  * vendor: the firmware's thrGetVendorApiVersion() it was compiled with.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>

namespace starling::npu {

inline constexpr const char* kQwen3EncFingerprintTensors[] = {
    "enc.conv1.bias", "enc.ln_post.weight", "enc.ln_post.bias", "proj.linear_2.bias", "enc.blk.0.attn_norm.weight",
};

inline uint64_t fnv1a_update(uint64_t h, const void* data, size_t size) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}
inline constexpr uint64_t kFnvBasis = 1469598103934665603ull;

inline std::string qwen3_enc_key(uint32_t n_mels, uint32_t frames, uint32_t tokens, uint32_t d, uint32_t layers,
                                 uint32_t heads, uint32_t ff, uint32_t ds, uint32_t out) {
    char b[200];
    std::snprintf(b, sizeof b, "qwen3-asr-enc/1 mels=%u frames=%u tokens=%u d=%u L=%u H=%u ff=%u ds=%u out=%u", n_mels,
                  frames, tokens, d, layers, heads, ff, ds, out);
    return b;
}

inline std::string qwen3_enc_artifact_name(uint64_t model_hash) {
    char b[64];
    std::snprintf(b, sizeof b, "qwen3-asr-enc-%016llx.stnpu", (unsigned long long)model_hash);
    return b;
}

}  // namespace starling::npu
