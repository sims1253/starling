// Copyright 2026 The Starling Authors. Apache-2.0.
//
// What an NPU encoder artifact promises, shared by the engine (which looks for
// and checks artifacts) and npu_tool (which writes them):
//
//  * key: the graph — one Qwen3-ASR audio-encoder attention window
//    (n_window_infer mel frames -> tokens rows of the projector output), with
//    inputs [1, n_mels, frames] f32 mel and [1, 1, 1, tokens] f32 additive key
//    mask (0 valid, -1e4 padding) and output [1, tokens, out_dim] f32;
//  * model_hash: the GGUF it was built for — a digest of every audio-encoder
//    and projector tensor exactly as the GGUF stores it (name, type, shape
//    and raw bytes, any quant; qwen3_encoder_digest). The graph's weights are
//    that GGUF's own encoder tensors (scripts/npu/export_encoder.py exports
//    from the file and computes the same digest), so an artifact never runs
//    beside encoder weights other than the ones it was built from, a
//    weights-only fine-tune included;
//  * vendor: the firmware's thrGetVendorApiVersion() it was compiled with.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace starling::npu {

// FNV-1a 64 (standard offset basis and prime; export_encoder.py uses the same).
inline constexpr uint64_t kFnvBasis = 0xcbf29ce484222325ull;
inline constexpr uint64_t kFnvPrime = 0x100000001b3ull;
inline uint64_t fnv1a_update(uint64_t h, const void* data, size_t size) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i) h = (h ^ p[i]) * kFnvPrime;
    return h;
}

// XXH64 (seed 0), the per-tensor hash of the encoder digest (export_encoder.py
// has the same): hashes 8-byte words, a few GB/s per core, where byte-wise
// FNV would make hashing a few hundred MB of weights a visible part of the
// model load.
inline uint64_t xxh64(const void* data, size_t len, uint64_t seed = 0) {
    constexpr uint64_t P1 = 0x9e3779b185ebca87ull, P2 = 0xc2b2ae3d27d4eb4full, P3 = 0x165667b19e3779f9ull,
                       P4 = 0x85ebca77c2b2ae63ull, P5 = 0x27d4eb2f165667c5ull;
    auto rotl = [](uint64_t x, int r) { return (x << r) | (x >> (64 - r)); };
    auto rd64 = [](const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; };   // little-endian hosts
    auto rd32 = [](const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return (uint64_t)v; };
    auto mix = [&](uint64_t acc, uint64_t in) { return rotl(acc + in * P2, 31) * P1; };
    auto merge = [&](uint64_t acc, uint64_t v) { return (acc ^ mix(0, v)) * P1 + P4; };
    const uint8_t* p = static_cast<const uint8_t*>(data);
    const uint8_t* const end = p + len;
    uint64_t h;
    if (len >= 32) {
        uint64_t v1 = seed + P1 + P2, v2 = seed + P2, v3 = seed, v4 = seed - P1;
        for (const uint8_t* const limit = end - 32; p <= limit; p += 32) {
            v1 = mix(v1, rd64(p));
            v2 = mix(v2, rd64(p + 8));
            v3 = mix(v3, rd64(p + 16));
            v4 = mix(v4, rd64(p + 24));
        }
        h = rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18);
        h = merge(merge(merge(merge(h, v1), v2), v3), v4);
    } else {
        h = seed + P5;
    }
    h += (uint64_t)len;
    for (; end - p >= 8; p += 8) h = rotl(h ^ mix(0, rd64(p)), 27) * P1 + P4;
    if (end - p >= 4) {
        h = rotl(h ^ (rd32(p) * P1), 23) * P2 + P3;
        p += 4;
    }
    for (; p < end; ++p) h = rotl(h ^ (*p * P5), 11) * P1;
    h ^= h >> 33;
    h *= P2;
    h ^= h >> 29;
    h *= P3;
    h ^= h >> 32;
    return h;
}

// ---- encoder digest (the artifact's model_hash) ----
// The tensors it covers: the audio encoder ("enc.") and the projector ("proj.").
inline bool is_qwen3_encoder_tensor(const std::string& name) {
    return name.rfind("enc.", 0) == 0 || name.rfind("proj.", 0) == 0;
}

// One tensor as the GGUF stores it: ggml type, ne[0..3] (unused dims 1),
// byte size and xxh64 of those bytes.
struct EncoderTensor {
    std::string name;
    uint32_t type = 0;
    uint64_t ne[4] = {1, 1, 1, 1};
    uint64_t nbytes = 0;
    uint64_t data_hash = 0;
};

// FNV-1a 64 over, per tensor in name order: the name with its NUL, the
// type (u32), ne[4] (u64), the byte size (u64) and the xxh64 of the bytes
// (u64), all little-endian.
inline uint64_t qwen3_encoder_digest(std::vector<EncoderTensor> ts) {
    std::sort(ts.begin(), ts.end(), [](const EncoderTensor& a, const EncoderTensor& b) { return a.name < b.name; });
    uint64_t h = kFnvBasis;
    for (const EncoderTensor& t : ts) {
        h = fnv1a_update(h, t.name.c_str(), t.name.size() + 1);
        h = fnv1a_update(h, &t.type, sizeof t.type);
        h = fnv1a_update(h, t.ne, sizeof t.ne);
        h = fnv1a_update(h, &t.nbytes, sizeof t.nbytes);
        h = fnv1a_update(h, &t.data_hash, sizeof t.data_hash);
    }
    return h;
}

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
