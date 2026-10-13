// Copyright 2026 The Starling Authors. Apache-2.0.
//
// The model hash of encoder_contract.hpp from a Qwen3-ASR model's ggml
// tensors and metadata. The engine passes its loaded tensors (the mapped
// GGUF) and the metadata ModelLoader read, npu_tool a GGUF file read tensor by
// tensor (gguf_fingerprint.cpp); both see each tensor's bytes exactly as the
// GGUF stores them and each metadata value as ModelLoader normalizes it, so
// they agree (and agree with scripts/npu/export_encoder.py, which reads the
// file the same way).
#pragma once

#include "encoder_contract.hpp"
#include "ggml.h"

#include <functional>
#include <string>
#include <vector>

namespace starling::npu {

// tensor(name): that tensor with `data` pointing at its bytes, or nullptr;
// the pointer only has to stay valid until the next call.
using TensorLookup = std::function<const ggml_tensor*(const std::string& name)>;

// The encoder metadata (is_qwen3_encoder_key) of a ModelLoader::kv() map
// (key -> GgufValue; a template so this header needs no runtime include).
// An array under an encoder key is an error: the digest has no form for it.
template <class KvMap>
bool qwen3_encoder_meta(const KvMap& kv, std::vector<EncoderMeta>& out, std::string& err) {
    using Kind = typename KvMap::mapped_type::Kind;
    out.clear();
    for (const auto& [key, v] : kv) {
        if (!is_qwen3_encoder_key(key)) continue;
        EncoderMeta m;
        m.key = key;
        switch (v.kind) {
            case Kind::k_int: m.kind = 'i'; m.i = v.i; break;
            case Kind::k_float: m.kind = 'f'; m.f = v.f; break;
            case Kind::k_str: m.kind = 's'; m.s = v.s; break;
            default:
                err = "encoder metadata " + key + " is an array (not covered by the model hash)";
                return false;
        }
        out.push_back(std::move(m));
    }
    return true;
}

// names: every tensor of the model (the encoder's are picked out here);
// meta: its encoder metadata (qwen3_encoder_meta).
inline bool qwen3_model_hash(const std::vector<std::string>& names, const TensorLookup& tensor,
                             const std::vector<EncoderMeta>& meta, uint64_t& hash, std::string& err) {
    std::vector<EncoderTensor> ts;
    for (const std::string& name : names) {
        if (!is_qwen3_encoder_tensor(name)) continue;
        const ggml_tensor* t = tensor(name);
        if (!t || !t->data) {
            err = "cannot read " + name;
            return false;
        }
        EncoderTensor e;
        e.name = name;
        e.type = (uint32_t)t->type;
        for (int d = 0; d < 4; ++d) e.ne[d] = (uint64_t)t->ne[d];
        e.nbytes = ggml_nbytes(t);
        e.data_hash = xxh64(t->data, e.nbytes);
        ts.push_back(std::move(e));
    }
    if (ts.empty()) {
        err = "no audio-encoder tensors (enc.*, proj.*)";
        return false;
    }
    hash = qwen3_encoder_digest(std::move(ts), meta);
    return true;
}

}  // namespace starling::npu
