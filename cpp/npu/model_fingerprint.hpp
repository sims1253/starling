// Copyright 2026 The Starling Authors. Apache-2.0.
//
// The two fingerprints of encoder_contract.hpp from a Qwen3-ASR model's ggml
// tensors. The engine passes its loaded tensors (the mapped GGUF), npu_tool
// a GGUF file read tensor by tensor (gguf_fingerprint.cpp); both see each
// tensor's bytes exactly as the GGUF stores them, so they agree.
#pragma once

#include "encoder_contract.hpp"
#include "ggml.h"

#include <functional>
#include <string>
#include <vector>

namespace starling::npu {

struct ModelFingerprint {
    uint64_t model_hash = 0;   // qwen3_encoder_digest: names the artifact
    uint64_t identity = 0;     // the checkpoint identity, compared with export_encoder.py's
};

// tensor(name): that tensor with `data` pointing at its bytes, or nullptr;
// the pointer only has to stay valid until the next call.
using TensorLookup = std::function<const ggml_tensor*(const std::string& name)>;

inline bool qwen3_identity(const TensorLookup& tensor, uint64_t& identity, std::string& err) {
    uint64_t h = kFnvBasis;
    for (const char* name : kQwen3EncFingerprintTensors) {
        const ggml_tensor* t = tensor(name);
        if (!t || t->type != GGML_TYPE_F32 || !t->data) {
            err = std::string("no checkpoint identity: ") + name + " missing or not F32";
            return false;
        }
        h = fnv1a_update(h, t->data, ggml_nbytes(t));
    }
    identity = h;
    return true;
}

// names: every tensor of the model (the encoder's are picked out here).
inline bool qwen3_model_hash(const std::vector<std::string>& names, const TensorLookup& tensor, uint64_t& hash,
                             std::string& err) {
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
    hash = qwen3_encoder_digest(std::move(ts));
    return true;
}

}  // namespace starling::npu
