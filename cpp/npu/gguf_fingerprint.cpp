// Copyright 2026 The Starling Authors. Apache-2.0.
//
// The model hash of model_fingerprint.hpp from a GGUF file (npu_tool on the
// phone, the tests on the host): ggml's GGUF parser for the metadata and the
// tensor table, then each tensor's bytes read from the file as stored.
#include "gguf_fingerprint.hpp"

#include "gguf.h"

#include <fstream>
#include <memory>
#include <vector>

namespace starling::npu {

namespace {

// The encoder metadata, normalized as ModelLoader reads it (model_loader.cpp:
// integers to int64, floats to double, strings; booleans skipped).
bool encoder_meta(const gguf_context* g, std::vector<EncoderMeta>& out, std::string& err) {
    for (int64_t i = 0; i < gguf_get_n_kv(g); ++i) {
        const std::string key = gguf_get_key(g, i);
        if (!is_qwen3_encoder_key(key)) continue;
        EncoderMeta m;
        m.key = key;
        switch (gguf_get_kv_type(g, i)) {
            case GGUF_TYPE_UINT8: m.i = gguf_get_val_u8(g, i); break;
            case GGUF_TYPE_INT8: m.i = gguf_get_val_i8(g, i); break;
            case GGUF_TYPE_UINT16: m.i = gguf_get_val_u16(g, i); break;
            case GGUF_TYPE_INT16: m.i = gguf_get_val_i16(g, i); break;
            case GGUF_TYPE_UINT32: m.i = gguf_get_val_u32(g, i); break;
            case GGUF_TYPE_INT32: m.i = gguf_get_val_i32(g, i); break;
            case GGUF_TYPE_UINT64: m.i = (int64_t)gguf_get_val_u64(g, i); break;
            case GGUF_TYPE_INT64: m.i = gguf_get_val_i64(g, i); break;
            case GGUF_TYPE_FLOAT32: m.kind = 'f'; m.f = gguf_get_val_f32(g, i); break;
            case GGUF_TYPE_FLOAT64: m.kind = 'f'; m.f = gguf_get_val_f64(g, i); break;
            case GGUF_TYPE_STRING: m.kind = 's'; m.s = gguf_get_val_str(g, i); break;
            case GGUF_TYPE_ARRAY:
                err = "encoder metadata " + key + " is an array (not covered by the model hash)";
                return false;
            default: continue;   // bool: ModelLoader does not read it
        }
        out.push_back(std::move(m));
    }
    return true;
}

}  // namespace

bool gguf_model_hash(const std::string& path, uint64_t& hash, std::string& err) {
    ggml_context* meta = nullptr;
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = &meta;
    std::unique_ptr<gguf_context, decltype(&gguf_free)> g(gguf_init_from_file(path.c_str(), params), gguf_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> hold(meta, ggml_free);
    if (!g || !meta) {
        err = "not a readable GGUF file: " + path;
        return false;
    }
    std::ifstream f(path, std::ios::binary);
    f.seekg(0, std::ios::end);
    const uint64_t file_size = f ? (uint64_t)f.tellg() : 0;
    if (!f) {
        err = "cannot open " + path;
        return false;
    }
    const uint64_t data0 = gguf_get_data_offset(g.get());
    std::vector<std::string> names;
    for (int64_t i = 0; i < gguf_get_n_tensors(g.get()); ++i) names.emplace_back(gguf_get_tensor_name(g.get(), i));
    // Reads `name` into buf; the tensor's data points there until the next read.
    std::vector<uint8_t> buf;
    TensorLookup read = [&](const std::string& name) -> const ggml_tensor* {
        const int64_t id = gguf_find_tensor(g.get(), name.c_str());
        ggml_tensor* t = id < 0 ? nullptr : ggml_get_tensor(meta, name.c_str());
        if (!t) return nullptr;
        const uint64_t off = data0 + gguf_get_tensor_offset(g.get(), id), n = ggml_nbytes(t);
        if (off > file_size || n > file_size - off) return nullptr;
        buf.resize(n);
        f.clear();
        f.seekg((std::streamoff)off);
        if (!f.read(reinterpret_cast<char*>(buf.data()), (std::streamsize)n)) return nullptr;
        t->data = buf.data();
        return t;
    };
    std::vector<EncoderMeta> enc_meta;
    if (!encoder_meta(g.get(), enc_meta, err) || !qwen3_model_hash(names, read, enc_meta, hash, err)) {
        err += " (" + path + ")";
        return false;
    }
    return true;
}

}  // namespace starling::npu
