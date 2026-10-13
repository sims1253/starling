// Copyright 2026 The Starling Authors. Apache-2.0.
//
// The model hash of model_fingerprint.hpp from a GGUF file (npu_tool on the
// phone, the tests on the host): ggml's GGUF parser for the tensor table, then
// each tensor's bytes read from the file as stored.
#include "gguf_fingerprint.hpp"

#include "gguf.h"

#include <fstream>
#include <memory>
#include <vector>

namespace starling::npu {

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
    if (!qwen3_model_hash(names, read, hash, err)) {
        err += " (" + path + ")";
        return false;
    }
    return true;
}

}  // namespace starling::npu
