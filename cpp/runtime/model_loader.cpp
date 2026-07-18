// model_loader.cpp — GGUF parse + weight realize.
//
// Opens the GGUF with gguf_init_from_file(no_alloc=false) so the whole weight
// region lives in one contiguous memory-backed ggml_context (zero-copy on CPU).
// Reads KV metadata + maps tensor name -> ggml_tensor*. realize_weights() gives
// each weight a backend buffer (zero-copy borrow on CPU, mirrored upload on
// GPU) so clone_weight() can hand them to graphs as already-allocated leaves.

#include "model_loader.hpp"
#include "backend.hpp"

#include "ggml.h"
#include "gguf.h"
#include "ggml-backend.h"

#include <cstring>
#include <string>
#include <vector>

namespace starling::ggml {

ModelLoader::ModelLoader() = default;

ModelLoader::~ModelLoader() {
    // The ggml context owns the weight tensors; freeing it frees them all.
    if (ctx_) ggml_free(ctx_);
    if (gguf_ctx_) gguf_free(gguf_ctx_);
}

bool ModelLoader::load(const char* path) {
    // gguf_init_params { no_alloc, ctx } — ctx!=NULL means allocate the weight
    // tensors into a freshly-created ggml_context (zero-copy mmap-backed).
    gguf_init_params params = {
        /*.no_alloc =*/ false,
        /*.ctx      =*/ &ctx_,
    };
    gguf_ctx_ = gguf_init_from_file(path, params);
    if (!gguf_ctx_) {
        error_ = std::string("failed to open GGUF: ") + path;
        return false;
    }
    // Map every tensor name -> tensor*.
    const int64_t n_tensors = gguf_get_n_tensors(gguf_ctx_);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char* name = gguf_get_tensor_name(gguf_ctx_, i);
        ggml_tensor* t = name ? ggml_get_tensor(ctx_, name) : nullptr;
        if (name && t) tensors_[name] = t;
    }
    // Read KV metadata.
    const int64_t n_kv = gguf_get_n_kv(gguf_ctx_);
    for (int64_t i = 0; i < n_kv; ++i) {
        const char* key = gguf_get_key(gguf_ctx_, i);
        const enum gguf_type t = gguf_get_kv_type(gguf_ctx_, i);
        if (!key) continue;
        GgufValue v;
        switch (t) {
            case GGUF_TYPE_STRING:
                v.kind = GgufValue::Kind::k_str;
                v.s = gguf_get_val_str(gguf_ctx_, i);
                break;
            case GGUF_TYPE_UINT8: case GGUF_TYPE_INT8:
            case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16:
            case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32:
            case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64:
                v.kind = GgufValue::Kind::k_int;
                v.i = gguf_get_val_i64(gguf_ctx_, i);
                break;
            case GGUF_TYPE_FLOAT32: case GGUF_TYPE_FLOAT64:
                v.kind = GgufValue::Kind::k_float;
                v.f = gguf_get_val_f64(gguf_ctx_, i);
                break;
            case GGUF_TYPE_ARRAY: {
                const enum gguf_type at = gguf_get_arr_type(gguf_ctx_, i);
                const size_t n = gguf_get_arr_n(gguf_ctx_, i);
                if (at == GGUF_TYPE_STRING) {
                    v.kind = GgufValue::Kind::k_arr_str;
                    v.arr_s.reserve(n);
                    for (size_t j = 0; j < n; ++j)
                        v.arr_s.emplace_back(gguf_get_arr_str(gguf_ctx_, i, j));
                } else {
                    // Integer array: gguf_get_arr_data returns the raw buffer;
                    // interpret according to `at`.
                    v.kind = GgufValue::Kind::k_arr_int;
                    const void* raw = gguf_get_arr_data(gguf_ctx_, i);
                    v.arr_i.resize(n);
                    for (size_t j = 0; j < n; ++j) {
                        switch (at) {
                            case GGUF_TYPE_INT8:  v.arr_i[j] = ((const int8_t*)raw)[j]; break;
                            case GGUF_TYPE_UINT8: v.arr_i[j] = ((const uint8_t*)raw)[j]; break;
                            case GGUF_TYPE_INT16: v.arr_i[j] = ((const int16_t*)raw)[j]; break;
                            case GGUF_TYPE_UINT16:v.arr_i[j] = ((const uint16_t*)raw)[j]; break;
                            case GGUF_TYPE_INT32: v.arr_i[j] = ((const int32_t*)raw)[j]; break;
                            case GGUF_TYPE_UINT32:v.arr_i[j] = ((const uint32_t*)raw)[j]; break;
                            case GGUF_TYPE_INT64: v.arr_i[j] = ((const int64_t*)raw)[j]; break;
                            case GGUF_TYPE_UINT64:v.arr_i[j] = ((const uint64_t*)raw)[j]; break;
                            default: v.arr_i[j] = 0; break;
                        }
                    }
                }
                break;
            }
            default: continue;
        }
        kv_[key] = std::move(v);
    }
    return true;
}

// ---- typed KV accessors -----------------------------------------------------
bool ModelLoader::kv_str(const std::string& key, std::string& out) const {
    auto it = kv_.find(key);
    if (it == kv_.end() || it->second.kind != GgufValue::Kind::k_str) return false;
    out = it->second.s; return true;
}
bool ModelLoader::kv_int(const std::string& key, int64_t& out) const {
    auto it = kv_.find(key);
    if (it == kv_.end() || it->second.kind != GgufValue::Kind::k_int) return false;
    out = it->second.i; return true;
}
bool ModelLoader::kv_float(const std::string& key, double& out) const {
    auto it = kv_.find(key);
    if (it == kv_.end() || it->second.kind != GgufValue::Kind::k_float) return false;
    out = it->second.f; return true;
}
bool ModelLoader::kv_arr_str(const std::string& key, std::vector<std::string>& out) const {
    auto it = kv_.find(key);
    if (it == kv_.end() || it->second.kind != GgufValue::Kind::k_arr_str) return false;
    out = it->second.arr_s; return true;
}
bool ModelLoader::kv_arr_int(const std::string& key, std::vector<int64_t>& out) const {
    auto it = kv_.find(key);
    if (it == kv_.end() || it->second.kind != GgufValue::Kind::k_arr_int) return false;
    out = it->second.arr_i; return true;
}

// ---- tensors ----------------------------------------------------------------
ggml_tensor* ModelLoader::tensor(const char* name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : it->second;
}
std::vector<std::string> ModelLoader::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensors_.size());
    for (const auto& kv : tensors_) names.push_back(kv.first);
    return names;
}

bool ModelLoader::realize_weights(Backend& backend) {
    if (realized_) return true;
    if (!ctx_) { error_ = "realize_weights: no loaded context"; return false; }
    // On CPU: borrow the loader's memory-backed context zero-copy.
    // On GPU: mirror each weight into a device-side context + upload.
    if (!backend.is_gpu()) {
        // Borrow the loader's memory-backed ggml_context zero-copy. The public
        // accessors ggml_get_mem_buffer / ggml_get_mem_size give the mmap'd region.
        ggml_backend_buffer_t buf = ggml_backend_cpu_buffer_from_ptr(
            ggml_get_mem_buffer(ctx_), ggml_get_mem_size(ctx_));
        // Point each tensor's ->buffer at the borrowed buffer.
        for (const auto& kv : tensors_) {
            ggml_tensor* t = kv.second;
            t->buffer = buf;
            // data pointer already set by the gguf mmap.
        }
    } else {
        // GPU: allocate a no_alloc device context, mirror tensors, upload.
        struct ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead() * (tensors_.size() + 16),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context* dev_ctx = ggml_init(params);
        // Snapshot the CPU tensors (name -> src data) before repointing the map.
        std::vector<std::pair<std::string, ggml_tensor*>> cpu_tensors;
        cpu_tensors.reserve(tensors_.size());
        for (const auto& kv : tensors_) cpu_tensors.emplace_back(kv.first, kv.second);
        // Repoint the name map at device tensors.
        for (const auto& kv : cpu_tensors) {
            ggml_tensor* src = kv.second;
            ggml_tensor* dst = ggml_dup_tensor(dev_ctx, src);
            ggml_set_name(dst, src->name);
            tensors_[kv.first] = dst;
        }
        ggml_backend_buffer_t dev_buf = ggml_backend_alloc_ctx_tensors(dev_ctx, backend.handle());
        if (!dev_buf) { error_ = "realize_weights: device alloc failed"; return false; }
        // Upload each tensor's bytes from the original CPU mmap'd data.
        for (const auto& kv : cpu_tensors) {
            ggml_tensor* dst = tensor(kv.first.c_str());
            if (dst && kv.second->data)
                ggml_backend_tensor_set(dst, kv.second->data, 0, ggml_nbytes(kv.second));
        }
    }
    realized_ = true;
    return true;
}

} // namespace starling::ggml
