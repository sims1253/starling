// model_loader.cpp — GGUF parse + weight realize.
//
// Parses the GGUF header (gguf_init_from_file, no_alloc) and maps the file
// privately: every tensor points into the mapping, so the weight region is
// one contiguous host range (zero-copy on CPU) whose pages load on first
// touch. The fast engines repack straight from it and drop the pages they
// consumed; nothing holds a second copy. Platforms without mmap, and
// STARLING_GGUF_MMAP=0, read the file into the ggml context instead.
// Reads KV metadata + maps tensor name -> ggml_tensor*. realize_weights() gives
// each weight a backend buffer (zero-copy borrow on CPU, mirrored upload on
// GPU) so clone_weight() can hand them to graphs as already-allocated leaves.

#include "model_loader.hpp"
#include "backend.hpp"
#include "cpu_repack.hpp"
#include "graph.hpp"

#include "ggml.h"
#include "gguf.h"
#include "ggml-backend.h"

#if defined(__unix__) || defined(__APPLE__)
#define STARLING_LOADER_MMAP 1
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#else
#define STARLING_LOADER_MMAP 0
#endif

#include <chrono>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_set>
#include <stdexcept>
#include <string>
#include <vector>

namespace starling::ggml {

namespace {
std::unordered_set<ModelLoader*> loaders;
std::unordered_set<ModelLoader*>& live_loaders() { return loaders; }
}

ModelLoader::ModelLoader() {
    std::lock_guard<std::recursive_mutex> lock(runtime_mutex());
    live_loaders().insert(this);
}

void ModelLoader::release_runtime_resources() {
    while (!caches_.empty()) caches_.pop_back();
    if (weight_buffer_) cpu_repack::detach(weight_buffer_);
    if (weight_buffer_) ggml_backend_buffer_free(weight_buffer_);
    weight_buffer_ = nullptr;
    if (device_ctx_) {
        tensors_.swap(host_tensors_);
        host_tensors_.clear();
    }
    for (auto& item : tensors_) item.second->buffer = nullptr;
    if (device_ctx_) ggml_free(device_ctx_);
    device_ctx_ = nullptr;
    realized_ = false;
}

void ModelLoader::release_all_runtime_resources() {
    std::lock_guard<std::recursive_mutex> lock(runtime_mutex());
    for (auto* loader : live_loaders()) loader->release_runtime_resources();
}

ModelLoader::~ModelLoader() {
    std::lock_guard<std::recursive_mutex> lock(runtime_mutex());
    release_runtime_resources();
    live_loaders().erase(this);
    // The ggml context owns the weight tensors (and, unless the file is
    // mapped, their data); freeing it frees them all.
    if (data_) cpu_repack::forget(data_, data_size_);
    if (ctx_) ggml_free(ctx_);
    if (compat_ctx_) ggml_free(compat_ctx_);
    if (gguf_ctx_) gguf_free(gguf_ctx_);
#if STARLING_LOADER_MMAP
    if (map_addr_) munmap(map_addr_, map_len_);
#endif
}

namespace {
// STARLING_GGUF_MMAP=0 reads the weights into anonymous memory instead of
// mapping the file (A/B measurements; platforms without mmap always read).
bool want_mmap() {
#if STARLING_LOADER_MMAP
    const char* e = std::getenv("STARLING_GGUF_MMAP");
    return !(e && e[0] == '0');
#else
    return false;
#endif
}
} // namespace

bool ModelLoader::map_weights(const char* path) {
#if STARLING_LOADER_MMAP
    // Metadata only: the tensors get no data until pointed into the map.
    gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &ctx_,
    };
    gguf_ctx_ = gguf_init_from_file(path, params);
    if (!gguf_ctx_) {
        error_ = std::string("failed to open GGUF: ") + path;
        return false;
    }
    // ggml's CPU buffers need 32-byte aligned tensor data: a file aligned
    // more loosely is read instead (empty error_ = fall back).
    if (gguf_get_alignment(gguf_ctx_) % 32 != 0) {
        gguf_free(gguf_ctx_);
        gguf_ctx_ = nullptr;
        if (ctx_) ggml_free(ctx_);
        ctx_ = nullptr;
        return false;
    }
    const int fd = open(path, O_RDONLY | O_CLOEXEC);
    struct stat st;
    if (fd < 0 || fstat(fd, &st) != 0) {
        if (fd >= 0) close(fd);
        error_ = std::string("failed to open GGUF: ") + path;
        return false;
    }
    const size_t file_size = (size_t)st.st_size;
    const size_t data_off = gguf_get_data_offset(gguf_ctx_);
    if (data_off > file_size) {
        close(fd);
        error_ = std::string("GGUF data offset past the end of ") + path;
        return false;
    }
    // Private and writable: pages stay shared with the page cache until
    // written (ggml's CPU repack rewrites weights in place: copy-on-write),
    // and the kernel can drop untouched pages under pressure instead of
    // swapping them.
    void* p = file_size ? mmap(nullptr, file_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0) : nullptr;
#if defined(__linux__)
    // Cold loads: stream the weight region from storage in the background
    // (readahead on a descriptor of its own) while the engine sets up and
    // plans; the repack then mostly finds the pages cached. A file already
    // in the page cache (sampled with mincore) is left alone: walking its
    // cached pages contends with the repack's page faults (host: +250 ms on
    // a warm Qwen3 load). STARLING_GGUF_READAHEAD=0/1 never/always.
    const char* ra = std::getenv("STARLING_GGUF_READAHEAD");
    if (p != MAP_FAILED && file_size > data_off && !(ra && ra[0] == '0')) {
        const int rfd = dup(fd);
        const bool always = ra && ra[0] == '1';
        if (rfd >= 0) {
            try {
                std::thread([rfd, p, data_off, file_size, always] {
                    // 256 one-page probes over the weight region (mincore
                    // reports page-cache residency for file mappings and
                    // reads nothing; a mapping gone meanwhile just fails).
                    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
                    const size_t first = (data_off + page - 1) / page, last = file_size / page;
                    int cached = 0, probes = 0;
                    for (int k = 0; k < 256 && last > first; ++k, ++probes) {
                        unsigned char v = 0;
                        const size_t pg = first + (last - first) * (size_t)k / 256;
                        if (mincore((uint8_t*)p + pg * page, page, &v) == 0 && (v & 1)) ++cached;
                    }
                    if (always || cached < probes * 9 / 10)
                        posix_fadvise(rfd, (off_t)data_off, (off_t)(file_size - data_off), POSIX_FADV_WILLNEED);
                    close(rfd);
                }).detach();
            } catch (...) {
                close(rfd);
            }
        }
    }
#endif
    close(fd);
    if (file_size && p == MAP_FAILED) {
        error_ = std::string("failed to map GGUF: ") + path;
        return false;
    }
    map_addr_ = p;
    map_len_ = file_size;
    data_ = (uint8_t*)p + data_off;
    data_size_ = file_size - data_off;
    const int64_t n_tensors = gguf_get_n_tensors(gguf_ctx_);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char* name = gguf_get_tensor_name(gguf_ctx_, i);
        ggml_tensor* t = name ? ggml_get_tensor(ctx_, name) : nullptr;
        if (!t) continue;
        const size_t off = gguf_get_tensor_offset(gguf_ctx_, i);
        if (off > data_size_ || ggml_nbytes(t) > data_size_ - off) {
            error_ = std::string("GGUF tensor ") + name + " extends past the end of " + path;
            return false;
        }
        t->data = data_ + off;
    }
    return true;
#else
    (void)path;
    error_ = "GGUF mapping is not supported on this platform";
    return false;
#endif
}

bool ModelLoader::load(const char* path) {
    if (ctx_ || gguf_ctx_) { error_ = "model loader is already loaded"; return false; }
    if (!path || !*path) { error_ = "empty GGUF path"; return false; }
    const auto t0 = std::chrono::steady_clock::now();
    if (want_mmap() && !map_weights(path) && !error_.empty()) return false;
    if (!gguf_ctx_) {
        // gguf_init_params { no_alloc, ctx } — ctx!=NULL means allocate the
        // weight tensors into a freshly-created ggml_context (host memory).
        gguf_init_params params = {
            /*.no_alloc =*/ false,
            /*.ctx      =*/ &ctx_,
        };
        gguf_ctx_ = gguf_init_from_file(path, params);
        if (!gguf_ctx_) {
            error_ = std::string("failed to open GGUF: ") + path;
            return false;
        }
        data_ = (uint8_t*)ggml_get_mem_buffer(ctx_);
        data_size_ = ggml_get_mem_size(ctx_);
    }
    if (const char* e = std::getenv("STARLING_FAST_TIMING"); e && e[0] == '1')
        std::fprintf(stderr, "[load] gguf %s %.1f ms (%.1f MiB)\n", map_addr_ ? "map" : "read",
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
                     data_size_ / 1048576.0);
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
                // Use the type-specific accessor (gguf_get_val_i64 aborts on
                // non-INT64 types). Map each width to its accessor.
                switch (t) {
                    case GGUF_TYPE_UINT8:  v.i = gguf_get_val_u8(gguf_ctx_, i); break;
                    case GGUF_TYPE_INT8:   v.i = gguf_get_val_i8(gguf_ctx_, i); break;
                    case GGUF_TYPE_UINT16: v.i = gguf_get_val_u16(gguf_ctx_, i); break;
                    case GGUF_TYPE_INT16:  v.i = gguf_get_val_i16(gguf_ctx_, i); break;
                    case GGUF_TYPE_UINT32: v.i = gguf_get_val_u32(gguf_ctx_, i); break;
                    case GGUF_TYPE_INT32:  v.i = gguf_get_val_i32(gguf_ctx_, i); break;
                    case GGUF_TYPE_UINT64: v.i = (int64_t)gguf_get_val_u64(gguf_ctx_, i); break;
                    case GGUF_TYPE_INT64:  v.i = gguf_get_val_i64(gguf_ctx_, i); break;
                    default: continue;
                }
                break;
            case GGUF_TYPE_FLOAT32: case GGUF_TYPE_FLOAT64:
                v.kind = GgufValue::Kind::k_float;
                v.f = (t == GGUF_TYPE_FLOAT32) ? (double)gguf_get_val_f32(gguf_ctx_, i)
                                                : gguf_get_val_f64(gguf_ctx_, i);
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

// ---- community-dialect compat ---------------------------------------------
void ModelLoader::add_tensor_alias(const char* alias, const char* existing) {
    auto it = tensors_.find(existing);
    if (it != tensors_.end()) tensors_[alias] = it->second;
}

void ModelLoader::add_owned_tensor(const char* name, const std::vector<float>& data,
                                   int64_t ne0, int64_t ne1) {
    if (!compat_ctx_) {
        // Small scratch context for compat-fabricated tensors (mel
        // filterbank/window, zero bias rows). Freed with the loader.
        struct ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead() * 8 + (1 << 20),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ false,
        };
        compat_ctx_ = ggml_init(params);
    }
    ggml_tensor* t = (ne1 > 1) ? ggml_new_tensor_2d(compat_ctx_, GGML_TYPE_F32, ne0, ne1)
                               : ggml_new_tensor_1d(compat_ctx_, GGML_TYPE_F32, ne0);
    if (!t || (size_t)ggml_nelements(t) != data.size()) return;
    std::memcpy(t->data, data.data(), data.size() * sizeof(float));
    ggml_set_name(t, name);
    tensors_[name] = t;
}

void ModelLoader::add_kv_int(const std::string& key, int64_t v) {
    GgufValue val; val.kind = GgufValue::Kind::k_int; val.i = v;
    kv_[key] = std::move(val);
}
void ModelLoader::add_kv_float(const std::string& key, double v) {
    GgufValue val; val.kind = GgufValue::Kind::k_float; val.f = v;
    kv_[key] = std::move(val);
}
void ModelLoader::add_kv_str(const std::string& key, const std::string& v) {
    GgufValue val; val.kind = GgufValue::Kind::k_str; val.s = v;
    kv_[key] = std::move(val);
}
void ModelLoader::add_kv_arr_int(const std::string& key, const std::vector<int64_t>& v) {
    GgufValue val; val.kind = GgufValue::Kind::k_arr_int; val.arr_i = v;
    kv_[key] = std::move(val);
}
void ModelLoader::add_kv_arr_str(const std::string& key, const std::vector<std::string>& v) {
    GgufValue val; val.kind = GgufValue::Kind::k_arr_str; val.arr_s = v;
    kv_[key] = std::move(val);
}

bool ModelLoader::realize_weights(Backend& backend) {
    std::lock_guard<std::recursive_mutex> lock(runtime_mutex());
    if (realized_) return true;
    if (!ctx_) { error_ = "realize_weights: no loaded context"; return false; }
    // On CPU: borrow the loader's memory-backed context zero-copy.
    // On GPU: mirror each weight into a device-side context + upload.
    if (!backend.is_gpu()) {
        // Borrow the loader's weight memory zero-copy (the file mapping, or
        // the ggml context's allocation when the file was read).
        ggml_backend_buffer_t buf = ggml_backend_cpu_buffer_from_ptr(data_, data_size_);
        if (!buf) { error_ = "realize_weights: CPU buffer alloc failed"; return false; }
        weight_buffer_ = buf;
        // Point each tensor's ->buffer at the borrowed buffer.
        for (const auto& kv : tensors_) {
            ggml_tensor* t = kv.second;
            t->buffer = buf;
            // data pointer already set by the GGUF allocation.
        }
        // Opt the weights into ggml's repacked CPU kernels as graphs use
        // them (no-op unless enabled; see cpu_repack.hpp).
        try {
            cpu_repack::attach(buf);
        } catch (const std::exception& e) {
            error_ = std::string("realize_weights: ") + e.what();
            ggml_backend_buffer_free(buf);
            weight_buffer_ = nullptr;
            for (const auto& kv : tensors_) kv.second->buffer = nullptr;
            return false;
        }
    } else {
        // GPU: allocate a no_alloc device context, mirror tensors, upload.
        struct ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead() * (tensors_.size() + 16),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        std::unique_ptr<ggml_context, decltype(&ggml_free)> dev_ctx_owner(ggml_init(params), ggml_free);
        ggml_context* dev_ctx = dev_ctx_owner.get();
        if (!dev_ctx) { error_ = "realize_weights: context alloc failed"; return false; }
        std::unordered_map<std::string, ggml_tensor*> device_tensors;
        std::unordered_map<ggml_tensor*, ggml_tensor*> mirrored;
        // Build replacement tensors without changing the loaded model on failure.
        for (const auto& kv : tensors_) {
            ggml_tensor* src = kv.second;
            auto& dst = mirrored[src];
            if (!dst) { dst = ggml_dup_tensor(dev_ctx, src); ggml_set_name(dst, src->name); }
            device_tensors[kv.first] = dst;
        }
        std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> dev_buf_owner(
            ggml_backend_alloc_ctx_tensors(dev_ctx, backend.handle()), ggml_backend_buffer_free);
        ggml_backend_buffer_t dev_buf = dev_buf_owner.get();
        if (!dev_buf) { error_ = "realize_weights: device alloc failed"; return false; }
        // Aliases share a device tensor; upload each source only once.
        for (const auto& item : mirrored) {
            if (item.first->data)
                ggml_backend_tensor_set(item.second, item.first->data, 0, ggml_nbytes(item.first));
        }
        tensors_.swap(device_tensors);
        host_tensors_.swap(device_tensors);
        device_ctx_ = dev_ctx_owner.release();
        weight_buffer_ = dev_buf_owner.release();
    }
    realized_ = true;
    return true;
}

} // namespace starling::ggml
