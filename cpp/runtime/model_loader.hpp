// model_loader.hpp — GGUF parse + weight realize, shared by every model.
//
// The mechanism (open the GGUF, read KV metadata, map tensor-name -> ggml_tensor*,
// give each weight a backend buffer) is model-agnostic. Only the specific KV key
// strings (e.g. parakeet.* vs moss_transcribe.*) and the config struct built
// from them are model-specific — those live in parakeet/loader.hpp and
// moss/loader.hpp.

#pragma once

#include <cstdint>
#include <memory>
#include <typeindex>
#include <utility>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct gguf_context;
struct ggml_backend_buffer;

namespace starling::ggml {

class Backend;

// Key/value metadata read from the GGUF (the `*.type`, `*.value` map). Typed
// accessors below; stored as strings/integers/floats as-read.
struct GgufValue {
    enum class Kind { k_str, k_int, k_float, k_arr_str, k_arr_int };
    Kind kind;
    std::string s;                  // k_str
    int64_t i = 0;                  // k_int
    double f = 0.0;                 // k_float
    std::vector<std::string> arr_s; // k_arr_str
    std::vector<int64_t> arr_i;     // k_arr_int
};

// Loads a GGUF file and owns the weight tensors. The GGUF context + the weight
// data stay memory-backed for the loader's lifetime (zero-copy on CPU; mirrored
// to the device by realize_weights on GPU).
class ModelLoader {
public:
    ModelLoader();
    ~ModelLoader();

    ModelLoader(const ModelLoader&) = delete;
    ModelLoader& operator=(const ModelLoader&) = delete;

    // Open + parse `path`. Reads the KV metadata into kv() and maps every tensor
    // name -> tensor* into tensors(). Returns true on success; on failure sets
    // an internal error string (see last_error).
    bool load(const char* path);

    const std::string& last_error() const { return error_; }

    // ---- metadata -------------------------------------------------------
    const std::unordered_map<std::string, GgufValue>& kv() const { return kv_; }

    // Typed KV accessors. Return false if `key` is absent or the wrong kind.
    bool kv_str(const std::string& key, std::string& out) const;
    bool kv_int(const std::string& key, int64_t& out) const;
    bool kv_float(const std::string& key, double& out) const;
    bool kv_arr_str(const std::string& key, std::vector<std::string>& out) const;
    bool kv_arr_int(const std::string& key, std::vector<int64_t>& out) const;

    // ---- tensors --------------------------------------------------------
    // Lookup by exact name (the verbatim NeMo / moss_transcribe key). nullptr if
    // absent.
    ggml_tensor* tensor(const char* name) const;

    // All tensor names (for diagnostics / "is the expected set present?").
    std::vector<std::string> tensor_names() const;

    // The ggml context that owns the weight tensors (memory-backed by the GGUF
    // allocation on CPU; realized to the device by realize_weights on GPU).
    ggml_context* ctx() const { return ctx_; }

    // Give every weight a backend buffer on `backend` (zero-copy on CPU, a
    // mirrored upload on GPU). Idempotent. Called automatically by clone_weight
    // (via ensure_weights_realized) the first time a graph references a weight;
    // models also call it up-front at load to surface missing tensors early.
    bool realize_weights(Backend& backend);

    // Private graph/cache types stay in their implementation files. Slots belong
    // to this loaded model and are destroyed in reverse creation order, before
    // its weight buffers. Callers serialize access with the runtime lock.
    template<class T> std::unique_ptr<T>& cache() const {
        const std::type_index key(typeid(T));
        for (auto& slot : caches_)
            if (slot.first == key) return static_cast<Cache<T>&>(*slot.second).value;
        auto slot = std::make_unique<Cache<T>>();
        auto& value = slot->value;
        caches_.emplace_back(key, std::move(slot));
        return value;
    }
    template<class T> const T* find_cache() const {
        const std::type_index key(typeid(T));
        for (auto& slot : caches_)
            if (slot.first == key) return static_cast<Cache<T>&>(*slot.second).value.get();
        return nullptr;
    }

    // Release caches and weights while the backend is alive. Shutdown visits
    // every live loader, including models whose caller has not freed them.
    static void release_all_runtime_resources();

    // ---- community-dialect compat (see cpp/parakeet/compat.cpp) ----------
    // Register an additional lookup name for an already-present tensor
    // (zero-copy: both names map to the same ggml_tensor*).
    void add_tensor_alias(const char* alias, const char* existing);

    // Register a compat-owned F32 tensor (1-D or 2-D) copied from `data`.
    // Used for the transcribe.cpp dialect's synthesized mel filterbank/window
    // and the zero side of its fused LSTM-bias split. Owned until destruction.
    void add_owned_tensor(const char* name, const std::vector<float>& data,
                          int64_t ne0, int64_t ne1);

    // Synthesize KV entries (existing keys are overwritten).
    void add_kv_int(const std::string& key, int64_t v);
    void add_kv_float(const std::string& key, double v);
    void add_kv_str(const std::string& key, const std::string& v);
    void add_kv_arr_int(const std::string& key, const std::vector<int64_t>& v);
    void add_kv_arr_str(const std::string& key, const std::vector<std::string>& v);

private:
    struct CacheBase { virtual ~CacheBase() = default; };
    template<class T> struct Cache : CacheBase { std::unique_ptr<T> value; };
    mutable std::vector<std::pair<std::type_index, std::unique_ptr<CacheBase>>> caches_;
    void release_runtime_resources();
    ggml_context* device_ctx_ = nullptr;
    ggml_backend_buffer* weight_buffer_ = nullptr;
    ggml_context* ctx_ = nullptr;          // owns the weight tensors
    ggml_context* compat_ctx_ = nullptr;   // compat-fabricated tensors (see compat.cpp)
    gguf_context* gguf_ctx_ = nullptr;      // the gguf_init_from_file handle
    std::unordered_map<std::string, ggml_tensor*> tensors_;
    std::unordered_map<std::string, ggml_tensor*> host_tensors_;
    std::unordered_map<std::string, GgufValue> kv_;
    std::string error_;
    bool realized_ = false;
};

} // namespace starling::ggml
