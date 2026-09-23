// cpu_repack.cpp — see cpu_repack.hpp.
#include "cpu_repack.hpp"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
// Internal headers (third_party/ggml/src): the buffer struct is needed to
// build the repack-typed alias over existing memory, the same way ggml's own
// CPU_REPACK allocator derives its buffers from plain CPU buffers.
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <cstdio>  // std::fprintf (unrecognized gate value)
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace starling::ggml::cpu_repack {
namespace {

struct Region {
    ggml_backend_buffer_t plain = nullptr;  // the loader's borrowed CPU buffer
    ggml_backend_buffer_t alias = nullptr;  // CPU_REPACK-typed view of the same memory
    const char* base = nullptr;
    size_t size = 0;
};

struct State {
    std::mutex mu;
    std::vector<Region> regions;
    // Per-tensor decisions keyed by the host tensor (it lives as long as its
    // loader's ggml context; forget() drops them before that goes away).
    std::unordered_map<ggml_tensor*, void*> repacked;  // tensor -> ggml tensor traits
    std::unordered_set<ggml_tensor*> never;
    int64_t repacked_bytes = 0;
};

State& state() {
    static State s;
    return s;
}

bool platform_default() {
#if defined(__ANDROID__)
    return true;
#else
    return false;
#endif
}

bool env_enabled() {
    const char* v = std::getenv("STARLING_GGML_CPU_REPACK");
    if (!v || !*v) return platform_default();
    std::string value(v);
    for (char& c : value) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (value == "1" || value == "true" || value == "on" || value == "yes") return true;
    if (value == "0" || value == "false" || value == "off" || value == "no") return false;
    // The gate latches on first use; a typo would otherwise be invisible.
    std::fprintf(stderr, "[starling] STARLING_GGML_CPU_REPACK=%s not understood (use 1/true/on/yes or 0/false/off/no); keeping the default (%s)\n",
                 v, platform_default() ? "on" : "off");
    return platform_default();
}

// A repacked weight cannot be read back as plain rows. ggml's own
// CPU_REPACK buffers leave these hooks null, which ggml_backend_tensor_get
// would call unconditionally; fail with the diagnosis instead.
void alias_get_tensor(ggml_backend_buffer_t, const ggml_tensor* tensor, void*, size_t, size_t) {
    GGML_ABORT("cpu_repack: weight '%s' is repacked for MUL_MAT and cannot be read back to the host; "
               "set STARLING_GGML_CPU_REPACK=0", tensor->name);
}

bool alias_cpy_tensor(ggml_backend_buffer_t, const ggml_tensor* src, ggml_tensor*) {
    GGML_ABORT("cpu_repack: weight '%s' is repacked for MUL_MAT and cannot be copied as plain rows; "
               "set STARLING_GGML_CPU_REPACK=0", src->name);
}

ggml_backend_buffer_type_t repack_buffer_type() {
    static ggml_backend_buffer_type_t cached = [] () -> ggml_backend_buffer_type_t {
        ggml_backend_reg_t reg = ggml_backend_cpu_reg();
        if (!reg) return nullptr;
        auto get_extra = (ggml_backend_dev_get_extra_bufts_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!get_extra || !dev) return nullptr;
        for (ggml_backend_buffer_type_t* it = get_extra(dev); it && *it; ++it) {
            if (std::strcmp(ggml_backend_buft_name(*it), "CPU_REPACK") == 0) return *it;
        }
        return nullptr;
    }();
    return cached;
}

// A CPU buffer over existing memory that carries the CPU_REPACK type and its
// init/set hooks. Mirrors ggml_backend_cpu_repack_buffer_type_alloc_buffer,
// which derives a repack buffer from a plain CPU buffer the same way; the
// hooks are taken from a probe buffer of the real type rather than copied.
//
// Borrowing the probe's hooks after freeing it is safe: the hooks are always
// invoked with the alias itself (`alias->iface.init_tensor(alias, t)`), so
// anything they read from `buffer` is the alias's own state, and a CPU
// buffer's context is its data pointer (exactly as in ggml's own repack
// buffers, which are CPU buffers too). The probe only donates function
// pointers. At the pinned ggml the hooks read only the tensor anyway.
ggml_backend_buffer_t make_alias(ggml_backend_buffer_t plain) {
    ggml_backend_buffer_type_t buft = repack_buffer_type();
    if (!buft) return nullptr;
    ggml_backend_buffer_t probe = ggml_backend_buft_alloc_buffer(buft, 64);
    if (!probe) return nullptr;
    ggml_backend_buffer_t alias = ggml_backend_cpu_buffer_from_ptr(
        ggml_backend_buffer_get_base(plain), ggml_backend_buffer_get_size(plain));
    if (alias) {
        alias->buft = buft;
        alias->iface.init_tensor = probe->iface.init_tensor;
        alias->iface.set_tensor = probe->iface.set_tensor;
        alias->iface.get_tensor = alias_get_tensor;
        alias->iface.cpy_tensor = alias_cpy_tensor;
    }
    ggml_backend_buffer_free(probe);
    return alias;
}

Region* region_of(State& s, const ggml_tensor* t) {
    if (!t || !t->buffer || !t->data) return nullptr;
    const char* p = static_cast<const char*>(t->data);
    for (Region& r : s.regions) {
        if ((t->buffer == r.plain || t->buffer == r.alias) && p >= r.base &&
            p + ggml_nbytes(t) <= r.base + r.size) {
            return &r;
        }
    }
    return nullptr;
}

// The exact contract of ggml's repack MUL_MAT (repack.cpp supports_op +
// forward_mul_mat asserts), checked per use.
bool supported_use(const ggml_tensor* node, int src_index, const ggml_tensor* weight) {
    if (node->op != GGML_OP_MUL_MAT || src_index != 0 || node->src[0] != weight) return false;
    if (ggml_n_dims(weight) != 2) return false;
    const ggml_tensor* x = node->src[1];
    if (!x || x->type != GGML_TYPE_F32 || x->ne[3] != 1) return false;
    // The kernel quantizes each activation row as ne10 consecutive floats
    // (any row/plane stride is fine), so a transposed activation would be
    // read wrong without an assert.
    if (x->nb[0] != sizeof(float)) return false;
    // forward_mul_mat asserts both separately (an F32 result can still be a
    // permuted view), so both are checked here too, with its stride order.
    if (node->type != GGML_TYPE_F32 || node->nb[0] != sizeof(float)) return false;
    if (node->nb[0] > node->nb[1] || node->nb[1] > node->nb[2]) return false;
    return true;
}

bool try_repack(State& s, Region& r, ggml_tensor* w) {
    w->buffer = r.alias;
    w->extra = nullptr;
    if (r.alias->iface.init_tensor) r.alias->iface.init_tensor(r.alias, w);
    if (!w->extra) {  // no repacked layout for this type/shape on this CPU
        w->buffer = r.plain;
        return false;
    }
    const size_t n = ggml_nbytes(w);
    std::vector<uint8_t> original;
    try {
        original.resize(n);
    } catch (...) {
        // Keep an allocation failure recoverable without copying weights
        // that have no repack kernel on this CPU.
        w->buffer = r.plain;
        w->extra = nullptr;
        throw;
    }
    std::memcpy(original.data(), w->data, n);
    // Dispatches to the CPU_REPACK set_tensor: rewrites w->data in place.
    ggml_backend_tensor_set(w, original.data(), 0, n);
    s.repacked[w] = w->extra;
    // In place: CPU_REPACK's get_alloc_size is ggml_nbytes, so the repacked
    // footprint is exactly the plain one.
    s.repacked_bytes += static_cast<int64_t>(n);
    return true;
}

}  // namespace

bool enabled() {
    static const bool on = env_enabled();
    return on;
}

void attach(ggml_backend_buffer* buffer) {
    if (!enabled() || !buffer) return;
    State& s = state();
    std::lock_guard<std::mutex> lk(s.mu);
    // Idempotent per buffer: a second attach without a detach in between
    // must not stack a second alias over the same memory.
    for (const Region& existing : s.regions) {
        if (existing.plain == buffer) return;
    }
    Region r;
    r.plain = buffer;
    r.alias = make_alias(buffer);
    if (!r.alias) return;  // CPU_REPACK not compiled in: nothing to do
    r.base = static_cast<const char*>(ggml_backend_buffer_get_base(buffer));
    r.size = ggml_backend_buffer_get_size(buffer);
    // Bytes repacked before an earlier release are still repacked.
    for (auto& [t, traits] : s.repacked) {
        const char* p = static_cast<const char*>(t->data);
        if (t->buffer == buffer && p >= r.base && p + ggml_nbytes(t) <= r.base + r.size) {
            t->buffer = r.alias;
            t->extra = traits;
        }
    }
    s.regions.push_back(r);
}

void detach(ggml_backend_buffer* buffer) {
    if (!buffer) return;
    State& s = state();
    std::lock_guard<std::mutex> lk(s.mu);
    for (size_t i = 0; i < s.regions.size(); ++i) {
        Region& r = s.regions[i];
        if (r.plain != buffer) continue;
        // Called under runtime_mutex immediately before the loader frees the
        // plain buffer and clears its tensor buffers. No graph can run in
        // this detach-to-attach window; the repack decision and bytes remain.
        for (auto& entry : s.repacked) {
            ggml_tensor* t = entry.first;
            if (t->buffer == r.alias) t->buffer = r.plain;
        }
        ggml_backend_buffer_free(r.alias);
        s.regions.erase(s.regions.begin() + static_cast<std::ptrdiff_t>(i));
        return;
    }
}

void forget(const void* base, size_t size) {
    if (!base) return;
    State& s = state();
    std::lock_guard<std::mutex> lk(s.mu);
    const char* lo = static_cast<const char*>(base);
    const char* hi = lo + size;
    auto inside = [&](const ggml_tensor* t) {
        const char* p = static_cast<const char*>(t->data);
        return p >= lo && p < hi;
    };
    for (auto it = s.repacked.begin(); it != s.repacked.end();) {
        if (inside(it->first)) {
            s.repacked_bytes -= static_cast<int64_t>(ggml_nbytes(it->first));
            it = s.repacked.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = s.never.begin(); it != s.never.end();) {
        it = inside(*it) ? s.never.erase(it) : std::next(it);
    }
}

void prepare_graph(ggml_cgraph* gf) {
    if (!enabled() || !gf) return;
    State& s = state();
    std::lock_guard<std::mutex> lk(s.mu);
    if (s.regions.empty()) return;

    // weight -> (region, every use in this graph is supported)
    struct Use {
        Region* region;
        bool ok;
        const ggml_tensor* bad_node;
    };
    std::unordered_map<ggml_tensor*, Use> uses;
    for (int i = 0; i < gf->n_nodes; ++i) {
        ggml_tensor* node = gf->nodes[i];
        for (int k = 0; k < GGML_MAX_SRC; ++k) {
            ggml_tensor* src = node->src[k];
            if (!src) continue;
            ggml_tensor* weight = src->view_src ? src->view_src : src;
            Region* region = region_of(s, weight);
            if (!region) continue;
            const bool ok = src == weight && supported_use(node, k, weight);
            auto [it, inserted] = uses.try_emplace(weight, Use{region, ok, ok ? nullptr : node});
            if (!inserted && !ok && it->second.ok) {
                it->second.ok = false;
                it->second.bad_node = node;
            }
        }
    }

    for (auto& [weight, use] : uses) {
        if (!use.ok) {
            if (s.repacked.count(weight)) {
                throw std::runtime_error(
                    std::string("cpu_repack: weight '") + weight->name +
                    "' is repacked for MUL_MAT but this graph reads it through op " +
                    ggml_op_name(use.bad_node->op) +
                    " (node '" + use.bad_node->name + "'); set STARLING_GGML_CPU_REPACK=0");
            }
            s.never.insert(weight);
            continue;
        }
        if (s.repacked.count(weight) || s.never.count(weight)) continue;
        if (!try_repack(s, *use.region, weight)) s.never.insert(weight);
    }
}

Stats stats() {
    State& s = state();
    std::lock_guard<std::mutex> lk(s.mu);
    return Stats{static_cast<int64_t>(s.repacked.size()), s.repacked_bytes};
}

}  // namespace starling::ggml::cpu_repack
