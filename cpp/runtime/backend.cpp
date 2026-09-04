// backend.cpp — Backend + ReplayGraph implementation.
//
// Mirrors the proven parakeet.cpp architecture (src/backend.cpp): registry-
// driven device selection, persistent gallocr reused across calls (never
// backend_sched on the fast path), zero-copy weights via clone_weight, private
// gallocr per ReplayGraph, stable cgraph uid (skips ggml-cuda's per-replay
// O(n_nodes) memcmp), async set_input + async readback + single sync.
//
// The stable uid reaches into ggml's internal cgraph->uid field (declared in
// the public ggml.h on this pinned ggml version, no internal header needed).

#include "backend.hpp"

#include "graph.hpp"
#include "imatrix.hpp"
#include "model_loader.hpp"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

// ggml-impl.h is ggml's internal header (under third_party/ggml/src/); it's the
// only place the cgraph `uid` field is declared. The public ggml.h leaves
// ggml_cgraph opaque. The uid lets ggml-cuda skip its O(n_nodes) per-replay
// memcmp (patch 0002) — load-bearing for ReplayGraph's steady-state perf.
#include "ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace {
// F1 instrumentation gate. When on, ReplayGraph::compute prints a per-replay
// split of host wall between graph_compute (the async graph launch — returns to
// host almost immediately on the fast CUDA path) and readback (async D2H + the
// SINGLE trailing sync — host blocks here until the GPU finishes, so this is
// the steady-state proxy for per-replay GPU time). Also prints n_nodes + the
// stable cgraph uid so callers can tell encoder vs decode replays apart and
// confirm capture reached steady state (uid non-zero = patch-0008 fast-path).
bool replay_timing_on() {
    static const bool on = std::getenv("STARLING_REPLAY_TIMING") != nullptr;
    return on;
}
} // namespace

// ggml backend headers (registered devices).
#include "ggml-cuda.h"
#include "ggml-cpu.h"
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif
#ifdef GGML_USE_HIP
// HIP registers through the CUDA-compatible path; ggml-cuda.h covers it.
#endif

namespace starling::ggml {
namespace {

// --------------------------------------------------------------------------- //
// Thread-local pointer to the Backend driving the current compute (set by
// Backend::compute / ReplayGraph ctor, read by add_graph_input /
// capture_graph_output so model build lambdas don't need a Backend ref).
// --------------------------------------------------------------------------- //
thread_local Backend* t_active_backend = nullptr;
// Inputs/captures registered during the current build; drained by the driver.
struct PendingInput  { ggml_tensor* t; const void* host; size_t nbytes; };
struct PendingCapture{ ggml_tensor* t; std::vector<float>* dst; };
thread_local std::vector<PendingInput>*   t_pending_inputs = nullptr;
thread_local std::vector<PendingCapture>* t_pending_captures = nullptr;
// Side-effect expansion roots registered during the current build (Wave D
// decode-state write-back cpys): nodes that must execute but are not read back.
thread_local std::vector<ggml_tensor*>*    t_pending_roots = nullptr;

constexpr size_t kGraphSize = 32768;  // max nodes in one ggml_cgraph (bumped for
                                          // the moss K-step multistep decode graph)

// The ggml_backend_sched_eval_callback trampoline feeding the ImatrixCollector
// (see imatrix.hpp). NOTE: returning true does NOT force the node onto the
// CPU backend — the sched consults the callback only for batching/
// observation; placement is decided without it. Host-side activation reads
// are sound because the collect tool pins STARLING_GGML_DEVICE=cpu.
bool imatrix_eval_cb(ggml_tensor* t, bool ask, void* /*user_data*/) {
    return ImatrixCollector::instance().observe(t, ask);
}

} // namespace

// --------------------------------------------------------------------------- //
// Backend::Impl
// --------------------------------------------------------------------------- //
struct Backend::Impl {
    ggml_backend_t backend = nullptr;       // primary (GPU or CPU)
    ggml_backend_t cpu_backend = nullptr;   // CPU fallback for GPU offload
    ggml_gallocr_t galloc = nullptr;        // persistent, reused across one-shot computes
    ggml_backend_sched_t sched = nullptr;   // lazy; GPU op offload + imatrix collection
    bool use_sched = false;                 // true iff a GPU backend is active

    // Create the sched on first need: {GPU, CPU} when a GPU is active, the
    // lone CPU backend otherwise (a duplicated entry would confuse sched's
    // backend list). When imatrix collection is enabled the eval callback is
    // attached so every MUL_MAT with a named weight src[0] is observed.
    ggml_backend_sched_t ensure_sched() {
        if (sched) return sched;
        ggml_backend_t backends[2];
        ggml_backend_buffer_type_t bufs[2];
        int n = 0;
        backends[n] = backend;
        bufs[n] = ggml_backend_get_default_buffer_type(backend);
        ++n;
        if (use_sched) {
            backends[n] = cpu_backend;
            bufs[n] = ggml_backend_get_default_buffer_type(cpu_backend);
            ++n;
        }
        // NOTE: the 4th arg is the graph size in NODES (see ggml-backend.h),
        // not bytes — the historical code here passed tensor_overhead()*
        // kGraphSize, which only worked because this branch never executed
        // (no unsupported ops on the CUDA builds).
        sched = ggml_backend_sched_new(backends, bufs, n,
                                       (int)kGraphSize,
                                       /*parallel=*/false, /*op_offload=*/true);
        // In the vendored ggml the eval callback gates observation only, not
        // backend placement; host-side activation reads are sound because
        // collection runs pin STARLING_GGML_DEVICE=cpu.
        if (sched && ImatrixCollector::enabled()) {
            ggml_backend_sched_set_eval_callback(sched, imatrix_eval_cb, nullptr);
        }
        return sched;
    }
};

Backend::Backend(int n_threads) : impl_(new Impl()), n_threads_(n_threads < 1 ? 1 : n_threads) {
    // Device selection (registry-driven; mirrors parakeet.cpp src/backend.cpp:91-166).
    // STARLING_GGML_DEVICE selects a named device ("cpu", "CUDA0", "Vulkan0",
    // "Metal", ...); unset auto-picks the first GPU/IGPU, else CPU.
    ggml_backend_dev_t chosen = nullptr;
    if (const char* dev_env = std::getenv("STARLING_GGML_DEVICE")) {
        std::string want = dev_env;
        std::transform(want.begin(), want.end(), want.begin(),
                       [](unsigned char c){ return std::tolower(c); });
        if (want == "cpu") {
            chosen = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        } else {
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                ggml_backend_dev_t d = ggml_backend_dev_get(i);
                std::string name = ggml_backend_dev_name(d);
                std::transform(name.begin(), name.end(), name.begin(),
                               [](unsigned char c){ return std::tolower(c); });
                if (name == want) { chosen = d; break; }
            }
        }
    }
    if (!chosen) {
        // Auto: prefer a GPU/IGPU, else CPU (ggml's registry decides what's
        // compiled in — CUDA/Metal/Vulkan/HIP all register here).
        chosen = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!chosen) chosen = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
        if (!chosen) chosen = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    }
    if (chosen) {
        device_name_ = ggml_backend_dev_name(chosen);
        impl_->backend = ggml_backend_dev_init(chosen, nullptr);
    }
    if (!impl_->backend) {
        // CPU fallback (always available).
        impl_->backend = ggml_backend_cpu_init();
        device_name_ = "cpu";
    }
    // GPU -> also create a CPU fallback backend for op offload + set the sched flag.
    // Compare case-insensitively: the CPU device reports its name as "CPU" (and
    // some platforms use other casings), so a bare != "cpu" mis-classifies a
    // selected CPU backend as GPU and routes mel to the GPU path.
    {
        std::string lname = device_name_;
        std::transform(lname.begin(), lname.end(), lname.begin(),
                       [](unsigned char c){ return std::tolower(c); });
        impl_->use_sched = (lname != "cpu");
    }
    if (impl_->use_sched) {
        impl_->cpu_backend = ggml_backend_cpu_init();
    } else {
        // CPU-only: the primary backend is already CPU.
        impl_->cpu_backend = impl_->backend;
    }
    if (ggml_backend_is_cpu(impl_->backend)) {
        ggml_backend_cpu_set_n_threads(impl_->backend, n_threads_);
    }
    // Persistent gallocr (shared by one-shot computes; ReplayGraph has its own).
    impl_->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(impl_->backend));
}

Backend::~Backend() {
    // Skip frees if the CUDA driver is already gone (post-shutdown teardown).
    if (shutting_down()) { delete impl_; return; }
    if (impl_) {
        if (impl_->sched)    ggml_backend_sched_free(impl_->sched);
        if (impl_->galloc)   ggml_gallocr_free(impl_->galloc);
        // On GPU we own cpu_backend separately; on CPU it aliases backend.
        if (impl_->use_sched && impl_->cpu_backend) ggml_backend_free(impl_->cpu_backend);
        if (impl_->backend)  ggml_backend_free(impl_->backend);
        delete impl_;
    }
}

void Backend::set_n_threads(int n_threads) {
    n_threads_ = n_threads < 1 ? 1 : n_threads;
    if (impl_ && ggml_backend_is_cpu(impl_->backend)) {
        ggml_backend_cpu_set_n_threads(impl_->backend, n_threads_);
    }
}

bool Backend::is_gpu() const { return impl_ && impl_->use_sched; }
ggml_backend_t Backend::handle() const { return impl_ ? impl_->backend : nullptr; }

void Backend::register_input(ggml_tensor* t, const void* host, size_t nbytes) {
    if (t_pending_inputs) t_pending_inputs->push_back({t, host, nbytes});
}
void Backend::register_capture(ggml_tensor* t, std::vector<float>* dst) {
    if (t_pending_captures) t_pending_captures->push_back({t, dst});
}

bool Backend::compute(const std::function<ggml_tensor*(ggml_context*)>& build,
                      std::vector<float>& out) {
    // 1. Build in a no_alloc=true metadata context.
    //    ggml_graph_overhead_custom(kGraphSize, ...) reserves node/leaf slot
    //    capacity matching the cgraph allocated below (ggml_new_graph_custom);
    //    the default ggml_new_graph() caps at GGML_DEFAULT_GRAPH_SIZE (2048)
    //    nodes, which the parakeet conformer encoder (24 layers, ~2400 ops)
    //    overflows.
    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * kGraphSize
                         + ggml_graph_overhead_custom(kGraphSize, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context* ctx = ggml_init(params);
    if (!ctx) return false;

    std::vector<PendingInput>   pin;
    std::vector<PendingCapture> pcap;
    t_active_backend = this;
    t_pending_inputs = &pin;
    t_pending_captures = &pcap;
    std::vector<ggml_tensor*> roots;
    t_pending_roots = &roots;

    ggml_tensor* out_t = build(ctx);

    t_pending_inputs = nullptr;
    t_pending_captures = nullptr;
    t_pending_roots = nullptr;
    t_active_backend = nullptr;

    if (!out_t) { ggml_free(ctx); return false; }
    ggml_set_output(out_t);  // mark output so the allocator keeps it

    // 2. Build the cgraph (capacity = kGraphSize; default 2048 is too small
    //    for the full conformer encoder graph).
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, kGraphSize, false);
    ggml_build_forward_expand(gf, out_t);
    // Captured intermediates are dead branches of out_t's tree: expand the
    // graph over them too so they're computed + allocated for readback.
    for (const auto& c : pcap) ggml_build_forward_expand(gf, c.t);
    // Side-effect roots (decode-state write-backs): expand so they execute.
    for (ggml_tensor* r : roots) ggml_build_forward_expand(gf, r);

    // 3. Allocate (persistent gallocr path, or sched fallback if some op is
    // unsupported by the primary backend / imatrix collection is active).
    bool need_sched = ImatrixCollector::enabled();
    if (impl_->use_sched) {
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            if (!ggml_backend_supports_op(impl_->backend, ggml_graph_node(gf, i))) {
                need_sched = true; break;
            }
        }
    }
    bool ok = false;
    if (!need_sched) {
        if (!impl_->galloc) impl_->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(impl_->backend));
        if (ggml_gallocr_alloc_graph(impl_->galloc, gf)) {
            // 4. Push host inputs AFTER alloc (->data was NULL until now).
            //    Inputs not reachable from the expanded graph get no
            //    allocation (data == NULL); skip those (diagnostic early-
            //    return graphs legitimately drop inputs).
            for (const auto& in : pin)
                if (in.t->data) ggml_backend_tensor_set(in.t, in.host, 0, in.nbytes);
            // 5. Compute.
            ok = (ggml_backend_graph_compute(impl_->backend, gf) == GGML_STATUS_SUCCESS);
        }
    } else {
        ggml_backend_sched_t sched = impl_->ensure_sched();
        if (!sched) { ggml_free(ctx); return false; }
        ggml_backend_sched_reset(sched);
        if (ggml_backend_sched_alloc_graph(sched, gf)) {
            for (const auto& in : pin)
                if (in.t->data) ggml_backend_tensor_set(in.t, in.host, 0, in.nbytes);
            ok = (ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS);
        }
    }
    if (ok) {
        // 6. Read back the output + captures.
        size_t n = (size_t)ggml_nelements(out_t);
        out.resize(n);
        ggml_backend_tensor_get(out_t, out.data(), 0, n * ggml_element_size(out_t));
        for (const auto& c : pcap) {
            size_t cn = (size_t)ggml_nelements(c.t);
            c.dst->resize(cn);
            ggml_backend_tensor_get(c.t, c.dst->data(), 0, cn * sizeof(float));
        }
    }
    pin.clear(); pcap.clear();
    ggml_free(ctx);
    return ok;
}

// --------------------------------------------------------------------------- //
// Free helpers routed via thread-locals (model build lambdas call these).
// --------------------------------------------------------------------------- //
void add_graph_input(ggml_tensor* t, const void* host, size_t nbytes) {
    ggml_set_input(t);
    if (t_active_backend) t_active_backend->register_input(t, host, nbytes);
}
ggml_tensor* graph_input_tensor(ggml_context* ctx, int type, int n_dims,
                                const int64_t* ne, const void* host, size_t nbytes) {
    ggml_tensor* t = (n_dims == 1) ? ggml_new_tensor_1d(ctx, (ggml_type)type, ne[0])
                   : (n_dims == 2) ? ggml_new_tensor_2d(ctx, (ggml_type)type, ne[0], ne[1])
                   : (n_dims == 3) ? ggml_new_tensor_3d(ctx, (ggml_type)type, ne[0], ne[1], ne[2])
                   : ggml_new_tensor_4d(ctx, (ggml_type)type, ne[0], ne[1], ne[2], ne[3]);
    add_graph_input(t, host, nbytes);
    return t;
}
void capture_graph_output(ggml_tensor* t, std::vector<float>* dst) {
    ggml_set_output(t);
    if (t_active_backend) t_active_backend->register_capture(t, dst);
}

void add_graph_root(ggml_tensor* t) {
    // The node is NOT marked output (no readback); the caller just needs it
    // expanded into the cgraph so it executes as a side effect.
    if (t_pending_roots) t_pending_roots->push_back(t);
}

// --------------------------------------------------------------------------- //
// Zero-copy weight referencing.
// --------------------------------------------------------------------------- //
void ensure_weights_realized(const ModelLoader& ml) {
    const_cast<ModelLoader&>(ml).realize_weights(global_backend());
}
ggml_tensor* clone_weight_opt(ggml_context* /*ctx*/, const ModelLoader& ml, const char* name) {
    ggml_tensor* t = ml.tensor(name);
    if (!t) return nullptr;
    // Lazily give it a backend buffer on first use.
    if (!t->buffer) ensure_weights_realized(ml);
    return t;
}
ggml_tensor* clone_weight(ggml_context* ctx, const ModelLoader& ml, const char* name) {
    ggml_tensor* t = clone_weight_opt(ctx, ml, name);
    return t;  // (loader-level asserts presence; callers assert by use)
}
void weight_to_host_f32(const ModelLoader& ml, const char* name, std::vector<float>& out) {
    ggml_tensor* t = ml.tensor(name);
    if (!t) { out.clear(); return; }
    ensure_weights_realized(ml);
    size_t n = (size_t)ggml_nelements(t);
    out.resize(n);
    // Backend tensors may be non-f32 (f16); for host-side math the loader keeps
    // the CPU f32 view authoritative when present. Fetch raw bytes, cast if f32.
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    } else {
        // Dequant via a temporary graph (rare path; used for host folds).
        std::vector<char> raw(ggml_row_size(t->type, n));
        ggml_backend_tensor_get(t, raw.data(), 0, ggml_row_size(t->type, n));
        ggml_fp16_to_fp32_row((const ggml_fp16_t*)raw.data(), out.data(), n);
    }
}

// --------------------------------------------------------------------------- //
// ReplayGraph
// --------------------------------------------------------------------------- //
ReplayGraph::ReplayGraph(Backend& backend,
                         const std::function<ggml_tensor*(ggml_context*)>& build)
    : backend_(backend) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * kGraphSize
                         + ggml_graph_overhead_custom(kGraphSize, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx_ = ggml_init(params);

    std::vector<PendingInput>   pin;
    std::vector<PendingCapture> pcap;
    t_active_backend = &backend_;
    t_pending_inputs = &pin;
    t_pending_captures = &pcap;
    std::vector<ggml_tensor*> roots;
    t_pending_roots = &roots;

    out_ = build(ctx_);

    t_pending_inputs = nullptr;
    t_pending_captures = nullptr;
    t_pending_roots = nullptr;
    t_active_backend = nullptr;

    if (out_) {
        ggml_set_output(out_);
        gf_ = ggml_new_graph_custom(ctx_, kGraphSize, false);
        // Stable uid: lets ggml-cuda skip its O(n_nodes) per-replay memcmp
        // (patch 0002: "skip per-node replay validation when uid is stable").
        gf_->uid = ggml_graph_next_uid();
        ggml_build_forward_expand(gf_, out_);
        for (const auto& c : pcap) ggml_build_forward_expand(gf_, c.t);
        // Side-effect roots (Wave D decode-state write-backs into persistent
        // device buffers): expand so they execute each replay without readback.
        for (ggml_tensor* r : roots) ggml_build_forward_expand(gf_, r);
        // Record inputs + captures in registration order (kept across calls).
        for (const auto& in : pin) {
            inputs_.push_back(in.t);
            input_hosts_.push_back(in.host);
        }
        for (const auto& c : pcap) {
            captures_.emplace_back(c.t, c.dst);
        }
        alloc_internal();
        // F1: one-time op histogram for this graph (gated). Counts nodes by op
        // name and, for MUL_MAT, the source (weight) dtype so we can confirm the
        // f16 tensor-core cuBLAS path vs an f32 fallback. Printed once per ctor.
        if (replay_timing_on()) {
            std::fprintf(stderr, "[enc-graph] uid=%u n_nodes=%d\n",
                         (unsigned)gf_->uid, gf_->n_nodes);
            std::map<ggml_op, int> hist;
            int mm_f32 = 0, mm_f16 = 0, mm_other = 0;
            for (int i = 0; i < gf_->n_nodes; ++i) {
                ggml_tensor* n = ggml_graph_node(gf_, i);
                if (!n) continue;
                hist[n->op]++;
                if (n->op == GGML_OP_MUL_MAT) {
                    ggml_tensor* src0 = n->src[0];
                    if (src0->type == GGML_TYPE_F32) mm_f32++;
                    else if (src0->type == GGML_TYPE_F16) mm_f16++;
                    else mm_other++;
                }
            }
            for (const auto& kv : hist) {
                std::fprintf(stderr, "[enc-graph]   op=%-22s count=%d\n",
                             ggml_op_name(kv.first), kv.second);
            }
            std::fprintf(stderr,
                "[enc-graph]   MUL_MAT dtype: f16=%d f32=%d other=%d\n",
                mm_f16, mm_f32, mm_other);
        }
    }
    pin.clear(); pcap.clear();
}

bool ReplayGraph::alloc_internal() {
    // Decide sched vs gallocr (same logic as Backend::compute).
    need_sched_ = ImatrixCollector::enabled();
    if (backend_.is_gpu() && !need_sched_) {
        for (int i = 0; i < ggml_graph_n_nodes(gf_); ++i) {
            if (!ggml_backend_supports_op(backend_.handle(), ggml_graph_node(gf_, i))) {
                need_sched_ = true; break;
            }
        }
    }
    if (!need_sched_) {
        // PRIVATE gallocr per ReplayGraph: a shared one would invalidate other
        // graphs' device pointers when a larger graph reallocs its buffer.
        if (!galloc_) galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_.handle()));
        return ggml_gallocr_alloc_graph(galloc_, gf_);
    }
    // sched path (GPU op offload, or imatrix collection on any device).
    ggml_backend_sched_t sched = backend_.impl_->ensure_sched();
    if (!sched) return false;
    ggml_backend_sched_reset(sched);
    return ggml_backend_sched_alloc_graph(sched, gf_);
}

ReplayGraph::~ReplayGraph() {
    if (shutting_down()) return;  // driver gone -> leak (fine at exit)
    if (galloc_) ggml_gallocr_free(galloc_);
    if (ctx_)    ggml_free(ctx_);
}

void ReplayGraph::set_input(size_t i, const void* host, size_t nbytes) {
    if (i >= inputs_.size()) return;
    // Async H2D on the backend's compute stream; stream ordering guarantees it
    // completes before the next graph_compute reads it. No sync here.
    ggml_backend_tensor_set_async(backend_.handle(), inputs_[i], host, 0, nbytes);
}

size_t ReplayGraph::input_nbytes(size_t i) const {
    if (i >= inputs_.size()) return 0;
    return (size_t)ggml_nbytes(inputs_[i]);
}
const void* ReplayGraph::input_host(size_t i) const {
    return (i < input_hosts_.size()) ? input_hosts_[i] : nullptr;
}

void ReplayGraph::readback_async_then_sync(Backend::Impl* impl,
                                           ggml_tensor* out_t,
                                           void* out_host, size_t out_nbytes) {
    // Async D2H for the output, then async D2H for each capture, then ONE sync.
    // Collapses N syncs (each ~150-200us on WSL2) into one. The async variant
    // runs on the backend stream so copies queue behind the graph launch.
    ggml_backend_tensor_get_async(impl->backend, out_t, out_host, 0, out_nbytes);
    for (const auto& c : captures_) {
        size_t cn = (size_t)ggml_nelements(c.first);
        if (c.second) {
            c.second->resize(cn);
            ggml_backend_tensor_get_async(impl->backend, c.first,
                                          c.second->data(), 0, cn * sizeof(float));
        }
    }
    ggml_backend_synchronize(impl->backend);
}

bool ReplayGraph::compute(std::vector<float>& out) {
    if (!gf_ || !out_) return false;
    Backend::Impl* impl = backend_.impl_;
    // Fast path: graph_compute_async (skip the sync-wrapping graph_compute so the
    // readbacks can pipeline behind the graph on the same stream), then async
    // readback + single sync.
    const bool t_on = replay_timing_on();
    const int64_t t_gc0 = t_on ? ggml_time_us() : 0;
    bool ok;
    if (!need_sched_) {
        ok = (ggml_backend_graph_compute_async(impl->backend, gf_) == GGML_STATUS_SUCCESS);
    } else {
        ok = (ggml_backend_sched_graph_compute(impl->sched, gf_) == GGML_STATUS_SUCCESS);
    }
    if (!ok) return false;
    const int64_t t_gc1 = t_on ? ggml_time_us() : 0;
    size_t n = (size_t)ggml_nelements(out_);
    out.resize(n);
    readback_async_then_sync(impl, out_, out.data(), n * ggml_element_size(out_));
    if (t_on) {
        const int64_t t_rb1 = ggml_time_us();
        std::fprintf(stderr,
            "[enc-timing]   graph_compute=%lldus readback=%lldus n_nodes=%d uid=%u\n",
            (long long)(t_gc1 - t_gc0), (long long)(t_rb1 - t_gc1),
            gf_ ? gf_->n_nodes : -1, gf_ ? (unsigned)gf_->uid : 0u);
    }
    return true;
}

bool ReplayGraph::compute_with_captures(std::vector<float>& out) {
    // Same as compute(); the readback path already drains captures.
    return compute(out);
}

} // namespace starling::ggml
