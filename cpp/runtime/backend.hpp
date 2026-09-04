// backend.hpp — the ggml backend abstraction + the ReplayGraph capture/replay
// primitive for Starling's ggml engine.
//
// This is the GENERIC, model-agnostic engine infrastructure shared by every
// model implementation (parakeet, moss). It mirrors the proven architecture of
// parakeet.cpp's Backend/ReplayGraph (one good way to wrap ggml for an
// inference engine that relies on CUDA-graph capture/replay), written here as
// Starling-owned first-party code.
//
// Two compute paths:
//   - Backend::compute : build + run a ONE-SHOT graph (allocate, run, read back,
//     free the ggml context). Used by tests and one-off graph builds.
//   - ReplayGraph      : build ONCE, replay many times with fresh inputs.
//     Keeps the ggml context + cgraph alive across calls so ggml-cuda can key
//     its CUDA-graph capture on the (stable) first-node pointer and actually
//     warm up the capture. This is the per-shape graph-capture primitive the
//     encoder, mel, and K-step decode all cache on.
//
// Critical invariants (load-bearing for correctness + perf):
//   - no_alloc=true build context: tensor ->data is NULL until gallocr alloc,
//     so host inputs MUST be pushed AFTER alloc (register_input defers them).
//   - persistent gallocr reused across calls (never backend_sched on the fast
//     path: sched re-plans every call and regressed CUDA 7-23% in parakeet.cpp).
//   - private gallocr PER ReplayGraph (sharing one invalidates coexisting
//     graphs' device pointers when a larger graph reallocs the shared buffer).
//   - stable cgraph uid: skips ggml-cuda's O(n_nodes) per-replay memcmp.
//   - async set_input + async readback + SINGLE sync per replay.

#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <utility>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;
struct ggml_backend;
typedef struct ggml_backend* ggml_backend_t;
struct ggml_gallocr;
typedef struct ggml_gallocr* ggml_gallocr_t;

namespace starling::ggml {

class ModelLoader;

// A persistent compute backend (CPU or a GPU via ggml's device registry) with a
// reusable graph allocator. Constructed once per process (see global_backend)
// and shared by every model.
class Backend {
public:
    explicit Backend(int n_threads = 8);
    ~Backend();

    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;

    void set_n_threads(int n_threads);
    int  n_threads() const { return n_threads_; }

    // Name of the selected device ("cpu", or the GPU's registry name e.g.
    // "CUDA0"). Driven by the STARLING_GGML_DEVICE env var or auto-detected.
    const char* device_name() const { return device_name_.c_str(); }

    // True iff a GPU backend is active (the replay optimisations only help on
    // GPU, where launch overhead dominates; callers gate on this).
    bool is_gpu() const;

    // The underlying ggml backend handle. Exposed so the loader can place
    // weight tensors in a buffer on the SAME backend graphs run on.
    ggml_backend_t handle() const;

    // Build + run a one-shot graph: `build(ctx)` constructs the graph in a
    // no_alloc=true context (register host inputs via add_graph_input; reference
    // loader weights directly as leaves), Backend allocates it on the persistent
    // gallocr, pushes the registered inputs, runs it, and reads the output
    // tensor's f32 contents into `out`. Returns true on success.
    bool compute(const std::function<ggml_tensor*(ggml_context*)>& build,
                 std::vector<float>& out);

    // Internal hook used by add_graph_input / capture_graph_output (routed via
    // a thread-local pointer to the Backend driving the current compute).
    void register_input(ggml_tensor* t, const void* host, size_t nbytes);
    void register_capture(ggml_tensor* t, std::vector<float>* dst);

private:
    struct Impl;
    Impl* impl_;
    int   n_threads_ = 1;
    std::string device_name_ = "cpu";

    friend class ReplayGraph;
};

// Register a host-backed graph input for the currently-active Backend::compute
// or ReplayGraph build. Marks `t` as a graph input and records an H2D copy of
// `nbytes` from `host` to perform AFTER allocation. Must be called from inside
// a build lambda (routes via thread-local). `host` must stay valid until the
// compute returns.
void add_graph_input(ggml_tensor* t, const void* host, size_t nbytes);

// Create a graph input tensor in `ctx` and register its host backing in one
// call (the common case). Returns the new tensor.
ggml_tensor* graph_input_tensor(ggml_context* ctx, int type, int n_dims,
                                const int64_t* ne, const void* host,
                                size_t nbytes);

// Capture an intermediate tensor for readback after compute. `*dst` is resized
// and filled with the tensor's f32 contents once the graph has run. Must be
// called from inside a build lambda.
void capture_graph_output(ggml_tensor* t, std::vector<float>* dst);

// Mark `t` as a graph node that MUST execute as a side effect but is NOT read
// back to host. The Wave D decode-state write-back uses this: the K-step graph
// cpy's its final (h/c/cc/frame/last_token) state into persistent device
// buffers (graph leaves) so the next replay reads it in-graph with no host
// round-trip; those cpy nodes are unreachable from the output/captures, so they
// must be registered as expansion roots. Must be called from inside a build
// lambda; the node is expanded into the cgraph alongside the output + captures.
void add_graph_root(ggml_tensor* t);

// Reference a loader weight DIRECTLY as a graph leaf (zero per-call copy). The
// loader gives every weight a backend buffer once (realize_weights); with
// ->data set, the gallocr treats the weight as already-allocated and never
// touches it. Allowlisted linears may be f16/q8_0 (ggml_mul_mat dequantizes).
// clone_weight asserts the name is present; clone_weight_opt returns nullptr.
ggml_tensor* clone_weight(ggml_context* ctx, const ModelLoader& ml,
                          const char* name);
ggml_tensor* clone_weight_opt(ggml_context* ctx, const ModelLoader& ml,
                              const char* name);

// Ensure the loader's weights have a backend buffer (zero-copy) on the
// process-global Backend. Idempotent.
void ensure_weights_realized(const ModelLoader& ml);

// Copy a weight's f32 contents to `out` on the host (for host-side math like
// batch-norm folding). NOT for graph leaves (use clone_weight).
void weight_to_host_f32(const ModelLoader& ml, const char* name,
                        std::vector<float>& out);

// A graph built once and replayed many times, keeping the same ggml context +
// cgraph alive so ggml-cuda can capture + replay it. Callers feed fresh input
// data each call via set_input.
class ReplayGraph {
public:
    ReplayGraph(Backend& backend,
                const std::function<ggml_tensor*(ggml_context*)>& build);
    ~ReplayGraph();

    ReplayGraph(const ReplayGraph&) = delete;
    ReplayGraph& operator=(const ReplayGraph&) = delete;

    // Feed `nbytes` from `host` into input #`i` (registration-order).
    void set_input(size_t i, const void* host, size_t nbytes);

    // Recompute + read the output tensor's f32 contents into `out`.
    bool compute(std::vector<float>& out);

    // Recompute + read the output AND every capture registered during build
    // (into the caller's stable dst vectors). Used by the decode loop to pull
    // new per-layer state out of each replayed step.
    bool compute_with_captures(std::vector<float>& out);

    size_t n_inputs() const { return inputs_.size(); }
    size_t input_nbytes(size_t i) const;
    const void* input_host(size_t i) const;

private:
    Backend& backend_;
    ggml_context* ctx_ = nullptr;
    ggml_cgraph*  gf_  = nullptr;
    ggml_tensor*  out_ = nullptr;
    std::vector<ggml_tensor*> inputs_;
    std::vector<const void*> input_hosts_;
    std::vector<std::pair<ggml_tensor*, std::vector<float>*>> captures_;
    bool need_sched_ = false;
    ggml_gallocr_t galloc_ = nullptr;

    bool alloc_internal();
    void readback_async_then_sync(struct Backend::Impl* impl,
                                  ggml_tensor* out_t,
                                  std::vector<float>& out);
};

} // namespace starling::ggml
