// graph.cpp — process-global Backend + run_graph + shutdown ordering.
//
// See graph.hpp. The shutdown sequence is load-bearing: ggml's destructors
// (~Backend, ~ReplayGraph, ~ggml_cuda_graph) call into the CUDA driver; if they
// run at process exit AFTER the driver's own atexit handler has torn the driver
// down, they abort with "driver shutting down". shutdown_backend() runs first
// (clear caches -> backend.reset -> flag) and is registered via std::atexit on
// first backend creation so the ctypes path (which never calls shutdown
// explicitly) still exits cleanly.

#include "graph.hpp"

#include "backend.hpp"

#include <atomic>
#include <cstdlib>
#include <memory>
#include <mutex>

namespace starling::ggml {
namespace {

constexpr int kDefaultThreads = 8;

// Recursive because run_graph() holds this mutex across Backend::compute() —
// and compute's BUILD lambda may legitimately call global_backend() again
// (e.g. parakeet/ark's rel-pos attention probe is_gpu() to pick the flash
// path). A plain std::mutex self-deadlocks there on the CPU backend, where
// the one-shot run_graph path is the only encoder route (GPU builds go
// through ReplayGraph, which does not hold the mutex).
std::recursive_mutex g_backend_mutex;
std::unique_ptr<Backend> g_backend;
std::atomic<bool> g_shutting_down{false};
std::atomic<bool> g_atexit_registered{false};
std::atomic<int> g_num_threads{kDefaultThreads};
std::atomic<bool> g_threads_set{false};

// Hook each model registers so shutdown_backend() can free process-global
// decode-graph caches (parakeet's K-step multistep, moss's decode caches)
// BEFORE the backend is reset. Set by register_decode_cache_clearer().
std::mutex g_clearer_mutex;
std::vector<std::function<void()>> g_decode_cache_clearers;

void atexit_shutdown() {
    shutdown_backend();
}

} // namespace

void register_decode_cache_clearer(std::function<void()> clearer) {
    std::lock_guard<std::mutex> lk(g_clearer_mutex);
    g_decode_cache_clearers.push_back(std::move(clearer));
}

void set_num_threads(int n_threads) {
    g_num_threads.store(n_threads < 1 ? 1 : n_threads);
    g_threads_set.store(true);
    // Also apply to an already-created backend.
    std::lock_guard<std::recursive_mutex> lk(g_backend_mutex);
    if (g_backend) g_backend->set_n_threads(g_num_threads.load());
}

Backend& global_backend() {
    std::lock_guard<std::recursive_mutex> lk(g_backend_mutex);
    if (g_backend) return *g_backend;
    int n = g_threads_set.load() ? g_num_threads.load() : kDefaultThreads;
    g_backend = std::make_unique<Backend>(n);
    // Register the atexit handler exactly once. The CUDA driver registers ITS
    // atexit handler lazily on the first CUDA call, which happens inside the
    // Backend ctor above (before this line). atexit handlers run in REVERSE
    // registration order, so ours runs BEFORE the driver's teardown -> the
    // driver is still alive when our destructors call into it.
    bool expected = false;
    if (g_atexit_registered.compare_exchange_strong(expected, true)) {
        std::atexit(atexit_shutdown);
    }
    return *g_backend;
}

bool run_graph(const std::function<ggml_tensor*(ggml_context*)>& build,
               std::vector<float>& out) {
    std::lock_guard<std::recursive_mutex> lk(g_backend_mutex);
    return g_backend ? g_backend->compute(build, out) : false;
}

void shutdown_backend() {
    // Idempotent: a flag exchange gates the whole body so the atexit handler +
    // any explicit call don't double-run.
    bool expected = false;
    if (!g_shutting_down.compare_exchange_strong(expected, true)) return;

    // 1. Clear every process-global decode-graph cache (each holds ReplayGraphs
    //    that reference the backend's device buffers) WHILE the driver is alive.
    {
        std::vector<std::function<void()>> clearers;
        {
            std::lock_guard<std::mutex> lk(g_clearer_mutex);
            clearers.swap(g_decode_cache_clearers);
        }
        for (auto& c : clearers) c();
    }
    // 2. Reset the global Backend (frees its device buffers + captured CUDA
    //    graphs + the ggml_cuda_graph instances).
    {
        std::lock_guard<std::recursive_mutex> lk(g_backend_mutex);
        g_backend.reset();
    }
}

bool shutting_down() {
    return g_shutting_down.load();
}

} // namespace starling::ggml
