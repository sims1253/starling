// Process backend lifetime and serialized execution. See graph.hpp.

#include "graph.hpp"

#include "backend.hpp"
#include "model_loader.hpp"

#include <atomic>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <stdexcept>

namespace starling::ggml {
namespace {

constexpr int kDefaultThreads = 8;

// Graph builders may re-enter global_backend() while the runtime lock is held.
std::recursive_mutex g_backend_mutex;
std::unique_ptr<Backend> g_backend;
std::atomic<bool> g_shutting_down{false};
std::atomic<bool> g_atexit_registered{false};
std::atomic<int> g_num_threads{kDefaultThreads};
std::atomic<bool> g_threads_set{false};

void atexit_shutdown() {
    shutdown_backend();
}

} // namespace

std::recursive_mutex& runtime_mutex() { return g_backend_mutex; }

void set_num_threads(int n_threads) {
    g_num_threads.store(n_threads < 1 ? 1 : n_threads);
    g_threads_set.store(true);
    // Also apply to an already-created backend.
    std::lock_guard<std::recursive_mutex> lk(g_backend_mutex);
    if (g_backend) g_backend->set_n_threads(g_num_threads.load());
}

Backend& global_backend() {
    std::lock_guard<std::recursive_mutex> lk(g_backend_mutex);
    if (g_shutting_down.load()) throw std::runtime_error("Starling backend has been shut down");
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

std::optional<std::string> try_global_backend_device_name() {
    std::lock_guard<std::recursive_mutex> lk(g_backend_mutex);
    if (!g_backend) return std::nullopt;
    return std::string(g_backend->device_name());  // copy under the lock
}

bool run_graph(const std::function<ggml_tensor*(ggml_context*)>& build,
               std::vector<float>& out) {
    std::lock_guard<std::recursive_mutex> lk(g_backend_mutex);
    return g_backend ? g_backend->compute(build, out) : false;
}

void shutdown_backend() {
    std::lock_guard<std::recursive_mutex> lock(g_backend_mutex);
    if (g_shutting_down.load()) return;
    ModelLoader::release_all_runtime_resources();
    g_backend.reset();
    // Destructors above must free resources while the backend is alive.
    g_shutting_down.store(true);
}

bool shutting_down() {
    return g_shutting_down.load();
}

} // namespace starling::ggml
