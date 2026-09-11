// Process backend lifetime and serialized execution. See graph.hpp.

#include "graph.hpp"

#include "backend.hpp"
#include "model_loader.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <stdexcept>
#include <utility>
#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#endif

namespace starling::ggml {
namespace {

constexpr int kDefaultThreads = 8;

// Default CPU thread count: physical cores (SMT siblings share execution
// units and only add barrier/dequant contention for this workload; measured
// 6 > 4 > 8 > 12 on a 6C/12T box, worse at every stage with SMT on). Linux:
// unique (package, core) among the sched-affine CPUs (respects cpusets).
// Elsewhere: 0 (unknown) -> the kDefaultThreads fallback below.
int physical_core_default() {
#ifdef __linux__
    cpu_set_t mask;
    CPU_ZERO(&mask);
    const long ncpu = std::min(sysconf(_SC_NPROCESSORS_CONF), (long)CPU_SETSIZE);
    if (ncpu < 1) return 0;
    if (sched_getaffinity(0, sizeof(mask), &mask) != 0) return 0;
    std::set<std::pair<std::string, std::string>> cores;
    char path[128], buf[64];
    for (int i = 0; i < ncpu; ++i) {
        if (!CPU_ISSET(i, &mask)) continue;
        std::snprintf(path, sizeof(path),
            "/sys/devices/system/cpu/cpu%d/topology/core_id", i);
        FILE* f = std::fopen(path, "r");
        if (!f) return 0;
        const bool ok1 = std::fgets(buf, sizeof(buf), f) != nullptr;
        std::fclose(f);
        if (!ok1) return 0;
        std::string core(buf);
        std::snprintf(path, sizeof(path),
            "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", i);
        f = std::fopen(path, "r");
        if (!f) return 0;
        const bool ok2 = std::fgets(buf, sizeof(buf), f) != nullptr;
        std::fclose(f);
        if (!ok2) return 0;
        cores.emplace(buf, core);
    }
    return cores.empty() ? 0 : (int)cores.size();
#else
    return 0;
#endif
}

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
    // [starling] env override for CPU thread count (default: physical cores;
    // SMT siblings measured slower at every stage). STARLING_GGML_THREADS wins.
    int n_env = 0;
    if (const char* e = std::getenv("STARLING_GGML_THREADS")) n_env = atoi(e);
    const int n_phys = physical_core_default();
    int n = g_threads_set.load() ? g_num_threads.load()
          : (n_env > 0 ? n_env : (n_phys > 0 ? n_phys : kDefaultThreads));
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
