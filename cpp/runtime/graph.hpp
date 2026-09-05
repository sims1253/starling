// graph.hpp — the process-global Backend + one-shot run_graph + shutdown.
//
// Starling's ggml backend is a single process-global Backend (ggml backends are
// not designed for concurrent use; callers serialise). global_backend() lazily
// creates it. run_graph() is the one-shot entry model code uses for graphs that
// don't merit a persistent ReplayGraph.
//
// Shutdown ordering (the teardown-crash fix): shutdown_backend() frees the
// global Backend + every process-global graph cache BEFORE the CUDA driver's
// own atexit handler tears the driver down. It's registered via std::atexit on
// first backend creation so a caller that never calls shutdown (e.g. the
// ctypes path) still exits cleanly. Idempotent; safe to call alongside the
// atexit handler.

#pragma once

#include <functional>
#include <optional>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace starling::ggml {

class Backend;

// Register a function that clears a process-global decode-graph cache (e.g.
// parakeet's K-step multistep cache, moss's decode caches). shutdown_backend()
// calls each registered clearer BEFORE resetting the Backend, so the cached
// graphs' device buffers are freed while the CUDA driver is still alive. Models
// call this once at their cache definition site.
void register_decode_cache_clearer(std::function<void()> clearer);

// lazily creates the process-global Backend on first call, then returns it.
// `STARLING_GGML_DEVICE` env var selects the device ("cpu", "CUDA0",
// "Vulkan0", "Metal", ...); unset auto-picks the first GPU/IGPU, else CPU.
Backend& global_backend();

// The runtime-selected device's name (e.g. "CPU", "Vulkan0", "CUDA0") if the
// global Backend already exists, else nullopt. Never creates the Backend —
// use for build-vs-runtime reporting before any model load. Returns BY
// VALUE: shutdown_backend() may destroy the Backend concurrently, so a
// pointer into its device_name_ could dangle mid-copy at the caller.
std::optional<std::string> try_global_backend_device_name();

// Override the worker-thread count (CPU backend). Honored on the next
// global_backend() creation if called before first use.
void set_num_threads(int n_threads);

// Build + run a one-shot graph on the global Backend and read the output
// tensor's f32 contents into `out`. Convenience wrapper over
// global_backend().compute(build, out) under the global mutex (serialises
// concurrent callers). Returns true on success.
bool run_graph(const std::function<ggml_tensor*(ggml_context*)>& build,
               std::vector<float>& out);

// Tear down the global Backend + every process-global graph cache. Frees device
// buffers + captured CUDA graphs while the driver is alive. Idempotent. See
// the file header for the atexit registration.
void shutdown_backend();

// True once shutdown_backend() has run. Read by ~Backend / ~ReplayGraph as a
// belt-and-suspenders guard so a Backend/ReplayGraph destroyed after shutdown
// (e.g. by static destruction in unspecified order) skips its ggml frees
// instead of aborting inside a dead driver.
bool shutting_down();

} // namespace starling::ggml
