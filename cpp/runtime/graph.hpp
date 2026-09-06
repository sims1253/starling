// Shared process backend. Public C API calls serialize inference, model
// lifetime and shutdown with runtime_mutex(). Each model owns its weight
// buffers and replay caches; shutdown releases those before the backend.
// The atexit hook is registered after backend creation, while the driver is
// alive. Shutdown is terminal: subsequent model loads/inference fail.

#pragma once

#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace starling::ggml {

class Backend;

// Serializes public inference, model lifetime, and backend teardown.
std::recursive_mutex& runtime_mutex();

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

// Release all live models' runtime resources, then the backend. Idempotent.
void shutdown_backend();

// True after orderly resource teardown has completed.
bool shutting_down();

} // namespace starling::ggml
