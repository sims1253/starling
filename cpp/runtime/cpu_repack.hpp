// cpu_repack.hpp — in-place CPU weight repacking for ggml's interleaved GEMM
// kernels (the CPU_REPACK extra buffer type: Q4_K/Q6_K/Q8_0/... laid out
// 4 or 8 rows at a time for the NEON dotprod/i8mm and AVX2 kernels).
//
// On CPU, ModelLoader borrows the GGUF weights zero-copy in a plain CPU
// buffer, so ggml's repacked kernels never engage. This module opts weights
// into them without extra persistent weight memory: bytes are rewritten in place
// and the tensor is re-pointed at an alias buffer over the same memory that
// carries the CPU_REPACK buffer type, which is what ggml's CPU backend
// dispatches on.
//
// A repacked weight is only valid as the direct src0 of a MUL_MAT with F32
// activations; any other reader (reshape/view, get_rows, conv, a copy)
// would read the interleaved bytes as plain rows and compute garbage. So
// the decision is made from observed usage, never from names:
//
//   * prepare_graph() runs before every graph whose primary backend is the
//     CPU is allocated, whichever allocator then runs it (gallocr, or the
//     sched route imatrix collection takes); GPU backends never attach
//     weights, so nothing there is ever repacked. A weight is
//     repacked the first time a graph uses it, and only if every use in that
//     graph is a supported MUL_MAT; any other use marks it never-repack.
//   * A graph that uses an already-repacked weight any other way throws
//     (std::runtime_error) instead of computing wrong results, and a host
//     readback of a repacked weight (ggml_backend_tensor_get/copy) aborts
//     with the same diagnosis instead of calling a null hook.
//
// Repacking is idempotent per tensor and survives release/re-realize cycles
// of the loader (the host bytes stay repacked, so re-attached tensors are
// re-pointed at the alias immediately). The plain bytes are not retained;
// disabling repacking after a weight has changed requires a fresh process
// and a model reload.
//
// Gate: STARLING_GGML_CPU_REPACK=1/true/on/yes enables, =0/false/off/no
// disables (anything else warns and keeps the default). Default: enabled on
// Android (phones are CPU-only and gain the most), disabled elsewhere.
#pragma once

#include <cstddef>
#include <cstdint>

struct ggml_backend_buffer;
struct ggml_cgraph;
struct ggml_context;

namespace starling::ggml::cpu_repack {

// Whether repacking is active for this process (env gate, read once).
bool enabled();

// ModelLoader::realize_weights (CPU path): `buffer` borrows the weight memory
// [base, base + size). Tensors in that range already repacked by an earlier
// attach are re-pointed at the repack alias right away. Throws if an alias
// cannot be recreated for weights whose bytes were already repacked.
void attach(ggml_backend_buffer* buffer);

// Before `buffer` is freed (ModelLoader::release_runtime_resources).
void detach(ggml_backend_buffer* buffer);

// The memory [base, base + size) is going away (ModelLoader destructor):
// drop every per-tensor decision for tensors inside it.
void forget(const void* base, size_t size);

// Scan a CPU-backend graph before allocation; repacks newly eligible weights.
// Throws std::runtime_error when a repacked weight has an unsupported use.
void prepare_graph(ggml_cgraph* gf);

// Observability (tests, logs): weights and bytes currently repacked.
struct Stats {
    int64_t tensors = 0;
    int64_t bytes = 0;
};
Stats stats();

}  // namespace starling::ggml::cpu_repack
