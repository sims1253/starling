// cpu_repack_test — in-place CPU weight repacking (cpp/runtime/cpu_repack.hpp).
//
// Self-contained (random weights, no model files), CPU backend only:
//   1. For each quantized type, MUL_MAT through the repacked layout matches
//      the plain layout (batched GEMM rows and the M=1 GEMV path).
//   2. A weight read by any other op (get_rows) is never repacked, and stays
//      correct for both uses.
//   3. A graph that reads an already-repacked weight through a view, or
//      multiplies it with a transposed activation, throws instead of
//      computing garbage.
//   4. detach/attach (loader release + re-realize) keeps repacked weights
//      working; forget drops the bookkeeping.
//
// Whether a type repacks depends on the CPU's kernels (x86 AVX2: q4_0,
// q4_K, iq4_nl; arm64 dotprod/i8mm: q4_0, q4_K, q5_K, q6_K, q8_0,
// iq4_nl). Correctness is checked
// either way; STARLING_REPACK_TEST_EXPECT=q4_0,q4_K,... (ggml_type_name
// spelling) additionally requires those types to have been repacked (set
// by the CI jobs).
#include "runtime/cpu_repack.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace repack = starling::ggml::cpu_repack;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_failures;
}

struct Weights {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor* w = nullptr;
};

// A quantized [K, N] weight in its own memory-backed context, borrowed into
// a CPU buffer exactly like ModelLoader::realize_weights does.
Weights make_weights(ggml_type type, int K, int N, uint32_t seed) {
    Weights out;
    const size_t bytes = ggml_row_size(type, K) * N;
    ggml_init_params params = {ggml_tensor_overhead() * 4 + bytes + 1024, nullptr, false};
    out.ctx = ggml_init(params);
    out.w = ggml_new_tensor_2d(out.ctx, type, K, N);
    ggml_set_name(out.w, (std::string("w_") + ggml_type_name(type)).c_str());
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.5f);
    std::vector<float> f((size_t)K * N);
    for (float& v : f) v = dist(rng);
    ggml_quantize_chunk(type, f.data(), out.w->data, 0, N, K, nullptr);
    out.buffer = ggml_backend_cpu_buffer_from_ptr(ggml_get_mem_buffer(out.ctx), ggml_get_mem_size(out.ctx));
    out.w->buffer = out.buffer;
    return out;
}

void free_weights(Weights& w) {
    repack::detach(w.buffer);
    repack::forget(ggml_get_mem_buffer(w.ctx), ggml_get_mem_size(w.ctx));
    ggml_backend_buffer_free(w.buffer);
    ggml_free(w.ctx);
}

// Frees on every exit path, so a throwing check cannot leak bookkeeping
// into later tests (or the final "forget() drops all bookkeeping" check).
struct WeightsGuard {
    Weights& w;
    ~WeightsGuard() { free_weights(w); }
};

std::vector<float> random_input(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> x(n);
    for (float& v : x) v = dist(rng);
    return x;
}

// Builds and computes one graph on the CPU backend. `build` returns the
// output; `inputs` are (tensor, host data) pairs set after allocation.
std::vector<float> run(ggml_backend_t backend,
                       const std::function<ggml_tensor*(ggml_context*, std::vector<std::pair<ggml_tensor*, const void*>>&)>& build) {
    ggml_init_params params = {ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true};
    // Owned so the expected-throw paths (a guarded view) release them too.
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx_owner(ggml_init(params), ggml_free);
    ggml_context* ctx = ctx_owner.get();
    std::vector<std::pair<ggml_tensor*, const void*>> inputs;
    ggml_tensor* out = build(ctx, inputs);
    ggml_set_output(out);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    repack::prepare_graph(gf);  // the Backend::compute hook
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> galloc_owner(
        ggml_gallocr_new(ggml_backend_cpu_buffer_type()), ggml_gallocr_free);
    ggml_gallocr_t galloc = galloc_owner.get();
    if (!ggml_gallocr_alloc_graph(galloc, gf)) throw std::runtime_error("alloc failed");
    for (auto& [t, host] : inputs) ggml_backend_tensor_set(t, host, 0, ggml_nbytes(t));
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) throw std::runtime_error("compute failed");
    std::vector<float> result((size_t)ggml_nelements(out));
    ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));
    return result;
}

std::vector<float> mul_mat(ggml_backend_t backend, ggml_tensor* w, const std::vector<float>& x, int K, int M) {
    return run(backend, [&](ggml_context* ctx, auto& inputs) {
        ggml_tensor* xt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
        inputs.emplace_back(xt, x.data());
        return ggml_mul_mat(ctx, w, xt);
    });
}

double max_rel_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double scale = 1e-6, diff = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        scale = std::max(scale, (double)std::fabs(a[i]));
        diff = std::max(diff, (double)std::fabs(a[i] - b[i]));
    }
    return diff / scale;
}

bool expected_to_repack(ggml_type type) {
    const char* expect = std::getenv("STARLING_REPACK_TEST_EXPECT");
    if (!expect) return false;
    std::string list = std::string(",") + expect + ",";
    return list.find(std::string(",") + ggml_type_name(type) + ",") != std::string::npos;
}

void test_type(ggml_backend_t backend, ggml_type type) {
    const int K = 512, N = 64;
    const std::string tag = ggml_type_name(type);
    Weights w = make_weights(type, K, N, 7);
    WeightsGuard guard{w};
    const auto x_batch = random_input((size_t)K * 37, 11);  // 37 rows: not a multiple of 4/8
    const auto x_single = random_input((size_t)K, 13);

    // Negative control for the view guard below: before anything is
    // attached, a view of the same weight computes normally.
    const auto view_graph = [&](ggml_context* ctx, auto&) {
        return ggml_cont(ctx, ggml_view_2d(ctx, w.w, K, 1, w.w->nb[1], 0));
    };
    bool plain_view_ok = true;
    try {
        run(backend, view_graph);
    } catch (const std::exception&) {
        plain_view_ok = false;
    }
    check(plain_view_ok, tag + " view of a plain (unattached) weight computes");

    // Reference: the weight's buffer is not attached, so nothing repacks.
    const auto ref_batch = mul_mat(backend, w.w, x_batch, K, 37);
    const auto ref_single = mul_mat(backend, w.w, x_single, K, 1);

    const auto before = repack::stats().tensors;
    repack::attach(w.buffer);
    const auto got_batch = mul_mat(backend, w.w, x_batch, K, 37);
    const bool repacked = repack::stats().tensors > before;
    const auto got_single = mul_mat(backend, w.w, x_single, K, 1);

    std::printf("  %s: %s\n", tag.c_str(), repacked ? "repacked" : "no repacked kernel on this CPU (plain path)");
    check(max_rel_diff(ref_batch, got_batch) < 2e-3, tag + " GEMM matches the plain layout");
    check(max_rel_diff(ref_single, got_single) < 2e-3, tag + " GEMV matches the plain layout");
    if (expected_to_repack(type)) check(repacked, tag + " was repacked (STARLING_REPACK_TEST_EXPECT)");

    // Loader release + re-realize: the bytes stay repacked and are re-pointed.
    repack::detach(w.buffer);
    w.w->buffer = nullptr;
    w.w->buffer = w.buffer;
    repack::attach(w.buffer);
    const auto again = mul_mat(backend, w.w, x_batch, K, 37);
    check(max_rel_diff(ref_batch, again) < 2e-3, tag + " still correct after detach/attach");

    if (repacked) {
        // A view of a repacked weight must be refused, not computed.
        bool threw = false;
        try {
            run(backend, view_graph);
        } catch (const std::runtime_error& e) {
            threw = std::string(e.what()).find("cpu_repack") != std::string::npos;
        }
        check(threw, tag + " view of a repacked weight throws");

        // The repacked kernel reads activation rows as contiguous floats; a
        // transposed activation must be refused too.
        threw = false;
        try {
            run(backend, [&](ggml_context* ctx, auto& inputs) {
                ggml_tensor* xt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 37, K);
                inputs.emplace_back(xt, x_batch.data());
                return ggml_mul_mat(ctx, w.w, ggml_transpose(ctx, xt));
            });
        } catch (const std::runtime_error& e) {
            threw = std::string(e.what()).find("cpu_repack") != std::string::npos;
        }
        check(threw, tag + " transposed activation with a repacked weight throws");
    }
}

// A weight whose first graph reads it with get_rows is never repacked, so
// both the lookup and a later MUL_MAT stay on the plain layout.
void test_non_matmul_use_blocks_repacking(ggml_backend_t backend) {
    const int K = 512, N = 64;
    Weights w = make_weights(GGML_TYPE_Q4_K, K, N, 21);
    WeightsGuard guard{w};
    const auto x = random_input((size_t)K * 16, 23);
    const auto ref = mul_mat(backend, w.w, x, K, 16);
    const int32_t ids[3] = {0, 5, 63};
    const auto rows_ref = run(backend, [&](ggml_context* ctx, auto& inputs) {
        ggml_tensor* it = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 3);
        inputs.emplace_back(it, ids);
        return ggml_get_rows(ctx, w.w, it);
    });

    repack::attach(w.buffer);
    const auto before = repack::stats().tensors;
    const auto rows = run(backend, [&](ggml_context* ctx, auto& inputs) {
        ggml_tensor* it = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 3);
        inputs.emplace_back(it, ids);
        return ggml_get_rows(ctx, w.w, it);
    });
    const auto got = mul_mat(backend, w.w, x, K, 16);
    check(repack::stats().tensors == before, "a get_rows weight is never repacked");
    // The never-repack decision leaves the same bytes and kernels in place,
    // so these two plain-path results should be bit-identical.
    check(max_rel_diff(rows_ref, rows) == 0.0, "get_rows reads the untouched layout");
    check(max_rel_diff(ref, got) == 0.0, "its later MUL_MAT stays on the plain path");
}

}  // namespace

int main() {
    // Opt in before the first enabled() query (the gate latches per process).
#ifdef _WIN32
    _putenv_s("STARLING_GGML_CPU_REPACK", "1");
#else
    setenv("STARLING_GGML_CPU_REPACK", "1", 1);
#endif
    check(repack::enabled(), "STARLING_GGML_CPU_REPACK=1 enables repacking");
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        std::printf("FAIL ggml_backend_cpu_init returned null\n");
        return 1;
    }
    ggml_backend_cpu_set_n_threads(backend, 4);
    try {
        for (ggml_type type : {GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_IQ4_NL}) {
            test_type(backend, type);
        }
        test_non_matmul_use_blocks_repacking(backend);
    } catch (const std::exception& e) {
        std::printf("FAIL unexpected exception: %s\n", e.what());
        ++g_failures;
    }
    check(repack::stats().tensors == 0 && repack::stats().bytes == 0, "forget() drops all bookkeeping");
    ggml_backend_free(backend);
    std::printf("%s: %d failure(s)\n", g_failures ? "FAILED" : "OK", g_failures);
    return g_failures ? 1 : 0;
}
