// Independent Q4_K x Q8_K CPU kernel versus ggml's persistent MUL_MAT graph.
// ggml is only used here for GGUF loading, synthetic data, and the baseline.
#include "independent_q4k.hpp"
#include "input_pattern.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
volatile float sink = 0.0f;

struct Options {
    std::string profile = "parakeet";
    std::string gguf_path;
    std::string tensor_name;
    int threads = 1;
    int iterations = 30;
    int batch = 0;
    std::string kernel = "packed-dot";
    std::string input = "cos";
    std::string output_path;
    bool list = false;
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--list") { o.list = true; continue; }
        if (i + 1 >= argc) throw std::runtime_error("missing value after " + arg);
        const std::string val = argv[++i];
        if (arg == "--profile") o.profile = val;
        else if (arg == "--gguf") o.gguf_path = val;
        else if (arg == "--tensor") o.tensor_name = val;
        else if (arg == "--threads") o.threads = std::stoi(val);
        else if (arg == "--iterations") o.iterations = std::stoi(val);
        else if (arg == "--batch") o.batch = std::stoi(val);
        else if (arg == "--kernel") o.kernel = val;
        else if (arg == "--input") o.input = val;
        else if (arg == "--output-candidate") o.output_path = val;
        else throw std::runtime_error("unknown option " + arg);
    }
    if (o.threads < 1 || o.iterations < 1 || o.batch < 0)
        throw std::runtime_error("threads and iterations must be positive; batch must be nonnegative");
    if (o.profile != "parakeet")
        throw std::runtime_error("this independent kernel PoC supports parakeet Q4_K only");
    if (o.kernel != "packed-dot" && o.kernel != "raw-dot")
        throw std::runtime_error("--kernel must be packed-dot or raw-dot");
    if (o.input != "cos" && o.input != "random" && o.input != "zero")
        throw std::runtime_error("--input must be cos, random, or zero");
    if (o.gguf_path.empty() != o.tensor_name.empty() && !o.list)
        throw std::runtime_error("--gguf and --tensor must be supplied together");
    return o;
}

struct Weights {
    ggml_type type = GGML_TYPE_Q4_0;
    int64_t cols = 0, rows = 0;
    const void* data = nullptr;
    size_t bytes = 0;
    std::unique_ptr<void, decltype(&std::free)> storage{nullptr, &std::free};
    void allocate() {
        // ggml_backend_cpu_buffer_from_ptr requires TENSOR_ALIGNMENT.
        const size_t rounded = (bytes + 63) & ~size_t(63);
        storage.reset(std::aligned_alloc(64, rounded));
        if (!storage) throw std::runtime_error("aligned weight allocation failed");
        data = storage.get();
    }
};

Weights make_synthetic(const std::string& profile) {
    Weights w;
    (void)profile;
    // Representative 1024 -> 4096 FastConformer FFN projection.
    w.type = GGML_TYPE_Q4_K;
    w.cols = 1024;
    w.rows = 4096;
    const auto* tr = ggml_get_type_traits_cpu(w.type);
    if (!tr->from_float) throw std::runtime_error("quantizer unavailable");
    const size_t row_bytes = ggml_row_size(w.type, w.cols);
    w.bytes = row_bytes * static_cast<size_t>(w.rows);
    w.allocate();
    std::vector<float> row(w.cols);
    for (int64_t r = 0; r < w.rows; ++r) {
        for (int64_t c = 0; c < w.cols; ++c)
            row[c] = std::sin(float((r * 17 + c * 13) % 211) * 0.031f) * 0.08f;
        tr->from_float(row.data(), static_cast<uint8_t*>(w.storage.get()) + r * row_bytes, w.cols);
    }
    return w;
}

void list_tensors(const std::string& path) {
    ggml_context* ctx = nullptr;
    gguf_init_params p = {true, &ctx};
    gguf_context* f = gguf_init_from_file(path.c_str(), p);
    if (!f) throw std::runtime_error("cannot open GGUF");
    int shown = 0, total = 0;
    for (int64_t i = 0; i < gguf_get_n_tensors(f); ++i) {
        const auto type = gguf_get_tensor_type(f, i);
        if (type != GGML_TYPE_Q4_K) continue;
        ++total;
        if (shown++ < 60) {
            const char* name = gguf_get_tensor_name(f, i);
            const ggml_tensor* t = ggml_get_tensor(ctx, name);
            std::printf("%s %s %lld x %lld\n", name, ggml_type_name(type),
                        static_cast<long long>(t->ne[0]), static_cast<long long>(t->ne[1]));
        }
    }
    std::printf("Q4_K tensors: %d (first 60 shown)\n", total);
    gguf_free(f);
    ggml_free(ctx);
}

Weights load_tensor(const std::string& path, const std::string& name) {
    Weights w;
    ggml_context* ctx = nullptr;
    gguf_init_params p = {false, &ctx};
    gguf_context* file = gguf_init_from_file(path.c_str(), p);
    if (!file) throw std::runtime_error("cannot load GGUF");
    const ggml_tensor* t = ggml_get_tensor(ctx, name.c_str());
    if (!t) throw std::runtime_error("tensor not found: " + name);
    if (t->type != GGML_TYPE_Q4_K)
        throw std::runtime_error("independent kernel supports Q4_K tensors only");
    if (t->ne[2] != 1 || t->ne[3] != 1) throw std::runtime_error("tensor must be 2-D");
    w.type = t->type;
    w.cols = t->ne[0];
    w.rows = t->ne[1];
    w.bytes = ggml_nbytes(t);
    w.allocate();
    std::memcpy(w.storage.get(), t->data, w.bytes);
    gguf_free(file);
    ggml_free(ctx);
    return w;
}

struct Graph {
    ggml_context* ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_tensor* x = nullptr;
    ggml_tensor* y = nullptr;
    ggml_cgraph* graph = nullptr;
    Graph(const Weights& w, int threads, int batch) {
        ggml_init_params p = {ggml_tensor_overhead() * 8 + ggml_graph_overhead() + 4096,
                              nullptr, true};
        ctx = ggml_init(p);
        if (!ctx) throw std::runtime_error("ggml context allocation failed");
        auto* wt = ggml_new_tensor_2d(ctx, w.type, w.cols, w.rows);
        x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.cols, batch);
        y = ggml_mul_mat(ctx, wt, x);
        ggml_set_input(x);
        ggml_set_output(y);
        graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, y);
        backend = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(backend, threads);
        weight_buffer = ggml_backend_cpu_buffer_from_ptr(const_cast<void*>(w.data), w.bytes);
        if (!weight_buffer || ggml_backend_tensor_alloc(weight_buffer, wt, const_cast<void*>(w.data)) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("CPU weight buffer allocation failed");
        allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(allocator, graph))
            throw std::runtime_error("ggml graph allocation failed");
    }
    ~Graph() {
        if (allocator) ggml_gallocr_free(allocator);
        if (weight_buffer) ggml_backend_buffer_free(weight_buffer);
        if (backend) ggml_backend_free(backend);
        if (ctx) ggml_free(ctx);
    }
    void run(const float* input, float* output) {
        ggml_backend_tensor_set(x, input, 0, ggml_nbytes(x));
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("ggml CPU graph compute failed");
        ggml_backend_tensor_get(y, output, 0, ggml_nbytes(y));
    }
};

template<class A, class B>
std::pair<double, double> paired_medians_us(A&& candidate, B&& baseline,
                                            int iterations) {
    std::vector<double> a, b;
    a.reserve(iterations);
    b.reserve(iterations);
    auto measure = [](auto&& run) {
        auto start = Clock::now();
        run();
        const auto end = Clock::now();
        return std::chrono::duration<double, std::micro>(end - start).count();
    };
    for (int i = 0; i < 5; ++i) { candidate(); baseline(); }
    for (int i = 0; i < iterations; ++i) {
        if (i & 1) {
            b.push_back(measure(baseline));
            a.push_back(measure(candidate));
        } else {
            a.push_back(measure(candidate));
            b.push_back(measure(baseline));
        }
    }
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    return {a[a.size() / 2], b[b.size() / 2]};
}

int argmax(const std::vector<float>& v) {
    return static_cast<int>(std::max_element(v.begin(), v.end()) - v.begin());
}
} // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);
        if (o.list) {
            if (o.gguf_path.empty()) throw std::runtime_error("--list requires --gguf");
            list_tensors(o.gguf_path);
            return 0;
        }
        Weights w = o.gguf_path.empty() ? make_synthetic(o.profile)
                                         : load_tensor(o.gguf_path, o.tensor_name);
        const int batch = o.batch > 0 ? o.batch : 50;
        std::vector<float> x(static_cast<size_t>(w.cols) * batch);
        std::vector<float> independent_out(static_cast<size_t>(w.rows) * batch);
        std::vector<float> graph_out(static_cast<size_t>(w.rows) * batch);
        independent_q4k::fill_input(x.data(), x.size(), o.input);
        const auto pack_start = Clock::now();
        independent_q4k::Matrix independent(w.data, w.bytes, w.cols, w.rows,
                                             o.kernel != "raw-dot");
        const double pack_us = std::chrono::duration<double, std::micro>(
            Clock::now() - pack_start).count();
        Graph graph(w, o.threads, batch);
        independent.run(x.data(), batch, independent_out.data(), o.threads,
                        o.kernel == "raw-dot");
        graph.run(x.data(), graph_out.data());
        float max_abs = 0.0f, max_rel = 0.0f;
        double sum_square_error = 0, sum_square_ref = 0;
        for (size_t i = 0; i < independent_out.size(); ++i) {
            const float d = std::abs(independent_out[i] - graph_out[i]);
            max_abs = std::max(max_abs, d);
            max_rel = std::max(max_rel, d / std::max(1e-5f, std::abs(graph_out[i])));
            sum_square_error += double(d) * d;
            sum_square_ref += double(graph_out[i]) * graph_out[i];
        }
        if (!std::isfinite(max_abs)) throw std::runtime_error("non-finite output difference");
        const auto [independent_us, graph_us] = paired_medians_us([&] {
            independent.run(x.data(), batch, independent_out.data(), o.threads,
                            o.kernel == "raw-dot");
            sink = independent_out[0];
        }, [&] {
            graph.run(x.data(), graph_out.data()); sink = graph_out[0];
        }, o.iterations);
        if (!o.output_path.empty()) {
            std::ofstream output(o.output_path, std::ios::binary);
            if (!output) throw std::runtime_error("cannot open candidate output file");
            output.write(reinterpret_cast<const char*>(independent_out.data()),
                         static_cast<std::streamsize>(independent_out.size() * sizeof(float)));
            if (!output) throw std::runtime_error("cannot write candidate output file");
        }
        std::printf("source=%s kernel=%s input=%s type=%s K=%lld N=%lld M=%d weights=%.1f MiB packed=%.1f MiB metadata=%.1f MiB pack=%.1f us threads=%d\n",
                    o.gguf_path.empty() ? "synthetic" : o.tensor_name.c_str(),
                    o.kernel.c_str(), o.input.c_str(), ggml_type_name(w.type),
                    static_cast<long long>(w.cols),
                    static_cast<long long>(w.rows), batch, w.bytes / 1048576.0,
                    independent.packed_bytes() / 1048576.0,
                    independent.metadata_bytes() / 1048576.0, pack_us, o.threads);
        std::printf("independent=%.1f us ggml_graph=%.1f us speedup=%.3fx\n",
                    independent_us, graph_us, graph_us / independent_us);
        std::printf("max_abs=%.7g max_rel=%.7g nrmse=%.7g argmax_independent=%d argmax_ggml=%d\n",
                    max_abs, max_rel, std::sqrt(sum_square_error / std::max(1e-20, sum_square_ref)),
                    argmax(independent_out), argmax(graph_out));
        return max_abs < 1e-3f && argmax(independent_out) == argmax(graph_out) ? 0 : 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "poc_q4_cpu: %s\n", e.what());
        return 1;
    }
}
