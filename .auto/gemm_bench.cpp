// gemm_bench.cpp — standalone Vulkan GEMM microbench for parakeet encoder shapes.
// Usage: ./gemm_bench <type> <m> <n> <k> [reps]   (type: q8_0|f16|f32|bf16|q5_0|q4_0)
// m = weight rows (ggml ne1), n = activation cols (tokens), k = reduction dim.
#include "ggml.h"
#include "ggml-quants.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s <type> <m> <n> <k> [reps]\n", argv[0]); return 1; }
    const char* ts = argv[1];
    ggml_type wt = GGML_TYPE_F32;
    if (!strcmp(ts, "q8_0")) wt = GGML_TYPE_Q8_0;
    else if (!strcmp(ts, "q5_0")) wt = GGML_TYPE_Q5_0;
    else if (!strcmp(ts, "q4_0")) wt = GGML_TYPE_Q4_0;
    else if (!strcmp(ts, "f16")) wt = GGML_TYPE_F16;
    else if (!strcmp(ts, "bf16")) wt = GGML_TYPE_BF16;
    else if (!strcmp(ts, "f32")) wt = GGML_TYPE_F32;
    else { fprintf(stderr, "bad type\n"); return 1; }
    const uint32_t m = atoi(argv[2]);
    const uint32_t n = atoi(argv[3]);
    const uint32_t k = atoi(argv[4]);
    int reps = argc > 5 ? atoi(argv[5]) : 50;

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        fprintf(stderr, "dev %zu: %s type=%d\n", i, ggml_backend_dev_name(d), (int)ggml_backend_dev_type(d));
    }
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("Vulkan0");
    ggml_backend_t bk = dev ? ggml_backend_dev_init(dev, nullptr) : nullptr;
    if (!bk) { fprintf(stderr, "no GPU backend\n"); return 1; }
    fprintf(stderr, "backend: %s\n", ggml_backend_name(bk));

    std::vector<float> wdata((size_t)m * k), xdata((size_t)k * n);
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.f, 0.05f);
    for (auto& v : wdata) v = nd(rng);
    for (auto& v : xdata) v = nd(rng);

    // Quantize weights on CPU.
    std::vector<uint8_t> wq(ggml_row_size(wt, (size_t)m * k));
    if (wt == GGML_TYPE_F32) {
        memcpy(wq.data(), wdata.data(), wdata.size() * sizeof(float));
    } else if (wt == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(wdata.data(), (ggml_fp16_t*)wq.data(), (int64_t)m * k);
    } else {
        switch (wt) {
        case GGML_TYPE_Q8_0: quantize_row_q8_0_ref(wdata.data(), (block_q8_0*)wq.data(), (int64_t)m * k); break;
        case GGML_TYPE_Q5_0: quantize_row_q5_0_ref(wdata.data(), (block_q5_0*)wq.data(), (int64_t)m * k); break;
        case GGML_TYPE_Q4_0: quantize_row_q4_0_ref(wdata.data(), (block_q4_0*)wq.data(), (int64_t)m * k); break;
        default: fprintf(stderr, "no quant path for %s\n", ts); return 1;
        }
    }

    ggml_init_params ip3 = { 16ull<<20, nullptr, /*no_alloc*/ true };
    ggml_context* gctx = ggml_init(ip3);
    ggml_tensor* Wg = ggml_new_tensor_2d(gctx, wt, k, m);
    ggml_tensor* Xg = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, k, n);
    ggml_tensor* Y = ggml_mul_mat(gctx, Wg, Xg);
    ggml_cgraph* graph = ggml_new_graph(gctx);
    ggml_build_forward_expand(graph, Y);
    ggml_set_input(Wg); ggml_set_input(Xg); ggml_set_output(Y);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(bk));
    if (!ggml_gallocr_alloc_graph(ga, graph)) { fprintf(stderr, "alloc failed\n"); return 1; }
    ggml_backend_tensor_set(Wg, wq.data(), 0, ggml_nbytes(Wg));
    ggml_backend_tensor_set(Xg, xdata.data(), 0, ggml_nbytes(Xg));

    for (int i = 0; i < 3; ++i)
        if (ggml_backend_graph_compute(bk, graph) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return 1; }
    ggml_backend_synchronize(bk);

    int64_t t0 = ggml_time_us();
    for (int i = 0; i < reps; ++i)
        ggml_backend_graph_compute(bk, graph);
    ggml_backend_synchronize(bk);
    int64_t t1 = ggml_time_us();
    double ms = (t1 - t0) / 1000.0 / reps;
    double flops = 2.0 * m * n * k;
    printf("RESULT type=%s m=%u n=%u k=%u ms=%.3f gflops=%.1f wbytes=%lld\n",
           ts, m, n, k, ms, flops / (ms * 1e-3) / 1e9, (long long)ggml_nbytes(Wg));
    return 0;
}
