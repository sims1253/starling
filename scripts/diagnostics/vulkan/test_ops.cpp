// Probe the qwen_decode stack's non-matmul ops on Vulkan vs CPU.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static ggml_backend_t cpu, vk;

template <typename F>
bool run(ggml_backend_t backend, F build, std::vector<float>& out) {
    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(4096, false), nullptr, true };
    ggml_context* ctx = ggml_init(ip);
    ggml_tensor* t = build(ctx);
    if (!t) { printf("  build failed\n"); ggml_free(ctx); return false; }
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(gf, t);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(ga, gf)) { printf("  alloc failed\n"); ggml_gallocr_free(ga); ggml_free(ctx); return false; }
    // Set all f32/bf16 leaf inputs to a deterministic pattern.
    for (int i = 0; i < gf->n_leafs; ++i) {
        ggml_tensor* l = gf->leafs[i];
        if (l->type == GGML_TYPE_F32) {
            std::vector<float> v(ggml_nelements(l));
            for (size_t j = 0; j < v.size(); ++j) v[j] = ((j * 37) % 199) / 199.0f - 0.5f;
            ggml_backend_tensor_set(l, v.data(), 0, v.size() * 4);
        } else if (l->type == GGML_TYPE_I64 || l->type == GGML_TYPE_I32) {
            int64_t n2 = ggml_nelements(l);
            std::vector<int64_t> v(n2);
            for (int64_t j = 0; j < n2; ++j) v[j] = (j * 5) % 13;  // valid row ids
            ggml_backend_tensor_set(l, v.data(), 0, n2 * ggml_type_size(l->type));
        } else if (l->type == GGML_TYPE_BF16) {
            std::vector<ggml_bf16_t> v(ggml_nelements(l));
            for (size_t j = 0; j < v.size(); ++j) v[j] = ggml_fp32_to_bf16(((j * 37) % 199) / 199.0f - 0.5f);
            ggml_backend_tensor_set(l, v.data(), 0, v.size() * 2);
        }
    }
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { printf("  compute failed\n"); ggml_gallocr_free(ga); ggml_free(ctx); return false; }
    // Read result as f32 (cast bf16 outputs).
    size_t n = ggml_nelements(t);
    out.resize(n);
    if (t->type == GGML_TYPE_F32) ggml_backend_tensor_get(t, out.data(), 0, n * 4);
    else {
        std::vector<ggml_bf16_t> b(n);
        ggml_backend_tensor_get(t, b.data(), 0, n * 2);
        for (size_t j = 0; j < n; ++j) out[j] = ggml_bf16_to_fp32(b[j]);
    }
    ggml_gallocr_free(ga); ggml_free(ctx);
    return true;
}

static void cmp(const char* name, auto build) {
    std::vector<float> a, b;
    bool ok1 = run(cpu, build, a), ok2 = run(vk, build, b);
    if (!ok1 || !ok2) { printf("%-28s RUN FAILED\n", name); return; }
    double md = 0; size_t arg_a = 0, arg_b = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        md = std::max(md, (double)std::fabs(a[i] - b[i]));
        if (a[i] > a[arg_a]) arg_a = i;
        if (b[i] > b[arg_b]) arg_b = i;
    }
    printf("%-28s maxdiff=%-12.6g argmax %s(%zu vs %zu) n=%zu\n", name, md,
           arg_a == arg_b ? "OK" : "DIFF", arg_a, arg_b, a.size());
}

int main() {
    cpu = ggml_backend_cpu_init();
    ggml_backend_dev_t vkdev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto ty = ggml_backend_dev_type(ggml_backend_dev_get(i));
        if (ty == GGML_BACKEND_DEVICE_TYPE_GPU || ty == GGML_BACKEND_DEVICE_TYPE_IGPU) { vkdev = ggml_backend_dev_get(i); break; }
    }
    vk = ggml_backend_dev_init(vkdev, nullptr);
    printf("device: %s\n", ggml_backend_dev_name(vkdev));

    const int D = 128, H = 4, HD = 32, T = 7, P = 13;

    cmp("rms_norm(f32)", [](ggml_context* c) {
        ggml_tensor* x = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, T);
        return ggml_rms_norm(c, x, 1e-6f); });

    cmp("rms_norm->mul(w_bf16->f32)", [](ggml_context* c) {
        ggml_tensor* x = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, T);
        ggml_tensor* w = ggml_new_tensor_1d(c, GGML_TYPE_BF16, D);
        return ggml_mul(c, ggml_rms_norm(c, x, 1e-6f), ggml_cast(c, w, GGML_TYPE_F32)); });

    cmp("add(f32,f32)", [](ggml_context* c) {
        ggml_tensor* a = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, T);
        ggml_tensor* b = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, T);
        return ggml_add(c, a, b); });

    cmp("soft_max(f32) ne0=64", [](ggml_context* c) {
        ggml_tensor* x = ggml_new_tensor_2d(c, GGML_TYPE_F32, 64, 8);
        return ggml_soft_max(c, x); });
    cmp("soft_max(f32) ne0=640", [](ggml_context* c) {
        ggml_tensor* x = ggml_new_tensor_2d(c, GGML_TYPE_F32, 640, 8);
        return ggml_soft_max(c, x); });

    cmp("rope(f32 q)", [](ggml_context* c) {
        ggml_tensor* q = ggml_new_tensor_3d(c, GGML_TYPE_F32, HD, H, T);
        ggml_tensor* pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, T);
        return ggml_rope_ext(c, q, pos, nullptr, HD, 2, 0, GGML_ROPE_TYPE_NORMAL, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f); });

    cmp("argmax(f32)", [](ggml_context* c) {
        ggml_tensor* x = ggml_new_tensor_1d(c, GGML_TYPE_F32, 1000);
        return ggml_argmax(c, x); });

    // mul_mat with bf16 weight and bf16 x across widths: GEMV (T=1, patch
    // 0009), narrow (T=7), and WIDE (T=64/512, the prefill path).
    auto mm = [](int T2) {
        return [T2](ggml_context* c) {
            ggml_tensor* W = ggml_new_tensor_2d(c, GGML_TYPE_BF16, D, 64);
            ggml_tensor* x = ggml_new_tensor_2d(c, GGML_TYPE_BF16, D, T2);
            return ggml_mul_mat(c, W, x);
        };
    };
    cmp("mulmat(bf16,bf16) T=1",  mm(1));
    cmp("mulmat(bf16,bf16) T=7",  mm(7));
    cmp("mulmat(bf16,bf16) T=64", mm(64));
    cmp("mulmat(bf16,bf16) T=512", mm(512));

    // KV-style: attention scores q f32 x K-cache bf16 then softmax then V
    cmp("attn scores(qf32,Kbf16)", [](ggml_context* c) {
        ggml_tensor* K = ggml_new_tensor_2d(c, GGML_TYPE_BF16, HD, P);
        ggml_tensor* q = ggml_new_tensor_2d(c, GGML_TYPE_BF16, HD, 1);
        return ggml_soft_max(c, ggml_scale(c, ggml_mul_mat(c, K, q), 1.0f / std::sqrt((float)HD))); });

    // K-step pattern: 3-D batched GEMV against a KV-cache-shaped tensor.
    cmp("kcache gemv 3D [HD,P,H]", [](ggml_context* c) {
        ggml_tensor* K = ggml_new_tensor_3d(c, GGML_TYPE_BF16, HD, P, H);
        ggml_tensor* q = ggml_new_tensor_3d(c, GGML_TYPE_BF16, HD, 1, H);
        return ggml_mul_mat(c, K, q); });
    // Masked soft_max_ext (the engines' attention softmax).
    cmp("soft_max_ext masked", [](ggml_context* c) {
        ggml_tensor* x = ggml_new_tensor_2d(c, GGML_TYPE_F32, P, H);
        ggml_tensor* m = ggml_new_tensor_2d(c, GGML_TYPE_F32, P, H);
        return ggml_soft_max_ext(c, x, m, 1.0f / std::sqrt((float)HD), 0.0f); });
    // Cache write-back: view of a slice then cpy q/k rows in.
    cmp("cache view+cpy write", [](ggml_context* c) {
        ggml_tensor* cache = ggml_new_tensor_3d(c, GGML_TYPE_BF16, HD, P, H);
        ggml_tensor* row  = ggml_new_tensor_3d(c, GGML_TYPE_BF16, HD, 1, H);
        ggml_tensor* dst  = ggml_view_3d(c, cache, HD, 1, H, cache->nb[1], cache->nb[2], 5 * cache->nb[1]);
        return ggml_cpy(c, row, dst); });
    // set_rows: the K/V cache write-back op.
    cmp("set_rows into bf16 cache", [](ggml_context* c) {
        ggml_tensor* cache = ggml_new_tensor_3d(c, GGML_TYPE_BF16, HD, P, H);
        ggml_tensor* x    = ggml_new_tensor_3d(c, GGML_TYPE_F32, HD, 2, H);
        ggml_tensor* ids  = ggml_new_tensor_1d(c, GGML_TYPE_I64, 2);
        return ggml_set_rows(c, cache, x, ids); });

    return 0;
}
