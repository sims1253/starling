// 3-D batched mul_mat (GQA broadcast) + 3-D soft_max_ext probes — the exact
// shapes the qwen_decode whole-model attention uses at prefill (S>1), which
// test_ops' 2-D / GEMV probes do not cover. Found the Vulkan divergence:
// bf16 [D,K,KV] x bf16 [D,S,H] with H % KV == 0 produces wrong values, while
// the same math per-head (2-D) is exact and CPU is correct.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"

#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
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
    // Deterministic injective fill for every float-ish leaf.
    for (int i = 0; i < gf->n_leafs; ++i) {
        ggml_tensor* l = gf->leafs[i];
        int64_t n = ggml_nelements(l);
        if (l->type == GGML_TYPE_F32) {
            std::vector<float> v(n);
            for (int64_t j = 0; j < n; ++j) v[j] = (float)((j % 199) + (j / 199) * 0.125) / 199.0f - 0.5f;
            ggml_backend_tensor_set(l, v.data(), 0, n * 4);
        } else if (l->type == GGML_TYPE_BF16) {
            std::vector<ggml_bf16_t> v(n);
            for (int64_t j = 0; j < n; ++j) v[j] = ggml_fp32_to_bf16(((j * 37) % 199) / 199.0f - 0.5f);
            ggml_backend_tensor_set(l, v.data(), 0, n * 2);
        } else if (l->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> v(n);
            for (int64_t j = 0; j < n; ++j) v[j] = ggml_fp32_to_fp16(((j * 37) % 199) / 199.0f - 0.5f);
            ggml_backend_tensor_set(l, v.data(), 0, n * 2);
        }
    }
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { printf("  compute failed\n"); ggml_gallocr_free(ga); ggml_free(ctx); return false; }
    size_t n = ggml_nelements(t);
    if (t->type == GGML_TYPE_F32) {
        out.resize(n);
        ggml_backend_tensor_get(t, out.data(), 0, n * 4);
    } else if (t->type == GGML_TYPE_BF16) {
        out.resize(n);
        std::vector<ggml_bf16_t> b(n);
        ggml_backend_tensor_get(t, b.data(), 0, n * 2);
        for (size_t j = 0; j < n; ++j) out[j] = ggml_bf16_to_fp32(b[j]);
    } else {
        printf("  unhandled output type %s\n", ggml_type_name(t->type));
        return false;
    }
    ggml_gallocr_free(ga);
    ggml_free(ctx);
    return true;
}

static int g_failures = 0;

// Per-batch (head) maxdiff so a broadcast-indexing bug (only some heads wrong)
// is distinguishable from uniform numeric drift. `tol` is dtype-dependent:
// bf16/f32 probes sit at fp-epsilon/pipeline noise (<1e-2), f16-STORED probes
// carry the documented ~4e-2 f16 storage noise (identical with/without
// striding), so they gate at 5e-2 — a mapping bug is O(1) and blows through
// either bound.
static void cmp(const char* name, int n_batch, double tol, auto build) {
    std::vector<float> a, b;
    bool ok1 = run(cpu, build, a), ok2 = run(vk, build, b);
    if (!ok1 || !ok2) { printf("%-40s RUN FAILED\n", name); ++g_failures; return; }
    double md = 0;
    int bad_batches = 0;
    size_t per = a.size() / std::max(1, n_batch);
    std::string per_b;
    for (int bi = 0; bi < n_batch && bi < 16; ++bi) {
        double bm = 0;
        for (size_t j = bi * per; j < (bi + 1) * per && j < a.size(); ++j)
            bm = std::max(bm, (double)std::fabs(a[j] - b[j]));
        if (bm > tol) ++bad_batches;
        if (bi < 8) per_b += " " + std::to_string((int)bm);
    }
    for (size_t i = 0; i < a.size(); ++i) md = std::max(md, (double)std::fabs(a[i] - b[i]));
    bool ok = md <= tol;
    if (!ok) ++g_failures;
    printf("%-40s maxdiff=%-10.4g tol=%-7.3g badbatches=%d/%d  per-batch:%s%s\n", name, md, tol,
           bad_batches, n_batch, per_b.c_str(), ok ? "" : "  <-- FAIL");
}

int main() {
    cpu = ggml_backend_cpu_init();
    ggml_backend_dev_t vkdev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (std::string(ggml_backend_dev_name(d)) == "Vulkan0") { vkdev = d; break; }
    }
    if (!vkdev) { printf("no Vulkan0 device\n"); return 1; }
    vk = ggml_backend_dev_init(vkdev, nullptr);
    printf("device: %s\n", ggml_backend_dev_name(vkdev));

    // ---- K^T Q: [D,K,KV] x [D,S,H] -> [K,S,H] (prefill attention scores) ----
    // moss dims: D=128 K=107 S=107 KV=8 H=16 (r2=2).
    cmp("kq bf16 D128 K107 S107 KV8 H16", 16, 2e-2, [](ggml_context* c) {
        ggml_tensor* k = ggml_new_tensor_3d(c, GGML_TYPE_BF16, 128, 107, 8);
        ggml_tensor* q = ggml_new_tensor_3d(c, GGML_TYPE_BF16, 128, 107, 16);
        return ggml_cast(c, ggml_mul_mat(c, k, q), GGML_TYPE_F32); });

    cmp("kq f32  D128 K107 S107 KV8 H16", 16, 2e-2, [](ggml_context* c) {
        ggml_tensor* k = ggml_new_tensor_3d(c, GGML_TYPE_F32, 128, 107, 8);
        ggml_tensor* q = ggml_new_tensor_3d(c, GGML_TYPE_F32, 128, 107, 16);
        return ggml_mul_mat(c, k, q); });

    cmp("kq f16  D128 K107 S107 KV8 H16", 16, 5e-2, [](ggml_context* c) {
        ggml_tensor* k = ggml_new_tensor_3d(c, GGML_TYPE_F16, 128, 107, 8);
        ggml_tensor* q = ggml_new_tensor_3d(c, GGML_TYPE_F16, 128, 107, 16);
        return ggml_cast(c, ggml_mul_mat(c, k, q), GGML_TYPE_F32); });

    cmp("kq bf16 r2=1 (KV16 H16)", 16, 2e-2, [](ggml_context* c) {
        ggml_tensor* k = ggml_new_tensor_3d(c, GGML_TYPE_BF16, 128, 107, 16);
        ggml_tensor* q = ggml_new_tensor_3d(c, GGML_TYPE_BF16, 128, 107, 16);
        return ggml_cast(c, ggml_mul_mat(c, k, q), GGML_TYPE_F32); });

    cmp("kq bf16 r2=4 (KV4 H16)", 16, 2e-2, [](ggml_context* c) {
        ggml_tensor* k = ggml_new_tensor_3d(c, GGML_TYPE_BF16, 128, 107, 4);
        ggml_tensor* q = ggml_new_tensor_3d(c, GGML_TYPE_BF16, 128, 107, 16);
        return ggml_cast(c, ggml_mul_mat(c, k, q), GGML_TYPE_F32); });

    cmp("kq bf16 K=S=128 (aligned)", 16, 2e-2, [](ggml_context* c) {
        ggml_tensor* k = ggml_new_tensor_3d(c, GGML_TYPE_BF16, 128, 128, 8);
        ggml_tensor* q = ggml_new_tensor_3d(c, GGML_TYPE_BF16, 128, 128, 16);
        return ggml_cast(c, ggml_mul_mat(c, k, q), GGML_TYPE_F32); });

    cmp("kq bf16 S=64 (n<128)", 16, 2e-2, [](ggml_context* c) {
        ggml_tensor* k = ggml_new_tensor_3d(c, GGML_TYPE_BF16, 128, 107, 8);
        ggml_tensor* q = ggml_new_tensor_3d(c, GGML_TYPE_BF16, 128, 64, 16);
        return ggml_cast(c, ggml_mul_mat(c, k, q), GGML_TYPE_F32); });

    cmp("kq bf16 D=64 K=S=107", 16, 2e-2, [](ggml_context* c) {
        ggml_tensor* k = ggml_new_tensor_3d(c, GGML_TYPE_BF16, 64, 107, 8);
        ggml_tensor* q = ggml_new_tensor_3d(c, GGML_TYPE_BF16, 64, 107, 16);
        return ggml_cast(c, ggml_mul_mat(c, k, q), GGML_TYPE_F32); });

    // ---- V^T probs: cont(permute(vall,1,0,2,3)) [K,D,KV] x [K,S,H] -> [D,S,H]
    cmp("vt bf16 (perm+cont) pr bf16", 16, 2e-2, [](ggml_context* c) {
        ggml_tensor* v = ggml_new_tensor_3d(c, GGML_TYPE_BF16, 128, 107, 8);
        ggml_tensor* vt = ggml_cont(c, ggml_permute(c, v, 1, 0, 2, 3));  // [107,128,8]
        ggml_tensor* pr = ggml_new_tensor_3d(c, GGML_TYPE_BF16, 107, 107, 16);
        return ggml_cast(c, ggml_mul_mat(c, vt, pr), GGML_TYPE_F32); });

    cmp("vt bf16 (plain [K,D,KV]) pr bf16", 16, 2e-2, [](ggml_context* c) {
        ggml_tensor* vt = ggml_new_tensor_3d(c, GGML_TYPE_BF16, 107, 128, 8);
        ggml_tensor* pr = ggml_new_tensor_3d(c, GGML_TYPE_BF16, 107, 107, 16);
        return ggml_cast(c, ggml_mul_mat(c, vt, pr), GGML_TYPE_F32); });

    // ---- 3-D soft_max_ext with 2-D f32 mask (batched attention softmax) ----
    cmp("softmax_ext 3D + 2D mask", 16, 2e-2, [](ggml_context* c) {
        ggml_tensor* sc = ggml_new_tensor_3d(c, GGML_TYPE_F32, 107, 107, 16);
        ggml_tensor* mk = ggml_new_tensor_2d(c, GGML_TYPE_F32, 107, 107);
        return ggml_soft_max_ext(c, sc, mk, 1.0f, 0.0f); });

    // ---- control: same KQ math per-head (2-D), the shape that IS exact ----
    cmp("kq bf16 PER-HEAD view2d (h=3)", 1, 2e-2, [](ggml_context* c) {
        ggml_tensor* k = ggml_new_tensor_3d(c, GGML_TYPE_BF16, 128, 107, 8);
        ggml_tensor* q = ggml_new_tensor_3d(c, GGML_TYPE_BF16, 128, 107, 16);
        ggml_tensor* kh = ggml_view_2d(c, k, 128, 107, k->nb[1], 1 * k->nb[2]);  // kv 1
        ggml_tensor* qh = ggml_view_2d(c, q, 128, 107, q->nb[1], 3 * q->nb[2]);  // head 3 -> kv 1
        return ggml_cast(c, ggml_mul_mat(c, kh, qh), GGML_TYPE_F32); });

    // ---- THE ENGINE SHAPE: src0 is the ggml_cpy result view of the persistent
    // [D, max_cache, KV] cache — dim01-contiguous but batch stride nb[2] =
    // D*max_cache != D*S. Host passes stride_batch_x = ne00*ne01 in
    // ggml_vk_mul_mat_q_f16 -> every kv batch != 0 reads the wrong slots.
    cmp("kq bf16 CACHE-VIEW src0 (kv0=0)", 16, 2e-2, [](ggml_context* c) {
        ggml_tensor* kc = ggml_new_tensor_3d(c, GGML_TYPE_BF16, 128, 512, 8);   // [D, max_cache, KV]
        ggml_tensor* k = ggml_view_3d(c, kc, 128, 107, 8, kc->nb[1], kc->nb[2], 0);
        ggml_tensor* q = ggml_new_tensor_3d(c, GGML_TYPE_BF16, 128, 107, 16);
        return ggml_cast(c, ggml_mul_mat(c, k, q), GGML_TYPE_F32); });

    cmp("kq f32  CACHE-VIEW src0", 16, 2e-2, [](ggml_context* c) {
        ggml_tensor* kc = ggml_new_tensor_3d(c, GGML_TYPE_F32, 128, 512, 8);
        ggml_tensor* k = ggml_view_3d(c, kc, 128, 107, 8, kc->nb[1], kc->nb[2], 0);
        ggml_tensor* q = ggml_new_tensor_3d(c, GGML_TYPE_F32, 128, 107, 16);
        return ggml_mul_mat(c, k, q); });

    cmp("kq f16  CACHE-VIEW src0", 16, 5e-2, [](ggml_context* c) {
        ggml_tensor* kc = ggml_new_tensor_3d(c, GGML_TYPE_F16, 128, 512, 8);
        ggml_tensor* k = ggml_view_3d(c, kc, 128, 107, 8, kc->nb[1], kc->nb[2], 0);
        ggml_tensor* q = ggml_new_tensor_3d(c, GGML_TYPE_F16, 128, 107, 16);
        return ggml_cast(c, ggml_mul_mat(c, k, q), GGML_TYPE_F32); });

    // Batch-strided src1 (q as a view of a taller [D, Sbig, H] buffer).
    cmp("kq bf16 CACHE-VIEW src1", 16, 2e-2, [](ggml_context* c) {
        ggml_tensor* k = ggml_new_tensor_3d(c, GGML_TYPE_BF16, 128, 107, 8);
        ggml_tensor* qc = ggml_new_tensor_3d(c, GGML_TYPE_BF16, 128, 256, 16);
        ggml_tensor* q = ggml_view_3d(c, qc, 128, 107, 16, qc->nb[1], qc->nb[2], 0);
        return ggml_cast(c, ggml_mul_mat(c, k, q), GGML_TYPE_F32); });

    printf("%s (%d failures)\n", g_failures ? "FAILURES FOUND" : "all ok", g_failures);
    return g_failures ? 1 : 0;
}
