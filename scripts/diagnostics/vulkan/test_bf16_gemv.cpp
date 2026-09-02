// Minimal repro: mul_mat(BF16 W [N,K], BF16 x [K,1]) -> F32 [N,1].
// Runs the same graph on the first Vulkan device and on CPU, compares.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    const int N = 64, K = 128;
    // Host data: deterministic floats in [-1, 1].
    std::vector<float> wh((size_t)N * K), xh(K);
    for (int i = 0; i < N * K; ++i) wh[i] = ((i * 37) % 255) / 255.0f - 0.5f;
    for (int i = 0; i < K; ++i) xh[i] = ((i * 91) % 255) / 255.0f - 0.5f;

    // bf16 bit patterns.
    std::vector<ggml_bf16_t> wb((size_t)N * K), xb(K);
    for (size_t i = 0; i < wb.size(); ++i) wb[i] = ggml_fp32_to_bf16(wh[i]);
    for (int i = 0; i < K; ++i) xb[i] = ggml_fp32_to_bf16(xh[i]);

    auto run = [&](ggml_backend_t backend, std::vector<float>& out, ggml_type xt) -> bool {
        ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
        ggml_context* ctx = ggml_init(ip);
        ggml_tensor* W = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, K, N);
        ggml_tensor* x = ggml_new_tensor_2d(ctx, xt, K, 1);
        ggml_tensor* mm = ggml_mul_mat(ctx, W, x);
        ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, mm);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(ga, gf)) { ggml_gallocr_free(ga); ggml_free(ctx); return false; }
        ggml_backend_tensor_set(W, wb.data(), 0, ggml_nbytes(W));
        if (xt == GGML_TYPE_F32) ggml_backend_tensor_set(x, xh.data(), 0, ggml_nbytes(x));
        else ggml_backend_tensor_set(x, xb.data(), 0, ggml_nbytes(x));
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { ggml_gallocr_free(ga); ggml_free(ctx); return false; }
        out.resize(N);
        ggml_backend_tensor_get(mm, out.data(), 0, N * sizeof(float));
        ggml_gallocr_free(ga);
        ggml_free(ctx);
        return true;
    };

    std::vector<float> ref, got;
    if (!run(ggml_backend_cpu_init(), ref, GGML_TYPE_BF16)) { printf("CPU run failed\n"); return 1; }

    // Pick the first Vulkan device.
    ggml_backend_dev_t vkdev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU ||
            ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_IGPU) { vkdev = d; break; }
    }
    if (!vkdev) { printf("no vulkan device\n"); return 1; }
    printf("device: %s\n", ggml_backend_dev_name(vkdev));
    ggml_backend_t vk = ggml_backend_dev_init(vkdev, nullptr);
    if (!run(vk, got, GGML_TYPE_BF16)) { printf("VULKAN run failed\n"); return 1; }
    std::vector<float> got_f32y;
    if (!run(vk, got_f32y, GGML_TYPE_F32)) { printf("VULKAN f32y run failed\n"); return 1; }
    { double md = 0; for (int i = 0; i < N; ++i) md = std::max(md, (double)std::fabs(ref[i] - got_f32y[i]));
      printf("CONTROL f32-y: maxdiff=%.6g\n", md); }

    double maxdiff = 0; int bad = 0;
    for (int i = 0; i < N; ++i) {
        double d = fabs((double)ref[i] - (double)got[i]);
        if (d > maxdiff) maxdiff = d;
        if (d > 1e-2) ++bad;
    }
    printf("maxdiff=%.6g bad(>1e-2)=%d/%d\n", maxdiff, bad, N);
    printf("ref[0..3]=%g %g %g %g\n", ref[0], ref[1], ref[2], ref[3]);
    printf("got[0..3]=%g %g %g %g\n", got[0], got[1], got[2], got[3]);
    // Pure CPY control: bf16 -> f32 through the backend.
    {
        ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
        ggml_context* ctx = ggml_init(ip);
        ggml_tensor* src = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, K, 1);
        ggml_tensor* dstt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 1);
        ggml_tensor* cp = ggml_cpy(ctx, src, dstt);
        ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, cp);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(vk));
        ggml_gallocr_alloc_graph(ga, gf);
        ggml_backend_tensor_set(src, xb.data(), 0, ggml_nbytes(src));
        ggml_backend_graph_compute(vk, gf);
        std::vector<float> cpyout(K);
        ggml_backend_tensor_get(cp, cpyout.data(), 0, K * sizeof(float));
        printf("CPY out[0..3]=%g %g %g %g (expect ~%g %g %g %g)\n", cpyout[0], cpyout[1], cpyout[2], cpyout[3],
               ggml_bf16_to_fp32(xb[0]), ggml_bf16_to_fp32(xb[1]), ggml_bf16_to_fp32(xb[2]), ggml_bf16_to_fp32(xb[3]));
        ggml_gallocr_free(ga); ggml_free(ctx);
    }
    // CAST probes: f32->bf16 and bf16->f32 through the backend.
    {
        auto cast_test = [&](ggml_type from, ggml_type to, std::vector<float>& srcv) {
            ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
            ggml_context* ctx = ggml_init(ip);
            ggml_tensor* src = ggml_new_tensor_2d(ctx, from, K, 1);
            ggml_tensor* cst = ggml_cast(ctx, src, to);
            ggml_cgraph* gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, cst);
            ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(vk));
            ggml_gallocr_alloc_graph(ga, gf);
            std::vector<uint8_t> raw(srcv.size() * (from == GGML_TYPE_F32 ? 4 : 2));
            if (from == GGML_TYPE_F32) memcpy(raw.data(), srcv.data(), raw.size());
            else for (int i = 0; i < K; ++i) { ggml_bf16_t b = ggml_fp32_to_bf16(srcv[i]); memcpy(raw.data()+2*i, &b, 2); }
            ggml_backend_tensor_set(src, raw.data(), 0, raw.size());
            ggml_backend_graph_compute(vk, gf);
            std::vector<float> outv(K);
            if (to == GGML_TYPE_F32) {
                ggml_backend_tensor_get(cst, outv.data(), 0, K*4);
            } else {
                std::vector<ggml_bf16_t> bf(K);
                ggml_backend_tensor_get(cst, bf.data(), 0, K*2);
                for (int i = 0; i < K; ++i) outv[i] = ggml_bf16_to_fp32(bf[i]);
            }
            double md = 0; for (int i = 0; i < K; ++i) {
                float ref = (float)ggml_bf16_to_fp32(ggml_fp32_to_bf16(srcv[i]));
                md = std::max(md, (double)std::fabs(ref - outv[i]));
            }
            printf("CAST %s->%s maxdiff=%.6g  out[0..2]=%g %g %g (in %g %g %g)\n",
                   ggml_type_name(from), ggml_type_name(to), md, outv[0], outv[1], outv[2], srcv[0], srcv[1], srcv[2]);
            ggml_gallocr_free(ga); ggml_free(ctx);
        };
        std::vector<float> sv(K);
        for (int i = 0; i < K; ++i) sv[i] = ((i * 37) % 255) / 255.0f - 0.5f;
        cast_test(GGML_TYPE_F32, GGML_TYPE_BF16, sv);
        cast_test(GGML_TYPE_BF16, GGML_TYPE_F32, sv);
    }
    return maxdiff > 1e-2;
}
