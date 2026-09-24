// fast_weights_test.cpp — the fast engine's weight repacking and CPU GEMV,
// checked against ggml's own dequantizers (no GPU needed).
//
// For every GGUF type the fast engine maps natively (Q4_0, Q4_1, Q4_K, Q8_0,
// Q6_K, F16) plus one fallback type (Q5_K -> requantized W8), random rows are
// quantized with ggml, repacked into the GPU layouts, dequantized again from
// those layouts on the host, and compared with ggml's to_float. The CPU int8
// GEMV is checked against an f64 reference.

#include "ggml.h"

#if defined(STARLING_HAVE_FAST)
#include "cpu_kernels.hpp"
#include "weights.hpp"
#endif

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#if !defined(STARLING_HAVE_FAST)
int main() {
    std::printf("fast_weights_test: skipped (built without STARLING_FAST)\n");
    return 0;
}
#else

using namespace starling::fast;

namespace {

float h2f(uint32_t h) {
    ggml_fp16_t v = (ggml_fp16_t)h;
    return ggml_fp16_to_fp32(v);
}

// Host-side dequantization of the GPU layouts (mirrors dequant_row.glsl).
float deq(const HostMatrix& m, uint32_t n, uint32_t k) {
    const uint32_t G = m.K / 32, g = n * G + k / 32, j = k % 32;
    switch (m.fmt) {
    case GpuFmt::W4: {
        const float s = h2f(m.s[g] & 0xffff), o = h2f(m.s[g] >> 16);
        return s * (float)((m.q[g * 4 + j / 8] >> (4 * (j % 8))) & 15) + o;
    }
    case GpuFmt::W8: {
        const float s = h2f(j < 16 ? m.s[g] & 0xffff : m.s[g] >> 16);
        return s * (float)(int8_t)(m.q[g * 8 + j / 4] >> (8 * (j % 4)));
    }
    case GpuFmt::F16: {
        const size_t e = (size_t)n * m.K + k;
        return h2f((m.q[e / 2] >> (16 * (e & 1))) & 0xffff);
    }
    }
    return 0.0f;
}

int check_type(ggml_type type, GpuFmt want_fmt, float rel_tol) {
    const uint32_t N = 8, K = 512;
    std::mt19937 rng(42 + (int)type);
    std::normal_distribution<float> nd(0.0f, 0.05f);
    std::vector<float> src((size_t)N * K);
    for (float& v : src) v = nd(rng);
    const size_t rb = ggml_row_size(type, K);
    std::vector<uint8_t> q(rb * N);
    ggml_quantize_chunk(type, src.data(), q.data(), 0, N, K, nullptr);
    std::vector<float> ref((size_t)N * K);
    ggml_get_type_traits(type)->to_float(q.data(), ref.data(), (int64_t)N * K);

    HostMatrix m;
    std::string err;
    if (!pack_gpu_matrix_raw(type, q.data(), N, K, m, err)) {
        std::printf("FAIL %s: pack: %s\n", ggml_type_name(type), err.c_str());
        return 1;
    }
    if (m.fmt != want_fmt) {
        std::printf("FAIL %s: format %s, want %s\n", ggml_type_name(type), fmt_name(m.fmt), fmt_name(want_fmt));
        return 1;
    }
    double max_err = 0.0, max_ref = 0.0;
    for (uint32_t n = 0; n < N; ++n)
        for (uint32_t k = 0; k < K; ++k) {
            max_err = std::max(max_err, (double)std::fabs(deq(m, n, k) - ref[(size_t)n * K + k]));
            max_ref = std::max(max_ref, (double)std::fabs(ref[(size_t)n * K + k]));
        }
    const bool ok = max_err <= rel_tol * max_ref;
    std::printf("%s %-5s -> %-3s max|err| %.3g (%.3g of max|w|)\n", ok ? "ok  " : "FAIL",
                ggml_type_name(type), fmt_name(m.fmt), max_err, max_err / max_ref);
    return ok ? 0 : 1;
}

int check_gemv() {
    const uint32_t N = 37, K = 640;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 0.1f);
    std::vector<float> w((size_t)N * K), x(K), b(N);
    for (float& v : w) v = nd(rng);
    for (float& v : x) v = nd(rng) * 10.0f;
    for (float& v : b) v = nd(rng);
    const size_t rb = ggml_row_size(GGML_TYPE_Q8_0, K);
    std::vector<uint8_t> q(rb * N);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, w.data(), q.data(), 0, N, K, nullptr);
    std::vector<float> wd((size_t)N * K);
    ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(q.data(), wd.data(), (int64_t)N * K);

    // Wrap the quantized rows in a ggml tensor view for pack_cpu_q8.
    ggml_init_params ip{ggml_tensor_overhead() * 4, nullptr, true};
    ggml_context* ctx = ggml_init(ip);
    ggml_tensor* t = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, N);
    t->data = q.data();
    CpuQ8 m;
    std::string err;
    if (!pack_cpu_q8(t, m, err)) { std::printf("FAIL gemv pack: %s\n", err.c_str()); return 1; }
    ggml_free(ctx);
    cpu::QVec xq;
    cpu::quantize(x.data(), K, xq);
    std::vector<float> y(N);
    cpu::gemv(m, xq, b.data(), y.data());
    double max_rel = 0.0;
    for (uint32_t n = 0; n < N; ++n) {
        double r = b[n], mag = 0.0;
        for (uint32_t k = 0; k < K; ++k) {
            r += (double)wd[(size_t)n * K + k] * x[k];
            mag += std::fabs((double)wd[(size_t)n * K + k] * x[k]);
        }
        max_rel = std::max(max_rel, std::fabs(y[n] - r) / (mag + 1e-9));
    }
    // int8 activations: ~1/254 relative error per element, far less in sum.
    const bool ok = max_rel < 5e-3;
    std::printf("%s cpu gemv (%s) max relative error %.3g\n", ok ? "ok  " : "FAIL", cpu::isa_name(), max_rel);
    return ok ? 0 : 1;
}

} // namespace

int main() {
    int fails = 0;
    // Q4_0 / Q4_1 / Q8_0 / F16 are exact; the k-quants round their scale
    // products to f16 once (<= 2^-11 relative); Q5_K takes the W8 fallback.
    fails += check_type(GGML_TYPE_Q4_0, GpuFmt::W4, 1e-6f);
    fails += check_type(GGML_TYPE_Q4_1, GpuFmt::W4, 2e-3f);
    fails += check_type(GGML_TYPE_Q8_0, GpuFmt::W8, 1e-6f);
    fails += check_type(GGML_TYPE_F16, GpuFmt::F16, 1e-6f);
    fails += check_type(GGML_TYPE_Q4_K, GpuFmt::W4, 2e-3f);
    fails += check_type(GGML_TYPE_Q6_K, GpuFmt::W8, 2e-3f);
    fails += check_type(GGML_TYPE_Q5_K, GpuFmt::W8, 1e-2f);
    fails += check_gemv();
    std::printf(fails ? "fast_weights_test: %d FAILED\n" : "fast_weights_test: all passed\n", fails);
    return fails ? 1 : 0;
}

#endif
