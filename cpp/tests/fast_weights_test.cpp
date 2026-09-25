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
#include "layout.hpp"
#include "weights.hpp"
#endif

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <chrono>
#include <thread>
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

// GemvHelper splits big products across a worker that spins, parks when idle
// and is woken by the next job or by hold(true). Every handoff path must give
// the single-threaded result bit for bit.
int check_gemv_helper() {
    CpuQ8 m;
    m.N = 512;
    m.K = 4096;   // 2M MACs: above the split threshold
    std::mt19937 rng(11);
    m.q.resize((size_t)m.N * m.K);
    for (int8_t& v : m.q) v = (int8_t)((int)(rng() % 255) - 127);
    m.s.resize((size_t)m.N * (m.K / 32));
    for (float& v : m.s) v = (float)(rng() % 1000) * 1e-5f;
    std::vector<float> x(m.K), b(m.N, 0.25f), ref(m.N), y(m.N);
    for (float& v : x) v = (float)((int)(rng() % 2001) - 1000) * 1e-3f;
    cpu::QVec xq;
    cpu::quantize(x.data(), m.K, xq);
    cpu::gemv(m, xq, b.data(), ref.data());
    auto idle = [] { std::this_thread::sleep_for(std::chrono::milliseconds(20)); };
    int bad = 0;
    cpu::GemvHelper h;
    const char* steps[] = {"first", "spinning", "after park", "held", "held after idle", "released+parked"};
    for (int i = 0; i < 6; ++i) {
        if (i == 2 || i == 5) idle();              // worker parks
        if (i == 3) h.hold(true);
        if (i == 4) idle();                        // held: keeps spinning
        if (i == 5) { h.hold(false); idle(); }
        std::fill(y.begin(), y.end(), -1.0f);
        h.run(m, xq, b.data(), y.data());
        if (std::memcmp(y.data(), ref.data(), y.size() * sizeof(float)) != 0) {
            std::printf("FAIL gemv helper: %s differs from single-threaded\n", steps[i]);
            ++bad;
        }
    }
    if (!bad) std::printf("ok   gemv helper: split / park / hold paths bit-exact\n");
    return bad ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Layout descriptors (#319)
// ---------------------------------------------------------------------------

// Quantize + dequant round trip through a descriptor: the reference dequant
// of the packed bytes must (a) be reproducible from the stored scale halves
// through the documented f32 expression and (b) stay a sane approximation.
// Regression (#323 review): f16 subnormal round trips through the REAL
// conversion paths (layout_store_scales -> layout_scale_at uses f2h/h2f_),
// plus the formula directly. ggml is the reference.
int check_f16_subnormals() {
    {   // end-to-end: a tiny scale produces an f16-subnormal stored half;
        // reading it back must equal ggml's round trip.
        LayoutDesc d;
        std::string err;
        if (!layout_from_string("w4g32sym", &d, &err)) return 1;
        const float tiny = 1.0e-7f;                 // f16-subnormal when packed
        const float s[1] = {tiny}, o[1] = {0.0f};
        std::vector<uint8_t> sb(layout_scale_bytes(d, 32));
        layout_store_scales(d, s, o, 32, sb.data());
        const float back = layout_scale_at(d, sb.data(), nullptr, nullptr, 0);
        const float ref = ggml_fp16_to_fp32(ggml_fp32_to_fp16(tiny));
        if (back != ref) {
            std::printf("FAIL f16 subnormal scale round trip: %g vs ggml %g\n", back, ref);
            return 1;
        }
    }
    for (uint32_t m = 1; m < 1024; m += 97) {
        for (uint16_t h16 : {(uint16_t)m, (uint16_t)(0x8000u | m)}) {
            const float ref = ggml_fp16_to_fp32(h16);
            const uint32_t sign = (uint32_t)(h16 & 0x8000) << 16;
            int e = -1;
            uint32_t mm = h16 & 0x3ff;
            while (!(mm & 0x400)) { mm <<= 1; --e; }
            const uint32_t bits = sign | ((uint32_t)(e + 114) << 23) | ((mm & 0x3ff) << 13);
            float got;
            std::memcpy(&got, &bits, 4);
            if (got != ref) {
                std::printf("FAIL f16 subnormal %04x: %g vs ggml %g\n", h16, got, ref);
                return 1;
            }
        }
    }
    std::printf("ok   f16 subnormal conversion == ggml (e+114 fix regression)\n");
    return 0;
}

int check_layout_roundtrip(const starling::fast::LayoutDesc& d) {
    using namespace starling::fast;
    const uint32_t K = 256, N = 4;
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 0.05f);
    std::vector<float> w((size_t)N * K);
    for (float& v : w) v = nd(rng);
    std::vector<float> im(K);
    for (float& v : im) v = 0.5f + (float)(rng() % 1000) / 1000.0f;

    const uint64_t cb = layout_code_bytes(d, K), sb = layout_scale_bytes(d, K);
    const bool sup = d.scale_dtype == ScaleDtype::U8Super;
    double max_rel = 0, max_rel_im = 0;
    for (uint32_t r = 0; r < N; ++r) {
        std::vector<uint8_t> codes(cb), scales(sb);
        std::vector<uint16_t> super(sup ? 1 : 0);
        layout_quant_row(d, w.data() + (size_t)r * K, K, nullptr, codes.data(), scales.data(),
                         super.data(), sup ? scales.data() : nullptr);
        // (a) elementwise reference: recompute from the raw halves exactly as
        // the shader would and compare bit for bit.
        for (uint32_t k = 0; k < K; ++k) {
            const float got = layout_dequant(d, codes.data(), scales.data(), super.data(),
                                             sup ? scales.data() : nullptr, k);
            const uint32_t g = k / d.group;
            const int16_t q = layout_decode_code(d, codes.data(), k);
            float s;
            if (sup) {
                // super * u8 in exactly this order (see layout.hpp)
                uint16_t h = super[0];
                uint32_t bits = ((uint32_t)(h & 0x8000) << 16) |
                               (((uint32_t)((h >> 10) & 31) + 112) << 23) | ((uint32_t)(h & 0x3ff) << 13);
                float sf;
                std::memcpy(&sf, &bits, 4);
                s = sf * (float)scales[g];
            } else {
                const bool pair = !d.symmetric || d.store_pair;
                const size_t sbyte = pair ? (size_t)g * 4 : (size_t)g * 2;
                const uint16_t h = (uint16_t)(scales[sbyte] | (scales[sbyte + 1] << 8));
                const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
                const uint32_t e = (h >> 10) & 31, m = h & 0x3ff;
                uint32_t bits = e ? (sign | ((e + 112) << 23) | (m << 13)) : sign;
                float sf;
                std::memcpy(&sf, &bits, 4);
                s = sf;
            }
            float want = d.symmetric ? (d.bits == 4 ? s * ((float)q - 8.0f) : s * (float)q) : 0.0f;
            if (!d.symmetric || d.store_pair) {
                // offset from the second half of the group's word (asym rows,
                // and sym rows stored in the legacy pair, where it is -8s)
                const size_t i = ((size_t)g * 2 + 1) * 2;
                const uint16_t ho = (uint16_t)(scales[i] | (scales[i + 1] << 8));
                const uint32_t sign = (uint32_t)(ho & 0x8000) << 16;
                const uint32_t e = (ho >> 10) & 31, m = ho & 0x3ff;
                uint32_t bits = e ? (sign | ((e + 112) << 23) | (m << 13)) : sign;
                float of;
                std::memcpy(&of, &bits, 4);
                want = s * (float)q + of;
            }
            if (std::memcmp(&got, &want, 4) != 0) {
                std::printf("FAIL %s: elementwise dequant differs at r=%u k=%u (%g vs %g)\n",
                            layout_to_string(d).c_str(), r, k, got, want);
                return 1;
            }
        }
        // (b) approximation sanity
        double e2 = 0, w2 = 0;
        for (uint32_t k = 0; k < K; ++k) {
            const float dq = layout_dequant(d, codes.data(), scales.data(), super.data(),
                                            sup ? scales.data() : nullptr, k);
            e2 += (double)(w[(size_t)r * K + k] - dq) * (w[(size_t)r * K + k] - dq);
            w2 += (double)w[(size_t)r * K + k] * w[(size_t)r * K + k];
        }
        max_rel = std::max(max_rel, std::sqrt(e2 / w2));
        // weighted run: its WEIGHTED error must not be worse than the
        // unweighted run's weighted error (the search does its job).
        double we_un = 0, we_im = 0;
        for (int pass = 0; pass < 2; ++pass) {
            std::vector<uint8_t> c2(cb), s2(sb);
            std::vector<uint16_t> sp2(sup ? 1 : 0);
            layout_quant_row(d, w.data() + (size_t)r * K, K, pass ? im.data() : nullptr, c2.data(),
                             s2.data(), sp2.data(), sup ? s2.data() : nullptr);
            double we = 0;
            for (uint32_t k = 0; k < K; ++k) {
                const float dq = layout_dequant(d, c2.data(), s2.data(), sp2.data(),
                                                sup ? s2.data() : nullptr, k);
                const double dd = w[(size_t)r * K + k] - dq;
                we += (double)im[k] * dd * dd;
            }
            if (pass == 0) we_un = we; else we_im = we;
        }
        max_rel_im = std::max(max_rel_im, we_im - we_un);
    }
    const bool ok = max_rel < 0.25 && max_rel_im <= 1e-9;
    std::printf("%s layout %-12s round trip: rel-rms %.4f, weighted search %s\n",
                ok ? "ok  " : "FAIL", layout_to_string(d).c_str(), max_rel,
                max_rel_im <= 1e-9 ? "improves" : "DOES NOT improve");
    return ok ? 0 : 1;
}

// The legacy descriptors must reproduce pack_gpu_matrix's bytes exactly,
// and their reference dequant must equal ggml's dequant of the same block
// bit for bit — that is what lets STARLING_FAST_PACKED load them into the
// existing kernels.
int check_layout_legacy_bytes() {
    using namespace starling::fast;
    const uint32_t N = 6, K = 512;
    std::mt19937 rng(99);
    std::normal_distribution<float> nd(0.0f, 0.05f);
    std::vector<float> src((size_t)N * K);
    for (float& v : src) v = nd(rng);
    int fails = 0;
    for (int ty = 0; ty < 2; ++ty) {
        const ggml_type type = ty == 0 ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0;
        const size_t rb = ggml_row_size(type, K);
        std::vector<uint8_t> q(rb * N);
        ggml_quantize_chunk(type, src.data(), q.data(), 0, N, K, nullptr);
        std::vector<float> ref((size_t)N * K);
        ggml_get_type_traits(type)->to_float(q.data(), ref.data(), (int64_t)N * K);
        HostMatrix m;
        std::string err;
        if (!pack_gpu_matrix_raw(type, q.data(), N, K, m, err)) {
            std::printf("FAIL legacy bytes: pack: %s\n", err.c_str());
            return 1;
        }
        LayoutDesc d;
        const char* spec = ty == 0 ? "w4g32asym" : "w8g16sym";
        if (!layout_from_string(spec, &d, &err)) {
            std::printf("FAIL legacy bytes: %s\n", err.c_str());
            return 1;
        }
        // Re-encode the ggml blocks through the layout API with the blocks'
        // own codes and scales; the bytes must match the engine repack.
        std::vector<uint8_t> codes((size_t)layout_code_bytes(d, K) * N);
        std::vector<uint8_t> scales((size_t)layout_scale_bytes(d, K) * N);
        for (uint32_t r = 0; r < N; ++r) {
            int16_t qc[K];
            float sv[K / 16], ov[K / 16];   // entries per scale group (K/group <= K/16)
            if (ty == 0) {
                for (uint32_t g = 0; g < K / 32; ++g) {
                    const size_t bo = (size_t)r * rb + (size_t)g * 18;
                    const uint16_t dh = (uint16_t)(q[bo] | (q[bo + 1] << 8));
                    uint32_t bits = ((uint32_t)(dh & 0x8000) << 16) |
                                   (((uint32_t)((dh >> 10) & 31) + 112) << 23) | ((uint32_t)(dh & 0x3ff) << 13);
                    float dsc;
                    std::memcpy(&dsc, &bits, 4);
                    sv[g] = dsc;
                    ov[g] = -8.0f * dsc;
                    const uint8_t* qs = q.data() + bo + 2;
                    for (int j = 0; j < 16; ++j) {
                        qc[g * 32 + j] = qs[j] & 15;
                        qc[g * 32 + 16 + j] = qs[j] >> 4;
                    }
                }
            } else {
                for (uint32_t g = 0; g < K / 32; ++g) {
                    const size_t bo = (size_t)r * rb + (size_t)g * 34;
                    const uint16_t dh = (uint16_t)(q[bo] | (q[bo + 1] << 8));
                    uint32_t bits = ((uint32_t)(dh & 0x8000) << 16) |
                                   (((uint32_t)((dh >> 10) & 31) + 112) << 23) | ((uint32_t)(dh & 0x3ff) << 13);
                    float dsc;
                    std::memcpy(&dsc, &bits, 4);
                    const int8_t* qs = (const int8_t*)(q.data() + bo + 2);
                    for (int j = 0; j < 32; ++j) qc[g * 32 + j] = qs[j];
                    sv[2 * g] = sv[2 * g + 1] = dsc;   // scale per 16, duplicated
                    ov[2 * g] = ov[2 * g + 1] = 0.0f;
                }
            }
            layout_encode_row(d, qc, K, codes.data() + (size_t)r * layout_code_bytes(d, K));
            layout_store_scales(d, sv, ov, K, scales.data() + (size_t)r * layout_scale_bytes(d, K));
        }
        if (codes.size() != m.q.size() * 4 || scales.size() != m.s.size() * 4 ||
            std::memcmp(codes.data(), m.q.data(), codes.size()) != 0 ||
            std::memcmp(scales.data(), m.s.data(), scales.size()) != 0) {
            std::printf("FAIL %s: layout bytes differ from pack_gpu_matrix\n", spec);
            ++fails;
            continue;
        }
        // Bit-for-bit reference dequant == ggml dequant (0 and -0 count as
        // equal: ggml's d*(q-8) yields -0 where the engine's s*q+o yields
        // +0; both are the value the kernels produce).
        for (uint32_t r = 0; r < N; ++r) {
            const uint8_t* cs = codes.data() + (size_t)r * layout_code_bytes(d, K);
            const uint8_t* sc = scales.data() + (size_t)r * layout_scale_bytes(d, K);
            for (uint32_t k = 0; k < K; ++k) {
                const float got = layout_dequant(d, cs, sc, nullptr, nullptr, k);
                const float want = ref[(size_t)r * K + k];
                const bool same = std::memcmp(&got, &want, 4) == 0 || got == want;
                if (!same) {
                    std::printf("FAIL %s: dequant differs from ggml at r=%u k=%u (%g vs %g)\n",
                                spec, r, k, got, want);
                    ++fails;
                    r = N;
                    break;
                }
            }
        }
        if (!fails) std::printf("ok   layout %-9s bytes == pack_gpu_matrix, dequant == ggml bit for bit\n", spec);
    }
    return fails ? 1 : 0;
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
    fails += check_gemv_helper();
    // Layout descriptors (#319): every descriptor round-trips through its
    // reference dequant bit for bit, and the two legacy specs are byte- and
    // bit-identical to the GPU layouts the engines already run.
    fails += check_layout_legacy_bytes();
    fails += check_f16_subnormals();
    {
        using namespace starling::fast;
        const char* specs[] = {"w4g32asym", "w4g32sym",  "w4g64sym",  "w4g128sym",
                               "w4g128symu8s", "w8g16sym",  "w8g32sym",
                               "w4g64sym-p1", "w4g32sym-a", "w4g64sym-a"};
        for (const char* s : specs) {
            LayoutDesc d;
            std::string err;
            if (!layout_from_string(s, &d, &err)) {
                std::printf("FAIL parse %s: %s\n", s, err.c_str());
                ++fails;
                continue;
            }
            fails += check_layout_roundtrip(d);
        }
        // Round-trip of the spec string itself.
        const char* round[] = {"w4g32asym", "w8g16sym", "w4g64sym", "w4g128symu8s", "w4g32sym-p1",
                               "w4g32sym-a"};
        for (const char* s : round) {
            LayoutDesc d;
            std::string err;
            if (!layout_from_string(s, &d, &err) || layout_to_string(d) != s) {
                std::printf("FAIL spec round trip: %s -> %s\n", s, layout_to_string(d).c_str());
                ++fails;
            }
        }
    }
    std::printf(fails ? "fast_weights_test: %d FAILED\n" : "fast_weights_test: all passed\n", fails);
    return fails ? 1 : 0;
}

#endif
