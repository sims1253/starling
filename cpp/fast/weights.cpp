// weights.cpp — see weights.hpp.

#include "weights.hpp"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace starling::fast {

namespace {

// ggml block layouts (mirrors ggml-common.h; asserted against ggml's sizes).
struct BlockQ4_0 { uint16_t d; uint8_t qs[16]; };
struct BlockQ4_1 { uint16_t d, m; uint8_t qs[16]; };
struct BlockQ8_0 { uint16_t d; int8_t qs[32]; };
struct BlockQ4_K { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[128]; };
struct BlockQ6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; };
static_assert(sizeof(BlockQ4_0) == 18, "q4_0");
static_assert(sizeof(BlockQ4_1) == 20, "q4_1");
static_assert(sizeof(BlockQ8_0) == 34, "q8_0");
static_assert(sizeof(BlockQ4_K) == 144, "q4_K");
static_assert(sizeof(BlockQ6_K) == 210, "q6_K");

float h2f(uint16_t h) { ggml_fp16_t v; std::memcpy(&v, &h, 2); return ggml_fp16_to_fp32(v); }
uint16_t f2h(float f) { ggml_fp16_t v = ggml_fp32_to_fp16(f); uint16_t h; std::memcpy(&h, &v, 2); return h; }
uint32_t pack2(float a, float b) { return (uint32_t)f2h(a) | ((uint32_t)f2h(b) << 16); }

void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// Eight 4-bit values, one per byte of x (low nibbles), packed sequentially
// into one word: value i at bits 4i.
inline uint32_t pack8_nibbles(uint64_t x) {
    x &= 0x0F0F0F0F0F0F0F0Full;
    x = (x | (x >> 4)) & 0x00FF00FF00FF00FFull;
    x = (x | (x >> 8)) & 0x0000FFFF0000FFFFull;
    return (uint32_t)(x | (x >> 16));
}
inline uint64_t load8(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

// One W4 group from 16 bytes whose low nibbles are values 0-15 and high
// nibbles values 16-31 (Q4_0 / Q4_1).
inline void put_w4_split(uint32_t* q, const uint8_t* qs) {
    const uint64_t a = load8(qs), b = load8(qs + 8);
    q[0] = pack8_nibbles(a);
    q[1] = pack8_nibbles(b);
    q[2] = pack8_nibbles(a >> 4);
    q[3] = pack8_nibbles(b >> 4);
}

// Quantize 32 floats to a W8 group (absmax per 16) — the requant path.
// Values: round-half-away(clamp(x / s, -127, 127)) with an IEEE division,
// bit for bit on every path (NEON: frinta is exactly roundf).
uint32_t quant_w8(uint32_t* q, const float* x) {
    int8_t v[32];
    float sc[2];
    for (int h = 0; h < 2; ++h) {
        const float* xh = x + h * 16;
        int8_t* vh = v + h * 16;
#if defined(__aarch64__)
        float32x4_t a[4];
        for (int j = 0; j < 4; ++j) a[j] = vld1q_f32(xh + 4 * j);
        const float amax = vmaxvq_f32(vmaxq_f32(vmaxq_f32(vabsq_f32(a[0]), vabsq_f32(a[1])),
                                                vmaxq_f32(vabsq_f32(a[2]), vabsq_f32(a[3]))));
#else
        float amax = 0.0f;
        for (int i = 0; i < 16; ++i) amax = std::max(amax, std::fabs(xh[i]));
#endif
        // f16-representable scale so the kernel sees exactly what we used.
        float sv = h2f(f2h(amax / 127.0f));
        if (sv == 0.0f && amax > 0.0f) sv = amax / 127.0f;
        sc[h] = sv;
        if (!(sv > 0)) {
            std::memset(vh, 0, 16);
            continue;
        }
#if defined(__aarch64__)
        const float32x4_t d = vdupq_n_f32(sv), lo = vdupq_n_f32(-127.0f), hi = vdupq_n_f32(127.0f);
        int32x4_t r[4];
        for (int j = 0; j < 4; ++j)
            r[j] = vcvtq_s32_f32(vrndaq_f32(vminq_f32(vmaxq_f32(vdivq_f32(a[j], d), lo), hi)));
        const int16x8_t r01 = vcombine_s16(vmovn_s32(r[0]), vmovn_s32(r[1]));
        const int16x8_t r23 = vcombine_s16(vmovn_s32(r[2]), vmovn_s32(r[3]));
        vst1q_s8(vh, vcombine_s8(vmovn_s16(r01), vmovn_s16(r23)));
#else
        // roundf without a libm call (vectorizes with plain SSE2): truncate,
        // then step away from zero on a fraction of at least one half (the
        // fraction of |v| <= 127 is exact).
        for (int i = 0; i < 16; ++i) {
            const float v = std::clamp(xh[i] / sv, -127.0f, 127.0f);
            const int t = (int)v;
            const float f = v - (float)t;
            vh[i] = (int8_t)(t + (f >= 0.5f) - (f <= -0.5f));
        }
#endif
    }
    std::memcpy(q, v, 32);
    return pack2(sc[0], sc[1]);   // the group-scale word
}

bool row_to_f32(int type, const void* row, float* out, uint32_t K, std::string& err) {
    if (type == GGML_TYPE_F32) { std::memcpy(out, row, (size_t)K * 4); return true; }
    const ggml_type_traits* tr = ggml_get_type_traits((ggml_type)type);
    if (!tr || !tr->to_float) { err = std::string("no dequantizer for type ") + (tr ? tr->type_name : "?"); return false; }
    tr->to_float(row, out, K);
    return true;
}

// W4/W8 group-scale words or F16 weight pairs times f, re-rounded to f16.
void scale_words_impl(uint32_t* w, size_t n, float f) {
    for (size_t i = 0; i < n; ++i)
        w[i] = pack2(h2f((uint16_t)(w[i] & 0xffffu)) * f, h2f((uint16_t)(w[i] >> 16)) * f);
}

} // namespace

const char* fmt_name(GpuFmt f) {
    switch (f) {
    case GpuFmt::W4: return "W4";
    case GpuFmt::W8: return "W8";
    case GpuFmt::F16: return "F16";
    }
    return "?";
}

std::vector<uint32_t> pack_f16(const float* x, size_t n) {
    std::vector<uint32_t> out((n + 1) / 2);
    for (size_t i = 0; i < n; i += 2) out[i / 2] = pack2(x[i], i + 1 < n ? x[i + 1] : 0.0f);
    return out;
}

void scale_f16_words(uint32_t* w, size_t n, float f) { scale_words_impl(w, n, f); }

uint32_t q_row_words(GpuFmt f, uint32_t K) {
    switch (f) {
    case GpuFmt::W4: return K / 8;
    case GpuFmt::W8: return K / 4;
    case GpuFmt::F16: return K / 2;
    }
    return 0;
}

uint32_t s_row_words(GpuFmt f, uint32_t K) { return f == GpuFmt::F16 ? 0 : K / 32; }

bool plan_gpu_rows(int type, uint32_t K, bool force_w8, GpuFmt& fmt, bool& lossless, std::string& err) {
    lossless = true;
    const bool k32 = K % 32 == 0, k256 = K % 256 == 0;
    if (!force_w8) {
        switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1: if (k32) { fmt = GpuFmt::W4; return true; } break;
        case GGML_TYPE_Q4_K: if (k256) { fmt = GpuFmt::W4; lossless = false; return true; } break;
        case GGML_TYPE_F16: fmt = GpuFmt::F16; return true;
        case GGML_TYPE_F32:
        case GGML_TYPE_BF16: fmt = GpuFmt::F16; lossless = false; return true;
        default: break;
        }
    }
    if (type == GGML_TYPE_Q8_0 && k32) { fmt = GpuFmt::W8; return true; }
    if (type == GGML_TYPE_Q6_K && k256) { fmt = GpuFmt::W8; lossless = false; return true; }
    // Requantized to W8 (dequantized through ggml's type traits).
    if (!k32) { err = "fast engine: K must be a multiple of 32"; return false; }
    const ggml_type_traits* tr = ggml_get_type_traits((ggml_type)type);
    if (type != GGML_TYPE_F32 && (!tr || !tr->to_float)) {
        err = std::string("no dequantizer for type ") + (tr && tr->type_name ? tr->type_name : "?");
        return false;
    }
    fmt = GpuFmt::W8;
    lossless = false;
    return true;
}

bool pack_gpu_rows(int type, const void* src, uint32_t n, uint32_t K, GpuFmt fmt, float scale,
                   uint32_t* q, uint32_t* s, const RowMap& map, std::string& err) {
    const size_t rb = ggml_row_size((ggml_type)type, K);
    const uint32_t qw = q_row_words(fmt, K), sw = s_row_words(fmt, K), G = K / 32;
    const uint8_t* base = (const uint8_t*)src;
    std::vector<float> row;   // dequantized row (requant / f32 / bf16 paths)
    // Group-scale words with the multiplier folded in before the one store
    // (mapped device memory may be uncached for CPU reads).
    auto S = [scale](uint32_t w) {
        if (scale != 1.0f) scale_words_impl(&w, 1, scale);
        return w;
    };
    for (uint32_t i = 0; i < n; ++i) {
        const uint8_t* in = base + (size_t)i * rb;
        const uint64_t d = map(i);
        uint32_t* qr = q + d * qw;
        uint32_t* sr = s ? s + d * sw : nullptr;
        if (fmt == GpuFmt::W4 && type == GGML_TYPE_Q4_0) {
            const BlockQ4_0* b = (const BlockQ4_0*)in;
            for (uint32_t g = 0; g < G; ++g) {
                put_w4_split(qr + g * 4, b[g].qs);
                const float dd = h2f(b[g].d);
                sr[g] = S(pack2(dd, -8.0f * dd));
            }
        } else if (fmt == GpuFmt::W4 && type == GGML_TYPE_Q4_1) {
            const BlockQ4_1* b = (const BlockQ4_1*)in;
            for (uint32_t g = 0; g < G; ++g) {
                put_w4_split(qr + g * 4, b[g].qs);
                sr[g] = S(pack2(h2f(b[g].d), h2f(b[g].m)));
            }
        } else if (fmt == GpuFmt::W4 && type == GGML_TYPE_Q4_K) {
            // d*sc and dmin*m products rounded to f16.
            const BlockQ4_K* b = (const BlockQ4_K*)in;
            for (uint32_t sb = 0; sb < K / 256; ++sb) {
                const float dd = h2f(b[sb].d), dmin = h2f(b[sb].dmin);
                const uint8_t* qs = b[sb].qs;
                uint32_t* qg = qr + sb * 32;   // 8 groups of 4 words
                for (int j = 0; j < 4; ++j) {          // 64 weights per step: two groups
                    uint8_t sc, mn;
                    const uint64_t a0 = load8(qs), a1 = load8(qs + 8), a2 = load8(qs + 16), a3 = load8(qs + 24);
                    qg[0] = pack8_nibbles(a0); qg[1] = pack8_nibbles(a1);
                    qg[2] = pack8_nibbles(a2); qg[3] = pack8_nibbles(a3);
                    qg[4] = pack8_nibbles(a0 >> 4); qg[5] = pack8_nibbles(a1 >> 4);
                    qg[6] = pack8_nibbles(a2 >> 4); qg[7] = pack8_nibbles(a3 >> 4);
                    get_scale_min_k4(2 * j, b[sb].scales, sc, mn);
                    sr[sb * 8 + 2 * j] = S(pack2(dd * sc, -dmin * mn));
                    get_scale_min_k4(2 * j + 1, b[sb].scales, sc, mn);
                    sr[sb * 8 + 2 * j + 1] = S(pack2(dd * sc, -dmin * mn));
                    qs += 32;
                    qg += 8;
                }
            }
        } else if (fmt == GpuFmt::W8 && type == GGML_TYPE_Q8_0) {
            const BlockQ8_0* b = (const BlockQ8_0*)in;
            for (uint32_t g = 0; g < G; ++g) {
                std::memcpy(qr + g * 8, b[g].qs, 32);
                const float dd = h2f(b[g].d);
                sr[g] = S(pack2(dd, dd));
            }
        } else if (fmt == GpuFmt::W8 && type == GGML_TYPE_Q6_K) {
            // d*sc rounded to f16 (the int values are exact).
            const BlockQ6_K* b = (const BlockQ6_K*)in;
            for (uint32_t sb = 0; sb < K / 256; ++sb) {
                const float dd = h2f(b[sb].d);
                int8_t v[256];
                const uint8_t* ql = b[sb].ql;
                const uint8_t* qh = b[sb].qh;
                for (int half = 0; half < 2; ++half) {
                    int8_t* y = v + half * 128;
                    for (int l = 0; l < 32; ++l) {
                        y[l + 0]  = (int8_t)(((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
                        y[l + 32] = (int8_t)(((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
                        y[l + 64] = (int8_t)(((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
                        y[l + 96] = (int8_t)(((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
                    }
                    ql += 64; qh += 32;
                }
                std::memcpy(qr + sb * 64, v, 256);
                // 16 scales, one per 16 weights, in order.
                for (int g = 0; g < 8; ++g)
                    sr[sb * 8 + g] = S(pack2(dd * b[sb].scales[2 * g], dd * b[sb].scales[2 * g + 1]));
            }
        } else if (fmt == GpuFmt::F16 && type == GGML_TYPE_F16) {
            std::memcpy(qr, in, (size_t)K * 2);
        } else if (fmt == GpuFmt::F16) {
            row.resize(K);
            if (!row_to_f32(type, in, row.data(), K, err)) return false;
            for (uint32_t k = 0; k < K; k += 2) qr[k / 2] = pack2(row[k], row[k + 1]);
        } else if (fmt == GpuFmt::W8) {
            row.resize(K);
            if (!row_to_f32(type, in, row.data(), K, err)) return false;
            for (uint32_t g = 0; g < G; ++g) sr[g] = S(quant_w8(qr + g * 8, row.data() + g * 32));
        } else {
            err = "pack_gpu_rows: no W4 repack for this type";
            return false;
        }
        if (scale != 1.0f && fmt == GpuFmt::F16) scale_words_impl(qr, qw, scale);
    }
    return true;
}

bool pack_gpu_matrix_raw(int type, const void* data, uint32_t N, uint32_t K,
                         HostMatrix& m, std::string& err) {
    m = HostMatrix{};
    m.N = N; m.K = K;
    if (!plan_gpu_rows(type, K, false, m.fmt, m.lossless, err)) return false;
    if (m.fmt == GpuFmt::W8) m.layout = LayoutDesc{8, 16, true};   // w8g16sym
    if (m.fmt == GpuFmt::F16 && K % 2) {
        // Odd rows: f16 pairs straddle rows, so convert the matrix flat.
        m.q.assign(((size_t)N * K + 1) / 2, 0);
        const size_t rb = ggml_row_size((ggml_type)type, K);
        std::vector<float> row(K);
        for (uint32_t n = 0; n < N; ++n) {
            if (type == GGML_TYPE_F16) {
                const uint16_t* h = (const uint16_t*)((const uint8_t*)data + n * rb);
                for (uint32_t k = 0; k < K; ++k) {
                    const size_t i = (size_t)n * K + k;
                    m.q[i / 2] |= (i & 1) ? (uint32_t)h[k] << 16 : h[k];
                }
                continue;
            }
            if (!row_to_f32(type, (const uint8_t*)data + n * rb, row.data(), K, err)) return false;
            for (uint32_t k = 0; k < K; ++k) {
                const size_t i = (size_t)n * K + k;
                const uint32_t h = f2h(row[k]);
                m.q[i / 2] |= (i & 1) ? h << 16 : h;
            }
        }
        return true;
    }
    m.q.assign((size_t)N * q_row_words(m.fmt, K), 0);
    m.s.assign((size_t)N * s_row_words(m.fmt, K), 0);
    return pack_gpu_rows(type, data, N, K, m.fmt, 1.0f, m.q.data(), m.s.empty() ? nullptr : m.s.data(),
                         RowMap{}, err);
}

bool pack_gpu_matrix(const ggml_tensor* t, HostMatrix& out, std::string& err) {
    if (!t || !t->data) { err = "fast engine: tensor data missing"; return false; }
    const uint32_t K = (uint32_t)t->ne[0];
    const uint32_t N = (uint32_t)(ggml_nelements(t) / t->ne[0]);
    if (!pack_gpu_matrix_raw((int)t->type, t->data, N, K, out, err)) {
        err = std::string(t->name) + ": " + err;
        return false;
    }
    return true;
}

bool pack_gpu_matrix_w8(const ggml_tensor* t, HostMatrix& m, std::string& err) {
    if (!t || !t->data) { err = "fast engine: tensor data missing"; return false; }
    const uint32_t K = (uint32_t)t->ne[0];
    const uint32_t N = (uint32_t)(ggml_nelements(t) / t->ne[0]);
    m = HostMatrix{};
    m.N = N; m.K = K;
    m.layout = LayoutDesc{8, 16, true};   // w8g16sym
    if (!plan_gpu_rows((int)t->type, K, true, m.fmt, m.lossless, err)) {
        err = std::string(t->name) + ": " + err;
        return false;
    }
    m.q.assign((size_t)N * q_row_words(m.fmt, K), 0);
    m.s.assign((size_t)N * s_row_words(m.fmt, K), 0);
    if (!pack_gpu_rows((int)t->type, t->data, N, K, m.fmt, 1.0f, m.q.data(), m.s.data(), RowMap{}, err)) {
        err = std::string(t->name) + ": " + err;
        return false;
    }
    return true;
}

bool scale_matrix(HostMatrix& m, float f, std::string& err) {
    if (!m.x.empty() || (m.fmt != GpuFmt::F16 && !m.layout.is_legacy_w4() && !m.layout.is_legacy_w8())) {
        err = "scale_matrix: packed-file layout " + layout_to_string(m.layout);
        return false;
    }
    // W4 (scale, offset) and W8 (scale, scale) words, or F16 weight pairs:
    // both halves scale.
    std::vector<uint32_t>& w = m.fmt == GpuFmt::F16 ? m.q : m.s;
    scale_words_impl(w.data(), w.size(), f);
    m.lossless = m.lossless && f == 1.0f;
    return true;
}

void permute_rows(HostMatrix& m, const std::vector<uint32_t>& src) {
    const size_t qw = m.q.size() / m.N, sw = m.N ? m.s.size() / m.N : 0,
                 xw = m.N ? m.x.size() / m.N : 0;
    std::vector<uint32_t> q(m.q.size()), s(m.s.size()), x(m.x.size());
    for (size_t r = 0; r < src.size(); ++r) {
        std::copy_n(m.q.begin() + src[r] * qw, qw, q.begin() + r * qw);
        if (sw) std::copy_n(m.s.begin() + src[r] * sw, sw, s.begin() + r * sw);
        if (xw) std::copy_n(m.x.begin() + src[r] * xw, xw, x.begin() + r * xw);
    }
    m.q.swap(q);
    m.s.swap(s);
    m.x.swap(x);
}

bool concat_rows(HostMatrix& a, const HostMatrix& b, std::string& err) {
    if (a.fmt != b.fmt || a.K != b.K) { err = "concat_rows: format/K mismatch"; return false; }
    if (a.fmt == GpuFmt::F16 && ((size_t)a.N * a.K) % 2) { err = "concat_rows: odd f16 size"; return false; }
    if (a.x.empty() != b.x.empty()) { err = "concat_rows: super-scale mismatch"; return false; }
    // Legacy W4/W8 bytes are kernel-identical whatever spec produced them
    // (w4g32asym vs w4g32sym-a); any other descriptor must match exactly.
    auto legacy = [](const HostMatrix& m) {
        return m.fmt == GpuFmt::F16 || m.layout.is_legacy_w4() || m.layout.is_legacy_w8();
    };
    if (!(legacy(a) && legacy(b)) && layout_to_string(a.layout) != layout_to_string(b.layout)) {
        err = "concat_rows: layout mismatch (" + layout_to_string(a.layout) + " vs " +
              layout_to_string(b.layout) + ")";
        return false;
    }
    a.q.insert(a.q.end(), b.q.begin(), b.q.end());
    a.s.insert(a.s.end(), b.s.begin(), b.s.end());
    a.x.insert(a.x.end(), b.x.begin(), b.x.end());
    a.N += b.N;
    a.lossless = a.lossless && b.lossless;
    return true;
}

bool tensor_to_f32(const ggml_tensor* t, std::vector<float>& out, std::string& err) {
    if (!t || !t->data) { err = "fast engine: tensor data missing"; return false; }
    const size_t n = (size_t)ggml_nelements(t);
    out.resize(n);
    const size_t K = (size_t)t->ne[0];
    const size_t rows = n / K;
    const size_t rb = ggml_row_size(t->type, (int64_t)K);
    for (size_t r = 0; r < rows; ++r)
        if (!row_to_f32((int)t->type, (const uint8_t*)t->data + r * rb, out.data() + r * K,
                        (uint32_t)K, err))
            return false;
    return true;
}

bool pack_cpu_q8(const ggml_tensor* t, CpuQ8& out, std::string& err) {
    if (!t || !t->data) { err = "fast engine: tensor data missing"; return false; }
    const uint32_t K = (uint32_t)t->ne[0];
    const uint32_t N = (uint32_t)(ggml_nelements(t) / t->ne[0]);
    if (K % 32) { err = std::string(t->name) + ": K must be a multiple of 32"; return false; }
    out = CpuQ8{};
    out.N = N; out.K = K;
    out.q.resize((size_t)N * K);
    out.s.resize((size_t)N * K / 32);
    const size_t rb = ggml_row_size(t->type, K);
    const uint8_t* base = (const uint8_t*)t->data;
    if (t->type == GGML_TYPE_Q8_0) {
        for (uint32_t n = 0; n < N; ++n) {
            const BlockQ8_0* b = (const BlockQ8_0*)(base + n * rb);
            for (uint32_t g = 0; g < K / 32; ++g) {
                out.s[(size_t)n * (K / 32) + g] = h2f(b[g].d);
                std::memcpy(&out.q[(size_t)n * K + g * 32], b[g].qs, 32);
            }
        }
        return true;
    }
    if (t->type == GGML_TYPE_Q4_0) {
        for (uint32_t n = 0; n < N; ++n) {
            const BlockQ4_0* b = (const BlockQ4_0*)(base + n * rb);
            for (uint32_t g = 0; g < K / 32; ++g) {
                out.s[(size_t)n * (K / 32) + g] = h2f(b[g].d);
                int8_t* q = &out.q[(size_t)n * K + g * 32];
                for (int j = 0; j < 16; ++j) {
                    q[j] = (int8_t)((b[g].qs[j] & 15) - 8);
                    q[j + 16] = (int8_t)((b[g].qs[j] >> 4) - 8);
                }
            }
        }
        return true;
    }
    out.lossless = false;
    std::vector<float> row(K);
    for (uint32_t n = 0; n < N; ++n) {
        if (!row_to_f32((int)t->type, base + n * rb, row.data(), K, err)) return false;
        for (uint32_t g = 0; g < K / 32; ++g) {
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(row[g * 32 + i]));
            const float s = amax / 127.0f;
            out.s[(size_t)n * (K / 32) + g] = s;
            for (int i = 0; i < 32; ++i)
                out.q[(size_t)n * K + g * 32 + i] =
                    (int8_t)(s > 0 ? std::lround(std::clamp(row[g * 32 + i] / s, -127.0f, 127.0f)) : 0);
        }
    }
    return true;
}

} // namespace starling::fast
