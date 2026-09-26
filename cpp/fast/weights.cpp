// weights.cpp — see weights.hpp.

#include "weights.hpp"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>

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

// Emit one W4 group: 32 unsigned 4-bit values (sequential), scale, offset.
void put_w4(HostMatrix& m, size_t g, const uint8_t* q32, float scale, float offset) {
    for (int w = 0; w < 4; ++w) {
        uint32_t word = 0;
        for (int i = 0; i < 8; ++i) word |= (uint32_t)(q32[w * 8 + i] & 15) << (4 * i);
        m.q[g * 4 + w] = word;
    }
    m.s[g] = pack2(scale, offset);
}

// Emit one W8 group: 32 int8 values and two scales (weights 0-15, 16-31).
void put_w8(HostMatrix& m, size_t g, const int8_t* q32, float s_lo, float s_hi) {
    for (int w = 0; w < 8; ++w) {
        uint32_t word = 0;
        for (int i = 0; i < 4; ++i) word |= (uint32_t)(uint8_t)q32[w * 4 + i] << (8 * i);
        m.q[g * 8 + w] = word;
    }
    m.s[g] = pack2(s_lo, s_hi);
}

// Quantize 32 floats to a W8 group (absmax per 16) — the fallback path.
void quant_w8(HostMatrix& m, size_t g, const float* x) {
    int8_t q[32];
    float sc[2];
    for (int h = 0; h < 2; ++h) {
        float amax = 0.0f;
        for (int i = 0; i < 16; ++i) amax = std::max(amax, std::fabs(x[h * 16 + i]));
        // f16-representable scale so the kernel sees exactly what we used.
        float s = h2f(f2h(amax / 127.0f));
        if (s == 0.0f && amax > 0.0f) s = amax / 127.0f;
        sc[h] = s;
        for (int i = 0; i < 16; ++i)
            q[h * 16 + i] = (int8_t)std::lround(s > 0 ? std::clamp(x[h * 16 + i] / s, -127.0f, 127.0f) : 0.0f);
    }
    put_w8(m, g, q, sc[0], sc[1]);
}

bool row_to_f32(int type, const void* row, float* out, uint32_t K, std::string& err) {
    const ggml_type_traits* tr = ggml_get_type_traits((ggml_type)type);
    if (type == GGML_TYPE_F32) { std::memcpy(out, row, K * 4); return true; }
    if (!tr || !tr->to_float) { err = std::string("no dequantizer for type ") + (tr ? tr->type_name : "?"); return false; }
    tr->to_float(row, out, K);
    return true;
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

bool pack_gpu_matrix_raw(int type, const void* data, uint32_t N, uint32_t K,
                         HostMatrix& m, std::string& err) {
    m = HostMatrix{};
    m.N = N; m.K = K;
    const size_t row_bytes = ggml_row_size((ggml_type)type, K);
    const uint8_t* base = (const uint8_t*)data;
    const size_t G = K / 32;
    auto alloc = [&](GpuFmt f) {
        m.fmt = f;
        if (f == GpuFmt::W8) m.layout = LayoutDesc{8, 16, true};   // w8g16sym
        if (f == GpuFmt::W4) { m.q.assign((size_t)N * G * 4, 0); m.s.assign((size_t)N * G, 0); }
        if (f == GpuFmt::W8) { m.q.assign((size_t)N * G * 8, 0); m.s.assign((size_t)N * G, 0); }
        if (f == GpuFmt::F16) { m.q.assign(((size_t)N * K + 1) / 2, 0); m.s.clear(); }
    };
    const bool k32 = (K % 32) == 0;
    switch (type) {
    case GGML_TYPE_Q4_0: {
        if (!k32) break;
        alloc(GpuFmt::W4);
        for (uint32_t n = 0; n < N; ++n) {
            const BlockQ4_0* b = (const BlockQ4_0*)(base + n * row_bytes);
            for (size_t g = 0; g < G; ++g) {
                uint8_t q[32];
                for (int j = 0; j < 16; ++j) { q[j] = b[g].qs[j] & 15; q[j + 16] = b[g].qs[j] >> 4; }
                const float d = h2f(b[g].d);
                put_w4(m, n * G + g, q, d, -8.0f * d);
            }
        }
        return true;
    }
    case GGML_TYPE_Q4_1: {
        if (!k32) break;
        alloc(GpuFmt::W4);
        for (uint32_t n = 0; n < N; ++n) {
            const BlockQ4_1* b = (const BlockQ4_1*)(base + n * row_bytes);
            for (size_t g = 0; g < G; ++g) {
                uint8_t q[32];
                for (int j = 0; j < 16; ++j) { q[j] = b[g].qs[j] & 15; q[j + 16] = b[g].qs[j] >> 4; }
                put_w4(m, n * G + g, q, h2f(b[g].d), h2f(b[g].m));
            }
        }
        return true;
    }
    case GGML_TYPE_Q4_K: {
        if (K % 256) break;
        alloc(GpuFmt::W4);
        m.lossless = false;   // d*sc and dmin*m products rounded to f16
        for (uint32_t n = 0; n < N; ++n) {
            const BlockQ4_K* b = (const BlockQ4_K*)(base + n * row_bytes);
            for (size_t sb = 0; sb < K / 256; ++sb) {
                const float d = h2f(b[sb].d), dmin = h2f(b[sb].dmin);
                const uint8_t* qs = b[sb].qs;
                for (int j = 0; j < 4; ++j) {          // 64 weights per step
                    uint8_t sc, mn;
                    uint8_t q[32];
                    get_scale_min_k4(2 * j, b[sb].scales, sc, mn);
                    for (int l = 0; l < 32; ++l) q[l] = qs[l] & 15;
                    put_w4(m, n * G + sb * 8 + 2 * j, q, d * sc, -dmin * mn);
                    get_scale_min_k4(2 * j + 1, b[sb].scales, sc, mn);
                    for (int l = 0; l < 32; ++l) q[l] = qs[l] >> 4;
                    put_w4(m, n * G + sb * 8 + 2 * j + 1, q, d * sc, -dmin * mn);
                    qs += 32;
                }
            }
        }
        return true;
    }
    case GGML_TYPE_Q8_0: {
        if (!k32) break;
        alloc(GpuFmt::W8);
        for (uint32_t n = 0; n < N; ++n) {
            const BlockQ8_0* b = (const BlockQ8_0*)(base + n * row_bytes);
            for (size_t g = 0; g < G; ++g) {
                const float d = h2f(b[g].d);
                put_w8(m, n * G + g, b[g].qs, d, d);
            }
        }
        return true;
    }
    case GGML_TYPE_Q6_K: {
        if (K % 256) break;
        alloc(GpuFmt::W8);
        m.lossless = false;   // d*sc rounded to f16 (int values exact)
        for (uint32_t n = 0; n < N; ++n) {
            const BlockQ6_K* b = (const BlockQ6_K*)(base + n * row_bytes);
            for (size_t sb = 0; sb < K / 256; ++sb) {
                const float d = h2f(b[sb].d);
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
                // 16 scales, one per 16 weights, in order.
                for (int g = 0; g < 8; ++g)
                    put_w8(m, n * G + sb * 8 + g, v + g * 32,
                           d * b[sb].scales[2 * g], d * b[sb].scales[2 * g + 1]);
            }
        }
        return true;
    }
    case GGML_TYPE_F16: {
        alloc(GpuFmt::F16);
        const uint16_t* h = (const uint16_t*)data;
        const size_t total = (size_t)N * K;
        for (size_t i = 0; i < total; i += 2)
            m.q[i / 2] = (uint32_t)h[i] | (i + 1 < total ? (uint32_t)h[i + 1] << 16 : 0u);
        return true;
    }
    case GGML_TYPE_F32:
    case GGML_TYPE_BF16: {
        alloc(GpuFmt::F16);
        std::vector<float> row(K);
        for (uint32_t n = 0; n < N; ++n) {
            if (!row_to_f32(type, base + n * row_bytes, row.data(), K, err)) return false;
            for (uint32_t k = 0; k < K; ++k) {
                const size_t i = (size_t)n * K + k;
                const uint32_t h = f2h(row[k]);
                m.q[i / 2] |= (i & 1) ? h << 16 : h;
            }
        }
        m.lossless = false;   // f32 -> f16 rounds
        return true;
    }
    default:
        break;
    }
    // Fallback: dequantize then requantize to W8.
    if (!k32) { err = "fast engine: K must be a multiple of 32"; return false; }
    alloc(GpuFmt::W8);
    m.lossless = false;
    std::vector<float> row(K);
    for (uint32_t n = 0; n < N; ++n) {
        if (!row_to_f32(type, base + n * row_bytes, row.data(), K, err)) return false;
        for (size_t g = 0; g < G; ++g) quant_w8(m, n * G + g, row.data() + g * 32);
    }
    return true;
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
