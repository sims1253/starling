// layout.cpp — see layout.hpp.

#include "layout.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <vector>

namespace starling::fast {

namespace {

// Round-to-nearest-even f32 -> f16 bit pattern, matching the GPU's f16
// conversion (and ggml_fp32_to_fp16's RNE). Implemented locally so layout
// code has no ggml dependency.
uint16_t f2h(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t exp32 = (x >> 23) & 0xffu;
    uint32_t mant = x & 0x7fffffu;
    if (exp32 == 0xffu) return (uint16_t)(sign | 0x7c00u);  // inf/nan
    if (exp32 == 0) {
        if (mant == 0) return (uint16_t)sign;                // +-0
        // Subnormal f32: renormalize into a normalized f16 candidate.
        int e = -1;
        while (!(mant & 0x800000u)) { mant <<= 1; --e; }
        // Renormalized subnormal: value = m * 2^(e-148) with m in
        // [2^23, 2^24), i.e. (1+frac) * 2^(e-125); f16 field = e - 110.
        const int f16e = e - 110;
        mant &= 0x7fffffu;
        if (f16e > 0) {
            uint32_t out = ((uint32_t)f16e << 10) | (mant >> 13);
            const uint32_t rem = mant & 0x1fffu;
            if (rem > 0x1000u || (rem == 0x1000u && (out & 1))) ++out;
            return (uint16_t)(sign | out);
        }
        // f16 subnormal: value = (1.mant) * 2^(e-127) = mant' * 2^(f16e-24)
        const uint32_t sub = mant | 0x800000u;
        const int shift = 1 - f16e + 13;
        uint32_t out = sub >> shift;
        const uint32_t rem = sub & ((1u << shift) - 1);
        const uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (out & 1))) ++out;
        return (uint16_t)(sign | out);
    }
    const int f16e = (int)exp32 - 127 + 15;
    if (f16e >= 31) return (uint16_t)(sign | 0x7c00u);       // overflow -> inf
    if (f16e <= 0) {
        if (f16e < -10) return (uint16_t)sign;               // underflow -> 0
        const uint32_t sub = mant | 0x800000u;               // implicit bit
        const int shift = 1 - f16e + 13;
        uint32_t out = sub >> shift;
        const uint32_t rem = sub & ((1u << shift) - 1);
        const uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (out & 1))) ++out;
        return (uint16_t)(sign | out);
    }
    uint32_t out = ((uint32_t)f16e << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (out & 1))) ++out;
    return (uint16_t)(sign | out);
}

float h2f_(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {  // f16 subnormal -> normalized f32
            // value = m*2^-24; after the shift loop e = (leading pos) - 11,
            // so the f32 exponent field is 127 + (e + 11) - 24 = e + 114.
            int e = -1;
            uint32_t m = mant;
            while (!(m & 0x400u)) { m <<= 1; --e; }
            bits = sign | ((uint32_t)(e + 114) << 23) | ((m & 0x3ffu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

inline uint32_t f32bits(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    return b;
}

// One quantize-and-score candidate: codes from (s, o), weighted error.
struct ScoreCtx {
    const float* w;
    const float* im;   // null -> weight 1
    uint32_t n;        // group size
    double norm = 0.0; // Σ im (or n) — for tie-breaking irrelevant, unused
};

int clamp_code(int bits, long q) {
    const int lo = bits == 4 ? 0 : -128;
    const int hi = bits == 4 ? 15 : 127;
    return (int)std::min<long>(std::max<long>(q, lo), hi);
}

// Quantize group with scale s / offset o (o unused for sym) and return the
// weighted squared error. Codes are returned in qc when non-null.
double score_candidate(const ScoreCtx& c, int bits, bool sym, float s, float o, int16_t* qc) {
    if (!(s > 0.0f) && !(s < 0.0f)) {  // 0 or NaN scale: everything codes 0
        if (qc)
            for (uint32_t i = 0; i < c.n; ++i) qc[i] = 0;
        double e = 0;
        for (uint32_t i = 0; i < c.n; ++i) {
            const double d = sym ? c.w[i] : c.w[i] - o;
            e += c.im ? c.im[i] * d * d : d * d;
        }
        return e;
    }
    double err = 0;
    for (uint32_t i = 0; i < c.n; ++i) {
        long q;
        float wq;
        if (bits == 4) {
            q = sym ? (long)std::lround(c.w[i] / s) + 8 : (long)std::lround((c.w[i] - o) / s);
            q = clamp_code(4, q);
            wq = sym ? s * ((float)q - 8.0f) + o /* o==0 for sym */
                     : s * (float)q + o;
        } else {
            q = (long)std::lround((c.w[i] - o) / s);
            q = clamp_code(8, q);
            wq = s * (float)q + o;
        }
        if (qc) qc[i] = (int16_t)q;
        const double d = (double)c.w[i] - (double)wq;
        err += c.im ? (double)c.im[i] * d * d : d * d;
    }
    return err;
}

} // namespace

std::string layout_to_string(const LayoutDesc& d) {
    std::ostringstream s;
    s << 'w' << d.bits << 'g' << d.group << (d.symmetric ? "sym" : "asym");
    if (d.scale_dtype == ScaleDtype::U8Super) s << "u8s";
    if (d.store_pair) s << "-a";
    if (d.order) s << "-p" << d.order;
    return s.str();
}

bool layout_from_string(const std::string& spec, LayoutDesc* out, std::string* err) {
    LayoutDesc d;
    const char* p = spec.c_str();
    auto fail = [&](const char* why) {
        if (err) *err = spec + ": " + why;
        return false;
    };
    auto num = [&](const char* q) -> long {
        char* end = nullptr;
        const long v = std::strtol(q, &end, 10);
        if (end == q) { fail("expected a number"); p = q; return -1; }
        p = end;
        return v;
    };
    if (*p++ != 'w') return fail("must start with 'w'");
    d.bits = (uint32_t)num(p);
    if (*p == 'g') { ++p; d.group = (uint32_t)num(p); }
    if (std::strncmp(p, "sym", 3) == 0) { d.symmetric = true; p += 3; }
    else if (std::strncmp(p, "asym", 4) == 0) { p += 4; }
    else return fail("need 'sym' or 'asym'");
    while (*p) {
        if (std::strncmp(p, "u8s", 3) == 0) { d.scale_dtype = ScaleDtype::U8Super; p += 3; }
        else if (*p == '-' && p[1] == 'a') { d.store_pair = true; p += 2; }
        else if (*p == '-' && p[1] == 'p') { p += 2; d.order = (uint32_t)num(p); }
        else { if (err) *err = spec + ": trailing junk at '" + p + "'"; return false; }
    }
    if (!d.valid()) return fail("invalid combination (see layout.hpp)");
    *out = d;
    return true;
}

uint64_t layout_hash(const LayoutDesc& d) {
    const std::string s = layout_to_string(d);
    uint64_t h = 1469598103934665603ull;
    for (char c : s) { h ^= (uint8_t)c; h *= 1099511628211ull; }
    return h;
}

uint64_t layout_code_bytes(const LayoutDesc& d, uint32_t K) {
    return ((uint64_t)K * d.bits + 7) / 8;
}

uint64_t layout_scale_bytes(const LayoutDesc& d, uint32_t K) {
    const uint64_t groups = (uint64_t)K / d.group;
    if (d.scale_dtype == ScaleDtype::U8Super) return (groups + 3) / 4 * 4;  // u8, word padded
    const bool pair = !d.symmetric || d.store_pair;
    const uint64_t halves = groups * (pair ? 2 : 1);
    return (halves + 1) / 2 * 4;  // f16 halves, two per word, word padded
}

uint64_t layout_super_bytes(const LayoutDesc& d) {
    return d.scale_dtype == ScaleDtype::U8Super ? 2 : 0;
}

void layout_encode_row(const LayoutDesc& d, const int16_t* q, uint32_t K, uint8_t* codes) {
    std::memset(codes, 0, (size_t)layout_code_bytes(d, K));
    if (d.bits == 8) {
        for (uint32_t k = 0; k < K; ++k) codes[k] = (uint8_t)q[k];
        return;
    }
    if (d.order == 0) {
        for (uint32_t k = 0; k < K; ++k)
            codes[k >> 1] |= (uint8_t)(q[k] & 15) << (4 * (k & 1));
    } else {
        // Byte-wise order: byte b of each word holds code b in its low nibble
        // and code b+4 in its high nibble, so (word & 0x0f0f0f0f) unpacks to
        // codes 0..3 and ((word >> 4) & 0x0f0f0f0f) to codes 4..7 — four
        // K-consecutive values per byte-wise unpack, no activation reorder.
        for (uint32_t c = 0; c < K; ++c) {
            const uint32_t word = c >> 3, b = c & 3, hi = (c & 4) ? 4 : 0;
            codes[word * 4 + b] |= (uint8_t)(q[c] & 15) << hi;
        }
    }
}

int16_t layout_decode_code(const LayoutDesc& d, const uint8_t* codes, uint32_t k) {
    if (d.bits == 8) return (int16_t)(int8_t)codes[k];
    if (d.order == 0) return (int16_t)((codes[k >> 1] >> (4 * (k & 1))) & 15);
    const uint32_t word = k >> 3, b = k & 3, hi = (k & 4) ? 4 : 0;
    return (int16_t)((codes[word * 4 + b] >> hi) & 15);
}

void layout_store_scales(const LayoutDesc& d, const float* s, const float* o, uint32_t K,
                         uint8_t* scale_bytes) {
    const uint32_t groups = K / d.group;
    std::memset(scale_bytes, 0, (size_t)layout_scale_bytes(d, K));
    std::vector<uint16_t> halves(groups * 2);
    uint32_t i = 0;
    for (uint32_t g = 0; g < groups; ++g) {
        halves[i++] = f2h(s[g]);
        if (!d.symmetric) halves[i++] = f2h(o[g]);
        else if (d.store_pair) halves[i++] = f2h(-8.0f * s[g]);   // legacy pair
    }
    for (uint32_t j = 0; j < i; ++j) {
        scale_bytes[j + j] = (uint8_t)(halves[j] & 0xff);
        scale_bytes[j + j + 1] = (uint8_t)(halves[j] >> 8);
    }
}

float layout_scale_at(const LayoutDesc& d, const uint8_t* scale_bytes, const uint16_t* super,
                      const uint8_t* u8s, uint32_t g) {
    if (d.scale_dtype == ScaleDtype::U8Super)
        return h2f_(super[0]) * (float)u8s[g];   // exactly this f32 product
    // F16 storage: halves are packed in K order (scale, offset?, scale, ...).
    const bool pair = !d.symmetric || d.store_pair;
    const size_t byte = pair ? (size_t)g * 4 : (size_t)g * 2;
    const uint16_t h = (uint16_t)(scale_bytes[byte] | (scale_bytes[byte + 1] << 8));
    return h2f_(h);
}

float layout_offset_at(const LayoutDesc& d, const uint8_t* scale_bytes, uint32_t g) {
    // store_pair rows keep the legacy (s, -8s) second half readable the same
    // way the W4 kernel does.
    const bool pair = !d.symmetric || d.store_pair;
    if (!pair || d.scale_dtype == ScaleDtype::U8Super) return 0.0f;
    const size_t byte = (size_t)g * 4 + 2;   // second half of the group's pair
    const uint16_t h = (uint16_t)(scale_bytes[byte] | (scale_bytes[byte + 1] << 8));
    return h2f_(h);
}

float layout_dequant(const LayoutDesc& d, const uint8_t* codes, const uint8_t* scale_bytes,
                     const uint16_t* super, const uint8_t* u8s, uint32_t k) {
    const uint32_t g = k / d.group;
    const int16_t q = layout_decode_code(d, codes, k);
    if (d.store_pair) {
        // Legacy pair (s, o = -8s): exactly the expression the W4 kernel
        // computes for these bytes.
        const float s = layout_scale_at(d, scale_bytes, super, u8s, g);
        const float o = layout_offset_at(d, scale_bytes, g);
        return s * (float)q + o;
    }
    const float s = layout_scale_at(d, scale_bytes, super, u8s, g);
    if (d.symmetric) {
        if (d.bits == 4) return s * ((float)q - 8.0f);
        return s * (float)q;
    }
    const float o = layout_offset_at(d, scale_bytes, g);
    // Plain mul + add in f32, exactly the expression dequant_row.glsl spells
    // out (`so.x * float(q) + so.y`) — no fused op, so any GPU that also
    // issues separate mul/add (or an fma-free translation) matches bit for
    // bit, and the quantizer optimizes precisely the values the kernels see.
    return s * (float)q + o;
}

double layout_quant_row(const LayoutDesc& d, const float* w, uint32_t K, const float* im,
                        uint8_t* codes, uint8_t* scale_bytes, uint16_t* super, uint8_t* u8s) {
    const uint32_t groups = K / d.group;
    std::vector<float> s_v(groups), o_v(groups);
    int16_t* q_all = new int16_t[K];
    double sum_w2 = 0, sum_e2 = 0;

    // Search grid: multiplicative scale factors around the absmax-derived
    // base (mirrors ggml's imatrix-weighted q4_0 search granularity).
    static const float kAlpha[] = {1.0f, 0.98f, 1.02f, 0.95f, 1.05f, 0.90f, 1.10f};

    for (uint32_t g = 0; g < groups; ++g) {
        const float* wg = w + (size_t)g * d.group;
        const float* img = im ? im + (size_t)g * d.group : nullptr;
        ScoreCtx ctx{wg, img, d.group};
        float amax = 0, lo = wg[0], hi = wg[0];
        for (uint32_t i = 0; i < d.group; ++i) {
            amax = std::max(amax, std::fabs(wg[i]));
            lo = std::min(lo, wg[i]);
            hi = std::max(hi, wg[i]);
        }
        double best_err = 1e300;
        float best_s = 0, best_o = 0;

        int16_t best_q[1024];   // LayoutDesc::valid() permits group <= 1024
        int16_t qc[1024];
        auto try_cand = [&](float s_raw, float o_raw) {
            // The deployed scale is the f16-rounded one; optimize that.
            const float s = d.scale_dtype == ScaleDtype::F16 ? h2f_(f2h(s_raw)) : s_raw;
            const float o = h2f_(f2h(o_raw));
            const double e = score_candidate(ctx, d.bits, d.symmetric, s, o, qc);
            if (e < best_err) {
                best_err = e;
                best_s = s;
                best_o = d.symmetric ? 0.0f : o;
                std::memcpy(best_q, qc, sizeof(int16_t) * d.group);
            }
        };
        if (d.symmetric) {
            // Base scales: amax/7 (sym4 code range -7..7 uses the positive
            // side fully) and amax/8 (the -8 side; the Q4_0 convention).
            const float bases[2] = {d.bits == 4 ? amax / 7.0f : amax / 127.0f,
                                    d.bits == 4 ? amax / 8.0f : amax / 128.0f};
            for (float b : bases)
                for (float a : kAlpha) try_cand(b * a, 0.0f);
            if (amax == 0.0f) try_cand(0.0f, 0.0f);
        } else {
            // Asymmetric: offset at min, and the centered variant.
            const float span = hi - lo;
            const float base = span / (d.bits == 4 ? 15.0f : 255.0f);
            for (float a : kAlpha) {
                const float s = base * a;
                try_cand(s, lo);
                try_cand(s, lo + 0.5f * (span - (d.bits == 4 ? 15.0f : 255.0f) * s));
            }
            if (span == 0.0f) try_cand(0.0f, lo);
        }
        s_v[g] = best_s;
        o_v[g] = best_o;
        for (uint32_t i = 0; i < d.group; ++i)
            q_all[(size_t)g * d.group + i] = best_q[i];
    }

    // U8Super: ideal f16 scales -> super = max/255 (f16), u8 = round(s/super).
    if (d.scale_dtype == ScaleDtype::U8Super) {
        float smax = 0;
        for (uint32_t g = 0; g < groups; ++g) smax = std::max(smax, s_v[g]);
        const float sup = h2f_(f2h(smax / 255.0f));
        super[0] = f2h(sup);
        for (uint32_t g = 0; g < groups; ++g) {
            int u = (int)std::lround(s_v[g] / sup);
            u = std::min(255, std::max(1, u));
            if (s_v[g] == 0.0f) u = 0;
            u8s[g] = (uint8_t)u;
            // Recompute codes against the effective (coarser) scale.
            const float s = sup * (float)u;
            const float* wg = w + (size_t)g * d.group;
            for (uint32_t i = 0; i < d.group; ++i) {
                long q = (long)std::lround(wg[i] / s) + 8;
                q_all[(size_t)g * d.group + i] = (int16_t)clamp_code(4, q);
            }
        }
    } else {
        layout_store_scales(d, s_v.data(), o_v.data(), K, scale_bytes);
    }
    layout_encode_row(d, q_all, K, codes);

    // Report error against the exact reference dequant.
    for (uint32_t k = 0; k < K; ++k) {
        const float dq = layout_dequant(d, codes, scale_bytes, super, u8s, k);
        sum_w2 += (double)w[k] * w[k];
        sum_e2 += (double)(w[k] - dq) * (w[k] - dq);
    }
    delete[] q_all;
    return sum_w2 > 0 ? std::sqrt(sum_e2 / sum_w2) : 0.0;
}

} // namespace starling::fast
