// cpu_kernels.cpp — see cpu_kernels.hpp.

#include "cpu_kernels.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <thread>

// ISA-specific kernels are compiled with function-level target attributes
// and picked at run time, so the library itself needs no -march flags.
#if defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
#include <arm_neon.h>
#if defined(__linux__) || defined(__ANDROID__)
#include <sys/auxv.h>
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1 << 20)
#endif
#endif
#define STARLING_FAST_NEON_DOT 1
#define STARLING_TARGET_DOT __attribute__((target("dotprod")))
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__clang__) || defined(__GNUC__))
#include <immintrin.h>
#define STARLING_FAST_AVX2 1
#define STARLING_TARGET_AVX2 __attribute__((target("avx2,fma")))
#endif

namespace starling::fast::cpu {

namespace {
bool have_simd() {
#if defined(STARLING_FAST_NEON_DOT)
#if defined(__APPLE__)
    return true;
#elif defined(__linux__) || defined(__ANDROID__)
    static const bool ok = (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
    return ok;
#else
    return false;
#endif
#elif defined(STARLING_FAST_AVX2)
    static const bool ok = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    return ok;
#else
    return false;
#endif
}
} // namespace

const char* isa_name() {
#if defined(STARLING_FAST_NEON_DOT)
    return have_simd() ? "neon-dotprod" : "scalar";
#elif defined(STARLING_FAST_AVX2)
    return have_simd() ? "avx2" : "scalar";
#else
    return "scalar";
#endif
}

void quantize(const float* x, uint32_t K, QVec& out) {
    const uint32_t G = K / 32;
    out.q.resize(K);
    out.s.resize(G);
#if defined(STARLING_FAST_NEON_DOT)
    for (uint32_t g = 0; g < G; ++g) {
        const float* v = x + g * 32;
        float32x4_t am = vdupq_n_f32(0.0f);
        for (uint i = 0; i < 32; i += 4) am = vmaxq_f32(am, vabsq_f32(vld1q_f32(v + i)));
        const float amax = vmaxvq_f32(am);
        const float s = amax / 127.0f;
        const float inv = s > 0.0f ? 1.0f / s : 0.0f;
        out.s[g] = s;
        int8_t* q = out.q.data() + g * 32;
        // lrintf rounds to nearest-even; so does vcvtnq_s32_f32 (but the
        // saturating variant clamps at ±127, which is what we want anyway).
        for (uint i = 0; i < 32; i += 8) {
            const float32x4_t a = vmulq_n_f32(vld1q_f32(v + i), inv);
            const float32x4_t b = vmulq_n_f32(vld1q_f32(v + i + 4), inv);
            const int16x8_t h = vcombine_s16(vqmovn_s32(vcvtnq_s32_f32(a)),
                                             vqmovn_s32(vcvtnq_s32_f32(b)));
            vst1_s8(q + i, vqmovn_s16(h));
        }
    }
#elif defined(STARLING_FAST_AVX2)
    for (uint32_t g = 0; g < G; ++g) {
        const float* v = x + g * 32;
        __m128 am = _mm_setzero_ps();
        for (uint i = 0; i < 32; i += 4) am = _mm_max_ps(am, _mm_andnot_ps(_mm_set1_ps(-0.0f), _mm_loadu_ps(v + i)));
        am = _mm_max_ps(am, _mm_movehl_ps(am, am));
        am = _mm_max_ss(am, _mm_movehdup_ps(am));
        const float amax = _mm_cvtss_f32(am);
        const float s = amax / 127.0f;
        const float inv = s > 0.0f ? 1.0f / s : 0.0f;
        out.s[g] = s;
        int8_t* q = out.q.data() + g * 32;
        for (uint i = 0; i < 32; i += 8) {
            const __m256i d = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(v + i), _mm256_set1_ps(inv)));
            const __m128i h = _mm_packs_epi32(_mm256_castsi256_si128(d), _mm256_extracti128_si256(d, 1));
            _mm_storel_epi64((__m128i*)(q + i), _mm_packs_epi16(h, h));
        }
    }
#else
    for (uint32_t g = 0; g < G; ++g) {
        const float* v = x + g * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(v[i]));
        const float s = amax / 127.0f;
        const float inv = s > 0.0f ? 1.0f / s : 0.0f;
        out.s[g] = s;
        for (int i = 0; i < 32; ++i) out.q[g * 32 + i] = (int8_t)std::lrintf(v[i] * inv);
    }
#endif
}

namespace {

void cpu_relax() {
#if defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#else
    _mm_pause();
#endif
}

#if defined(STARLING_FAST_NEON_DOT)

inline float hsum(float32x4_t v) { return vaddvq_f32(v); }

STARLING_TARGET_DOT void gemv_simd(const CpuQ8& W, const QVec& x, const float* bias, float* y,
               uint32_t r0, uint32_t r1) {
    const uint32_t K = W.K, G = K / 32;
    uint32_t n = r0;
    for (; n + 4 <= r1; n += 4) {
        float32x4_t acc[4] = {vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0)};
        const int8_t* w[4];
        const float* ws[4];
        for (int r = 0; r < 4; ++r) {
            w[r] = W.q.data() + (size_t)(n + r) * K;
            ws[r] = W.s.data() + (size_t)(n + r) * G;
        }
        for (uint32_t g = 0; g < G; ++g) {
            const int8x16_t x0 = vld1q_s8(x.q.data() + g * 32);
            const int8x16_t x1 = vld1q_s8(x.q.data() + g * 32 + 16);
            const float xs = x.s[g];
            for (int r = 0; r < 4; ++r) {
                int32x4_t d = vdotq_s32(vdupq_n_s32(0), vld1q_s8(w[r] + g * 32), x0);
                d = vdotq_s32(d, vld1q_s8(w[r] + g * 32 + 16), x1);
                acc[r] = vmlaq_n_f32(acc[r], vcvtq_f32_s32(d), ws[r][g] * xs);
            }
        }
        for (int r = 0; r < 4; ++r) y[n + r] = hsum(acc[r]) + (bias ? bias[n + r] : 0.0f);
    }
    for (; n < r1; ++n) {
        float32x4_t acc = vdupq_n_f32(0);
        const int8_t* w = W.q.data() + (size_t)n * K;
        const float* ws = W.s.data() + (size_t)n * G;
        for (uint32_t g = 0; g < G; ++g) {
            int32x4_t d = vdotq_s32(vdupq_n_s32(0), vld1q_s8(w + g * 32), vld1q_s8(x.q.data() + g * 32));
            d = vdotq_s32(d, vld1q_s8(w + g * 32 + 16), vld1q_s8(x.q.data() + g * 32 + 16));
            acc = vmlaq_n_f32(acc, vcvtq_f32_s32(d), ws[g] * x.s[g]);
        }
        y[n] = hsum(acc) + (bias ? bias[n] : 0.0f);
    }
}

#elif defined(STARLING_FAST_AVX2)

STARLING_TARGET_AVX2 inline float hsum(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_movehdup_ps(lo));
    return _mm_cvtss_f32(lo);
}

// int8 x int8 -> 8 x int32 partial sums via the |a|, sign(b, a) trick.
STARLING_TARGET_AVX2 inline __m256i dot32(__m256i a, __m256i b) {
    const __m256i ax = _mm256_sign_epi8(a, a);
    const __m256i sb = _mm256_sign_epi8(b, a);
    return _mm256_madd_epi16(_mm256_maddubs_epi16(ax, sb), _mm256_set1_epi16(1));
}

STARLING_TARGET_AVX2 void gemv_simd(const CpuQ8& W, const QVec& x, const float* bias, float* y,
               uint32_t r0, uint32_t r1) {
    const uint32_t K = W.K, G = K / 32;
    uint32_t n = r0;
    for (; n + 4 <= r1; n += 4) {
        __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
        for (uint32_t g = 0; g < G; ++g) {
            const __m256i xv = _mm256_loadu_si256((const __m256i*)(x.q.data() + g * 32));
            const float xs = x.s[g];
            for (int r = 0; r < 4; ++r) {
                const __m256i wv = _mm256_loadu_si256((const __m256i*)(W.q.data() + (size_t)(n + r) * K + g * 32));
                const __m256 d = _mm256_cvtepi32_ps(dot32(wv, xv));
                acc[r] = _mm256_fmadd_ps(d, _mm256_set1_ps(W.s[(size_t)(n + r) * G + g] * xs), acc[r]);
            }
        }
        for (int r = 0; r < 4; ++r) y[n + r] = hsum(acc[r]) + (bias ? bias[n + r] : 0.0f);
    }
    for (; n < r1; ++n) {
        __m256 acc = _mm256_setzero_ps();
        for (uint32_t g = 0; g < G; ++g) {
            const __m256i xv = _mm256_loadu_si256((const __m256i*)(x.q.data() + g * 32));
            const __m256i wv = _mm256_loadu_si256((const __m256i*)(W.q.data() + (size_t)n * K + g * 32));
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot32(wv, xv)),
                                  _mm256_set1_ps(W.s[(size_t)n * G + g] * x.s[g]), acc);
        }
        y[n] = hsum(acc) + (bias ? bias[n] : 0.0f);
    }
}

#endif

void gemv_scalar(const CpuQ8& W, const QVec& x, const float* bias, float* y,
               uint32_t r0, uint32_t r1) {
    const uint32_t K = W.K, G = K / 32;
    for (uint32_t n = r0; n < r1; ++n) {
        float acc = 0.0f;
        const int8_t* w = W.q.data() + (size_t)n * K;
        for (uint32_t g = 0; g < G; ++g) {
            int32_t d = 0;
            for (int i = 0; i < 32; ++i) d += (int32_t)w[g * 32 + i] * (int32_t)x.q[g * 32 + i];
            acc += (float)d * W.s[(size_t)n * G + g] * x.s[g];
        }
        y[n] = acc + (bias ? bias[n] : 0.0f);
    }
}

} // namespace

void gemv(const CpuQ8& W, const QVec& x, const float* bias, float* y, uint32_t r0, uint32_t r1) {
    r1 = std::min(r1, W.N);
    if (r0 >= r1) return;
#if defined(STARLING_FAST_NEON_DOT) || defined(STARLING_FAST_AVX2)
    if (have_simd()) { gemv_simd(W, x, bias, y, r0, r1); return; }
#endif
    gemv_scalar(W, x, bias, y, r0, r1);
}

GemvHelper::~GemvHelper() {
    if (started_) {
        job_.seq.store(0xffffffffu, std::memory_order_release);   // poison: exit
        th_.join();
    }
}

void GemvHelper::worker() {
    uint32_t seen = 0;
    for (;;) {
        while (job_.seq.load(std::memory_order_acquire) == seen) cpu_relax();
        const uint32_t s = job_.seq.load(std::memory_order_relaxed);
        if (s == 0xffffffffu) return;
        gemv(*job_.W, *job_.x, job_.bias, job_.y, job_.r0, job_.r1);
        seen = s;
        job_.ack.store(s, std::memory_order_release);
    }
}

void GemvHelper::run(const CpuQ8& W, const QVec& x, const float* bias, float* y,
                     uint32_t r0, uint32_t r1) {
    r1 = std::min(r1, W.N);
    if (r0 >= r1) return;
    // Only the big products: the handoff costs a few us.
    const uint64_t macs = (uint64_t)(r1 - r0) * W.K;
    const bool split = macs > (1u << 20) && r1 - r0 >= 256;
    if (!split) { gemv(W, x, bias, y, r0, r1); return; }
    if (!started_) {
        th_ = std::thread([this] { worker(); });
        started_ = true;
    }
    const uint32_t mid = r0 + ((r1 - r0) / 8) * 4;   // bias toward this thread
    job_.W = &W;
    job_.x = &x;
    job_.bias = bias;
    job_.y = y;
    job_.r0 = r0;
    job_.r1 = mid;
    const uint32_t s = job_.seq.load(std::memory_order_relaxed) + 1;
    job_.seq.store(s, std::memory_order_release);
    gemv(W, x, bias, y, mid, r1);
    while (job_.ack.load(std::memory_order_acquire) != s) cpu_relax();
}

} // namespace starling::fast::cpu
