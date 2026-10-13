// host_conv.cpp — blocked, vectorized, threaded K=3 Conv1d (see
// host_conv.hpp for the bit-exactness argument).
//
// GEMM-style blocking over (oc, t) with f64 accumulators: a work item owns an
// [kMc oc x kTc t] output block and walks the input channels in kKc chunks,
// keeping its accumulators in a block buffer between chunks so each output
// still sums its 3*IC terms strictly in (c, k) order. Inside a chunk an
// [kMR x kNR] register tile streams packed f64 weight micro panels against
// a packed f64 input panel. Work items are claimed from an atomic counter so
// big and little cores share the load.
#include "host_conv.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <exception>
#include <mutex>
#include <thread>

namespace starling::ggml::lib {
namespace {

// The register tile uses GNU vector extensions (GCC, Clang, clang-cl); other
// compilers (MSVC) get a scalar tile with the same per-element arithmetic
// (one f64 multiply, then one f64 add, in step order), so both produce the
// same bits. STARLING_HOST_CONV_SCALAR forces the scalar tile (for testing it
// on GCC/Clang).
#if (defined(__GNUC__) || defined(__clang__)) && !defined(STARLING_HOST_CONV_SCALAR)
#define STARLING_HOST_CONV_VEC 1
typedef double v2d __attribute__((vector_size(16)));
typedef double v2du __attribute__((vector_size(16), aligned(8)));  // unaligned access
#endif

#if defined(__aarch64__)
constexpr int64_t kMR = 4, kNR = 8;  // 16 accumulator q-registers of 32
#else
constexpr int64_t kMR = 4, kNR = 4;  // fits the 16 SSE registers
#endif
constexpr int64_t kMc = 64;   // oc per work item (multiple of kMR)
constexpr int64_t kTc = 64;   // t per work item (multiple of kNR)
constexpr int64_t kKc = 64;   // input channels per chunk (3*kKc steps)

// acc[i][:] += sum over `steps` of wp[s][i] * xp[s][:] (s ascending).
// wp: [steps][kMR] f64; xp: [steps][ldx] f64 (this tile's kNR columns at
// xp + col); acc: [kMR][lda] f64.
inline void micro_tile(const double* wp, const double* xp, int64_t ldx, int64_t steps,
                       double* acc, int64_t lda) {
#if !defined(STARLING_HOST_CONV_VEC)
    double a[kMR][kNR];
    for (int64_t i = 0; i < kMR; ++i)
        for (int64_t j = 0; j < kNR; ++j) a[i][j] = acc[i * lda + j];
    for (int64_t s = 0; s < steps; ++s) {
        const double* xs = xp + s * ldx;
        const double* ws = wp + s * kMR;
        for (int64_t i = 0; i < kMR; ++i) {
            const double wi = ws[i];
            for (int64_t j = 0; j < kNR; ++j) a[i][j] += wi * xs[j];
        }
    }
    for (int64_t i = 0; i < kMR; ++i)
        for (int64_t j = 0; j < kNR; ++j) acc[i * lda + j] = a[i][j];
#else
    constexpr int64_t kNV = kNR / 2;
    // Element-wise vector loads/stores (not memcpy of whole rows) so GCC
    // keeps the accumulator tile in registers.
    v2d a[kMR][kNV];
    for (int64_t i = 0; i < kMR; ++i)
        for (int64_t j = 0; j < kNV; ++j) a[i][j] = *(const v2du*)(acc + i * lda + 2 * j);
    for (int64_t s = 0; s < steps; ++s) {
        v2d xv[kNV];
        for (int64_t j = 0; j < kNV; ++j) xv[j] = *(const v2du*)(xp + s * ldx + 2 * j);
        const double* ws = wp + s * kMR;
        for (int64_t i = 0; i < kMR; ++i) {
            const v2d wb = {ws[i], ws[i]};
            for (int64_t j = 0; j < kNV; ++j) a[i][j] += wb * xv[j];
        }
    }
    for (int64_t i = 0; i < kMR; ++i)
        for (int64_t j = 0; j < kNV; ++j) *(v2du*)(acc + i * lda + 2 * j) = a[i][j];
#endif
}

} // namespace

int64_t host_conv1d_k3(const float* w, const float* bias, int64_t OC, int64_t IC,
                       const ggml_bf16_t* x, int64_t L, int64_t pad_l, int64_t pad_r,
                       int64_t stride, std::vector<float>& y, int n_threads,
                       const ConvPost& post) {
    constexpr int64_t K = 3;
    const int64_t Lp = L + pad_l + pad_r;
    const int64_t OL = Lp >= K ? (Lp - K) / stride + 1 : 0;
    y.assign((size_t) OC * std::max<int64_t>(OL, 0), 0.0f);
    if (OL <= 0 || OC <= 0) return std::max<int64_t>(OL, 0);
    const int64_t s = stride;
    const int64_t n_tb = (OL + kTc - 1) / kTc;
    const int64_t OLr = n_tb * kTc;          // OL rounded up to whole t blocks
    const int64_t OCr = (OC + kMR - 1) / kMR * kMR;
    const int64_t n_ob = (OCr + kMc - 1) / kMc;
    const int nthr = std::max(1, n_threads);

    // Phase-split padded input: tap k of output t reads padded column t*s+k
    // = phase (k % s) at offset t + k / s. Row length covers every tile.
    const int64_t J = OLr + (K - 1) / s + 1;
    std::vector<float> xph((size_t) s * IC * J, 0.0f);
    // Weights packed per kMR-row micro panel: wp[ob4][c*3+k][i] (rows >= OC
    // stay zero and are discarded).
    std::vector<float> wpk((size_t) OCr * IC * K, 0.0f);
    // Work items from an atomic counter on nthr threads (this one included).
    // The first exception, from a body (scratch allocation, `post`) or from
    // starting a thread, stops the remaining items and is rethrown here once
    // every started thread has joined.
    auto run = [&](const auto& body, int64_t total) {
        std::atomic<int64_t> next{0};
        std::exception_ptr first;
        std::mutex mu;
        auto stop = [&](std::exception_ptr e) {
            next.store(total);
            std::lock_guard<std::mutex> lk(mu);
            if (!first) first = e;
        };
        auto worker = [&]() {
            try {
                for (int64_t it; (it = next.fetch_add(1)) < total;) body(it);
            } catch (...) {
                stop(std::current_exception());
            }
        };
        std::vector<std::thread> ths;
        try {
            const int64_t n = std::min<int64_t>(nthr, total);
            ths.reserve((size_t) std::max<int64_t>(n - 1, 0));
            for (int64_t i = 1; i < n; ++i) ths.emplace_back(worker);
        } catch (...) {
            stop(std::current_exception());
        }
        worker();
        for (auto& t : ths) t.join();
        if (first) std::rethrow_exception(first);
    };
    run([&](int64_t c) {
        const ggml_bf16_t* src = x + (size_t) c * L;
        for (int64_t p = 0; p < s; ++p) {
            float* row = xph.data() + ((size_t) p * IC + c) * J;
            for (int64_t j = 0; j < J; ++j) {
                const int64_t i = j * s + p - pad_l;  // unpadded input column
                if (i >= 0 && i < L) row[j] = ggml_bf16_to_fp32(src[i]);
            }
        }
    }, IC);
    run([&](int64_t m4) {
        float* dst = wpk.data() + (size_t) m4 * IC * K * kMR;
        for (int64_t i = 0; i < kMR; ++i) {
            const int64_t oc = m4 * kMR + i;
            if (oc >= OC) break;
            const float* src = w + (size_t) oc * IC * K;
            for (int64_t ck = 0; ck < IC * K; ++ck) dst[ck * kMR + i] = src[ck];
        }
    }, OCr / kMR);

    run([&](int64_t item) {
        const int64_t ob = item % n_ob, tb = item / n_ob;
        const int64_t oc0 = ob * kMc, t0 = tb * kTc;
        const int64_t mc = std::min(kMc, OCr - oc0);
        thread_local std::vector<double> accb, xp, wp;
        accb.resize((size_t) kMc * kTc);
        xp.resize((size_t) kKc * K * kTc);
        wp.resize((size_t) kMc * kKc * K);
        for (int64_t i = 0; i < mc; ++i) {
            const double b = oc0 + i < OC ? (double) bias[oc0 + i] : 0.0;
            std::fill_n(accb.data() + i * kTc, kTc, b);
        }
        for (int64_t c0 = 0; c0 < IC; c0 += kKc) {
            const int64_t kc = std::min(kKc, IC - c0);
            // Pack the f64 input panel xp[(c-c0)*3+k][0..kTc).
            for (int64_t cc = 0; cc < kc; ++cc)
                for (int64_t k = 0; k < K; ++k) {
                    const float* src = xph.data() +
                        ((size_t)(k % s) * IC + (c0 + cc)) * J + t0 + k / s;
                    double* dst = xp.data() + (cc * K + k) * kTc;
                    for (int64_t j = 0; j < kTc; ++j) dst[j] = (double) src[j];
                }
            // Widen this chunk's weight micro panels to f64 (contiguous per m4).
            const int64_t wn = kc * K * kMR;
            for (int64_t m4 = 0; m4 < mc / kMR; ++m4) {
                const float* src = wpk.data() + ((size_t)(oc0 / kMR + m4) * IC + c0) * K * kMR;
                double* dst = wp.data() + m4 * wn;
                for (int64_t i = 0; i < wn; ++i) dst[i] = (double) src[i];
            }
            for (int64_t m4 = 0; m4 < mc / kMR; ++m4)
                for (int64_t n = 0; n < kTc; n += kNR)
                    micro_tile(wp.data() + m4 * wn, xp.data() + n, kTc, kc * K,
                               accb.data() + m4 * kMR * kTc + n, kTc);
        }
        const int64_t tn = std::min(kTc, OL - t0);
        for (int64_t i = 0; i < mc && oc0 + i < OC; ++i) {
            float* dst = y.data() + (size_t)(oc0 + i) * OL + t0;
            const double* src = accb.data() + i * kTc;
            for (int64_t j = 0; j < tn; ++j) dst[j] = (float) src[j];
            if (post) post(dst, tn);
        }
    }, n_ob * n_tb);
    return OL;
}

} // namespace starling::ggml::lib
