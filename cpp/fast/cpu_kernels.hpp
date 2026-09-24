// cpu_kernels.hpp — CPU GEMV kernels for the sequential decoders.
//
// The transducer/LLM greedy loops are strictly sequential with tiny
// per-step work (a few matrix-vector products), where a GPU round trip per
// step costs more than the math. These kernels multiply int8 weights (CpuQ8:
// f32 scale per 32) by an int8-quantized activation vector with the
// dot-product instructions of the target (ARMv8.2 SDOT, x86 AVX2), four rows
// at a time so each activation load feeds four accumulators.

#pragma once

#include "weights.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace starling::fast::cpu {

// Activation quantized per 32 (absmax / 127, f32 scale).
struct QVec {
    std::vector<int8_t> q;
    std::vector<float> s;
};
void quantize(const float* x, uint32_t K, QVec& out);

// y[n] = sum_k W[n][k] * x[k] (+ bias[n] when non-null) for rows [r0, r1).
void gemv(const CpuQ8& W, const QVec& x, const float* bias, float* y,
          uint32_t r0 = 0, uint32_t r1 = UINT32_MAX);

// Name of the compiled kernel family ("neon-dotprod", "avx2", "scalar").
const char* isa_name();

// Second thread for the decoder's large matrix-vector products: one
// persistent worker spinning on a sequence counter (a futex/condvar wake
// costs ~10x a spin). While held (hold(true), for the length of one
// transcription) it spins even between jobs; otherwise it parks on a
// condition variable ~2 ms after its last job, so an idle engine costs no
// CPU. Holding matters on big.LITTLE phones: a thread woken cold at decode
// start lands on a little core (Pixel 10 Pro: decode +17 %), while one kept
// busy through the GPU encoder stays on a big core. The split is by rows,
// so every output element is computed exactly as in the single-threaded path.
// Not re-entrant: one decode at a time per helper (one helper per engine).
class GemvHelper {
public:
    ~GemvHelper();
    // Splits [r0, r1) across this thread and the worker when the product
    // is large enough to amortize the handoff (~2 us).
    void run(const CpuQ8& W, const QVec& x, const float* bias, float* y, uint32_t r0 = 0,
             uint32_t r1 = UINT32_MAX);
    // Keep the worker spinning (true) until released (false); starts it.
    void hold(bool on);

private:
    struct Job {
        std::atomic<uint32_t> seq{0};
        std::atomic<uint32_t> ack{0};
        const CpuQ8* W = nullptr;
        const QVec* x = nullptr;
        const float* bias = nullptr;
        float* y = nullptr;
        uint32_t r0 = 0, r1 = 0;
    };
    void worker();
    Job job_;
    std::thread th_;
    bool started_ = false;
    std::atomic<bool> parked_{false};
    std::atomic<bool> held_{false};
    std::mutex park_m_;
    std::condition_variable park_cv_;
};

} // namespace starling::fast::cpu
