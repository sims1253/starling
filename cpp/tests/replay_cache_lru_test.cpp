// replay_cache_lru_test.cpp — permanent regression test for the Wave H
// unbounded per-shape ReplayGraph cache OOM (VRAM-exhaustion class).
//
// Root cause (see plans/wave-h-graph-cache-lru.md): three process-global
// per-shape capture caches (parakeet Encoder::replay_cache_, moss
// g_encoder_cache, moss g_prefill_cache) were std::unordered_maps with NO
// eviction. Real audio produces a near-continuous length distribution, so each
// distinct mel length / prompt length permanently pinned its own captured CUDA
// graph + private gallocr (its own device buffer) until process exit -> linear
// VRAM growth -> OOM on the second dataset of a real-audio leaderboard run.
// Comments claimed the caches were "LRU-bounded"; they were not.
//
// Fix: a shared bounded LruCache (runtime/lru_cache.hpp) backs all three, with
// capacity STARLING_REPLAY_CACHE_SIZE (default 16); on a miss at capacity it
// evicts the least-recently-used shape (freeing its device buffer) first.
//
// This test proves the bound is real. It drives the moss encoder+adapter cache
// (encode_audio_and_adapt) and the moss prefill cache (llm_prefill) across N >
// capacity DISTINCT shapes in one process and asserts:
//   (1) the cache entry count NEVER exceeds capacity, and saturates at capacity
//       (every miss past capacity evicts one, so size is flat, not growing);
//   (2) device VRAM PLATEAUS once the cache is full, instead of growing
//       linearly with N (the signature of the unbounded bug). Without LRU each
//       new distinct shape would add one gallocr forever; with LRU each new
//       shape reclaims ~one old one, so the marginal VRAM cost per new shape
//       collapses to ~0 past capacity.
//
// This is the test that WOULD have caught the bug. The capacity is forced small
// (STARLING_REPLAY_CACHE_SIZE=4) via setenv so the plateau is dramatic and the
// test is fast; the default capacity (16) is exercised end-to-end by the
// real-audio leaderboard bench (benchmarks/bench_leaderboard.py).
//
// GPU-only: the captured-graph cache path exists only on GPU (CPU uses the
// one-shot build, which allocates+ frees per call and has no cache). On CPU the
// bug class cannot occur, so the test is a vacuous pass there.
//
// Usage: replay_cache_lru_test [repo_root]
//   (loads <repo_root>/models/moss-transcribe-preview-2b-bf16-exact.gguf)
// Exit 0 = pass, 1 = bound/plateau failure, 2 = model load failure.

#include "moss/audio_encoder.hpp"
#include "moss/config.hpp"
#include "moss/llm.hpp"
#include "moss/loader.hpp"
#include "runtime/backend.hpp"
#include "runtime/graph.hpp"
#include "runtime/lru_cache.hpp"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace starling::ggml;

namespace {

// Used VRAM on the active GPU, in bytes (total - free), via ggml's device
// registry (cudaMemGetInfo-equivalent; not CUDA-header-coupled). 0 if no GPU.
size_t gpu_used_bytes() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) return 0;
    size_t free_b = 0, total_b = 0;
    ggml_backend_dev_memory(dev, &free_b, &total_b);
    return (total_b > free_b) ? (total_b - free_b) : 0;
}

double mib(size_t bytes) { return (double)bytes / (1024.0 * 1024.0); }

// Build a synthetic MOSS mel of T frames (128 mels). Finite, small non-zero
// values; magnitude is irrelevant (the encoder normalizes). Only the SHAPE
// (which keys the cache) matters for this test.
moss::MelFeatures synth_mel(int64_t T) {
    moss::MelFeatures mel;
    mel.n_mels = 128;
    mel.n_frames = T;
    mel.data.resize((size_t)128 * T);
    for (size_t i = 0; i < mel.data.size(); ++i)
        mel.data[i] = ggml_fp32_to_bf16(1e-3f * (float)((i % 13) - 6));
    return mel;
}

// Encoder+adapter phase: N distinct mel lengths through encode_audio_and_adapt.
// Returns 0 on success. Records cache size + VRAM per step into the out-vectors.
// `first_out` (if non-null) receives the byte-exact output of the FIRST shape, for
// the eviction+rebuild correctness check.
int run_encoder_phase(moss::MossModel& m, int cap, int n,
                      std::vector<size_t>& sizes, std::vector<size_t>& vrams,
                      std::vector<float>* first_out, std::string& e) {
    sizes.assign((size_t)n, 0);
    vrams.assign((size_t)n, 0);
    // All lengths share chunk count C=11 (T in [1001,1012] -> (T+99)/100 == 11)
    // so the per-shape gallocr sizes are nearly identical: eviction past
    // capacity then reclaims ~exactly as much as the new entry allocates, giving
    // a clean plateau (size-neutral churn) instead of a monotonic-size drift
    // that would mask the bound. Distinct tail (T%100) -> distinct cache keys.
    for (int i = 0; i < n; ++i) {
        int64_t T = 1001 + i;  // C=11, tail = 1..n  (distinct keys)
        moss::MelFeatures mel = synth_mel(T);
        moss::AudioEncoding out;
        if (!moss::encode_audio_and_adapt(m, mel, out, e)) {
            e = "encode_audio_and_adapt failed at i=" + std::to_string(i) +
                " (T=" + std::to_string(T) + "): " + e;
            return 1;
        }
        if (i == 0 && first_out) *first_out = out.data;
        sizes[(size_t)i] = moss::encoder_replay_cache_size(m);
        vrams[(size_t)i] = gpu_used_bytes();
        std::printf("[enc] i=%2d T=%lld cache_size=%zu vram_used=%.1fMiB\n",
                    i, (long long)T, sizes[(size_t)i], mib(vrams[(size_t)i]));
    }
    return 0;
}

int run_prefill_phase(moss::MossModel& m, int cap, int n,
                      std::vector<size_t>& sizes, std::string& e) {
    sizes.assign((size_t)n, 0);
    const int64_t hidden = m.config.llm.hidden;
    for (int i = 0; i < n; ++i) {
        int64_t S = 16 + (int64_t)i * 4;  // distinct prompt lengths
        moss::InputsEmbeds in;
        in.width = hidden;
        in.n_tokens = S;
        in.data.resize((size_t)S * (size_t)hidden);
        for (size_t z = 0; z < in.data.size(); ++z)
            in.data[z] = 1e-3f * (float)((z % 17) - 8);
        moss::PrefillResult pr;
        // max_cache_len must be >= S; the model's max_cache (2048) covers it.
        if (!moss::llm_prefill(m, in, (int32_t)m.config.llm.max_cache, pr, e)) {
            e = "llm_prefill failed at i=" + std::to_string(i) +
                " (S=" + std::to_string(S) + "): " + e;
            return 1;
        }
        sizes[(size_t)i] = moss::prefill_replay_cache_size(m);
        std::printf("[pfx] i=%2d S=%lld cache_size=%zu\n",
                    i, (long long)S, sizes[(size_t)i]);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    // Force a SMALL capacity so the bound (and the plateau) are obvious and the
    // test is fast. Read by each cache at its first construction (lazy), so the
    // env must be set before the first encode/prefill. Capacity 4 is well below
    // the default 16 (also valid); the leaderboard bench covers the default.
    const int cap = 4;
    ::setenv("STARLING_REPLAY_CACHE_SIZE", "4", 1);
    const int n_enc = 3 * cap;   // 12 distinct encoder shapes (> cap)
    const int n_pfx = 2 * cap;   // 8 distinct prefill shapes (> cap)

    std::string root = argc > 1 ? argv[1] : ".";
    std::string model = root + "/models/moss-transcribe-preview-2b-bf16-exact.gguf";

    moss::MossModel m;
    std::string e;
    if (!m.load(model.c_str(), e)) {
        std::fprintf(stderr, "load: %s\n", e.c_str());
        return 2;
    }

    const bool gpu = global_backend().is_gpu();
    std::printf("replay_cache_lru_test: device=%s cap=%d n_enc=%d n_pfx=%d\n",
                global_backend().device_name(), cap, n_enc, n_pfx);
    if (!gpu) {
        // The captured-graph cache path is GPU-only; on CPU the one-shot build
        // allocates+frees per call and there is no cache to bound, so this bug
        // class cannot occur. Vacuous pass on CPU (regressions are caught on
        // CUDA, where the bug lives).
        std::printf("GPU cache path inactive on this backend; LRU test is a no-op pass.\n");
        shutdown_backend();
        return 0;
    }

    int rc = 0;

    // ---------- Encoder+adapter cache ----------
    {
        std::vector<size_t> enc_sizes, enc_vrams;
        std::vector<float> first_out;  // byte-exact output of shape i=0
        if (int r = run_encoder_phase(m, cap, n_enc, enc_sizes, enc_vrams, &first_out, e)) {
            std::fprintf(stderr, "[enc] FAIL: %s\n", e.c_str());
            shutdown_backend();
            return r;
        }

        // (0) Eviction+rebuild correctness: shape i=0 was captured first, then
        //     evicted once the cache filled (cap entries). Re-encode it -> the
        //     cache rebuilds it from scratch. The output MUST be byte-identical
        //     to the first capture: same weights + same graph build code =>
        //     deterministic. A non-identical result would mean eviction left
        //     stale state (e.g. a shared gallocr) corrupting a rebuild -- the
        //     subtle correctness hazard the plan flags. The LruCache gives each
        //     entry its own ReplayGraph (own private gallocr), so this holds.
        {
            moss::MelFeatures mel0 = synth_mel(1001);
            moss::AudioEncoding rebuilt;
            if (!moss::encode_audio_and_adapt(m, mel0, rebuilt, e)) {
                std::fprintf(stderr, "[enc] FAIL: rebuild encode failed: %s\n", e.c_str());
                shutdown_backend();
                return 1;
            }
            if (rebuilt.data.size() != first_out.size() ||
                !std::equal(rebuilt.data.begin(), rebuilt.data.end(),
                            first_out.begin())) {
                std::fprintf(stderr,
                    "[enc] FAIL: eviction+rebuild NOT byte-identical "
                    "(first=%zu floats, rebuilt=%zu floats) -- stale state after eviction\n",
                    first_out.size(), rebuilt.data.size());
                rc = 1;
            } else {
                std::printf("[enc] OK: evicted shape rebuild is byte-identical (%zu floats)\n",
                            first_out.size());
            }
        }

        // (1a) Hard bound: cache size never exceeds capacity.
        for (int i = 0; i < n_enc; ++i) {
            if ((int)enc_sizes[(size_t)i] > cap) {
                std::fprintf(stderr,
                    "[enc] FAIL: cache size %zu > cap %d at i=%d (unbounded growth)\n",
                    enc_sizes[(size_t)i], cap, i);
                rc = 1;
            }
        }
        // (1b) Saturation: once full, size stays exactly at capacity for every
        //      subsequent distinct shape (a miss must evict one, not grow).
        for (int i = cap; i < n_enc; ++i) {
            if ((int)enc_sizes[(size_t)i] != cap) {
                std::fprintf(stderr,
                    "[enc] FAIL: expected cache saturated at cap=%d after fill, got %zu at i=%d\n",
                    cap, enc_sizes[(size_t)i], i);
                rc = 1;
            }
        }

        // (2) VRAM plateau: marginal VRAM cost per new shape in the EVICTION
        //     phase must be a small fraction of the cost in the FILLING phase.
        //     Filling (steps 1..cap-1): each new shape adds one gallocr, no
        //     eviction. Eviction (steps cap..n-1): each new shape reclaims ~one
        //     old one, so the marginal cost collapses. Without LRU the eviction
        //     phase would grow at the SAME rate as the filling phase (linear).
        const size_t fill_growth = enc_vrams[(size_t)cap - 1] - enc_vrams[0];
        const size_t evict_growth = enc_vrams[(size_t)n_enc - 1] - enc_vrams[(size_t)cap - 1];
        const int fill_steps = cap - 1;          // deltas within the filling phase
        const int evict_steps = n_enc - cap;     // deltas within the eviction phase
        const double m_fill = fill_steps > 0 ? (double)fill_growth / fill_steps : 0.0;
        const double m_evict = evict_steps > 0 ? (double)evict_growth / evict_steps : 0.0;
        std::printf("[enc] VRAM: fill %.1fMiB (%.2fMiB/step over %d steps); "
                    "evict %.1fMiB (%.2fMiB/step over %d steps)\n",
                    mib(fill_growth), mib((size_t)m_fill), fill_steps,
                    mib(evict_growth), mib((size_t)m_evict), evict_steps);
        std::printf("[enc] ~per-entry gallocr cost (filling phase) = %.1fMiB\n",
                    mib((size_t)m_fill));
        // Eviction-phase marginal cost must be < 40% of filling-phase marginal
        // cost (plateau, not linear). 0.4 is generous vs measurement noise yet
        // cleanly separates bounded (~0) from unbounded (~1.0+).
        if (m_fill > 0 && !(m_evict < 0.4 * m_fill)) {
            std::fprintf(stderr,
                "[enc] FAIL: VRAM did not plateau: eviction marginal %.2fMiB/step >= "
                "40%% of fill marginal %.2fMiB/step (linear growth = unbounded cache)\n",
                mib((size_t)m_evict), mib((size_t)m_fill));
            rc = 1;
        } else if (m_fill > 0) {
            std::printf("[enc] OK: VRAM plateaued (evict marginal %.0f%% of fill marginal)\n",
                        100.0 * m_evict / m_fill);
        }
    }

    // ---------- Prefill cache ----------
    {
        std::vector<size_t> pfx_sizes;
        if (int r = run_prefill_phase(m, cap, n_pfx, pfx_sizes, e)) {
            std::fprintf(stderr, "[pfx] FAIL: %s\n", e.c_str());
            shutdown_backend();
            return r;
        }
        for (int i = 0; i < n_pfx; ++i) {
            if ((int)pfx_sizes[(size_t)i] > cap) {
                std::fprintf(stderr,
                    "[pfx] FAIL: prefill cache size %zu > cap %d at i=%d (unbounded growth)\n",
                    pfx_sizes[(size_t)i], cap, i);
                rc = 1;
            }
        }
        for (int i = cap; i < n_pfx; ++i) {
            if ((int)pfx_sizes[(size_t)i] != cap) {
                std::fprintf(stderr,
                    "[pfx] FAIL: expected prefill cache saturated at cap=%d, got %zu at i=%d\n",
                    cap, pfx_sizes[(size_t)i], i);
                rc = 1;
            }
        }
    }

    shutdown_backend();
    if (rc == 0) std::printf("replay_cache_lru_test: PASS (caches bounded + VRAM plateau)\n");
    return rc;
}
