// moss_kstep_oob_test.cpp — permanent regression test for the Wave G K-step
// decode out-of-bounds KV-cache / RoPE access (heap-buffer-overflow class).
//
// Root cause (see plans/wave-g-moss-kstep-oob-fix.md): run_kstep computes a
// block of K decode steps in ONE captured graph before truncating to the real
// token budget on the host. When a block's remaining budget < K, the wasted
// tail steps still execute and their positions (past + j) run past max_cache,
// writing KV slots (set_rows) and reading RoPE rows (get_rows) out of bounds.
// The outer `n_tokens + max_new_tokens <= max_cache` check only bounds the LAST
// NEEDED position, not the wasted lookahead.
//
// This test reproduces the exact boundary: n_tokens + max_new_tokens == max_cache
// (== 2048). With the run_kstep tail-cap fix every block's step count is capped
// to the remaining budget, so every device index stays < max_cache and
// greedy_generate completes cleanly. On UNPATCHED code the final K-step block's
// wasted tail steps write past max_cache and a sticky CUDA illegal-memory-access
// surfaces during the graph readback -> greedy_generate returns false. eos=-1
// forces the decode to run every step up to the boundary (no early stop).
//
// Usage: moss_kstep_oob_test [repo_root]
//   (loads <repo_root>/models/moss-transcribe-preview-2b-bf16-exact.gguf)
// Exit 0 = pass, 1 = boundary failure, 2 = model load failure.

#include "moss/config.hpp"
#include "moss/llm.hpp"
#include "moss/loader.hpp"
#include "runtime/backend.hpp"
#include "runtime/graph.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace starling::ggml;

namespace {

// One boundary generation: n_tokens chosen so n_tokens + max_new_tokens ==
// max_cache exactly (the tightest trigger). Returns 0 on success.
int run_boundary(moss::MossModel& m, int K, int max_new_tokens, std::string& e) {
    const auto& lc = m.config.llm;
    const int64_t max_cache = lc.max_cache;
    const int64_t hidden = lc.hidden;
    const int64_t n_tokens = max_cache - max_new_tokens;  // sums to max_cache

    if (n_tokens <= 0 || n_tokens > max_cache) {
        e = "invalid n_tokens=" + std::to_string(n_tokens);
        return 1;
    }

    // Synthetic inputs_embeds [hidden, n_tokens]: deterministic small non-zero
    // values. Magnitude is irrelevant (rms-norm rescales); we only need finite
    // logits and (via eos=-1) no early stop so the decode runs to the boundary.
    moss::InputsEmbeds in;
    in.width = hidden;
    in.n_tokens = n_tokens;
    in.data.resize((size_t)n_tokens * (size_t)hidden);
    for (size_t i = 0; i < in.data.size(); ++i)
        in.data[i] = 1e-3f * (float)((i % 17) - 8);  // [-8e-3, 8e-3]

    moss::GenerateOptions op;
    op.max_new_tokens = max_new_tokens;
    op.max_cache_len = (int32_t)max_cache;
    op.eos_token_id = -1;  // never matches an argmax in [0, vocab) -> full decode

    moss::GenerateResult gr;
    // Core regression assertion: greedy_generate must NOT raise a CUDA error at
    // the cache boundary. On unpatched code the wasted tail steps' OOB device
    // access makes the graph readback fail here.
    if (!moss::greedy_generate(m, in, op, gr, e)) {
        e = "greedy_generate FAILED at the cache boundary: " + e;
        return 1;
    }

    // eos=-1 => exactly max_new_tokens tokens emitted (1 from prefill +
    // max_new_tokens-1 decode).
    if ((int)gr.ids.size() != max_new_tokens) {
        e = "expected " + std::to_string(max_new_tokens) + " tokens, got " +
            std::to_string(gr.ids.size());
        return 1;
    }
    // Defense in depth: a non-faulting OOB KV write could silently corrupt the
    // cache and surface as an out-of-range argmax.
    for (int32_t t : gr.ids) {
        if (t < 0 || t >= (int32_t)lc.vocab) {
            e = "out-of-range token id " + std::to_string(t) +
                " (vocab=" + std::to_string(lc.vocab) + ")";
            return 1;
        }
    }
    std::printf("[K=%d] OK: n_tokens=%lld + max_new_tokens=%d == max_cache=%lld; %zu tokens; no OOB\n",
                K, (long long)n_tokens, max_new_tokens, (long long)max_cache, gr.ids.size());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::string root = argc > 1 ? argv[1] : ".";
    std::string model = root + "/models/moss-transcribe-preview-2b-bf16-exact.gguf";

    moss::MossModel m;
    std::string e;
    if (!m.load(model.c_str(), e)) {
        std::fprintf(stderr, "load: %s\n", e.c_str());
        return 2;
    }

    const bool gpu = global_backend().is_gpu();
    const bool kstep_active = gpu && std::getenv("STARLING_MOSS_NOKSTEP") == nullptr;
    std::printf("moss_kstep_oob_test: device=%s kstep_active=%d max_cache=%u\n",
                global_backend().device_name(), (int)kstep_active, m.config.llm.max_cache);
    if (!kstep_active) {
        // The K-step path is GPU-only. On CPU the single-step decode is used
        // (one position per loop iteration, naturally bounded, no wasted
        // lookahead) so this bug class cannot occur; the boundary test is a
        // vacuous pass there. It catches regressions on CUDA, where the bug lives.
        std::printf("K-step path inactive on this backend; boundary test is a no-op pass.\n");
        shutdown_backend();
        return 0;
    }

    // Two boundary cases, both with n_tokens + max_new_tokens == max_cache (2048)
    // and n_tokens=2038 (one captured prefill graph, reused across both):
    //   * K=4 (default), max_new_tokens=10 -> decode 9 steps in blocks [4,4,1];
    //     unpatched final block (K=4) writes slots 2046..2049 -> OOB at 2048,2049.
    //   * K=8 (max),     max_new_tokens=10 -> decode 9 steps in blocks [8,1];
    //     unpatched final block (K=8) writes slots 2046..2053 -> OOB at 2048..2053
    //     (the largest overreach, most likely to fault -> strongest gate).
    struct Case { int K; int mnt; };
    const Case cases[] = {{4, 10}, {8, 10}};
    int rc = 0;
    for (const auto& c : cases) {
        char kv[16];
        std::snprintf(kv, sizeof(kv), "%d", c.K);
        ::setenv("STARLING_MOSS_KSTEP", kv, 1);  // moss_kstep_K() reads this each call
        std::string err;
        int r = run_boundary(m, c.K, c.mnt, err);
        if (r != 0) {
            std::fprintf(stderr, "[K=%d] FAIL: %s\n", c.K, err.c_str());
            rc = r;
        }
    }

    shutdown_backend();
    if (rc == 0) std::printf("moss_kstep_oob_test: PASS (all boundary cases clean)\n");
    return rc;
}
