// tdt_slice_argmax_test.cpp — the parakeet TDT in-graph argmax split
// (parakeet::tdt_slice_argmax, shared by the fused step graph and the K-step
// multistep graph).
//
// Windows Vulkan on an RTX 5090 (#57 verification) aborted on the first
// transcription: ggml-vulkan's ARGMAX asserted in init_pushconst_tensor_offsets
// because its source, the duration slice of the joint logits, started at byte
// 8193 * 4 = 32772 — aligned to RADV's 4-byte minStorageBufferOffsetAlignment
// (where Linux Vulkan was validated) but not to NVIDIA's 16. supports_op does
// not check offsets, so nothing rejected the node before execution.
//
//   (1) layout: every ARGMAX source starts at an offset aligned to 256 bytes,
//       the Vulkan spec's upper bound for minStorageBufferOffsetAlignment, so
//       no conforming device can see it misaligned (fails on the old graph:
//       the duration view sat at view_offs 32772);
//   (2) values: token and duration indices match a host argmax over the
//       same slices at the v3 dimensions (8193 tokens + 5 durations),
//       including a duration maximum larger than every token logit.
//
// Runs on the global backend (CPU in CI). Exit 0 = pass, 1 = failure.

#include "parakeet/joint.hpp"
#include "runtime/backend.hpp"
#include "runtime/graph.hpp"
#include "ggml.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace starling::ggml;

namespace {

int g_failures = 0;

void check(bool cond, const std::string& what) {
    if (!cond) {
        std::printf("FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

constexpr int kTokenCount = 8193;  // parakeet-tdt-0.6b-v3: vocab 8192 + blank
constexpr int kNumDur = 5;         // tdt_durations {0, 1, 2, 3, 4}
constexpr size_t kMaxVulkanOffsetAlignment = 256;

size_t source_offset(const ggml_tensor* argmax) {
    const ggml_tensor* src = argmax->src[0];
    return src->view_src ? src->view_offs : 0;
}

void run_case(const char* name, const std::vector<float>& logits,
              int want_tok, int want_dur) {
    for (int which = 0; which < 2; ++which) {
        std::vector<float> out;
        const bool ok = run_graph([&](ggml_context* c) -> ggml_tensor* {
            int64_t ne[1] = { (int64_t)logits.size() };
            ggml_tensor* y = graph_input_tensor(c, GGML_TYPE_F32, 1, ne,
                                                logits.data(), logits.size() * sizeof(float));
            const parakeet::TdtSliceArgmax amax =
                parakeet::tdt_slice_argmax(c, y, kTokenCount, kNumDur);
            for (const ggml_tensor* t : { amax.tok, amax.dur }) {
                check(t->op == GGML_OP_ARGMAX && t->type == GGML_TYPE_I32,
                      std::string(name) + ": split must end in i32 ARGMAX nodes");
                check(source_offset(t) % kMaxVulkanOffsetAlignment == 0,
                      std::string(name) + ": ARGMAX source at byte offset "
                      + std::to_string(source_offset(t))
                      + " is not 256-byte aligned (ggml-vulkan asserts)");
            }
            return ggml_cast(c, which == 0 ? amax.tok : amax.dur, GGML_TYPE_F32);
        }, out);
        const int want = which == 0 ? want_tok : want_dur;
        check(ok && out.size() == 1 && (int)out[0] == want,
              std::string(name) + (which == 0 ? ": token" : ": duration")
              + " argmax = " + (ok && !out.empty() ? std::to_string((int)out[0]) : "<failed>")
              + ", want " + std::to_string(want));
    }
}

} // namespace

int main() {
    (void)global_backend();
    const size_t n = (size_t)kTokenCount + kNumDur;

    // Token maximum mid-slice; duration maximum inside the duration slice.
    std::vector<float> a(n);
    for (size_t i = 0; i < n; ++i) a[i] = -1.0f - 0.001f * (float)(i % 97);
    a[4321] = 3.0f;
    a[(size_t)kTokenCount + 2] = 1.5f;
    run_case("mid-slice", a, 4321, 2);

    // Global maximum in the duration slice: the token argmax must not see it,
    // and the duration argmax must index from the slice start.
    std::vector<float> b(n, 0.0f);
    b[kTokenCount - 1] = 2.0f;                 // last token (blank)
    b[(size_t)kTokenCount + 4] = 9.0f;         // last duration
    b[(size_t)kTokenCount] = 8.0f;
    run_case("duration-global-max", b, kTokenCount - 1, 4);

    if (g_failures) {
        std::printf("tdt_slice_argmax_test: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("tdt_slice_argmax_test: all checks passed\n");
    return 0;
}
