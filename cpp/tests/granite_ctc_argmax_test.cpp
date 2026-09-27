// BF16 CTC ties must select the first label, matching torch.argmax.
#include "granite/encoder.hpp"
#include "runtime/backend.hpp"
#include "runtime/graph.hpp"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    using namespace starling::ggml;
    using namespace starling::ggml::granite;
    // Raw ggml CPU argmax picks the last tie. The final three columns catch
    // threshold masks that treat a distinct near-zero BF16 value as a tie.
    const float tiny = std::ldexp(1.0f, -133);  // smallest BF16 subnormal
    if (ggml_bf16_to_fp32(ggml_fp32_to_bf16(tiny)) != tiny) return 2;
    const std::vector<float> values = {
        1, 2, 2, 0, -1,         // nonzero top tie -> 1
        0, 0, 0, 0, 0,          // all-label tie -> 0
        0, tiny, tiny, 0, 0,    // near-zero top tie -> 1
        0, tiny, 2*tiny, 0, 0,  // distinct subnormal max -> 2
        -tiny, 0, 0, -tiny, -tiny, // zero top tie -> 1
    };
    std::vector<ggml_bf16_t> logits;
    for (float v : values) logits.push_back(ggml_fp32_to_bf16(v));
    const std::vector<float> iota = {5, 4, 3, 2, 1};
    const float one = 1.0f;
    std::vector<float> out;
    (void) global_backend();
    const bool ok = run_graph([&](ggml_context* c) -> ggml_tensor* {
        int64_t ne[2] = {5, 5};
        ggml_tensor* input = graph_input_tensor(c, GGML_TYPE_BF16, 2, ne,
                                                logits.data(), logits.size() * sizeof(ggml_bf16_t));
        return ggml_cast(c, ctc_argmax_first(c, input, iota, one), GGML_TYPE_F32);
    }, out);
    if (!ok || out != std::vector<float>{1, 0, 1, 2, 1}) {
        std::fprintf(stderr, "Granite CTC first-index tie test failed:");
        for (float v : out) std::fprintf(stderr, " %g", v);
        std::fprintf(stderr, "\n");
        return 1;
    }
    std::puts("Granite CTC first-index tie test passed");
    return 0;
}
