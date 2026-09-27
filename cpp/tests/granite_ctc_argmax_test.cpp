// Match torch.softmax(bf16_logits.float()).argmax(-1), including F32 ties.
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
    // Oracle: torch.softmax(torch.tensor(values, dtype=torch.bfloat16)
    //                       .float(), dim=-1).argmax(dim=-1).
    // Raw ggml CPU argmax picks the last tie. The last three frames also
    // catch the case where distinct BF16 logits become equal after softmax.
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
    if (!ok || out != std::vector<float>{1, 0, 0, 0, 0}) {
        std::fprintf(stderr, "Granite CTC softmax argmax test failed:");
        for (float v : out) std::fprintf(stderr, " %g", v);
        std::fprintf(stderr, "\n");
        return 1;
    }
    // At the production vocabulary, a single BF16 subnormal logit also
    // rounds to the same F32 probability as every zero logit.
    std::vector<ggml_bf16_t> wide(100353, ggml_fp32_to_bf16(0.0f));
    wide[1] = ggml_fp32_to_bf16(tiny);
    std::vector<float> wide_iota(wide.size());
    for (size_t i = 0; i < wide.size(); ++i)
        wide_iota[i] = (float)(wide.size() - i);
    out.clear();
    const bool wide_ok = run_graph([&](ggml_context* c) -> ggml_tensor* {
        int64_t ne[2] = {(int64_t)wide.size(), 1};
        ggml_tensor* input = graph_input_tensor(c, GGML_TYPE_BF16, 2, ne,
                                                wide.data(), wide.size() * sizeof(ggml_bf16_t));
        return ggml_cast(c, ctc_argmax_first(c, input, wide_iota, one), GGML_TYPE_F32);
    }, out);
    if (!wide_ok || out != std::vector<float>{0}) {
        std::fprintf(stderr, "Granite CTC full-vocabulary softmax tie test failed\n");
        return 1;
    }
    std::puts("Granite CTC softmax argmax test passed");
    return 0;
}
