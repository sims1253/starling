// BF16 CTC ties must select the first label, matching torch.argmax.
#include "granite/encoder.hpp"
#include "runtime/backend.hpp"
#include "runtime/graph.hpp"
#include "ggml.h"

#include <cstdio>
#include <vector>

int main() {
    using namespace starling::ggml;
    using namespace starling::ggml::granite;
    // Two columns: a tied nonzero max and an all-tied row. Raw ggml CPU
    // argmax selects 2 and 4 respectively; torch selects 1 and 0.
    const std::vector<float> logits = {1, 2, 2, 0, -1, 0, 0, 0, 0, 0};
    const std::vector<float> iota = {5, 4, 3, 2, 1};
    const float one = 1.0f;
    std::vector<float> out;
    (void) global_backend();
    const bool ok = run_graph([&](ggml_context* c) -> ggml_tensor* {
        int64_t ne[2] = {5, 2};
        ggml_tensor* input = graph_input_tensor(c, GGML_TYPE_F32, 2, ne,
                                                logits.data(), logits.size() * sizeof(float));
        return ggml_cast(c, ctc_argmax_first(c, input, iota, one), GGML_TYPE_F32);
    }, out);
    if (!ok || out != std::vector<float>{1, 0}) {
        std::fprintf(stderr, "Granite CTC first-index tie test failed: [%g, %g]\n",
                     out.size() > 0 ? out[0] : -1.0f,
                     out.size() > 1 ? out[1] : -1.0f);
        return 1;
    }
    std::puts("Granite CTC first-index tie test passed");
    return 0;
}
