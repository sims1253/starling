#include "runtime/graph_snapshot.hpp"

#include "ggml.h"

#include <cstdio>
#include <string>

int main() {
    ggml_init_params params{ggml_tensor_overhead() * 8 + ggml_graph_overhead(),
                            nullptr, true};
    ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        std::printf("graph snapshot UTF-8: FAIL (ggml_init failed)\n");
        return 1;
    }
    ggml_tensor* input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_name(input, "r\xc3\xa9play_input");
    ggml_tensor* output = ggml_add(ctx, input, input);
    ggml_set_name(output, "bad-\xff");
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    const std::string json = starling::ggml::graph_snapshot_json(
        graph, output, "cpu", {}, {});
    ggml_free(ctx);
    const bool valid_utf8_preserved = json.find("r\xc3\xa9play_input") != std::string::npos;
    const bool invalid_byte_escaped = json.find("bad-\\u00ff") != std::string::npos;
    if (!valid_utf8_preserved) std::printf("valid UTF-8 name not preserved:\n%s\n", json.c_str());
    if (!invalid_byte_escaped) std::printf("invalid byte not escaped:\n%s\n", json.c_str());
    std::printf("graph snapshot UTF-8: %s\n",
                valid_utf8_preserved && invalid_byte_escaped ? "PASS" : "FAIL");
    return valid_utf8_preserved && invalid_byte_escaped ? 0 : 1;
}
