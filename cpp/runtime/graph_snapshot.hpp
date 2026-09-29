// Read-only ggml graph topology snapshot for issue #32 research.
#pragma once

#include <string>
#include <vector>

struct ggml_cgraph;
struct ggml_tensor;

namespace starling::ggml {

// Records graph structure and tensor metadata. Leaf values are not included,
// so a matching snapshot alone is never an equivalence proof.
std::string graph_snapshot_json(ggml_cgraph* graph, ggml_tensor* output,
                                const char* device,
                                const std::vector<ggml_tensor*>& captures,
                                const std::vector<ggml_tensor*>& side_effect_roots);

}  // namespace starling::ggml
