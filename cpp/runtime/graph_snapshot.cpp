#include "graph_snapshot.hpp"

#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace starling::ggml {
namespace {

void json_string(std::ostringstream& out, const char* value, size_t max_length = size_t(-1)) {
    out << '"';
    const auto* bytes = reinterpret_cast<const unsigned char*>(value);
    for (size_t i = 0; i < max_length && bytes[i]; ++i) {
        const unsigned char byte = bytes[i];
        if (byte == '"' || byte == '\\') out << '\\' << static_cast<char>(byte);
        else if (byte < 0x20) {
            char escaped[7];
            std::snprintf(escaped, sizeof(escaped), "\\u%04x", byte);
            out << escaped;
        } else out << static_cast<char>(byte);
    }
    out << '"';
}

void params_hex(std::ostringstream& out, const ggml_tensor* tensor) {
    static constexpr char digits[] = "0123456789abcdef";
    const auto* bytes = reinterpret_cast<const unsigned char*>(tensor->op_params);
    out << '"';
    for (size_t i = 0; i < GGML_MAX_OP_PARAMS; ++i) {
        out << digits[bytes[i] >> 4] << digits[bytes[i] & 15];
    }
    out << '"';
}

}  // namespace

std::string graph_snapshot_json(ggml_cgraph* graph, ggml_tensor* output,
                                const char* device,
                                const std::vector<ggml_tensor*>& captures,
                                const std::vector<ggml_tensor*>& side_effect_roots) {
    if (!graph || !output || !device)
        throw std::invalid_argument("graph snapshot needs graph, output and device");
    std::vector<ggml_tensor*> tensors;
    std::unordered_map<const ggml_tensor*, size_t> ids;
    auto add = [&](ggml_tensor* tensor) {
        if (!tensor || ids.find(tensor) != ids.end()) return;
        ids.emplace(tensor, tensors.size());
        tensors.push_back(tensor);
    };
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) add(ggml_graph_node(graph, i));
    add(output);
    for (ggml_tensor* capture : captures) add(capture);
    for (ggml_tensor* root : side_effect_roots) add(root);
    // Walk iteratively: real encoder graphs can contain long chains of views.
    // This also gives every shared leaf one ID without recursive stack growth.
    for (size_t i = 0; i < tensors.size(); ++i) {
        for (ggml_tensor* source : tensors[i]->src) add(source);
        add(tensors[i]->view_src);
    }

    std::ostringstream out;
    out << "{\"schema\":1,\"semantics\":\"not_encoded\",\"leaf_values\":\"not_exported\"";
    out << ",\"device\":";
    json_string(out, device);
    out << ",\"ggml_commit\":";
    json_string(out, ggml_commit());
    out << ",\"graph_nodes\":[";
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        if (i) out << ',';
        out << ids.at(ggml_graph_node(graph, i));
    }
    out << "],\"output\":" << ids.at(output) << ",\"captures\":[";
    for (size_t i = 0; i < captures.size(); ++i) {
        if (i) out << ',';
        out << ids.at(captures[i]);
    }
    out << "],\"side_effect_roots\":[";
    for (size_t i = 0; i < side_effect_roots.size(); ++i) {
        if (i) out << ',';
        out << ids.at(side_effect_roots[i]);
    }
    out << "],\"tensors\":[";
    for (size_t i = 0; i < tensors.size(); ++i) {
        const ggml_tensor* tensor = tensors[i];
        if (i) out << ',';
        out << "{\"id\":" << i << ",\"name\":";
        json_string(out, tensor->name, GGML_MAX_NAME);
        out << ",\"op\":";
        json_string(out, ggml_op_name(tensor->op));
        out << ",\"type\":";
        json_string(out, ggml_type_name(tensor->type));
        out << ",\"ne\":[";
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            if (d) out << ',';
            out << tensor->ne[d];
        }
        out << "],\"nb\":[";
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            if (d) out << ',';
            out << tensor->nb[d];
        }
        out << "],\"flags\":" << tensor->flags << ",\"op_params_hex\":";
        params_hex(out, tensor);
        out << ",\"src\":[";
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (s) out << ',';
            if (tensor->src[s]) out << ids.at(tensor->src[s]);
            else out << "null";
        }
        out << "],\"view_src\":";
        if (tensor->view_src) out << ids.at(tensor->view_src);
        else out << "null";
        out << ",\"view_offs\":" << tensor->view_offs;
        out << '}';
    }
    out << "]}\n";
    return out.str();
}

}  // namespace starling::ggml
