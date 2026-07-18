// graph_builder.hpp — host-scratch pool + the thread-local routing that lets a
// graph build lambda reach the Backend/ReplayGraph driving the current compute.
//
// Model code builds ggml graphs by appending ops to a ggml_context and
// registering host-backed inputs via add_graph_input / graph_input_tensor.
// Those free helpers route to the active Backend (set thread-locally by
// Backend::compute / ReplayGraph's ctor) so the model code doesn't need a
// Backend reference threaded through every build function.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace starling::ggml {

// A pool of host float32/int32 scratch buffers with STABLE addresses. A
// ReplayGraph remembers the host pointers it was built against (input_host());
// keeping the pool alive for the ReplayGraph's lifetime lets the caller
// recompute input contents in place each call and re-upload via set_input
// without re-deriving shapes.
class GraphInputPool {
public:
    // Allocate (or reuse) a contiguous f32 buffer of `count` floats and return
    // a mutable pointer into it. The pointer stays valid for the pool's life
    // (or until the pool is cleared).
    float* alloc_f32(size_t count) {
        f32_.emplace_back(count, 0.0f);
        return f32_.back().data();
    }
    int32_t* alloc_i32(size_t count) {
        i32_.emplace_back(count, 0);
        return i32_.back().data();
    }
    // Borrow a raw byte buffer (e.g. for a constant table).
    std::byte* alloc_bytes(size_t nbytes) {
        bytes_.emplace_back(nbytes, std::byte{0});
        return bytes_.back().data();
    }

private:
    std::vector<std::vector<float>>   f32_;
    std::vector<std::vector<int32_t>> i32_;
    std::vector<std::vector<std::byte>> bytes_;
};

} // namespace starling::ggml
