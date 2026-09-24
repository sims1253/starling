// parakeet_engine.hpp — Parakeet-TDT fast engine (Vulkan encoder + CPU
// transducer decoder).
//
// The FastConformer encoder runs on the GPU as ONE recorded command buffer
// per mel length (replayed without host work): subsampling convs, 24
// conformer layers with fused epilogues (residual adds, GLU, SiLU, pos-bias
// q projections, layer-boundary double LayerNorm), and the joint encoder
// projection. The positional projections depend only on the relative
// distance, so one table per layer is computed once for the longest input
// seen and sliced for shorter ones. The TDT greedy loop runs on the CPU with
// int8 dot-product GEMVs and reuses the prediction network and its joint
// projection across blank steps.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace starling::ggml::parakeet { struct ParakeetModel; }

namespace starling::fast {

class ParakeetEngine {
public:
    // Builds the engine from a loaded (not device-realized) model. Returns
    // nullptr with `err` when Vulkan or a tensor type is unsupported.
    static std::unique_ptr<ParakeetEngine> create(const ggml::parakeet::ParakeetModel& model,
                                                  std::string& err);
    ~ParakeetEngine();

    // Full pipeline: mel -> encoder -> TDT greedy. `ids` includes blanks
    // (the same stream as the ggml engine's tdt_greedy).
    bool decode_ids(const float* pcm, size_t n, std::vector<int32_t>& ids, std::string& err);
    // Text of the last decode_ids-equivalent run.
    bool transcribe(const float* pcm, size_t n, std::string& text, std::string& err);
    // Encoder only: row-major [Tp][joint_hidden] joint-projected encoder output.
    bool encode(const float* pcm, size_t n, std::vector<float>& enc, int& Tp, std::string& err);

    std::string describe() const;

private:
    ParakeetEngine();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace starling::fast
