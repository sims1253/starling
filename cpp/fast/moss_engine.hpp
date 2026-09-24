// moss_engine.hpp — MOSS-Transcribe fast engine (all-Vulkan).
//
// One recorded pass per mel shape covers the audio encoder (implicit-GEMM
// convs, 32 windowed-attention layers), the adapter (written straight into
// the LLM's input rows — the audio embeddings never visit the host) and the
// LLM prefill, which ends with the first greedy token chosen on the device.
// Decoding then replays a recording of K whole-model steps whose positions,
// KV-cache slots, argmax, EOS test and next-token embedding all live in a
// device state buffer: the CPU sleeps on one fence per K tokens, and the KV
// cache never leaves GPU memory.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace starling::ggml::moss { struct MossModel; }

namespace starling::fast {

class MossEngine {
public:
    static std::unique_ptr<MossEngine> create(const ggml::moss::MossModel& model, std::string& err);
    ~MossEngine();

    // Greedy transcription token ids (EOS included when reached, like the
    // ggml engine). `eos` reports whether generation stopped on EOS.
    bool generate(const float* pcm, size_t n, std::vector<int32_t>& ids, bool& eos,
                  std::string& err);
    std::string describe() const;

private:
    MossEngine();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace starling::fast
