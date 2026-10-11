// audio_llm_engine.hpp — fast engine (all-Vulkan) for the audio-encoder +
// Qwen3-decoder models: MOSS-Transcribe and Qwen3-ASR (1.7B and 0.6B).
//
// Both share the Qwen3-Omni-style audio tower (three k3/s2 convs over
// 100-frame mel chunks, 13 tokens per chunk, windowed-attention layers,
// LayerNorm + two-layer GELU projection) and a Qwen3 trunk with per-head q/k
// norm and a tied lm_head. MOSS adds a SwiGLU adapter between the two;
// Qwen3-ASR feeds the projection straight into the decoder. The mel
// frontends differ and stay on the CPU (each family's own implementation).
//
// One recorded pass per mel shape covers the audio encoder (implicit-GEMM
// convs, windowed-attention layers), the adapter (MOSS) and the projection
// (written straight into the LLM's input rows — the audio embeddings never
// visit the host) and the LLM prefill, which ends with the first greedy
// token chosen on the device. Decoding then replays a recording of K
// whole-model steps whose positions, KV-cache slots, argmax, EOS/budget test
// and next-token embedding all live in a device state buffer: the CPU sleeps
// on one fence per K tokens, and the KV cache never leaves GPU memory.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace starling::ggml::moss { struct MossModel; }
namespace starling::ggml::qwen3 { struct Qwen3Model; }

namespace starling::fast {

class AudioLlmEngine {
public:
    static std::unique_ptr<AudioLlmEngine> create(const ggml::moss::MossModel& model, std::string& err);
    static std::unique_ptr<AudioLlmEngine> create(const ggml::qwen3::Qwen3Model& model, std::string& err);
    ~AudioLlmEngine();

    // Greedy transcription token ids (EOS included when reached, like the
    // ggml engines). At most `budget` new tokens; 0 means the model's
    // max_new_tokens. `eos` reports whether generation stopped on EOS.
    bool generate(const float* pcm, size_t n, uint32_t budget, std::vector<int32_t>& ids, bool& eos,
                  std::string& err);
    std::string describe() const;

    struct Impl;   // opaque; public only so the factories' helper can build one

private:
    AudioLlmEngine();
    std::unique_ptr<Impl> impl_;
};

} // namespace starling::fast
