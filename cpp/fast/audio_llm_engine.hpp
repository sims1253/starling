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
//
// ARK-ASR (Whisper encoder with RoPE global attention, merge-by-4 GELU
// adapter, Qwen2.5 trunk with q/k/v biases and no q/k norm) runs whole on
// the device from the host's log-mel: conv stem, encoder layers and adapter
// fill the prompt's audio slots in the same recorded prefill.
// Granite-speech (granite-4.0 trunk with untied lm_head and muP-style
// multipliers) keeps its conformer + Q-Former on ggml: the host hands over
// their output rows and the prompt ids, and the engine runs the prompt
// embedding lookup, the prefill and the same on-device greedy loop (ARK
// can do the same with STARLING_FAST_ARK_HOST_ENCODER=1). The multipliers
// are folded into the repacked weights (residual -> o/down projections,
// embedding -> the input table) and the attention scale; the logit divisor
// cannot change a greedy argmax and is dropped.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace starling::ggml { class ModelLoader; }
namespace starling::ggml::moss { struct MossModel; }
namespace starling::ggml::qwen3 { struct Qwen3Model; }
namespace starling::ggml::ark { struct ArkModel; }
namespace starling::ggml::granite { struct GraniteModel; }

namespace starling::fast {

class AudioLlmEngine {
public:
    static std::unique_ptr<AudioLlmEngine> create(const ggml::moss::MossModel& model, std::string& err);
    static std::unique_ptr<AudioLlmEngine> create(const ggml::qwen3::Qwen3Model& model, std::string& err);
    // Decoder-only engines (the caller keeps running the audio encoder).
    static std::unique_ptr<AudioLlmEngine> create(const ggml::ark::ArkModel& model, std::string& err);
    static std::unique_ptr<AudioLlmEngine> create(const ggml::granite::GraniteModel& model, std::string& err);
    ~AudioLlmEngine();

    // Greedy transcription token ids (EOS included when reached, like the
    // ggml engines). At most `budget` new tokens; 0 means the model's
    // max_new_tokens. `eos` reports whether generation stopped on EOS.
    bool generate(const float* pcm, size_t n, uint32_t budget, std::vector<int32_t>& ids, bool& eos,
                  std::string& err);
    // Decoder-only engines: the prompt `prompt_ids` whose slots with
    // audio_mask != 0 take the encoder rows `audio` ([audio_rows][hidden]
    // f32) in order — bf16-rounded, and zero once the rows run out, like the
    // ggml engines' scatter (lib/embed_scatter.hpp).
    bool generate(const std::vector<int32_t>& prompt_ids, const std::vector<uint8_t>& audio_mask,
                  const float* audio, int64_t audio_rows, uint32_t budget, std::vector<int32_t>& ids,
                  bool& eos, std::string& err);
    // Whether the engine runs the audio encoder itself (MOSS, Qwen3; ARK and
    // Granite unless STARLING_FAST_{ARK,GRANITE}_HOST_ENCODER=1, where they
    // take the ggml encoder's rows through generate() above).
    bool encodes_audio() const;
    // ARK / Granite with the encoder on the device. `mel_bf16` holds bf16
    // bits: ARK's log-mel feature-major [n_mels][mel_T]; Granite's stacked
    // frames frame-major [mel_T][160]. The adapter / Q-Former rows fill the
    // (one contiguous run of) audio slots in order, zero past the last row.
    bool generate_from_mel(const std::vector<int32_t>& prompt_ids, const std::vector<uint8_t>& audio_mask,
                           const uint16_t* mel_bf16, int64_t mel_T, uint32_t budget,
                           std::vector<int32_t>& ids, bool& eos, std::string& err);
    std::string describe() const;

    struct Impl;   // opaque; public only so the factories' helper can build one

private:
    AudioLlmEngine();
    std::unique_ptr<Impl> impl_;
};

// Return the host pages of the loader's weights whose names start with one of
// `prefixes` ("llm.", and for ARK's on-device encoder "enc.", "adapter.") to
// the OS once the engine holds its own copy: the GGUF is read into anonymous
// memory, and on unified-memory phones the duplicate would cost those
// weights' size again. The tensors stay registered but read as zeros
// afterwards, so the caller must route that work through the engine.
// Returns the bytes released (0 where the platform has no madvise).
size_t release_host_weights(const ggml::ModelLoader& loader, const std::vector<std::string>& prefixes);

} // namespace starling::fast
