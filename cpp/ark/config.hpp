// config.hpp — architecture constants for the ARK-ASR-3B ggml port.
//
// ARK-ASR-3B (AutoArk-AI/ARK-ASR-3B) is an audio-encoder + MLP-adapter +
// Qwen2.5 decoder ASR model. The audio path is a Whisper encoder (32 layers,
// d_model 1280, 20 heads, head_dim 64) that uses RoPE attention (use_rope=True,
// rope_dim=32, base=10000), followed by an ARK-added LayerNorm and an MLP
// adapter that merges every 4 frames into one Qwen2.5 token. The decoder is a
// Qwen2.5 trunk (36 layers, d2048, 16 query / 2 KV GQA, head_dim 128, SwiGLU
// intermediate 11008, RMSNorm eps 1e-6, RoPE theta 1e6, tied embeddings) WITHOUT
// the q_norm/k_norm of the Qwen3 family, and WITH q/k/v attention biases.
//
// Defaults match the baked GGUF metadata (scripts/convert_ark_gguf.py); the
// loader overrides them from `ark.*` KV. Single source of truth: config.py +
// the HF config.json for AutoArk-AI/ARK-ASR-3B.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace starling::ggml::ark {

struct FrontendConfig {
    uint32_t sample_rate = 16000, n_fft = 400, win_length = 400;
    uint32_t hop_length = 160, n_mels = 128, power = 2;
    uint32_t nb_max_frames = 3000, n_samples = 480000, chunk_length = 30;
    bool center = true;
    std::string pad_mode = "reflect", mel_scale = "slaney", mel_norm = "slaney";
    std::string log = "log", output_dtype = "bf16";
    float mel_floor = 1e-10f, dynamic_range = 8.0f;
    // Whisper: x=(log+4)/4. The C++ mel uses (v+offset)/divisor, so offset=4.
    float normalization_offset = 4.0f, normalization_divisor = 4.0f;
};

struct EncoderConfig {
    uint32_t num_mel_bins = 128, n_layers = 32, d_model = 1280, n_heads = 20;
    uint32_t head_dim = 64, ffn_dim = 5120, max_source_positions = 1500;
    uint32_t conv_kernel = 3, merge_factor = 4, use_rope = 1;
    uint32_t rope_dim = 32;
    float rope_base = 10000.0f, layer_norm_eps = 1e-5f;
};

struct AdapterConfig {
    // adapting Sequential: Linear(input->hidden) GELU Linear(hidden->output).
    uint32_t input_size = 5120, hidden_size = 4096, output_size = 2048;
    uint32_t merge_factor = 4;
    std::string act = "gelu";
};

struct LlmConfig {
    uint32_t n_layers = 36, hidden = 2048, n_heads = 16, n_kv_heads = 2;
    uint32_t head_dim = 128, intermediate = 11008, vocab = 151936;
    uint32_t max_position_embeddings = 32768, max_cache = 4096;
    float rope_theta = 1000000.0f, rms_norm_eps = 1e-6f;
    std::string rope_scaling = "none";
    bool tied_embeddings = true, has_qk_norm = false;
};

struct Config {
    FrontendConfig frontend;
    EncoderConfig encoder;
    AdapterConfig adapter;
    LlmConfig llm;
    int32_t audio_token_id = 151663;
    int32_t begin_audio_id = 151666, end_audio_id = 151667;
    int32_t user_id = 151665, assistant_id = 151668;
    int32_t pad_token_id = 151643, bos_token_id = 151643, eos_token_id = 151645;
    uint32_t max_new_tokens = 200;
    std::string default_instruction = "Transcribe the audio to text.";
    std::vector<int32_t> prompt_prefix, prompt_suffix;  // baked token-id arrays
    // Greedy suppression ids (sorted; the model card bans every special and
    // codec id except EOS from generation). Empty when the GGUF carries no
    // ark.bad_words_ids key, which disables suppression entirely.
    std::vector<int32_t> bad_words_ids;
};

// Number of LLM audio tokens implied by an (uncapped) mel-frame count.
// Mirrors ArkasrProcessor.calculate_audio_token_count:
//   max(((mel_frames + 1) // 2) // merge_factor, 1)
int64_t audio_token_count(int64_t mel_frames, uint32_t merge_factor);

} // namespace starling::ggml::ark
