// config.hpp — architecture constants for the granite-speech-4.1-2b ggml port.
//
// ibm-granite/granite-speech-4.1-2b is a CTC-conformer encoder + BLIP2
// Q-Former projector + Granite-4.0-1b decoder. The encoder (16 blocks, hidden
// 1024, 8 heads x 128, block-local Shaw relative attention over 200-frame
// windows, depthwise-conv module with BatchNorm, self-conditioned mid CTC at
// 1-indexed block 8) consumes 160-dim mel frames (80-mel torchaudio frontend,
// odd-frame drop + consecutive-pair stack). The projector windows into 15-frame
// blocks, cross-attends with 3 learned queries (2 BERT-style qformer layers,
// erf GELU, LayerNorm eps 1e-12) and projects to the decoder's 2048 space,
// emitting window/downsample = 3 tokens per block. The decoder is a bias-free
// Qwen-family trunk WITHOUT q_norm/k_norm and an UNTIED lm_head, plus the
// Granite numerics (embedding x12.0, attention scale 0.0078125, residual x0.22,
// logits /8.0) carried by the shared decode stack's spec.
//
// Defaults match the baked GGUF metadata (scripts/convert_granite_gguf.py);
// the loader overrides them from `granite.*` KV. Single source of truth:
// src/starling/config.py + the HF config.json for granite-speech-4.1-2b.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace starling::ggml::granite {

struct FrontendConfig {
    uint32_t sample_rate = 16000, n_fft = 512, win_length = 400;
    uint32_t hop_length = 160, n_mels = 80, power = 2;
    uint32_t chunk_length = 30, n_samples = 0;  // n_samples unused (no cap)
    float mel_floor = 1e-10f, dynamic_range = 8.0f;
    // Granite normalizes max(x, mx-8)/4 + 1 == (x + 4)/4 bit-exactly in f32.
    float normalization_offset = 4.0f, normalization_divisor = 4.0f;
};

struct EncoderConfig {
    uint32_t input_dim = 160, hidden = 1024, n_layers = 16, n_heads = 8;
    uint32_t head_dim = 128, ffn_dim = 4096, conv_kernel = 15;
    uint32_t context_size = 200, max_pos_emb = 512, output_dim = 348;
    uint32_t mid_layer = 8;  // 1-indexed block after which the mid CTC fires
    float layer_norm_eps = 1e-5f;
    uint32_t conv_inner = 2048;  // hidden * conv_expansion_factor(2), post-GLU
};

struct ProjectorConfig {
    uint32_t window_size = 15, downsample_rate = 5, num_queries = 3;
    uint32_t hidden = 1024, qformer_layers = 2, qformer_heads = 16;
    uint32_t qformer_intermediate = 4096, output_dim = 2048;
    float layer_norm_eps = 1e-12f;
};

struct LlmConfig {
    uint32_t n_layers = 40, hidden = 2048, n_heads = 16, n_kv_heads = 4;
    uint32_t head_dim = 128, intermediate = 4096, vocab = 100353;
    uint32_t max_position_embeddings = 4096, max_cache = 640;
    float rope_theta = 10000.0f, rms_norm_eps = 1e-5f;
    // Granite numerics (consumed by the shared decode spec in llm.cpp).
    float attention_multiplier = 0.0078125f;
    float embedding_multiplier = 12.0f;
    float residual_multiplier = 0.22f;
    float logits_scaling = 8.0f;
    bool tied_embeddings = false, has_qk_norm = false;
};

struct Config {
    FrontendConfig frontend;
    EncoderConfig encoder;
    ProjectorConfig projector;
    LlmConfig llm;
    int32_t audio_token_id = 100352;
    int32_t pad_token_id = 100256, bos_token_id = 100257, eos_token_id = 100257;
    uint32_t max_new_tokens = 200;
    double chunk_seconds = 30.0;
    std::vector<int32_t> prompt_prefix, prompt_suffix;  // baked token-id arrays
};

// Audio-token count implied by a raw sample count, mirroring
// GraniteSpeechFeatureExtractor._get_num_audio_features:
//   mel = S/hop + 1; enc = mel/2 (odd drop + pair stack);
//   nblocks = ceil(enc / window); N = nblocks * (window/downsample).
int64_t audio_token_count(int64_t n_samples, const Config& c);

} // namespace starling::ggml::granite
