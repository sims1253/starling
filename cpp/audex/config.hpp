// config.hpp — architecture constants for the Nemotron-Labs-Audex-2B ggml
// port.
//
// nvidia/Nemotron-Labs-Audex-2B is a Qwen2AudioEncoder (whisper-large-v3
// shaped: 128-bin whisper log-mel, fixed 30 s clips = 3000 mel frames ->
// GELU Conv1d k3/s1/p1 + GELU Conv1d k3/s2/p1 over time -> LEARNED (1500,
// 1280) positional embedding -> 32 pre-norm layers, biased LayerNorms,
// biased q/v/out projections (k bias-free), Q pre-scaled by head_dim^-0.5,
// 20 heads x 64, FULL bidirectional attention (no mask -- padded tail
// frames attend like any other), erf-GELU FFN 1280 -> 5120 -> 1280 ->
// avg-pooler halving 1500 -> 750 -> final biased LayerNorm). The projector
// is RMSNorm(1280, eps 1e-5, F.rms_norm single-round) -> fc1 1280 -> 4096
// -> relu(x)^2 -> fc2 -> 2048 (all bias-free). The decoder is a
// Nemotron-Dense 2B trunk: 28 layers, bias-free GQA 16Q/8KV head_dim 128,
// hidden 2048, relu2 MLP (up -> relu^2 -> down, NO gate), UNTIED lm_head,
// RoPE theta 1e8, rms eps 1e-5 (F.rms_norm), vocab 205312, stock numerics.
//
// Every clip is exactly 750 audio tokens (the avg-pooler output) — FIXED,
// unlike qwen3's post-CNN length formula. Defaults match the baked GGUF
// metadata (scripts/convert_audex_gguf.py); the loader overrides them from
// `audex.*` KV. Single source of truth: src/starling/audex/config.py + the
// pinned HF checkpoint (revision 77b7e1a, checkpoint_folder_full).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace starling::ggml::audex {

struct FrontendConfig {
    uint32_t sample_rate = 16000, n_fft = 400, win_length = 400;
    uint32_t hop_length = 160, n_mels = 128, power = 2;
    uint32_t chunk_length = 30, n_samples = 480000;
    float mel_floor = 1e-10f, dynamic_range = 8.0f;
    float normalization_offset = 4.0f, normalization_divisor = 4.0f;
};

struct EncoderConfig {
    uint32_t n_mel = 128, hidden = 1280, n_layers = 32, n_heads = 20;
    uint32_t head_dim = 64, ffn_dim = 5120;
    uint32_t max_pos_emb = 1500, out_frames = 750;
    float layer_norm_eps = 1e-5f;
};

struct ProjectorConfig {
    uint32_t hidden = 1280, intermediate = 4096, output_dim = 2048;
    float norm_eps = 1e-5f;
};

struct LlmConfig {
    uint32_t n_layers = 28, hidden = 2048, n_heads = 16, n_kv_heads = 8;
    uint32_t head_dim = 128, intermediate = 9216, vocab = 205312;
    uint32_t max_position_embeddings = 131072, max_cache = 4096;
    float rope_theta = 100000000.0f, rms_norm_eps = 1e-5f;
    bool tied_embeddings = false, has_qk_norm = false;
};

struct Config {
    FrontendConfig frontend;
    EncoderConfig encoder;
    ProjectorConfig projector;
    LlmConfig llm;
    int32_t audio_token_id = 29;      // <so_embedding>
    int32_t pad_token_id = 0, eos_token_id = 11;  // eos = <|im_end|>
    int32_t sound_start_token_id = 30, sound_end_token_id = 31;
    uint32_t sound_embedding_size = 750;  // per 30 s clip (FIXED)
    uint32_t max_new_tokens = 200;
    double chunk_seconds = 30.0;
    std::vector<int32_t> prompt_prefix, prompt_suffix;  // baked token-id arrays
};

} // namespace starling::ggml::audex
