// config.hpp — architecture constants for the Qwen3-ASR-1.7B ggml port.
//
// Qwen/Qwen3-ASR-1.7B-hf is a windowed-attention conv encoder + 2-layer MLP
// projector + Qwen3 decoder. The frontend computes a 128-bin whisper-style
// log-mel (torch.stft n_fft=400, hop=160, periodic hann, center/reflect,
// slaney filterbank; log10, max-clamp 8, (x+4)/4), drops the trailing STFT
// frame (T = S/H) and right-pads the mel axis with ZEROS to a multiple of
// 2*n_window = 100 frames; clips under min_length samples are zero-padded
// first. The encoder chunks the mel into 100-frame blocks, runs three
// GELU Conv2d k3/s2/p1 layers (480 channels: freq 128->64->32->16, time
// 100->50->25->13) + a bias-free Linear(7680 -> 1024), adds a (13, 1024)
// sinusoidal position table, keeps the first post-CNN-length rows per chunk
// (triple ceil-halving: 13 per full chunk) and runs 24 windowed-attention
// layers (biased MHA 16 heads x 64, full attention within n_window_infer
// windows of 8 chunks = 104 packed rows) + final LayerNorm. The projector is
// Linear(1024 -> 1024) + erf GELU + Linear(1024 -> 2048). The decoder is a
// bias-free Qwen3 trunk WITH per-head q_norm/k_norm and a TIED lm_head, stock
// numerics (no multipliers) — the shared decode stack's moss variant.
//
// Defaults match the baked GGUF metadata (scripts/convert_qwen3_gguf.py); the
// loader overrides them from `qwen3.*` KV. Single source of truth:
// src/starling/qwen3/config.py + the HF config.json for Qwen3-ASR-1.7B-hf.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace starling::ggml::qwen3 {

struct FrontendConfig {
    uint32_t sample_rate = 16000, n_fft = 400, win_length = 400;
    uint32_t hop_length = 160, n_mels = 128, power = 2;
    uint32_t chunk_length = 30, min_length = 8000, n_window = 50;
    uint32_t n_samples = 0;  // unused (no waveform cap)
    float mel_floor = 1e-10f, dynamic_range = 8.0f;
    float normalization_offset = 4.0f, normalization_divisor = 4.0f;
};

struct EncoderConfig {
    uint32_t n_mel = 128, hidden = 1024, n_layers = 24, n_heads = 16;
    uint32_t head_dim = 64, ffn_dim = 4096, downsample_hidden = 480;
    uint32_t n_window = 50, n_window_infer = 800;
    uint32_t max_pos_emb = 13, output_dim = 2048;
    float layer_norm_eps = 1e-5f;
};

struct ProjectorConfig {
    uint32_t hidden = 1024, output_dim = 2048;
};

struct LlmConfig {
    uint32_t n_layers = 28, hidden = 2048, n_heads = 16, n_kv_heads = 8;
    uint32_t head_dim = 128, intermediate = 6144, vocab = 151936;
    uint32_t max_position_embeddings = 65536, max_cache = 4096;
    float rope_theta = 1000000.0f, rms_norm_eps = 1e-6f;
    bool tied_embeddings = true, has_qk_norm = true;
};

struct Config {
    FrontendConfig frontend;
    EncoderConfig encoder;
    ProjectorConfig projector;
    LlmConfig llm;
    int32_t audio_token_id = 151676;
    int32_t pad_token_id = 151645, eos_token_id = 151645;
    uint32_t max_new_tokens = 200;
    double chunk_seconds = 30.0;
    std::vector<int32_t> prompt_prefix, prompt_suffix;  // baked token-id arrays
};

// Audio-token count implied by a raw sample count, mirroring
// Qwen3ASRProcessor._get_audio_token_length over the valid frame count
// T = max(S, min_length)/hop: per chunk of 100 frames the conv stack keeps
// ceil-halved triple (13 per full chunk), so N = 13*(T/100) + c3(T%100).
int64_t audio_token_count(int64_t n_samples, const Config& c);

} // namespace starling::ggml::qwen3
