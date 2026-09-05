// config.hpp — architecture constants for the Voxtral-Mini-4B-Realtime ggml port.
//
// mistralai/Voxtral-Mini-4B-Realtime-2602 is a Whisper-style causal audio encoder
// (32 layers, d_model 1280) feeding a downsample-4 projector into a
// Ministral-3-class text decoder (26 layers, hidden 3072, tied lm_head).
//
// Defaults match the baked GGUF metadata (scripts/convert_voxtral_gguf.py); the
// loader overrides them from `voxtral.*` KV. Invariants the loader enforces:
// encoder attention width 32*64 = 2048 (NOT the hidden 1280); llm hidden 3072
// with GQA q width 32*128 = 4096 and kv width 8*128 = 1024; heads % kv == 0.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace starling::ggml::voxtral {

struct FrontendConfig {
    uint32_t sample_rate = 16000, n_fft = 400, win_length = 400;
    uint32_t hop_length = 160, n_mels = 128, center = 1;
    uint32_t unit_samples = 1280, left_pad_tokens = 32, right_pad_tokens = 17;
    float mel_floor = 1e-10f, log_mel_max = 1.5f;
    // Stock: x=(log+4)/4 with floor at (max-8). offset/divisor reproduce it.
    float normalization_offset = 4.0f, normalization_divisor = 4.0f;
    float dynamic_range = 8.0f;
    std::string mel_scale = "slaney", log = "log10", output_dtype = "bf16";
};

struct EncoderConfig {
    uint32_t num_mel_bins = 128, n_layers = 32, d_model = 1280, n_heads = 32;
    uint32_t head_dim = 64, ffn_dim = 5120, sliding_window = 750;
    uint32_t conv_kernel = 3, conv_left_pad1 = 2, conv_left_pad2 = 1;
    uint32_t conv_stride2 = 2;
    float rope_theta = 1000000.0f, rms_norm_eps = 1e-5f;
};

struct ProjectorConfig {
    // Linear(input->output, no bias) GELU Linear(output->output, no bias).
    uint32_t input_size = 5120, output_size = 3072, downsample = 4;
    uint32_t mel_per_token = 8;
    std::string act = "gelu";
};

struct LlmConfig {
    uint32_t n_layers = 26, hidden = 3072, n_heads = 32, n_kv_heads = 8;
    uint32_t head_dim = 128, intermediate = 9216, vocab = 131072;
    uint32_t sliding_window = 8192, tied = 1, num_delay_tokens = 6;
    uint32_t time_embedding_dim = 3072, ada_bottleneck = 32, max_cache = 4096;
    float rope_theta = 1000000.0f, rms_norm_eps = 1e-5f;
    float time_embedding_theta = 10000.0f;
};

struct Config {
    FrontendConfig frontend;
    EncoderConfig encoder;
    ProjectorConfig projector;
    LlmConfig llm;
    int32_t bos_token_id = 1, eos_token_id = 2, pad_token_id = 11;
    int32_t streaming_pad_id = 32;
    uint32_t left_pad_tokens = 32, right_pad_tokens = 17;
    uint32_t max_new_tokens = 200;
    std::vector<int32_t> prompt_prefix;  // baked [1] + [32]*38
};

// Waveform length after the offline streaming-pad: ceil to whole audio tokens
// (1280 samples = 8 mel frames) plus the 32 left + 17 right pad tokens.
int64_t offline_padded_samples(int64_t n_samples);
// Mel-frame count for the offline (padded) waveform. The extractor's STFT runs
// center=True (1 + padded//hop raw frames) but drops the last TIME frame,
// canceling the +1: mel_T == padded//hop exactly, always a multiple of 8.
int64_t mel_frames(int64_t n_samples);
// Audio-token count from a mel length via the conv chain (exact): conv1 (k3 s1,
// left-pad 2) preserves length, conv2 (k3 s2, left-pad 1) halves, the projector
// groups by 4. Offline lengths give mel_T//8 with no remainder.
int64_t audio_token_count(int64_t mel_T);

} // namespace starling::ggml::voxtral
