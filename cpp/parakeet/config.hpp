// config.hpp — parakeet-tdt-0.6b-v3 model config, read from the GGUF.
//
// The KV keys are the verbatim NeMo/converter strings (parakeet.preprocessor.*,
// parakeet.decoder.*, etc.) — the same names the GGUF converter wrote and that
// parakeet.cpp reads. See cpp/parakeet/loader.cpp for the read.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace starling::ggml::parakeet {

struct Config {
    // ---- preprocessor / mel ----
    uint32_t sample_rate = 16000;
    uint32_t n_mels      = 128;
    uint32_t n_fft       = 512;
    uint32_t win_length  = 400;
    uint32_t hop_length  = 160;
    float    preemph     = 0.0f;       // 0.97 for parakeet-tdt
    float    mag_power   = 2.0f;       // |STFT|**mag_power
    std::string normalize = "per_feature";
    float    log_zero_guard = 0.0f;    // 2**-24 for parakeet-tdt

    // ---- encoder ----
    uint32_t d_model     = 0;          // 1024 (pre-projection)
    uint32_t n_layers    = 0;          // 24
    uint32_t pred_out    = 0;          // 640 (encoder_projector output)
    uint32_t n_heads     = 0;          // 8 (attention heads)
    uint32_t ff_dim      = 0;          // 4096 (feed-forward inner dim)
    uint32_t conv_kernel = 0;          // 9 (conformer conv module kernel)
    uint32_t subsampling_conv_channels = 0;  // 256 (subsampling conv width)
    std::string conv_norm_type;        // "batch_norm" (offline) or "layer_norm"
    bool xscaling = false;             // x *= sqrt(d_model) before layers (OFF)

    // ---- decoder (prediction net) ----
    uint32_t pred_hidden     = 0;      // 640
    uint32_t pred_rnn_layers = 0;      // 2

    // ---- joint ----
    uint32_t joint_hidden    = 0;      // 640
    std::string joint_activation;      // "relu"

    // ---- decoding / vocab ----
    uint32_t max_symbols = 10;
    uint32_t vocab_size  = 0;          // 8193
    uint32_t blank_id    = 0;          // 8192
    std::vector<int32_t> tdt_durations;  // [0,1,2,3,4]

    // ---- tokenizer (SentencePiece pieces, verbatim from the GGUF) ----
    std::vector<std::string> tokenizer_pieces;
};

} // namespace starling::ggml::parakeet
