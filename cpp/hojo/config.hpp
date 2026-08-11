// config.hpp — architecture constants for the HojoAI/Hojo-ASR-V1 ggml port.
//
// Hojo-ASR-V1 = Whisper-large-v3 mel -> Qwen3-Omni audio tower -> WeNet
// Conformer bottleneck -> LayerNorm(2560) -> Qwen3-4B decoder (beam-4).
//
// The audio path:
//   log-mel[128] -> reshape (1,1,T,128) -> 3x Conv2d downsample (k3/s2/p1,
//     GELU between each: conv2d1 [480,1,3,3], conv2d2/conv2d3 [480,480,3,3]) ->
//     flatten freq (7680=480*16) -> conv_out Linear [1280,7680] (no bias) ->
//     add computed SinusoidsPositionEmbedding (sin/cos concat over 1280 ch) ->
//     32 Qwen3OmniMoeAudioEncoderLayer (pre-norm LayerNorm -> MHA 20 heads
//     head_dim 64 scaling 64^-0.5, q/k/v/out [1280,1280] WITH bias, bidirectional
//     -> residual -> final_layer_norm LayerNorm -> fc1 [5120,1280] GELU
//     fc2 [1280,5120] -> residual) -> ln_post LayerNorm -> proj1 [1280,1280]
//     GELU -> proj2 [2048,1280]. Output [n_speech, 2048].
//
// The tower runs the conv2d front-end over mel chunked into 200-frame windows
// (n_window*2=100 per window, conv_chunksize=500 windows batched), then runs the
// 32 transformer layers over the FULL packed sequence with a block-diagonal
// 4D attention mask built from cu_seqlens (bidirectional within each window).
//
// The bottleneck (WeNet Conformer, LinearNoSubsampling + RelPositionalEncoding):
//   Linear(2048->2560) + LayerNorm -> + RelPosEnc pe[1,5000,2560] (x scaled by
//     sqrt(2560)) -> 2 ConformerEncoderLayer (macaron FFN 0.5 -> rel-pos MHA
//     (4 heads head_dim 640, pos_bias_u/v, NO rel_shift) -> conv module
//     (pointwise1->GLU->depthwise k15 -> BatchNorm1d(inference fold) -> Swish ->
//     pointwise2) -> FFN 0.5 -> norm_final) -> after_norm LayerNorm.
//   Then ln_speech LayerNorm(2560).
//
// The decoder is a Qwen3-4B trunk (36 layers, d2560, GQA 32/8, head_dim 128,
// SwiGLU intermediate 9728, RMSNorm eps 1e-6, rope_theta 5e6, SEPARATE lm_head)
// WITH qk_norm and WITHOUT attention biases. Decode is beam-4 with
// repetition_penalty=2.0, length_penalty=1, eos=151645.
//
// Defaults match the baked GGUF metadata (scripts/convert_hojo_gguf.py); the
// loader overrides them from `hojo.*` KV. Single source of truth:
// hojo-asr-v1.tensors.json + the hojo_asr package forward path.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace starling::ggml::hojo {

struct FrontendConfig {
    uint32_t sample_rate = 16000, n_fft = 400, win_length = 400;
    uint32_t hop_length = 160, n_mels = 128, power = 2;
    uint32_t nb_max_frames = 3000, n_samples = 480000, chunk_length = 30;
    std::string pad_mode = "reflect", mel_scale = "slaney", mel_norm = "slaney";
    std::string log = "log";
    double mel_floor = 1e-10;
    double normalization_offset = 4.0, normalization_divisor = 4.0;
    double dynamic_range = 8.0;
};

struct TowerConfig {
    // Qwen3-Omni audio tower.
    uint32_t num_mel_bins = 128, d_model = 1280, encoder_layers = 32;
    uint32_t encoder_attention_heads = 20, head_dim = 64, encoder_ffn_dim = 5120;
    uint32_t downsample_hidden_size = 480, output_dim = 2048;
    uint32_t max_source_positions = 1500, conv_kernel = 3;
    double layer_norm_eps = 1e-5;
    // Conv chunking / windowing (hojo_asr_model + omni config).
    uint32_t n_window = 1500, n_window_infer = 3000, conv_chunksize = 500;
    std::string activation_function = "gelu";
};

struct BottleneckConfig {
    // WeNet Conformer bottleneck.
    uint32_t input_size = 2048, output_size = 2560, linear_units = 640;
    uint32_t num_blocks = 2, attention_heads = 4, cnn_module_kernel = 15;
    uint32_t max_len = 5000;
    double norm_eps = 1e-5;
    std::string input_layer = "linear", pos_enc_layer_type = "rel_pos";
    std::string selfattention_layer_type = "rel_selfattn";
    std::string activation_type = "swish", cnn_module_norm = "batch_norm";
    bool macaron_style = true, normalize_before = true;
};

struct LlmConfig {
    // Qwen3-4B decoder.
    uint32_t n_layers = 36, hidden = 2560, n_heads = 32, n_kv_heads = 8;
    uint32_t head_dim = 128, intermediate = 9728, vocab = 151670;
    uint32_t max_position_embeddings = 262144, max_cache = 4096;
    double rope_theta = 5000000.0, rms_norm_eps = 1e-6;
    std::string rope_scaling = "none";
    bool tied_embeddings = false, has_qk_norm = true;
};

struct DecodeConfig {
    // Beam search (hojo config.yaml generate).
    uint32_t num_beams = 4, min_length = 1, max_new_tokens = 200;
    double repetition_penalty = 2.0, length_penalty = 1.0;
    double temperature = 1.0, top_p = 0.9;
    bool do_sample = false;
};

struct Config {
    FrontendConfig frontend;
    TowerConfig tower;
    BottleneckConfig bottleneck;
    LlmConfig llm;
    DecodeConfig decode;
    int32_t bos_token_id = 151644;  // <|im_start|>
    int32_t eos_token_id = 151645;  // <|im_end|>
    int32_t pad_token_id = 151645;
};

// Hojo tower output length: `_get_feat_extract_output_lengths` (3 stride-2
// conv2d over mel frames), then a per-window re-aggregation. For a single
// window of `mel_T` frames (mel_T <= n_window*2=3000), the conv output length
// is computed by repeated ((L-1)//2+1) floor-div. The window/chunk packing is
// handled in audio_tower.cpp; this helper gives the raw conv-downsampled count
// for one window of `mel_frames`.
int64_t tower_conv_output_length(int64_t mel_frames);

} // namespace starling::ggml::hojo
