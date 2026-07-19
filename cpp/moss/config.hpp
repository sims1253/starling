#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace starling::ggml::moss {

struct FrontendConfig {
    uint32_t sample_rate = 16000, n_fft = 640, win_length = 640;
    uint32_t hop_length = 160, n_mels = 128, power = 2;
    bool center = true;
    std::string pad_mode = "reflect", mel_scale = "slaney", mel_norm = "slaney";
    std::string log = "log10", output_dtype = "bf16";
    float mel_floor = 1e-10f, dynamic_range = 8.0f;
    float normalization_offset = 4.0f, normalization_divisor = 4.0f;
};

struct EncoderConfig {
    uint32_t n_layers = 32, d_model = 1280, n_heads = 20, head_dim = 64;
    uint32_t ff_dim = 5120, downsample_hidden_size = 480;
    uint32_t max_source_positions = 1500, n_window = 50, n_window_infer = 800;
    uint32_t conv_chunksize = 500, output_dim = 2048;
    float layer_norm_eps = 1e-5f;
};

struct LlmConfig {
    uint32_t n_layers = 28, hidden = 2048, n_heads = 16, n_kv_heads = 8;
    uint32_t head_dim = 128, intermediate = 6144, vocab = 151936;
    uint32_t max_position_embeddings = 40960, max_cache = 2048;
    float rope_theta = 1000000.0f, rms_norm_eps = 1e-6f;
    std::string rope_scaling = "none";
    bool tied_embeddings = true;
};

struct Config {
    FrontendConfig frontend;
    EncoderConfig encoder;
    LlmConfig llm;
    uint32_t adapter_input = 2048, adapter_hidden = 8192, adapter_output = 2048;
    int32_t pad_token_id = 151643, eos_token_id = 151645, start_token_id = 151644;
    int32_t audio_start_id = 151669, audio_end_id = 151670, audio_placeholder_id = 0;
    uint32_t max_new_tokens = 200;
    std::vector<int32_t> prompt_prefix, prompt_suffix;
};

int64_t audio_token_length(int64_t mel_frames);

} // namespace starling::ggml::moss
