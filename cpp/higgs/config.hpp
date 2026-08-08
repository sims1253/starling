// config.hpp — architecture constants for the higgs-audio-v3-stt ggml port.
//
// bosonai/higgs-audio-v3-stt = Whisper-large-v3 mel frontend + Whisper audio
// encoder (absolute positional embeddings, NOT RoPE) + MLP projector + Qwen3-1.7B
// decoder. The audio path: log-mel[128] -> conv1(K3,s1,p1)+GELU -> conv2(K3,s2,p1)+
// GELU -> transpose -> add embed_positions[absolute, 1500] -> 32 WhisperEncoderLayers
// (global bidirectional attention) -> ln_post -> AvgPool1d(2,2) over the time dim
// -> depthwise temporal Conv1d(K3,s2,p1,groups=1280) -> Linear(1280->2048)+bias ->
// ReLU -> Linear(2048->2048)+bias -> [n_audio_tokens, 2048] audio embeddings. The
// decoder is a Qwen3-1.7B trunk (28 layers, d2048, 16 query / 8 KV GQA, head_dim
// 128, SwiGLU intermediate 6144, RMSNorm eps 1e-6, RoPE theta 1e6, SEPARATE
// lm_head) WITH qk_norm and WITHOUT attention biases (Qwen3 attention_bias=false).
//
// Defaults match the baked GGUF metadata (the separate convert_higgs_gguf.py task);
// the loader overrides them from `higgs.*` KV. Single source of truth:
// src/starling/higgs/config.py + the HF config.json for bosonai/higgs-audio-v3-stt,
// and the byte-for-byte upstream modeling in
// src/starling/higgs/vendor/modeling/modeling_higgs_audio.py.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace starling::ggml::higgs {

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
    uint32_t conv_kernel = 3, avg_pool_kernel = 2;
    uint32_t use_rope = 0;  // Higgs uses ABSOLUTE positional embeddings, not RoPE
    bool has_positional_embeddings = true;
    float layer_norm_eps = 1e-5f;  // Whisper LayerNorm eps
};

struct ProjectorConfig {
    // HiggsAudioFeatureProjector (projector_type="mlp", stride=2):
    //   temporal = depthwise Conv1d(in,in, K3, s2, p1, groups=in, bias=True)
    //   linear1  = Linear(in -> hidden, bias=True), ReLU
    //   linear2  = Linear(hidden -> out, bias=True)
    uint32_t temporal_kernel = 3, temporal_stride = 2, temporal_groups = 1280;
    uint32_t input_size = 1280, hidden_size = 2048, output_size = 2048;
    std::string act = "relu";
};

struct LlmConfig {
    uint32_t n_layers = 28, hidden = 2048, n_heads = 16, n_kv_heads = 8;
    uint32_t head_dim = 128, intermediate = 6144, vocab = 151936;
    uint32_t max_position_embeddings = 32768, max_cache = 4096;
    float rope_theta = 1000000.0f, rms_norm_eps = 1e-6f;
    std::string rope_scaling = "none";
    bool tied_embeddings = false, has_qk_norm = true;
};

struct Config {
    FrontendConfig frontend;
    EncoderConfig encoder;
    ProjectorConfig projector;
    LlmConfig llm;
    // Qwen3 / ChatML special-token ids (src/starling/higgs/config.py).
    int32_t audio_placeholder_id = 151672;  // <|AUDIO|>
    int32_t audio_bos_id = 151669;          // <|audio_bos|>
    int32_t audio_eos_id = 151670;          // <|audio_eos|>
    int32_t im_start_id = 151644;           // <|im_start|>
    int32_t im_end_id = 151645;             // <|im_end|>
    int32_t pad_token_id = 151643;          // <|endoftext|>
    int32_t eos_token_id = 151643;          // <|endoftext|>
    uint32_t max_new_tokens = 200;
    std::string default_instruction =
        "Transcribe the speech. Output only the spoken words in lowercase with no punctuation.";
    // Pre-tokenized ChatML prompt arrays (baked by the converter since the C++
    // tokenizer is decode-only). prefix = everything up to (not incl.) the AUDIO
    // placeholders; suffix = everything after. See build_transcribe_prompt.
    std::vector<int32_t> prompt_prefix, prompt_suffix;
};

// Number of LLM audio tokens implied by an (uncapped) mel-frame count.
//
// The audio tower downsamples time three times after the mel front-end:
//   1. conv2  (K3, stride 2, pad 1): T_enc   = floor((mel_T - 1)/2) + 1 = (mel_T+1)//2
//   2. avgpool(K2, stride 2)        : T_avg  = floor((T_enc - 2)/2) + 1
//   3. temporal depthwise (K3,s2,p1): T_proj = floor((T_avg - 1)/2) + 1 = (T_avg+1)//2
// For mel_T that are multiples of 8 (the realistic 100 Hz -> 25 Hz -> 12.5 Hz path;
// e.g. 30 s = 3000 mel frames -> 375 tokens), this collapses to exactly mel_T//8,
// matching modeling's `_get_feat_extract_output_lengths` (conv /2, then
// `(L-2)//2+1` avg-pool) followed by `downsample_lengths` (`(L-1)//stride+1`).
// `mel_frames//8` is the prompt-time token count the eager reference uses (the
// pipeline derives it from the mel frame count before the encoder runs).
int64_t audio_token_count(int64_t mel_frames);

} // namespace starling::ggml::higgs
