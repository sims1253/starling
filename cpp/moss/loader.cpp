#include "loader.hpp"
#include <cstdio>
#include <vector>

namespace starling::ggml::moss {
namespace {
uint32_t u32(const ModelLoader& m, const char* k, uint32_t d) { int64_t v; return m.kv_int(k,v)?(uint32_t)v:d; }
float f32(const ModelLoader& m, const char* k, float d) { double v; return m.kv_float(k,v)?(float)v:d; }
std::string str(const ModelLoader& m, const char* k, const char* d) { std::string v; return m.kv_str(k,v)?v:d; }
bool require(const ModelLoader& m, const std::string& n, std::string& err) {
    if (m.tensor(n.c_str())) return true; err = "GGUF missing required tensor: " + n; return false;
}
}

int64_t audio_token_length(int64_t T) {
    if (T <= 0) return 0;
    const int64_t r = T % 100;
    const int64_t tail = r ? (((((r - 1) / 2 + 1) - 1) / 2 + 1 - 1) / 2 + 1) : 0;
    return tail + 13 * (T / 100);
}

bool MossModel::load(const char* path, std::string& err) {
    if (!loader.load(path)) { err = loader.last_error(); return false; }
    auto& c=config; const auto& m=loader;
#define U(field,key) field=u32(m,"moss_transcribe." key,field)
#define F(field,key) field=f32(m,"moss_transcribe." key,field)
    U(c.frontend.sample_rate,"frontend.sample_rate"); U(c.frontend.n_fft,"frontend.n_fft"); U(c.frontend.win_length,"frontend.win_length"); U(c.frontend.hop_length,"frontend.hop_length"); U(c.frontend.n_mels,"frontend.n_mels"); U(c.frontend.power,"frontend.power");
    F(c.frontend.mel_floor,"frontend.mel_floor"); F(c.frontend.dynamic_range,"frontend.dynamic_range"); F(c.frontend.normalization_offset,"frontend.normalization_offset"); F(c.frontend.normalization_divisor,"frontend.normalization_divisor");
    c.frontend.pad_mode=str(m,"moss_transcribe.frontend.pad_mode","reflect"); c.frontend.mel_scale=str(m,"moss_transcribe.frontend.mel_scale","slaney"); c.frontend.mel_norm=str(m,"moss_transcribe.frontend.mel_norm","slaney"); c.frontend.log=str(m,"moss_transcribe.frontend.log","log10"); c.frontend.output_dtype=str(m,"moss_transcribe.frontend.output_dtype","bf16");
    U(c.encoder.n_layers,"enc.encoder_layers"); U(c.encoder.d_model,"enc.d_model"); U(c.encoder.n_heads,"enc.encoder_attention_heads"); U(c.encoder.head_dim,"enc.head_dim"); U(c.encoder.ff_dim,"enc.encoder_ffn_dim"); U(c.encoder.downsample_hidden_size,"enc.downsample_hidden_size"); U(c.encoder.max_source_positions,"enc.max_source_positions"); U(c.encoder.n_window,"enc.n_window"); U(c.encoder.n_window_infer,"enc.n_window_infer"); U(c.encoder.conv_chunksize,"enc.conv_chunksize"); U(c.encoder.output_dim,"enc.output_dim"); F(c.encoder.layer_norm_eps,"enc.layer_norm_eps");
    U(c.llm.hidden,"llm.hidden_size"); U(c.llm.n_layers,"llm.num_layers"); U(c.llm.n_heads,"llm.num_heads"); U(c.llm.n_kv_heads,"llm.num_kv_heads"); U(c.llm.head_dim,"llm.head_dim"); U(c.llm.intermediate,"llm.intermediate_size"); U(c.llm.vocab,"llm.vocab_size"); U(c.llm.max_position_embeddings,"llm.max_position_embeddings"); U(c.llm.max_cache,"max_cache_len"); F(c.llm.rope_theta,"llm.rope_theta"); F(c.llm.rms_norm_eps,"llm.rms_norm_eps"); c.llm.rope_scaling=str(m,"moss_transcribe.llm.rope_scaling","none");
    U(c.adapter_input,"adapter.input_size"); U(c.adapter_hidden,"adapter.hidden_size"); U(c.adapter_output,"adapter.output_size"); U(c.max_new_tokens,"max_new_tokens");
    c.pad_token_id=(int32_t)u32(m,"moss_transcribe.pad_token_id",c.pad_token_id); c.eos_token_id=(int32_t)u32(m,"moss_transcribe.eos_token_id",c.eos_token_id); c.start_token_id=(int32_t)u32(m,"moss_transcribe.start_token_id",c.start_token_id); c.audio_start_id=(int32_t)u32(m,"moss_transcribe.audio_start_id",c.audio_start_id); c.audio_end_id=(int32_t)u32(m,"moss_transcribe.audio_end_id",c.audio_end_id); c.audio_placeholder_id=(int32_t)u32(m,"moss_transcribe.audio_placeholder_id",c.audio_placeholder_id);
#undef U
#undef F
    std::vector<int64_t> a; if(m.kv_arr_int("moss_transcribe.prompt_prefix",a)) for(auto v:a)c.prompt_prefix.push_back((int32_t)v); a.clear(); if(m.kv_arr_int("moss_transcribe.prompt_suffix",a)) for(auto v:a)c.prompt_suffix.push_back((int32_t)v);

    // --- Validate untrusted GGUF metadata (spec ggml-moss-spec.md §11, bring-up
    // item 1). Fail fast with a clear message: the consumers divide by these
    // values (audio_encoder.cpp window W = M*(n_window_infer/100); llm.cpp
    // head grouping h / (n_heads/n_kv_heads)), so a zero or inconsistent value
    // means div-by-zero or an infinite window loop.
    if (std::string arch; m.kv_str("general.architecture", arch) && arch != "moss_transcribe") { err = "unsupported MOSS GGUF architecture: " + arch; return false; }
    if (std::string prof; m.kv_str("starling.numeric_profile", prof) && prof != "bf16_exact" && prof != "f16") { err = "unsupported MOSS numeric profile: " + prof; return false; }
    if (int64_t fv; m.kv_int("starling.format_version", fv) && fv != 1) { err = "unsupported Starling GGUF format version: " + std::to_string(fv); return false; }
    if (m.tensor_names().empty()) { err = "MOSS GGUF contains no tensors"; return false; }
#define POS(v, name) do { if (!(v)) { err = "MOSS GGUF " name " must be positive"; return false; } } while (0)
    POS(c.encoder.n_layers, "enc.encoder_layers"); POS(c.encoder.d_model, "enc.d_model");
    POS(c.encoder.n_heads, "enc.encoder_attention_heads"); POS(c.encoder.head_dim, "enc.head_dim");
    POS(c.encoder.ff_dim, "enc.encoder_ffn_dim"); POS(c.encoder.output_dim, "enc.output_dim");
    POS(c.encoder.downsample_hidden_size, "enc.downsample_hidden_size");
    POS(c.llm.hidden, "llm.hidden_size"); POS(c.llm.n_heads, "llm.num_heads");
    POS(c.llm.n_kv_heads, "llm.num_kv_heads"); POS(c.llm.head_dim, "llm.head_dim");
    POS(c.llm.intermediate, "llm.intermediate_size"); POS(c.llm.vocab, "llm.vocab_size");
#undef POS
    if (c.encoder.n_window_infer < 100) { err = "MOSS GGUF enc.n_window_infer must be >= 100 (attention window stride would be 0)"; return false; }
    if (c.llm.n_layers != 28) { err = "MOSS GGUF llm.num_layers must be 28"; return false; }
    if (c.encoder.d_model != c.encoder.n_heads * c.encoder.head_dim) { err = "MOSS GGUF enc.d_model != enc.encoder_attention_heads * enc.head_dim"; return false; }
    if (c.llm.hidden != c.llm.n_heads * c.llm.head_dim) { err = "MOSS GGUF llm.hidden_size != llm.num_heads * llm.head_dim"; return false; }
    if (c.llm.n_heads % c.llm.n_kv_heads != 0) { err = "MOSS GGUF llm.num_heads must be a multiple of llm.num_kv_heads"; return false; }

    if (c.frontend.n_fft != 640 || c.frontend.win_length != 640 || c.frontend.n_mels != 128 || c.frontend.hop_length != 160) { err="unsupported MOSS frontend metadata (requires n_fft/win=640, hop=160, n_mels=128)"; return false; }
    for (const char* n : {"audio.mel_filters","audio.mel_window","enc.conv1.weight","enc.conv1.bias","enc.conv2.weight","enc.conv2.bias","enc.conv3.weight","enc.conv3.bias","enc.conv_out.weight","enc.positional_embedding"}) if(!require(m,n,err)) return false;
    for(uint32_t i=0;i<c.encoder.n_layers;++i) { char n[128]; for(const char* tail:{"attn_norm.weight","attn_norm.bias","attn.q.weight","attn.q.bias","attn.k.weight","attn.k.bias","attn.v.weight","attn.v.bias","attn.o.weight","attn.o.bias","ffn_norm.weight","ffn_norm.bias","ffn.fc1.weight","ffn.fc1.bias","ffn.fc2.weight","ffn.fc2.bias"}) { std::snprintf(n,sizeof n,"enc.blk.%u.%s",i,tail); if(!require(m,n,err))return false; } }
    for(const char* n:{"enc.ln_post.weight","enc.ln_post.bias","enc.proj1.weight","enc.proj1.bias","enc.proj2.weight","enc.proj2.bias","adapter.gate.weight","adapter.up.weight","adapter.down.weight","llm.embed.weight"}) if(!require(m,n,err))return false;
    for(uint32_t i=0;i<c.llm.n_layers;++i) { char n[128]; for(const char* tail:{"attn_norm.weight","attn.q.weight","attn.k.weight","attn.v.weight","attn.o.weight","attn.q_norm.weight","attn.k_norm.weight","ffn_norm.weight","ffn.gate.weight","ffn.up.weight","ffn.down.weight"}) { std::snprintf(n,sizeof n,"llm.blk.%u.%s",i,tail); if(!require(m,n,err))return false; } }
    return require(m,"llm.final_norm.weight",err);
}
} // namespace starling::ggml::moss
