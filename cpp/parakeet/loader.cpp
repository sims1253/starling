// loader.cpp — read parakeet-tdt config from the loaded GGUF into Config.
//
// The shared ModelLoader parses the GGUF + KV map + tensors; this reads the
// parakeet.* keys (verbatim NeMo/converter strings, same as parakeet.cpp) into
// the Config struct. See config.hpp for the field meanings.

#include "loader.hpp"

#include "compat.hpp"

#include <cstdlib>

namespace starling::ggml::parakeet {

namespace {
// Tiny helpers over ModelLoader's typed accessors with defaults.
uint32_t kv_u32(const ModelLoader& ml, const std::string& key, uint32_t def) {
    int64_t v;
    return ml.kv_int(key, v) ? (uint32_t)v : def;
}
float kv_f32(const ModelLoader& ml, const std::string& key, float def) {
    double v;
    if (ml.kv_float(key, v)) return (float)v;
    // Some keys may be stored as int; fall back.
    int64_t iv;
    return ml.kv_int(key, iv) ? (float)iv : def;
}
std::string kv_str(const ModelLoader& ml, const std::string& key, const std::string& def) {
    std::string v;
    return ml.kv_str(key, v) ? v : def;
}
} // namespace

bool ParakeetModel::load(const char* gguf_path, std::string& err) {
    if (!loader.load(gguf_path)) {
        err = loader.last_error();
        return false;
    }
    // Normalize community GGUF dialects (parakeet.cpp / transcribe.cpp
    // naming) onto the native contract before any config is read.
    if (!apply_community_dialect_compat(loader, err)) return false;
    const ModelLoader& ml = loader;
    Config& c = config;

    // ---- preprocessor / mel ----
    c.sample_rate   = kv_u32(ml, "parakeet.preprocessor.sample_rate", 16000);
    c.n_mels        = kv_u32(ml, "parakeet.preprocessor.n_mels", 128);
    c.n_fft         = kv_u32(ml, "parakeet.preprocessor.n_fft", 512);
    c.win_length    = kv_u32(ml, "parakeet.preprocessor.win_length", 400);
    c.hop_length    = kv_u32(ml, "parakeet.preprocessor.hop_length", 160);
    c.preemph       = kv_f32(ml, "parakeet.preprocessor.preemph", 0.97f);
    c.mag_power     = kv_f32(ml, "parakeet.preprocessor.mag_power", 2.0f);
    c.normalize     = kv_str(ml, "parakeet.preprocessor.normalize", "per_feature");
    c.log_zero_guard= kv_f32(ml, "parakeet.preprocessor.log_zero_guard", 5.9604645e-08f);

    // ---- encoder ----
    c.d_model       = kv_u32(ml, "parakeet.encoder.d_model", 1024);
    c.n_layers      = kv_u32(ml, "parakeet.encoder.n_layers", 24);
    c.pred_out      = kv_u32(ml, "parakeet.encoder.pred_out", 640);
    c.n_heads       = kv_u32(ml, "parakeet.encoder.n_heads", 8);
    c.ff_dim        = kv_u32(ml, "parakeet.encoder.ff_dim", 4096);
    c.conv_kernel   = kv_u32(ml, "parakeet.encoder.conv_kernel", 9);
    c.subsampling_conv_channels = kv_u32(ml, "parakeet.encoder.subsampling_conv_channels", 256);
    c.conv_norm_type = kv_str(ml, "parakeet.encoder.conv_norm_type", "batch_norm");
    {
        int64_t xs = 0;
        if (ml.kv_int("parakeet.encoder.xscaling", xs)) c.xscaling = (xs != 0);
    }

    // ---- decoder ----
    c.pred_hidden     = kv_u32(ml, "parakeet.decoder.pred_hidden", 640);
    c.pred_rnn_layers = kv_u32(ml, "parakeet.decoder.pred_rnn_layers", 2);

    // ---- joint ----
    c.joint_hidden    = kv_u32(ml, "parakeet.joint.joint_hidden", 640);
    c.joint_activation = kv_str(ml, "parakeet.joint.activation", "relu");

    // ---- decoding / vocab ----
    c.max_symbols = kv_u32(ml, "parakeet.decoding.max_symbols", 10);
    c.vocab_size  = kv_u32(ml, "parakeet.vocab_size", 8193);
    c.blank_id    = kv_u32(ml, "parakeet.blank_id", 8192);

    // ---- TDT durations (INT32 array) ----
    {
        std::vector<int64_t> d;
        if (ml.kv_arr_int("parakeet.tdt.durations", d)) {
            c.tdt_durations.clear();
            for (int64_t v : d) c.tdt_durations.push_back((int32_t)v);
        }
    }
    // ---- tokenizer pieces (STRING array) ----
    ml.kv_arr_str("parakeet.tokenizer.pieces", c.tokenizer_pieces);

    // Sanity: must have at least the mel filterbank + window tensors + vocab.
    if (!ml.tensor("preprocessor.featurizer.fb")) {
        err = "GGUF missing preprocessor.featurizer.fb (mel filterbank)";
        return false;
    }
    if (c.vocab_size == 0) { err = "GGUF missing parakeet.vocab_size"; return false; }
    return true;
}

} // namespace starling::ggml::parakeet
