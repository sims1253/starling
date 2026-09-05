// compat.cpp — community-GGUF dialect normalization. See compat.hpp.

#include "compat.hpp"

#include "runtime/model_loader.hpp"

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace starling::ggml::parakeet {
namespace {

bool has_kv(const ModelLoader& ml, const std::string& k) {
    int64_t v;
    return ml.kv_int(k, v);
}

// ---- tensor-name translation ----------------------------------------------

// Apply a sequence of literal prefix/suffix rewrites to a dialect tensor
// name, returning the native name (or "" when unchanged).
std::string translate_tail(const std::string& s) {
    std::string t = s;
    auto sub = [&](const char* from, const char* to) {
        size_t p = t.find(from);
        if (p != std::string::npos) t.replace(p, strlen(from), to);
    };
    // Order matters only where substrings overlap; the tokens below are
    // distinct at their '.' boundaries.
    sub("norm_attn", "norm_self_att");
    sub("norm_ff", "norm_feed_forward");
    if (t.rfind("ff", 0) == 0) t.replace(0, 2, "feed_forward");  // ff1.linear1 -> feed_forward1.linear1
    sub("conv.bn.", "conv.batch_norm.");
    sub("conv.dw.", "conv.depthwise_conv.");
    sub("conv.pw1", "conv.pointwise_conv1");
    sub("conv.pw2", "conv.pointwise_conv2");
    sub("conv.pointwise1", "conv.pointwise_conv1");   // transcribe dialect spelling
    sub("conv.pointwise2", "conv.pointwise_conv2");
    sub("conv.depthwise.", "conv.depthwise_conv.");   // transcribe dialect spelling
    return t;
}

// parakeet.cpp/CrispASR dialect: full-name rewrites on top of the shared
// tail rules. Returns the native name, "" when unchanged.
std::string cstr_name(const std::string& n) {
    auto p = [&](const char* pre, std::string* tail) -> bool {
        size_t l = strlen(pre);
        if (n.compare(0, l, pre) == 0) { *tail = n.substr(l); return true; }
        return false;
    };
    std::string tail;
    if (n == "decoder.embed.weight") return "decoder.prediction.embed.weight";
    if (p("decoder.lstm.", &tail)) {
        // {i}.{w_ih|w_hh|b_ih|b_hh}
        size_t d = tail.find('.');
        if (d == std::string::npos) return "";
        std::string i = tail.substr(0, d), kind = tail.substr(d + 1);
        std::string out;
        if (kind == "w_ih") out = "weight_ih_l";
        else if (kind == "w_hh") out = "weight_hh_l";
        else if (kind == "b_ih") out = "bias_ih_l";
        else if (kind == "b_hh") out = "bias_hh_l";
        else return "";
        return "decoder.prediction.dec_rnn.lstm." + out + i;
    }
    if (p("encoder.layers.", &tail)) {
        size_t d = tail.find('.');
        if (d == std::string::npos) return "";
        std::string i = tail.substr(0, d), rest = tail.substr(d + 1);
        if (rest.rfind("attn.", 0) == 0) {
            rest = rest.substr(5);
            if (rest == "q.weight") rest = "linear_q.weight";
            else if (rest == "k.weight") rest = "linear_k.weight";
            else if (rest == "v.weight") rest = "linear_v.weight";
            else if (rest == "out.weight") rest = "linear_out.weight";
            else if (rest == "pos.weight") rest = "linear_pos.weight";
            return "encoder.layers." + i + ".self_attn." + rest;
        }
        return "encoder.layers." + i + "." + translate_tail(rest);
    }
    if (p("encoder.pre.", &tail)) return "encoder.pre_encode." + tail;
    if (n.rfind("joint.out.", 0) == 0) return "joint.joint_net.2." + n.substr(10);
    if (n == "preprocessor.fb") return "preprocessor.featurizer.fb";
    if (n == "preprocessor.window") return "preprocessor.featurizer.window";
    // joint.enc.* / joint.pred.* / encoder.pre_encode.*: already native.
    return "";
}

// transcribe.cpp dialect rewrites.
std::string transcribe_name(const std::string& n) {
    auto p = [&](const char* pre, std::string* tail) -> bool {
        size_t l = strlen(pre);
        if (n.compare(0, l, pre) == 0) { *tail = n.substr(l); return true; }
        return false;
    };
    std::string tail;
    if (n == "pred.embed.weight") return "decoder.prediction.embed.weight";
    if (p("pred.lstm.", &tail)) {
        size_t d = tail.find('.');
        if (d == std::string::npos) return "";
        std::string i = tail.substr(0, d), kind = tail.substr(d + 1);
        if (kind == "Wx") return "decoder.prediction.dec_rnn.lstm.weight_ih_l" + i;
        if (kind == "Wh") return "decoder.prediction.dec_rnn.lstm.weight_hh_l" + i;
        if (kind == "bias") return "decoder.prediction.dec_rnn.lstm.bias_ih_l" + i;
        return "";
    }
    if (p("enc.blocks.", &tail)) {
        size_t d = tail.find('.');
        if (d == std::string::npos) return "";
        std::string i = tail.substr(0, d), rest = tail.substr(d + 1);
        if (rest.rfind("attn.", 0) == 0)
            return "encoder.layers." + i + ".self_attn." + rest.substr(5);
        return "encoder.layers." + i + "." + translate_tail(rest);
    }
    if (p("enc.pre_encode.", &tail)) return "encoder.pre_encode." + tail;
    if (n.rfind("joint.out.", 0) == 0) return "joint.joint_net.2." + n.substr(10);
    return "";
}

// ---- mel frontend synthesis (transcribe dialect) ---------------------------

// librosa "slaney" mel scale (htk=False), matching the converter's fallback
// and NeMo's preprocessor.
double mel_slaney(double hz) {
    const double f_sp = 200.0 / 3.0;
    const double min_log_hz = 1000.0;
    const double min_log_mel = min_log_hz / f_sp;
    const double logstep = std::log(6.4) / 27.0;
    if (hz >= min_log_hz) return min_log_mel + std::log(hz / min_log_hz) / logstep;
    return hz / f_sp;
}
double mel_slaney_inv(double m) {
    const double f_sp = 200.0 / 3.0;
    const double min_log_hz = 1000.0;
    const double min_log_mel = min_log_hz / f_sp;
    const double logstep = std::log(6.4) / 27.0;
    if (m >= min_log_mel) return min_log_hz * std::exp(logstep * (m - min_log_mel));
    return f_sp * m;
}

// Hann window [win_length], F32.
std::vector<float> synth_window(int64_t win_length) {
    std::vector<float> w((size_t)win_length);
    for (int64_t n = 0; n < win_length; ++n)
        w[(size_t)n] = (float)(0.5 - 0.5 * std::cos(2.0 * M_PI * (double)n / (double)win_length));
    return w;
}

// Slaney-normalized mel filterbank, [n_mels, n_bins] row-major
// (fb[m * n_bins + b]).
std::vector<float> synth_filterbank(int n_mels, int n_fft, double sr,
                                    double fmin, double fmax) {
    const int n_bins = n_fft / 2 + 1;
    const double m_min = mel_slaney(fmin), m_max = mel_slaney(fmax);
    std::vector<double> mel_pts((size_t)n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i)
        mel_pts[(size_t)i] = mel_slaney_inv(m_min + (m_max - m_min) * i / (n_mels + 1));
    std::vector<float> fb((size_t)n_mels * n_bins, 0.0f);
    for (int m = 0; m < n_mels; ++m) {
        // slaney norm: 2 / (upper - lower mel points)
        const double enorm = 2.0 / (mel_pts[(size_t)m + 2] - mel_pts[(size_t)m]);
        for (int b = 0; b < n_bins; ++b) {
            const double f = (double)b * sr / (double)n_fft;
            double lower = mel_pts[(size_t)m], center = mel_pts[(size_t)m + 1],
                   upper = mel_pts[(size_t)m + 2];
            double w = 0.0;
            if (f > lower && f < upper)
                w = enorm * (f <= center ? (f - lower) / (center - lower)
                                         : (upper - f) / (upper - center));
            fb[(size_t)m * n_bins + b] = (float)w;
        }
    }
    return fb;
}

// ---- shared KV normalization ----------------------------------------------

// The joint head is [vocab | blank | durations] and the engine's vocab_size
// excludes the blank; recover it from geometry, then reconcile with the
// dialect's KV (which may count the blank).
bool normalize_vocab(ModelLoader& ml, const std::string& kv_vocab_key,
                     const std::string& pieces_key, std::string& err) {
    ggml_tensor* head = ml.tensor("joint.joint_net.2.weight");
    if (!head) head = ml.tensor("joint.out.weight");
    if (!head) { err = "compat: no joint output projection found"; return false; }
    std::vector<int64_t> durs;
    int64_t n_dur = 0;
    if (ml.kv_arr_int("parakeet.tdt.durations", durs) ||
        ml.kv_arr_int("stt.parakeet.tdt.durations", durs))
        n_dur = (int64_t)durs.size();
    const int64_t v_layout = head->ne[1] - 1 - n_dur;
    if (v_layout <= 0) { err = "compat: implausible joint layout"; return false; }

    int64_t kv_vocab = -1;
    ml.kv_int(kv_vocab_key, kv_vocab);  // -1 when the dialect omits it
    // Tripwire: a vocab count that matches NEITHER convention means the
    // dialect was mis-detected (e.g. a wrong durations array) — refuse
    // rather than silently trusting geometry.
    if (kv_vocab != -1 && kv_vocab != v_layout && kv_vocab != v_layout + 1) {
        err = "compat: dialect vocab_size " + std::to_string(kv_vocab) +
              " matches neither layout convention (expected " +
              std::to_string(v_layout) + " or " + std::to_string(v_layout + 1) +
              "); refusing to guess";
        return false;
    }
    ml.add_kv_int("parakeet.vocab_size", v_layout);

    // pieces: expect the blank-inclusive or blank-excluded spelling, and
    // REFUSE anything else — a silently missing/mis-sized token array yields
    // empty transcripts for every utterance (detokenize drops out-of-range
    // ids) with no error anywhere.
    std::vector<std::string> pieces;
    if (!ml.kv_arr_str(pieces_key, pieces) ||
        ((int64_t)pieces.size() != v_layout && (int64_t)pieces.size() != v_layout + 1)) {
        err = "compat: " + pieces_key + " missing or sized " +
              std::to_string(pieces.size()) + " (expected " +
              std::to_string(v_layout) + " or " + std::to_string(v_layout + 1) + ")";
        return false;
    }
    if ((int64_t)pieces.size() == v_layout + 1)
        pieces.pop_back();  // drop the trailing blank piece
    ml.add_kv_arr_str("parakeet.tokenizer.pieces", pieces);

    int64_t blank = v_layout;
    ml.kv_int("parakeet.blank_id", blank);
    ml.add_kv_int("parakeet.blank_id", blank);
    return true;
}

void copy_kv_int(ModelLoader& ml, const char* from, const char* to) {
    int64_t v;
    if (ml.kv_int(from, v)) ml.add_kv_int(to, v);
}
void copy_kv_str(ModelLoader& ml, const char* from, const char* to) {
    std::string v;
    if (ml.kv_str(from, v)) ml.add_kv_str(to, v);
}
void copy_kv_arr_int(ModelLoader& ml, const char* from, const char* to) {
    std::vector<int64_t> v;
    if (ml.kv_arr_int(from, v)) ml.add_kv_arr_int(to, v);
}
void copy_kv_float(ModelLoader& ml, const char* from, const char* to) {
    double v;
    if (ml.kv_float(from, v)) ml.add_kv_float(to, v);
}

} // namespace

bool apply_community_dialect_compat(ModelLoader& ml, std::string& err) {
    // Native files carry the namespaced config key; nothing to do.
    if (has_kv(ml, "parakeet.encoder.d_model")) return true;

    const std::vector<std::string> names = ml.tensor_names();

    // ---- dialect: parakeet.cpp / CrispASR (cstr & friends) ----------------
    if (has_kv(ml, "parakeet.d_model") && ml.tensor("decoder.lstm.0.w_ih")) {
        for (const auto& n : names) {
            std::string native = cstr_name(n);
            if (!native.empty() && native != n) ml.add_tensor_alias(native.c_str(), n.c_str());
        }
        copy_kv_int(ml, "parakeet.d_model", "parakeet.encoder.d_model");
        copy_kv_int(ml, "parakeet.n_layers", "parakeet.encoder.n_layers");
        copy_kv_int(ml, "parakeet.n_heads", "parakeet.encoder.n_heads");
        copy_kv_int(ml, "parakeet.ff_dim", "parakeet.encoder.ff_dim");
        copy_kv_int(ml, "parakeet.conv_kernel", "parakeet.encoder.conv_kernel");
        copy_kv_int(ml, "parakeet.subsampling_channels", "parakeet.encoder.subsampling_conv_channels");
        copy_kv_int(ml, "parakeet.pred_hidden", "parakeet.decoder.pred_hidden");
        copy_kv_int(ml, "parakeet.pred_layers", "parakeet.decoder.pred_rnn_layers");
        copy_kv_int(ml, "parakeet.joint_hidden", "parakeet.joint.joint_hidden");
        copy_kv_int(ml, "parakeet.sample_rate", "parakeet.preprocessor.sample_rate");
        copy_kv_int(ml, "parakeet.n_fft", "parakeet.preprocessor.n_fft");
        copy_kv_int(ml, "parakeet.n_mels", "parakeet.preprocessor.n_mels");
        copy_kv_int(ml, "parakeet.win_length", "parakeet.preprocessor.win_length");
        copy_kv_int(ml, "parakeet.hop_length", "parakeet.preprocessor.hop_length");
        copy_kv_arr_int(ml, "parakeet.tdt_durations", "parakeet.tdt.durations");
        // preemph / mag_power / normalize / log_zero_guard: loader defaults
        // already match NeMo parakeet (0.97 / 2.0 / per_feature / 5.96e-8).
        return normalize_vocab(ml, "parakeet.vocab_size", "tokenizer.ggml.tokens", err);
    }

    // ---- dialect: transcribe.cpp (handy-computer & friends) ---------------
    if (has_kv(ml, "stt.parakeet.joint.hidden") && ml.tensor("enc.blocks.0.attn.linear_q.weight")) {
        for (const auto& n : names) {
            std::string native = transcribe_name(n);
            if (!native.empty() && native != n) ml.add_tensor_alias(native.c_str(), n.c_str());
        }
        // Fused LSTM bias: bias_ih carries the sum, bias_hh is zero.
        for (int layer = 0; layer < 8; ++layer) {
            char fused[64], ih[96], hh[96];
            std::snprintf(fused, sizeof fused, "pred.lstm.%d.bias", layer);
            std::snprintf(ih, sizeof ih,
                          "decoder.prediction.dec_rnn.lstm.bias_ih_l%d", layer);
            std::snprintf(hh, sizeof hh,
                          "decoder.prediction.dec_rnn.lstm.bias_hh_l%d", layer);
            ggml_tensor* b = ml.tensor(fused);
            if (!b) break;
            std::vector<float> zeroes((size_t)ggml_nelements(b), 0.0f);
            ml.add_tensor_alias(ih, fused);
            ml.add_owned_tensor(hh, zeroes, b->ne[0], 1);
        }

        copy_kv_int(ml, "stt.parakeet.encoder.d_model", "parakeet.encoder.d_model");
        copy_kv_int(ml, "stt.parakeet.encoder.n_layers", "parakeet.encoder.n_layers");
        copy_kv_int(ml, "stt.parakeet.encoder.n_heads", "parakeet.encoder.n_heads");
        copy_kv_int(ml, "stt.parakeet.encoder.d_ff", "parakeet.encoder.ff_dim");
        copy_kv_int(ml, "stt.parakeet.encoder.conv_kernel", "parakeet.encoder.conv_kernel");
        copy_kv_int(ml, "stt.parakeet.encoder.subsampling_channels", "parakeet.encoder.subsampling_conv_channels");
        copy_kv_int(ml, "stt.parakeet.encoder.xscaling", "parakeet.encoder.xscaling");
        copy_kv_int(ml, "stt.parakeet.predictor.hidden", "parakeet.decoder.pred_hidden");
        copy_kv_int(ml, "stt.parakeet.predictor.n_layers", "parakeet.decoder.pred_rnn_layers");
        copy_kv_int(ml, "stt.parakeet.joint.hidden", "parakeet.joint.joint_hidden");
        copy_kv_str(ml, "stt.parakeet.joint.activation", "parakeet.joint.activation");
        copy_kv_int(ml, "stt.parakeet.tdt.max_symbols", "parakeet.decoding.max_symbols");
        copy_kv_arr_int(ml, "stt.parakeet.tdt.durations", "parakeet.tdt.durations");
        copy_kv_int(ml, "stt.frontend.sample_rate", "parakeet.preprocessor.sample_rate");
        copy_kv_int(ml, "stt.frontend.n_fft", "parakeet.preprocessor.n_fft");
        copy_kv_int(ml, "stt.frontend.num_mels", "parakeet.preprocessor.n_mels");
        copy_kv_int(ml, "stt.frontend.win_length", "parakeet.preprocessor.win_length");
        copy_kv_int(ml, "stt.frontend.hop_length", "parakeet.preprocessor.hop_length");
        copy_kv_float(ml, "stt.frontend.pre_emphasis", "parakeet.preprocessor.preemph");
        copy_kv_str(ml, "stt.frontend.normalize", "parakeet.preprocessor.normalize");

        if (!normalize_vocab(ml, "stt.parakeet.predictor.vocab",
                             "tokenizer.ggml.tokens", err))
            return false;

        // Synthesize the mel frontend (transcribe GGUFs carry no fb/window
        // tensors; the runtime computes mel from the stt.frontend.* metadata).
        if (!ml.tensor("preprocessor.featurizer.fb")) {
            int64_t n_mels = 128, n_fft = 512, win = 400, hop = 160, sr = 16000;
            double fmax = 8000.0, fmin = 0.0;
            ml.kv_int("parakeet.preprocessor.n_mels", n_mels);
            ml.kv_int("parakeet.preprocessor.n_fft", n_fft);
            ml.kv_int("parakeet.preprocessor.win_length", win);
            ml.kv_int("parakeet.preprocessor.sample_rate", sr);
            double f;
            if (ml.kv_float("stt.frontend.f_max", f)) fmax = f;
            if (ml.kv_float("stt.frontend.f_min", f)) fmin = f;
            ml.add_owned_tensor("preprocessor.featurizer.window",
                                synth_window(win), win, 1);
            ml.add_owned_tensor("preprocessor.featurizer.fb",
                                synth_filterbank((int)n_mels, (int)n_fft,
                                                 (double)sr, fmin, fmax),
                                n_fft / 2 + 1, n_mels);
            if (std::getenv("STARLING_COMPAT_DEBUG"))
                std::fprintf(stderr, "[compat] synthesized slaney fb (%lld mels, "
                                     "f_max=%g) + hann window (%lld)\n",
                             (long long)n_mels, fmax, (long long)win);
        }
        return true;
    }

    err = "unrecognized parakeet GGUF dialect (neither native, parakeet.cpp "
          "nor transcribe.cpp naming)";
    return false;
}

} // namespace starling::ggml::parakeet
