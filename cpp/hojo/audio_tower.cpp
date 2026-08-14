// audio_tower.cpp — Qwen3-Omni audio tower for Hojo-ASR-V1.
//
// The audio path (hojo_asr.qwen3_omni_audioencoder.ModifyQwen3OmniMoeAudioEncoder):
//   mel (T,128) -> chunk into windows of n_window*2=3000 frames (ceil(T/3000))
//   -> per window: reshape (1,1,win,128) -> conv2d1 [480,1,3,3] s2 p1 -> GELU
//      -> conv2d2 [480,480,3,3] s2 p1 -> GELU -> conv2d3 s2 p1 -> GELU
//      -> permute -> flatten freq (480*16=7680) -> conv_out Linear [1280,7680]
//      -> add SinusoidsPositionEmbedding (sin/cos concat, computed, per-window
//         position index resets to 0) -> extract valid frames
//   -> pack all windows' valid frames into one sequence
//   -> build block-diagonal 4D attention mask from cu_seqlens (bidirectional
//      within each window, no cross-window attention)
//   -> 32 pre-norm LayerNorm transformer layers (MHA 20 heads head_dim 64 WITH
//      bias, GELU FFN) -> ln_post -> proj1 GELU proj2
//   Output [n_speech, 2048].
//
// For single-utterance parity inference (batch=1): the conv chunking produces
// ceil(T/3000) windows; the block mask is full-bidirectional within each
// window's frame range and blocks cross-window attention. short/medium have one
// window (full bidirectional); long has two.
//
// Correctness-first: one-shot run_graph (no K-step capture). The conv2d runs
// via the host fallback (ggml_conv_2d under CUDA-graph capture is unvalidated,
// matching the higgs/ark host-conv discipline); the transformer layers + conv_out
// run as a graph.
#include "audio_tower.hpp"
#include "runtime/backend.hpp"
#include "runtime/graph.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace starling::ggml::hojo {
namespace {

ggml_tensor* weight(ggml_context* c, const ModelLoader& ml, const std::string& n) {
    return clone_weight(c, ml, n.c_str());
}
ggml_tensor* f32(ggml_context* c, ggml_tensor* x) {
    return x->type == GGML_TYPE_F32 ? x : ggml_cast(c, x, GGML_TYPE_F32);
}
ggml_tensor* linear(ggml_context* c, const ModelLoader& ml, ggml_tensor* x,
                    const std::string& n, bool bias) {
    ggml_tensor* y = ggml_mul_mat(c, weight(c, ml, n + ".weight"), f32(c, x));
    if (bias) y = ggml_add(c, f32(c, y), f32(c, weight(c, ml, n + ".bias")));
    return f32(c, y);
}
// exact GELU (erf) — config.activation_function="gelu", approximate="none".
ggml_tensor* exact_gelu(ggml_context* c, ggml_tensor* x) {
    return f32(c, ggml_gelu_erf(c, f32(c, x)));
}
// PyTorch LayerNorm: F32 reduction + affine.
ggml_tensor* layer_norm(ggml_context* c, const ModelLoader& ml, ggml_tensor* x,
                        const std::string& n, float eps) {
    ggml_tensor* y = ggml_norm(c, f32(c, x), eps);
    y = ggml_mul(c, y, f32(c, weight(c, ml, n + ".weight")));
    y = ggml_add(c, y, f32(c, weight(c, ml, n + ".bias")));
    return f32(c, y);
}

// Read a weight's f32 contents to host (host-conv path needs the raw values).
std::vector<float> read_f32(const ModelLoader& ml, const char* name) {
    ggml_tensor* t = ml.tensor(name);
    if (!t) return {};
    ensure_weights_realized(ml);
    size_t n = (size_t) ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> raw(n);
        ggml_backend_tensor_get(t, raw.data(), 0, n * sizeof(ggml_bf16_t));
        for (size_t i = 0; i < n; ++i) out[i] = ggml_bf16_to_fp32(raw[i]);
    }
    return out;
}

// 3 stride-2 conv2d + GELU between each, host-side (f32, double accumulation).
// Input: window mel [win, n_mels] time-major (element (t=time, c=freq) at
// t*n_mels+c). The reference feeds conv2d1 an input of shape
// (1, 1, freq=n_mels, time=win) -- i.e. freq is the H axis and time is the W
// axis (verified by hooking hojo_asr.speech_encoder.conv2d1: input is
// (1,1,128,743) for short). We must therefore treat H=freq, W=time so the
// 3x3 kernel weights wf[oc,ic,kh,kw] index kh over freq and kw over time,
// matching the eager path. Conv2d k3 s2 p1 on both axes; after 3 layers
// freq: 128->64->32->16, time: win->...->out_T. Output [out_T, 7680] where
// 7680 = 480*16. The reference flattens via permute(0,3,1,2).view(b,t,c*f) on
// (b, C=480, F=16, T) -> (b, T, C*F), so per time step the layout is c-outer,
// f-inner (element (t,c,f) at t*7680 + c*16 + f).
//
// Conv2d weight layout in HF/PyTorch: [out_C, in_C, kH, kW]. We compute the
// standard 2D convolution with kH=kW=3, stride 2, pad 1 on both axes.
std::vector<float> host_conv2d_stack(const ModelLoader& ml,
                                     const std::vector<float>& mel_win,
                                     int64_t win, int64_t n_mels) {
    // Track (C, H=freq, W=time). Build x from the time-major mel_win.
    int64_t C = 1, H = n_mels, W = win;
    std::vector<float> x((size_t) C * H * W);
    for (int64_t h = 0; h < H; ++h)        // h indexes freq
        for (int64_t w = 0; w < W; ++w)    // w indexes time
            x[((size_t)0 * H + h) * W + w] = mel_win[(size_t) w * n_mels + h];
    const char* names[3] = {"audio.conv2d1", "audio.conv2d2", "audio.conv2d3"};
    for (int li = 0; li < 3; ++li) {
        std::vector<float> wf = read_f32(ml, (std::string(names[li]) + ".weight").c_str());
        std::vector<float> bf = read_f32(ml, (std::string(names[li]) + ".bias").c_str());
        // wf: [OC, IC, 3, 3].
        const int64_t IC = C, OC = (int64_t) bf.size(), K = 3, p = 1, s = 2;
        const int64_t OH = (H + 2 * p - K) / s + 1;
        const int64_t OW = (W + 2 * p - K) / s + 1;
        // Pad x to (C, H+2p, W+2p).
        std::vector<float> xp((size_t) IC * (H + 2 * p) * (W + 2 * p), 0.0f);
        for (int64_t c = 0; c < IC; ++c)
            for (int64_t h = 0; h < H; ++h)
                for (int64_t w = 0; w < W; ++w)
                    xp[((size_t) c * (H + 2 * p) + (h + p)) * (W + 2 * p) + (w + p)] =
                        x[((size_t) c * H + h) * W + w];
        std::vector<double> acc((size_t) OC * OH * OW);
        for (int64_t oc = 0; oc < OC; ++oc) {
            for (int64_t oh = 0; oh < OH; ++oh) {
                for (int64_t ow = 0; ow < OW; ++ow) {
                    double a = (double) bf[(size_t) oc];
                    for (int64_t ic = 0; ic < IC; ++ic) {
                        for (int64_t kh = 0; kh < K; ++kh) {
                            for (int64_t kw = 0; kw < K; ++kw) {
                                int64_t ih = oh * s + kh;
                                int64_t iw = ow * s + kw;
                                a += (double) wf[((((size_t) oc * IC + ic) * K + kh) * K + kw)] *
                                     (double) xp[((size_t) ic * (H + 2 * p) + ih) * (W + 2 * p) + iw];
                            }
                        }
                    }
                    acc[((size_t) oc * OH + oh) * OW + ow] = a;
                }
            }
        }
        // GELU (exact erf) between conv layers.
        C = OC; H = OH; W = OW;
        x.resize((size_t) C * H * W);
        for (size_t i = 0; i < acc.size(); ++i) {
            float v = (float) acc[i];
            v = 0.5f * v * (1.0f + std::erf(v / (float) M_SQRT2));
            x[i] = v;
        }
    }
    // x is now (C=480, H=freq_down=16=F, W=time_down=out_T). The reference does
    // permute(0,3,1,2).view(b, t, c*f) on (b, C, F, T) -> (b, T=93, C*F=7680):
    // c-outer, f-inner per time step (element (t,c,f) at t*7680 + c*16 + f).
    const int64_t out_T = W, F = H;
    std::vector<float> out((size_t) out_T * C * F);
    for (int64_t t = 0; t < out_T; ++t)
        for (int64_t c = 0; c < C; ++c)
            for (int64_t f = 0; f < F; ++f)
                out[(size_t) t * (C * F) + c * F + f] =
                    x[((size_t) c * F + f) * out_T + t];
    return out;  // [out_T, 7680] c-outer, f-inner per step
}

// Compute the SinusoidsPositionEmbedding for `length` positions over `channels`
// (must be even): sin/cos concat, log_timescale_increment = log(10000)/(ch/2-1).
// Returns [length, channels] f32. Matches transformers' SinusoidsPositionEmbedding.
std::vector<float> sinusoids_pos_emb(int64_t length, int64_t channels) {
    const double log_inc = std::log(10000.0) / ((double)(channels / 2) - 1.0);
    std::vector<float> inv_ts(channels / 2);
    for (int64_t i = 0; i < channels / 2; ++i)
        inv_ts[i] = (float) std::exp(-log_inc * (double) i);
    std::vector<float> pe((size_t) length * channels);
    for (int64_t t = 0; t < length; ++t) {
        for (int64_t i = 0; i < channels / 2; ++i) {
            float a = (float) t * inv_ts[i];
            float s = std::sin(a), c = std::cos(a);
            pe[(size_t) t * channels + i] = s;
            pe[(size_t) t * channels + channels / 2 + i] = c;
        }
    }
    return pe;
}

// Append one tower transformer layer (pre-norm LayerNorm, MHA with bias,
// bidirectional, GELU FFN). x_in is [d_model, T] f32 (column-per-token).
// mask is the [T, T] f32 block-diagonal additive mask (0 in-block, -inf out).
ggml_tensor* build_tower_layer(ggml_context* c, const HojoModel& m,
                               int li, int64_t T, ggml_tensor* x,
                               ggml_tensor* mask, float ln_eps) {
    const auto& tc = m.config.tower;
    const ModelLoader& ml = m.loader;
    const std::string p = "audio.blk." + std::to_string(li) + ".";
    const int H = (int) tc.encoder_attention_heads, D = (int) tc.head_dim;
    const float scale = 1.0f / std::sqrt((float) D);
    ggml_tensor* r = x;
    ggml_tensor* n = layer_norm(c, ml, x, p + "attn_norm", ln_eps);
    ggml_tensor* q = linear(c, ml, n, p + "attn.q", true);   // [d_model, T]
    ggml_tensor* k = linear(c, ml, n, p + "attn.k", true);
    ggml_tensor* v = linear(c, ml, n, p + "attn.v", true);
    // Split into heads: [d_model, T] -> [D, H, T] -> [D, T, H].
    auto to_heads = [&](ggml_tensor* z) {
        z = ggml_reshape_3d(c, z, D, H, T);
        return ggml_cont(c, ggml_permute(c, z, 0, 2, 1, 3));  // [D, T, H]
    };
    ggml_tensor* qh = to_heads(q), * kh = to_heads(k), * vh = to_heads(v);
    // Bidirectional attention with the block-diagonal additive mask.
    // scores = kh^T @ qh -> [T, T, H]; scale; softmax(mask); @ vh.
    ggml_tensor* sc = ggml_mul_mat(c, kh, qh);                 // [T, T, H]
    sc = ggml_scale(c, f32(c, sc), scale);
    ggml_tensor* prob = ggml_soft_max_ext(c, f32(c, sc), f32(c, mask), 1.0f, 0.0f);
    ggml_tensor* vt = ggml_cont(c, ggml_permute(c, vh, 1, 0, 2, 3));  // [T, D, H]
    ggml_tensor* co = ggml_mul_mat(c, vt, prob);                      // [D, T, H]
    co = ggml_cont(c, ggml_permute(c, co, 0, 2, 1, 3));               // [D, H, T]
    ggml_tensor* joined = ggml_reshape_2d(c, co, (int64_t) D * H, T);  // [d_model, T]
    ggml_tensor* a = linear(c, ml, joined, p + "attn.o", true);
    x = ggml_add(c, f32(c, r), f32(c, a));
    r = x;
    n = layer_norm(c, ml, x, p + "ffn_norm", ln_eps);
    ggml_tensor* h = linear(c, ml, n, p + "ffn.fc1", true);
    h = exact_gelu(c, h);
    h = linear(c, ml, h, p + "ffn.fc2", true);
    x = ggml_add(c, f32(c, r), f32(c, h));
    return f32(c, x);
}

} // namespace

bool encode_audio_tower(const HojoModel& model, const MelFeatures& mel,
                        TowerOutput& out, std::string& err) {
    ensure_weights_realized(model.loader);
    const auto& tc = model.config.tower;
    if (mel.n_mels != (int64_t) tc.num_mel_bins || mel.n_frames <= 0 ||
        mel.data.size() != (size_t) mel.n_mels * mel.n_frames) {
        err = "invalid Hojo mel shape/data";
        return false;
    }
    const int64_t mel_T = mel.n_frames;
    const int64_t n_mels = mel.n_mels;
    const int64_t win_size = (int64_t) tc.n_window * 2;  // 3000

    // 1. Chunk the mel into windows of `win_size` frames (ceil(mel_T / win_size)).
    std::vector<int64_t> window_lens;
    for (int64_t off = 0; off < mel_T; off += win_size) {
        int64_t len = std::min(win_size, mel_T - off);
        window_lens.push_back(len);
    }

    // 2. Per-window host conv2d stack -> [conv_out_T, 7680] each.
    std::vector<std::vector<float>> per_window_conv;
    per_window_conv.reserve(window_lens.size());
    std::vector<int64_t> per_window_frames;
    for (size_t wi = 0; wi < window_lens.size(); ++wi) {
        const int64_t wl = window_lens[wi];
        std::vector<float> mel_win((size_t) wl * n_mels);
        const int64_t off = (int64_t) wi * win_size;
        for (int64_t t = 0; t < wl; ++t)
            for (int64_t c = 0; c < n_mels; ++c)
                mel_win[(size_t) t * n_mels + c] =
                    mel.data[(size_t) c * mel_T + (off + t)];
        std::vector<float> conv = host_conv2d_stack(model.loader, mel_win, wl, n_mels);
        // Flattened conv width = downsample_hidden_size (480) * (n_mels>>3) (freq
        // is halved 3x by the stride-2 convs: 128->64->32->16). Validate the conv
        // output divides evenly so a malformed window cannot yield a wrong conv_T.
        const int64_t conv_width = (int64_t) tc.downsample_hidden_size * (n_mels >> 3);
        if (conv_width <= 0 || conv.size() % (size_t) conv_width != 0) {
            err = "Hojo tower conv output size " + std::to_string(conv.size()) +
                  " is not divisible by the flattened width " +
                  std::to_string(conv_width);
            return false;
        }
        const int64_t conv_T = (int64_t) conv.size() / conv_width;
        per_window_conv.push_back(std::move(conv));
        per_window_frames.push_back(conv_T);
    }

    // 3. conv_out Linear (7680 -> 1280) per window, + sinusoidal pos emb, then
    //    pack valid frames into one sequence. Build cu_seqlens for the block mask.
    const int64_t d_model = tc.d_model;  // 1280
    const int64_t n_layers = tc.encoder_layers;
    const float ln_eps = (float) tc.layer_norm_eps;
    int64_t total_T = 0;
    for (auto ft : per_window_frames) total_T += ft;

    std::vector<float> packed((size_t) total_T * d_model, 0.0f);
    {
        int64_t pos = 0;
        for (size_t wi = 0; wi < per_window_conv.size(); ++wi) {
            const int64_t conv_T = per_window_frames[wi];
            // Run conv_out (Linear 7680->1280, no bias) as a graph on this
            // window's conv output, add the sinusoidal pos emb, read back.
            std::vector<float> conv_in = std::move(per_window_conv[wi]);
            std::vector<float> conv_out_f;
            bool ok = run_graph([&](ggml_context* c) -> ggml_tensor* {
                int64_t ne[2] = {7680, conv_T};
                ggml_tensor* in = graph_input_tensor(c, GGML_TYPE_F32, 2, ne,
                    conv_in.data(), conv_in.size() * sizeof(float));
                ggml_tensor* y = ggml_mul_mat(c, weight(c, model.loader, "audio.conv_out.weight"),
                                              f32(c, in));  // [1280, conv_T]
                return f32(c, y);
            }, conv_out_f);
            if (!ok) { err = "Hojo tower conv_out graph failed"; return false; }
            if (const char* dp = std::getenv("STARLING_HOJO_DUMP_CONVOUT")) {
                if (FILE* f = std::fopen(dp, "wb")) {
                    std::fwrite(conv_out_f.data(), sizeof(float), conv_out_f.size(), f);
                    std::fclose(f);
                }
            }
            // conv_out_f is [d_model, conv_T] (column-per-token). Add per-window
            // sinusoidal pos emb (position index 0..conv_T-1, reset per window).
            std::vector<float> pe = sinusoids_pos_emb(conv_T, d_model);
            for (int64_t t = 0; t < conv_T; ++t)
                for (int64_t d = 0; d < d_model; ++d)
                    packed[(size_t)(pos + t) * d_model + d] =
                        conv_out_f[(size_t) t * d_model + d] + pe[(size_t) t * d_model + d];
            pos += conv_T;
        }
    }

    // 4. Build the block-diagonal additive mask [total_T, total_T]: 0 within a
    //    window's frame range, -inf across windows.
    std::vector<float> mask((size_t) total_T * total_T);
    const float neg = -3.3895313892515355e38f;
    for (size_t i = 0; i < mask.size(); ++i) mask[i] = neg;
    {
        int64_t start = 0;
        for (auto ft : per_window_frames) {
            for (int64_t i = 0; i < ft; ++i)
                for (int64_t j = 0; j < ft; ++j)
                    mask[(size_t)(start + i) * total_T + (start + j)] = 0.0f;
            start += ft;
        }
    }

    // 5. Run the 32 transformer layers + ln_post + proj1 GELU proj2 as a graph.
    std::vector<float> body_out;
    bool ok = run_graph([&](ggml_context* c) -> ggml_tensor* {
        int64_t ne[2] = {d_model, total_T};
        ggml_tensor* x = graph_input_tensor(c, GGML_TYPE_F32, 2, ne,
                                            packed.data(), packed.size() * sizeof(float));
        int64_t mne[2] = {total_T, total_T};
        ggml_tensor* mt = graph_input_tensor(c, GGML_TYPE_F32, 2, mne,
                                             mask.data(), mask.size() * sizeof(float));
        for (int64_t li = 0; li < n_layers; ++li)
            x = build_tower_layer(c, model, (int) li, total_T, x, mt, ln_eps);
        x = layer_norm(c, model.loader, x, "audio.ln_post", ln_eps);
        x = linear(c, model.loader, x, "audio.proj1", true);
        x = exact_gelu(c, x);
        x = linear(c, model.loader, x, "audio.proj2", true);  // [output_dim, total_T]
        return f32(c, x);
    }, body_out);
    if (!ok) { err = "Hojo tower transformer graph failed"; return false; }

    // Optional dump (STARLING_HOJO_DUMP_TOWER) for byte-exactness debugging.
    if (const char* dp = std::getenv("STARLING_HOJO_DUMP_TOWER")) {
        if (FILE* f = std::fopen(dp, "wb")) {
            std::fwrite(body_out.data(), sizeof(float), body_out.size(), f);
            std::fclose(f);
        }
    }

    out.data = std::move(body_out);
    out.n_speech = total_T;
    out.width = tc.output_dim;
    out.per_window_frames = per_window_frames;
    return true;
}
} // namespace starling::ggml::hojo
