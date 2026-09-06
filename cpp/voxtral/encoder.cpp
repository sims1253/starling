// encoder.cpp — Voxtral causal audio encoder + projector (see encoder.hpp).
//
// The stock embedder is two CAUSAL conv1ds: conv1 (k3 s1) prepends 2 zero
// columns (left_pad = k - s = 2) and conv2 (k3 s2) prepends 1 (left_pad =
// 3 - 2 = 1), each followed by exact GELU, then a transpose to (T', 1280)
// with T' = mel_T/2. ggml_conv_1d applies SYMMETRIC padding only (no
// left-pad-only mode), so both convo paths run the scalar left-pad conv on
// the host and feed the padded time series in-graph:
//   - host path: scalar conv1+GELU -> scalar conv2+GELU -> one-shot graph for
//     layers + projector (the portable reference).
//   - replay path: bf16-rounded left-pad staging of the mel (conv1 pad 2)
//     and of the conv1 output (conv2 pad 1, stride-2 gather) on the host,
//     then ONE captured ReplayGraph per mel_T running gemm-conv1/gelu/
//     gemm-conv2/gelu/layers/projector end-to-end (the convs as in-graph GEMMs
//     over the padded inputs, mirroring ark's fused encoder structure). The
//     GEMM weights are [K, IC]-transposed to the im2col/staging row order
//     (channel inner, window outer); tensors stay channel-major [C, T]
//     throughout (never row-major), matching the host scalar conv exactly.
//
// Attention: full MHA (kv heads == q heads), rotate-half RoPE on the FULL
// head_dim (theta 1e6), and a band-causal sliding-window mask: position i
// attends j iff j <= i && i-j < window, blocked cells getting bf16-min
// (-3.3895313892515355e38, torch.finfo(bf16).min) added to the f32 scores
// before softmax. The mask is a per-T_enc [T_enc, T_enc] f32 graph input;
// its memory is the O(T^2) budget (see kVoxtralMaxMaskBytes).
//
// Numeric discipline: the bf16 oracle (activations in bf16, elementwise math
// in f32, rounding at the bf16 boundary), matching the checkpoint dtype.
#include "encoder.hpp"
#include "runtime/backend.hpp"
#include "runtime/graph.hpp"
#include "runtime/graph_builder.hpp"
#include "runtime/lru_cache.hpp"
#include "lib/graph_helpers.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace starling::ggml::voxtral {

namespace {

using lib::weight;
using lib::bf16;
using lib::f32;

// Blocked-attention additive constant: torch.finfo(bfloat16).min, the value
// the stock mask materializes in the encoder dtype.
constexpr float kMaskedBias = -3.389531388998e38f;  // nearest f32 to -3.3895313892515355e38

bool debug_enabled() {
    return lib::debug_enabled("STARLING_VOXTRAL_DEBUG");
}
bool force_replay() {
    const char* p = std::getenv("STARLING_VOXTRAL_FORCE_REPLAY");
    return p && std::string(p) == "1";
}

// ---- host math ------------------------------------------------------------
// Causal conv1d + exact GELU in f32 over a feat-major [IC, L] bf16 input with
// an explicit LEFT pad (zeros prepended, never symmetric). Returns the
// oc-contiguous [OC, OL] f32 output (element (oc,t) at oc*OL+t).
std::vector<float> host_causal_conv1d_gelu(const ModelLoader& ml,
                                           const std::vector<ggml_bf16_t>& in,
                                           int64_t IC, int64_t L, int64_t left_pad,
                                           int64_t stride, const std::string& wname) {
    std::vector<float> wf = lib::read_f32(ml, (wname + ".weight").c_str());  // [OC, IC, K]
    std::vector<float> bf = lib::read_f32(ml, (wname + ".bias").c_str());    // [OC]
    const int64_t OC = (int64_t) bf.size(), K = 3;
    const int64_t OL = (L + left_pad - K) / stride + 1;
    std::vector<float> x((size_t) IC * (L + left_pad), 0.0f);
    for (int64_t c = 0; c < IC; ++c)
        for (int64_t t = 0; t < L; ++t)
            x[(size_t) c * (L + left_pad) + (t + left_pad)] =
                ggml_bf16_to_fp32(in[(size_t) c * L + t]);
    std::vector<float> y((size_t) OC * OL, 0.0f);
    for (int64_t oc = 0; oc < OC; ++oc) {
        for (int64_t t = 0; t < OL; ++t) {
            double acc = (double) bf[(size_t) oc];
            for (int64_t c = 0; c < IC; ++c)
                for (int64_t k = 0; k < K; ++k)
                    acc += (double) wf[((size_t) oc * IC + c) * K + k] *
                           (double) x[(size_t) c * (L + left_pad) + t * stride + k];
            float v = (float) acc;
            v = 0.5f * v * (1.0f + std::erf(v / (float) M_SQRT2));  // exact GELU
            y[(size_t) oc * OL + t] = v;
        }
    }
    return y;
}

// Host RMSNorm: f32 normalize, affine in f32 with the bf16 weight, one bf16
// round (mirrors rms_single). `x` is row-major [T, W].
std::vector<float> host_rms_norm(const std::vector<float>& x, int64_t T, int64_t W,
                                 const std::vector<float>& w, float eps) {
    std::vector<float> y((size_t) T * W);
    for (int64_t t = 0; t < T; ++t) {
        double ms = 0.0;
        for (int64_t d = 0; d < W; ++d) {
            const double v = x[(size_t) t * W + d];
            ms += v * v;
        }
        const float s = (float)(1.0 / std::sqrt(ms / (double) W + (double) eps));
        for (int64_t d = 0; d < W; ++d) {
            const float v = x[(size_t) t * W + d] * s * w[d];
            y[(size_t) t * W + d] = ggml_bf16_to_fp32(ggml_fp32_to_bf16(v));
        }
    }
    return y;
}

// ---- graph builders --------------------------------------------------------
// RMSNorm in f32 with the affine in f32 and ONE bf16 round at the end (the
// F.rms_norm semantics the stock VoxtralRealtimeRMSNorm runs: weight * normed
// computed in the activation dtype, one store).
ggml_tensor* rms_single(ggml_context* c, const ModelLoader& ml, ggml_tensor* x,
                        const std::string& n, float eps) {
    return lib::rms_single(c, ml, x, n + ".weight", eps);
}
ggml_tensor* exact_gelu(ggml_context* c, ggml_tensor* x) {
    return lib::gelu_erf_bf16(c, x);
}

// nn.Linear in the bf16 oracle with the checkpoint's round order: the GEMM
// output rounds to bf16 BEFORE the bias adds in f32 (a bf16 matmul stores a
// bf16 result, so the bias never sees the f32 accumulator), then one more
// round. Unbiased linears use lib::linear_bf16 directly (identical: a single
// round of the GEMM output).
ggml_tensor* linear_bf16_oracle(ggml_context* c, const ModelLoader& ml,
                                ggml_tensor* x, const std::string& n, bool bias) {
    ggml_tensor* y = ggml_mul_mat(c, weight(c, ml, n + ".weight"), bf16(c, x));
    if (bias)
        y = ggml_add(c, f32(c, bf16(c, y)),
                     f32(c, weight(c, ml, n + ".bias")));
    return bf16(c, y);
}

// Rotate-half RoPE on the FULL head_dim (transformers rotate_half: the first
// half negated-swapped with the second half), theta from config. q/k are
// [AW, T] bf16 (heads folded into AW = H*head_dim); returns [AW, T] bf16.
// cos/sin are [T, head_dim/2] f32 tables: cos[pos, i] at pos*(hd/2)+i.
ggml_tensor* apply_vox_rope(ggml_context* c, ggml_tensor* x, ggml_tensor* cos_t,
                            ggml_tensor* sin_t, int64_t head_dim, int64_t n_heads,
                            int64_t T) {
    const int64_t half = head_dim / 2;
    ggml_tensor* xf = f32(c, x);
    // Split each head's [head_dim] row into halves: x1 = first half, x2 = last.
    // A head's dims are contiguous (d=0..D-1); reshape to [D, H, T] and view
    // the two halves as [half, H, T] slices.
    ggml_tensor* x3 = ggml_reshape_3d(c, xf, head_dim, n_heads, T);
    ggml_tensor* x1 = ggml_view_3d(c, x3, half, n_heads, T, x3->nb[1], x3->nb[2], 0);
    ggml_tensor* x2 = ggml_view_3d(c, x3, half, n_heads, T, x3->nb[1], x3->nb[2],
                                   (size_t) half * x3->nb[0]);
    // cos/sin rows [half, T] -> [half, 1, T] broadcast over heads... reshape
    // to 4D [half, 1, 1, T]-style via [half, 1, T] with a head broadcast: the
    // head axis is ne[2] of the [half, H, T] operands, so use [half, 1, T].
    // HF casts the tables to the activation dtype before the elementwise
    // rope, so they round to bf16 once here (the fill computes f32).
    ggml_tensor* cos_row = ggml_view_2d(c, cos_t, half, T, cos_t->nb[1], 0);
    ggml_tensor* sin_row = ggml_view_2d(c, sin_t, half, T, sin_t->nb[1], 0);
    // Repeat over heads explicitly: cosH/sinH are [half, H, T] with
    // (i,h,t) = table (i,t), so every op below is same-shape elementwise
    // (no reliance on broadcast dim-1 over heads).
    auto expand_heads = [&](ggml_tensor* row) {
        ggml_tensor* r3 = ggml_reshape_3d(c, row, half, 1, T);
        ggml_tensor* dst = ggml_new_tensor_3d(c, GGML_TYPE_F32, half, n_heads, T);
        return ggml_repeat(c, r3, dst);
    };
    ggml_tensor* cosH = expand_heads(f32(c, bf16(c, cos_row)));
    ggml_tensor* sinH = expand_heads(f32(c, bf16(c, sin_row)));
    // rotate_half with per-op bf16 rounds (torch rounds every elementwise op):
    // first  = bf16(bf16(x1*cos) - bf16(x2*sin))
    // second = bf16(bf16(x2*cos) + bf16(x1*sin))
    auto r16 = [&](ggml_tensor* z) { return f32(c, bf16(c, z)); };
    ggml_tensor* rx1 = r16(ggml_sub(c, r16(ggml_mul(c, x1, cosH)),
                                    r16(ggml_mul(c, x2, sinH))));
    ggml_tensor* rx2 = r16(ggml_add(c, r16(ggml_mul(c, x2, cosH)),
                                    r16(ggml_mul(c, x1, sinH))));
    ggml_tensor* rot = ggml_concat(c, rx1, rx2, 0);      // [head_dim, H, T]
    return ggml_reshape_2d(c, bf16(c, rot), head_dim * n_heads, T);
}

// Band-causal sliding-window attention (full MHA). q/k/v are [AW, T] bf16.
// `mask` is the [T, T] f32 additive band mask (query-major: element (i,j) at
// i*T+j); `cos_t`/`sin_t` are the [T, head_dim/2] f32 RoPE tables. Returns
// [AW, T] bf16. When `cap` is non-null and li==0, captures the layer-0 bisect
// stages RAW ([AW, T] ggml flat == torch [T, AW] row-major flat, so the test
// compares flats directly; prob is [T, T, H] with head 0 first in the same
// correspondence). Readback converts bf16; no extra graph nodes.
ggml_tensor* band_attention(ggml_context* ctx, const VoxtralModel& model,
                            ggml_tensor* q, ggml_tensor* k, ggml_tensor* v,
                            ggml_tensor* mask, ggml_tensor* cos_t,
                            ggml_tensor* sin_t, int64_t T,
                            EncoderDebug* cap = nullptr, int64_t li = -1) {
    const auto& ec = model.config.encoder;
    const int64_t H = ec.n_heads, D = ec.head_dim;
    const int64_t AW = H * D;
    const float scale = 1.0f / std::sqrt((float) D);
    q = apply_vox_rope(ctx, q, cos_t, sin_t, D, H, T);
    k = apply_vox_rope(ctx, k, cos_t, sin_t, D, H, T);
    if (cap && li == 0) {
        if (cap->l0_qr) capture_graph_output(q, cap->l0_qr);
        if (cap->l0_kr) capture_graph_output(k, cap->l0_kr);
    }
    // Head-split: [AW, T] -> [D, H, T] -> [D, T, H] (head batch in ne[2]).
    auto to_heads = [&](ggml_tensor* z) {
        z = ggml_reshape_3d(ctx, z, D, H, T);
        return ggml_cont(ctx, ggml_permute(ctx, z, 0, 2, 1, 3));  // [D, T, H]
    };
    ggml_tensor* qh = to_heads(q);
    ggml_tensor* kh = to_heads(k);
    ggml_tensor* vh = to_heads(v);
    // Scores [T, T, H]: K^T Q, scaled in f32. HF numerics: the bf16 matmul
    // rounds its output once, then the scalar scale rounds again (torch
    // elementwise bf16 ops); the f32 band mask is added AFTER, in f32, and
    // the softmax reduces in f32. The mask input is query-major rows
    // ([T queries, T keys], element (q,k) at q*T+k), which is exactly the
    // softmax kernel's access (row i01 = query i01, reduced contiguously
    // over keys), so it feeds soft_max_ext directly.
    ggml_tensor* scores = ggml_mul_mat(ctx, kh, qh);  // [T(keys), T(queries), H]
    scores = f32(ctx, bf16(ctx, scores));
    scores = f32(ctx, bf16(ctx, ggml_scale(ctx, scores, scale)));
    ggml_tensor* prob = ggml_soft_max_ext(ctx, scores, f32(ctx, mask), 1.0f, 0.0f);
    prob = bf16(ctx, prob);
    if (cap && li == 0 && cap->l0_prob) capture_graph_output(prob, cap->l0_prob);
    ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, vh, 1, 0, 2, 3));  // [T, D, H]
    ggml_tensor* co = bf16(ctx, ggml_mul_mat(ctx, vt, prob));             // [D, T, H]
    co = ggml_cont(ctx, ggml_permute(ctx, co, 0, 2, 1, 3));               // [D, H, T]
    ggml_tensor* cout = ggml_reshape_2d(ctx, co, AW, T);
    if (cap && li == 0 && cap->l0_ctx) capture_graph_output(cout, cap->l0_ctx);
    return cout;
}

// Encoder body: the N pre-norm layers + final RMSNorm. `x` is the [d_model,
// T_enc] conv output; `mask`/`cos_t`/`sin_t` are the per-T_enc tables.
// Returns the post-final-norm hidden state. Debug captures route through
// `cap` (null when disabled).
ggml_tensor* build_encoder_layers(ggml_context* ctx, const VoxtralModel& model,
                                  int64_t T_enc, ggml_tensor* x, ggml_tensor* mask,
                                  ggml_tensor* cos_t, ggml_tensor* sin_t,
                                  EncoderDebug* cap) {
    const auto& ec = model.config.encoder;
    const ModelLoader& ml = model.loader;
    for (uint32_t li = 0; li < ec.n_layers; ++li) {
        const std::string pre = "enc.blk." + std::to_string(li) + ".";
        ggml_tensor* r = x;
        ggml_tensor* n = rms_single(ctx, ml, x, pre + "attn_norm", ec.rms_norm_eps);
        // q/v/o have bias, k has NO bias (Whisper convention, kept here).
        ggml_tensor* q = linear_bf16_oracle(ctx, ml, n, pre + "attn.q", true);
        ggml_tensor* k = lib::linear_bf16(ctx, ml, n, pre + "attn.k", false);
        ggml_tensor* vv = linear_bf16_oracle(ctx, ml, n, pre + "attn.v", true);
        // Layer-0 bisect captures (raw bf16 graph tensors; the [AW, T] ggml
        // flat order matches the torch [T, AW] row-major flats; null-guarded,
        // off by default, no extra graph nodes).
        if (cap && (int64_t) li == 0) {
            if (cap->l0_n) capture_graph_output(n, cap->l0_n);
            if (cap->l0_q) capture_graph_output(q, cap->l0_q);
            if (cap->l0_k) capture_graph_output(k, cap->l0_k);
            if (cap->l0_v) capture_graph_output(vv, cap->l0_v);
        }
        ggml_tensor* joined = band_attention(ctx, model, q, k, vv, mask, cos_t,
                                             sin_t, T_enc, cap, (int64_t) li);
        ggml_tensor* a = linear_bf16_oracle(ctx, ml, joined, pre + "attn.o", true);
        if (cap && (int64_t) li == 0 && cap->l0_a)
            capture_graph_output(a, cap->l0_a);
        x = lib::addb(ctx, r, a);
        r = x;
        n = rms_single(ctx, ml, x, pre + "ffn_norm", ec.rms_norm_eps);
        // SwiGLU: silu(gate) * up computed in f32 with ONE bf16 round (the
        // oracle order: no intermediate store of the silu output), then down
        // WITH bias under the oracle round order.
        ggml_tensor* g = lib::linear_bf16(ctx, ml, n, pre + "ffn.gate", false);
        ggml_tensor* u = lib::linear_bf16(ctx, ml, n, pre + "ffn.up", false);
        ggml_tensor* h = bf16(ctx, ggml_mul(ctx, ggml_silu(ctx, f32(ctx, g)),
                                            f32(ctx, u)));
        h = linear_bf16_oracle(ctx, ml, h, pre + "ffn.down", true);
        if (cap && (int64_t) li == 0 && cap->l0_ffn)
            capture_graph_output(h, cap->l0_ffn);
        x = lib::addb(ctx, r, h);
        if (cap && cap->layer_outs) {
            for (size_t di = 0; di < cap->layer_idx.size(); ++di)
                if (cap->layer_idx[di] == (int64_t) li)
                    capture_graph_output(f32(ctx, x), &(*cap->layer_outs)[di]);
        }
    }
    return rms_single(ctx, ml, x, "enc.final_norm", ec.rms_norm_eps);
}

// Projector: group-by-downsample reshape [Dm, T_enc] -> [Dm*ds, N], then
// Linear(input->output, no bias) GELU Linear(output->output, no bias).
// Returns the F32 projector output [Po, N].
ggml_tensor* build_projector(ggml_context* ctx, const VoxtralModel& model,
                             ggml_tensor* x, int64_t T_enc, int64_t* N_out) {
    const auto& pc = model.config.projector;
    const int64_t N = T_enc / (int64_t) pc.downsample;
    *N_out = N;
    x = ggml_reshape_2d(ctx, x, (int64_t) pc.input_size, N);
    ggml_tensor* h = lib::linear_bf16(ctx, model.loader, x, "proj.fc0", false);
    h = lib::gelu_erf_bf16(ctx, h);
    h = lib::linear_bf16(ctx, model.loader, h, "proj.fc2", false);
    return f32(ctx, h);
}

} // namespace

// ---------------------------------------------------------------------------
// Fused encode + project with a per-T_enc bounded-LRU ReplayGraph cache
// (GPU / STARLING_VOXTRAL_FORCE_REPLAY=1). The causal convs run as in-graph
// GEMMs over host-left-padded inputs (ggml has no left-pad-only conv mode):
// the mel input is staged with conv1's left pad (2 zero columns) and the
// conv1 output with conv2's (1 zero column); the GEMM windows then implement
// the causal offsets exactly. Keyed on T_enc (the mask + RoPE tables + conv
// GEMM shapes all depend on it; the cache is LRU-bounded).
// ---------------------------------------------------------------------------
struct EncoderReplayEntry {
    int64_t T_enc = 0, N = 0;
    GraphInputPool pool;
    std::unique_ptr<ReplayGraph> conv1_graph;  // padded mel -> GELU [Dm, mel_T]
    std::unique_ptr<ReplayGraph> body_graph;   // staged conv1 cols -> projector out
    float* mel_pad_buf = nullptr;    // [(mel_T+2)*n_mels] f32, conv1's padded input
    float* c1_pad_buf = nullptr;     // [3*Dm*T_enc] f32, conv2's staged window columns
    std::vector<float> c1_rows;      // conv1 graph output staging [mel_T*Dm]
    float* mask_buf = nullptr;       // [T_enc*T_enc] f32 band mask
    float* cos_buf = nullptr;        // [T_enc*head_dim/2] f32 RoPE cos
    float* sin_buf = nullptr;        // [T_enc*head_dim/2] f32 RoPE sin
};

namespace {
struct ShapeKey {
    int64_t T_enc;
    bool operator==(const ShapeKey& o) const { return T_enc == o.T_enc; }
};
struct ShapeKeyHash {
    size_t operator()(const ShapeKey& k) const noexcept { return (size_t) k.T_enc; }
};
std::unique_ptr<LruCache<ShapeKey, EncoderReplayEntry, ShapeKeyHash>> g_encoder_cache;
std::once_flag g_encoder_cache_once;
void register_encoder_cache_clearer_once() {
    std::call_once(g_encoder_cache_once, [] {
        register_decode_cache_clearer([] { g_encoder_cache.reset(); });
    });
}
} // namespace

size_t encoder_replay_cache_size() {
    return g_encoder_cache ? g_encoder_cache->size() : 0;
}

bool encode_audio_and_project(const VoxtralModel& model, const MelFeatures& mel,
                              AudioEncoding& out, std::string& err,
                              EncoderDebug* dbg) {
    ensure_weights_realized(model.loader);
    const auto& ec = model.config.encoder;
    const auto& pc = model.config.projector;
    const int64_t Dm = ec.d_model, Pm = ec.num_mel_bins;
    if (mel.n_mels != Pm || mel.n_frames <= 0 ||
        mel.data.size() != (size_t) mel.n_mels * mel.n_frames ||
        mel.f32.size() != mel.data.size()) {
        err = "invalid VOXTRAL mel shape/data";
        return false;
    }
    const int64_t mel_T = mel.n_frames;
    // conv1 (k3 s1, left-pad 2) preserves length; conv2 (k3 s2, left-pad 1)
    // halves: OL = (L+1-3)/2+1. The offline mel is always a multiple of 8,
    // so mel_T is even and the projector's group-by-4 divides with no
    // remainder; a foreign mel_T is still rejected loudly.
    if (mel_T % 2 != 0) {
        err = "VOXTRAL mel length not even (conv stride-2 requires mel_T % 2 == 0)";
        return false;
    }
    const int64_t T_enc = mel_T / 2;
    if (T_enc % (int64_t) pc.downsample != 0) {
        err = "VOXTRAL encoder length not divisible by the projector downsample";
        return false;
    }
    // Mask-memory guard: the [T_enc, T_enc] f32 band mask is the O(T^2)
    // term; reject with a shorter-audio hint before allocating anything.
    {
        double bytes = (double) T_enc * (double) T_enc * 4.0;
        if (bytes > (double) kVoxtralMaxMaskBytes) {
            err = "VOXTRAL audio too long for the encoder mask budget "
                  "(T_enc=" + std::to_string(T_enc) +
                  "); use shorter audio (the [T,T] f32 mask would exceed 1 GiB)";
            return false;
        }
    }

    const bool use_replay = global_backend().is_gpu() || force_replay() || debug_enabled();

    // ---- shared host staging ------------------------------------------------
    // Mel f32, feat-major [Pm, mel_T] -> conv1's left-padded input
    // [(mel_T+2), Pm] f32 (element (t, c) at t*Pm+c: time-major rows, the
    // layout the in-graph GEMM consumes directly). Values round to bf16: the
    // stock embedder consumes bf16 mel (the host scalar path reads mel.data,
    // the torch reference rounds explicitly), so the replay GEMM must see the
    // same bf16 boundary or every downstream bf16 store flips independently.
    std::vector<float> mel_pad((size_t)(mel_T + 2) * Pm, 0.0f);
    for (int64_t c = 0; c < Pm; ++c)
        for (int64_t t = 0; t < mel_T; ++t)
            mel_pad[(size_t)(t + 2) * Pm + c] =
                ggml_bf16_to_fp32(ggml_fp32_to_bf16(mel.f32[(size_t) c * mel_T + t]));

    // RoPE tables [T_enc, head_dim/2] f32 row-major: inv_freq[i] =
    // theta^(-2i/head_dim), cos[pos, i] = cos(pos*inv_freq[i]).
    const int64_t Hd = ec.head_dim, half = Hd / 2;
    std::vector<float> cos_tab((size_t) T_enc * half), sin_tab((size_t) T_enc * half);
    for (int64_t p = 0; p < T_enc; ++p) {
        for (int64_t i = 0; i < half; ++i) {
            // Stock computes a float32 power, then reciprocal, then the
            // position product in float32 before sin/cos. Double precision
            // here can cross a later bf16 rounding boundary.
            const float inv = 1.0f / std::pow(ec.rope_theta,
                                             (2.0f * (float) i) / (float) Hd);
            const float a = (float) p * inv;
            cos_tab[(size_t) p * half + i] = (float) std::cos(a);
            sin_tab[(size_t) p * half + i] = (float) std::sin(a);
        }
    }
    // Band mask [T_enc, T_enc] f32 query-major (element (i,j) at i*T+j):
    // 0 when j <= i && i-j < window, else bf16-min.
    std::vector<float> mask((size_t) T_enc * T_enc);
    for (int64_t i = 0; i < T_enc; ++i)
        for (int64_t j = 0; j < T_enc; ++j)
            mask[(size_t) i * T_enc + j] =
                (j <= i && i - j < (int64_t) ec.sliding_window) ? 0.0f : kMaskedBias;

    // ---- CPU host path: scalar convs, one-shot layers+projector graph -------
    if (!use_replay) {
        std::vector<float> c1 = host_causal_conv1d_gelu(model.loader, mel.data, Pm,
                                                        mel_T, ec.conv_left_pad1,
                                                        1, "enc.conv1");
        std::vector<ggml_bf16_t> c1_bf16(c1.size());
        for (size_t i = 0; i < c1.size(); ++i) c1_bf16[i] = ggml_fp32_to_bf16(c1[i]);
        std::vector<float> c2 = host_causal_conv1d_gelu(model.loader, c1_bf16, Dm,
                                                        mel_T, ec.conv_left_pad2,
                                                        ec.conv_stride2, "enc.conv2");
        // Transpose oc-contig [Dm, T_enc] -> token-major [T_enc, Dm] for the
        // graph input (ggml ne0=Dm layout: element (d,t) at t*Dm+d), bf16.
        std::vector<ggml_bf16_t> conv_bf16((size_t) Dm * T_enc);
        for (int64_t t = 0; t < T_enc; ++t)
            for (int64_t d = 0; d < Dm; ++d)
                conv_bf16[(size_t) t * Dm + d] =
                    ggml_fp32_to_bf16(c2[(size_t) d * T_enc + t]);
        if (dbg && dbg->embedder) {
            dbg->embedder->resize((size_t) T_enc * Dm);
            for (size_t i = 0; i < dbg->embedder->size(); ++i)
                (*dbg->embedder)[i] = ggml_bf16_to_fp32(conv_bf16[i]);
        }
        int64_t N = 0;
        std::vector<float> body_out;
        if (dbg && dbg->layer_outs)
            dbg->layer_outs->resize(dbg->layer_idx.size());
        bool ok = run_graph([&](ggml_context* ctx) -> ggml_tensor* {
            int64_t cne[2] = {Dm, T_enc};
            ggml_tensor* x = graph_input_tensor(ctx, GGML_TYPE_BF16, 2, cne,
                                                conv_bf16.data(),
                                                conv_bf16.size() * sizeof(ggml_bf16_t));
            // The bf16 conv output feeds the layers through the bf16 oracle
            // boundary (linear_bf16 casts inputs explicitly; the input tensor
            // itself stays bf16 so no in-graph f32->bf16 pool conversion).
            int64_t mne[2] = {T_enc, T_enc};
            ggml_tensor* m = graph_input_tensor(ctx, GGML_TYPE_F32, 2, mne,
                                                mask.data(), mask.size() * sizeof(float));
            int64_t rne[2] = {half, T_enc};
            ggml_tensor* co = graph_input_tensor(ctx, GGML_TYPE_F32, 2, rne,
                                                 cos_tab.data(),
                                                 cos_tab.size() * sizeof(float));
            ggml_tensor* si = graph_input_tensor(ctx, GGML_TYPE_F32, 2, rne,
                                                 sin_tab.data(),
                                                 sin_tab.size() * sizeof(float));
            x = bf16(ctx, x);
            ggml_tensor* enc = build_encoder_layers(ctx, model, T_enc, x, m, co, si, dbg);
            return build_projector(ctx, model, enc, T_enc, &N);
        }, body_out);
        if (!ok) { err = "VOXTRAL encoder graph execution failed"; return false; }
        out.data = std::move(body_out);
        out.n_tokens = N;
        out.width = pc.output_size;
        return true;
    }

    // ---- replay path: captured per-T_enc graph ------------------------------
    // The causal convs run as in-graph GEMMs over host-left-padded inputs
    // (ggml has no left-pad-only conv mode): conv1 reads the mel staged with
    // 2 leading zero frames, conv2 reads the conv1 output staged with 1.
    // Staging conv2's pad needs conv1's OUTPUT values, which only exist after
    // conv1 runs -- so each replay is two steps: (1) run the small captured
    // conv1 graph, (2) stage conv2's padded input on the host, (3) replay the
    // captured layers+projector graph. The cached entry owns both graphs.
    register_encoder_cache_clearer_once();
    if (!g_encoder_cache)
        g_encoder_cache = std::unique_ptr<LruCache<ShapeKey, EncoderReplayEntry, ShapeKeyHash>>(
            new LruCache<ShapeKey, EncoderReplayEntry, ShapeKeyHash>(replay_cache_size()));

    ShapeKey key{T_enc};
    EncoderReplayEntry& e = *g_encoder_cache->get_or_init(key,
        [&](EncoderReplayEntry& entry) {
            entry.T_enc = T_enc;
            entry.mel_pad_buf = reinterpret_cast<float*>(entry.pool.alloc_bytes(
                (size_t)(mel_T + 2) * Pm * sizeof(float)));
            entry.c1_pad_buf = reinterpret_cast<float*>(entry.pool.alloc_bytes(
                (size_t) 3 * Dm * T_enc * sizeof(float)));
            entry.mask_buf = reinterpret_cast<float*>(entry.pool.alloc_bytes(
                (size_t) T_enc * T_enc * sizeof(float)));
            entry.cos_buf = reinterpret_cast<float*>(entry.pool.alloc_bytes(
                (size_t) T_enc * half * sizeof(float)));
            entry.sin_buf = reinterpret_cast<float*>(entry.pool.alloc_bytes(
                (size_t) T_enc * half * sizeof(float)));
            std::memcpy(entry.mel_pad_buf, mel_pad.data(),
                        (size_t)(mel_T + 2) * Pm * sizeof(float));
            std::memcpy(entry.mask_buf, mask.data(),
                        (size_t) T_enc * T_enc * sizeof(float));
            std::memcpy(entry.cos_buf, cos_tab.data(),
                        (size_t) T_enc * half * sizeof(float));
            std::memcpy(entry.sin_buf, sin_tab.data(),
                        (size_t) T_enc * half * sizeof(float));
            // Im2col helper: stack the 3 window columns per output frame. `pin`
            // is [IC, Lpad] (element (c, t) at c+IC*t, matching the host's
            // row-major [Lpad, IC] staging); the window for output t is
            // columns (t, t+1, t+2). Returns the [3*IC, Lout] f32 columns
            // (dim-0 concat: ne0 sums to 3*IC) with rows r = c+IC*k (channel
            // inner, window outer).
            auto im2col = [&](ggml_context* ctx, ggml_tensor* pin, int64_t Lout,
                              int64_t IC) {
                ggml_tensor* cols[3];
                for (int k = 0; k < 3; ++k)
                    cols[k] = ggml_view_2d(ctx, pin, IC, Lout, pin->nb[1],
                                           (size_t) k * IC * sizeof(float));
                ggml_tensor* cat = ggml_concat(ctx, cols[0], cols[1], 0);
                cat = ggml_concat(ctx, cat, cols[2], 0);  // [3*IC, Lout]
                return ggml_cont(ctx, cat);
            };
            // Conv1 GEMM weight: [K, IC, OC] (ne0=K, rows r = k+K*c, window
            // inner) reordered to the im2col row order r = c+IC*k (channel
            // inner, window outer) via a [K, IC] slice transpose.
            auto gemm_weight = [&](ggml_context* ctx, const std::string& n,
                                   int64_t IC, int64_t OC) {
                ggml_tensor* w = weight(ctx, model.loader, n);
                ggml_tensor* wt = ggml_cont(ctx, ggml_permute(ctx, f32(ctx, w),
                                                              1, 0, 2, 3));
                return ggml_reshape_2d(ctx, wt, 3 * IC, OC);  // rows c+IC*k
            };
            // Conv1 graph: padded mel -> GELU output [Dm, mel_T] (element
            // (d, t) at d+Dm*t; the host stages conv2's windows by reading
            // these channel-major columns directly).
            entry.conv1_graph = std::make_unique<ReplayGraph>(global_backend(),
                [&](ggml_context* ctx) -> ggml_tensor* {
                    int64_t pne[2] = {Pm, (int64_t)(mel_T + 2)};
                    ggml_tensor* pad1 = graph_input_tensor(ctx, GGML_TYPE_F32, 2, pne,
                        entry.mel_pad_buf, (size_t)(mel_T + 2) * Pm * sizeof(float));
                    ggml_tensor* w1m = gemm_weight(ctx, "enc.conv1.weight",
                                                   Pm, Dm);
                    ggml_tensor* c1 = ggml_mul_mat(ctx, w1m,
                                                   im2col(ctx, pad1, mel_T, Pm));
                    c1 = ggml_add(ctx, f32(ctx, c1),
                                  ggml_reshape_2d(ctx, f32(ctx, weight(ctx, model.loader,
                                                                      "enc.conv1.bias")),
                                                  Dm, 1));
                    return exact_gelu(ctx, c1);  // bf16 [Dm, mel_T]
                });
            // Body graph: conv2 (one GEMM over the host-staged [(mel_T+1),
            // Dm] padded conv1 rows, strided windows gathered on the host
            // into [3*Dm, T_enc] columns) -> transpose -> layers ->
            // projector. The stride-2 gather (rows (2t, 2t+1, 2t+2)) is not
            // expressible as ggml views, so the host stages the columns
            // directly; the GEMM + everything downstream stays in-graph.
            entry.c1_rows.resize((size_t) mel_T * Dm);
            entry.body_graph = std::make_unique<ReplayGraph>(global_backend(),
                [&](ggml_context* ctx) -> ggml_tensor* {
                    int64_t qne[2] = {3 * Dm, T_enc};
                    ggml_tensor* cols = graph_input_tensor(ctx, GGML_TYPE_F32, 2, qne,
                        entry.c1_pad_buf, (size_t) 3 * Dm * T_enc * sizeof(float));
                    ggml_tensor* w2m = gemm_weight(ctx, "enc.conv2.weight",
                                                   Dm, Dm);
                    ggml_tensor* c2 = ggml_mul_mat(ctx, w2m, cols);  // [Dm, T_enc]
                    c2 = ggml_add(ctx, f32(ctx, c2),
                                  ggml_reshape_2d(ctx, f32(ctx, weight(ctx, model.loader,
                                                                      "enc.conv2.bias")),
                                                  Dm, 1));
                    ggml_tensor* g2 = exact_gelu(ctx, c2);  // bf16 [Dm, T_enc]
                    if (dbg && dbg->embedder)
                        capture_graph_output(f32(ctx, g2), dbg->embedder);
                    int64_t mne[2] = {T_enc, T_enc};
                    ggml_tensor* m = graph_input_tensor(ctx, GGML_TYPE_F32, 2, mne,
                        entry.mask_buf, (size_t) T_enc * T_enc * sizeof(float));
                    int64_t rne[2] = {half, T_enc};
                    ggml_tensor* co = graph_input_tensor(ctx, GGML_TYPE_F32, 2, rne,
                        entry.cos_buf, (size_t) T_enc * half * sizeof(float));
                    ggml_tensor* si = graph_input_tensor(ctx, GGML_TYPE_F32, 2, rne,
                        entry.sin_buf, (size_t) T_enc * half * sizeof(float));
                    ggml_tensor* enc = build_encoder_layers(ctx, model, T_enc, g2,
                                                            m, co, si, dbg);
                    int64_t N = 0;
                    ggml_tensor* out = build_projector(ctx, model, enc, T_enc, &N);
                    entry.N = N;
                    return out;
                });
            entry.N = T_enc / (int64_t) pc.downsample;
        });
    // Refresh the varying inputs in their stable pool buffers, then re-upload.
    std::memcpy(e.mel_pad_buf, mel_pad.data(),
                (size_t)(mel_T + 2) * Pm * sizeof(float));
    std::memcpy(e.mask_buf, mask.data(), (size_t) T_enc * T_enc * sizeof(float));
    std::memcpy(e.cos_buf, cos_tab.data(), (size_t) T_enc * half * sizeof(float));
    std::memcpy(e.sin_buf, sin_tab.data(), (size_t) T_enc * half * sizeof(float));
    for (size_t i = 0; i < e.conv1_graph->n_inputs(); ++i)
        e.conv1_graph->set_input(i, e.conv1_graph->input_host(i),
                                 e.conv1_graph->input_nbytes(i));

    // Step 1: conv1 -> time-major GELU rows.
    if (!e.conv1_graph->compute(e.c1_rows)) {
        err = "VOXTRAL encoder conv1 replay failed";
        return false;
    }
    // Step 2: stage conv2's strided window columns on the host. The conv1
    // output is [Dm, mel_T] channel-major (element (d, t) at d+Dm*t); column
    // t of the [3*Dm, T_enc] input (element (r, t) at r+3*Dm*t, rows
    // r = k*Dm+d) gathers the 1-left-padded window rows (2t-1, 2t, 2t+1)
    // (row -1 = zeros). The stride-2 gather is not expressible as ggml
    // views, so the host stages the columns directly.
    std::memset(e.c1_pad_buf, 0, (size_t) 3 * Dm * T_enc * sizeof(float));
    {
        float* pad = e.c1_pad_buf;
        const float* g1 = e.c1_rows.data();  // (d, t) at d+Dm*t
        for (int64_t t = 0; t < T_enc; ++t)
            for (int k = 0; k < 3; ++k) {
                const int64_t src_t = 2 * t + k - 1;  // -1 -> zero (memset)
                if (src_t < 0) continue;
                const float* col = g1 + (size_t) Dm * src_t;
                float* dst = pad + (size_t) 3 * Dm * t + (size_t) k * Dm;
                std::memcpy(dst, col, (size_t) Dm * sizeof(float));
            }
    }
    if (dbg && dbg->embedder) {
        // The embedder capture fires inside the body graph (post-conv2 GELU);
        // size it now so the test sees the layout even on graph failure.
        dbg->embedder->assign((size_t) Dm * T_enc, 0.0f);
    }
    if (dbg && dbg->layer_outs) dbg->layer_outs->resize(dbg->layer_idx.size());
    for (size_t i = 0; i < e.body_graph->n_inputs(); ++i)
        e.body_graph->set_input(i, e.body_graph->input_host(i),
                                e.body_graph->input_nbytes(i));

    std::vector<float> tmp;
    if (!e.body_graph->compute_with_captures(tmp)) {
        err = "VOXTRAL encoder replay failed";
        return false;
    }
    out.data = std::move(tmp);
    out.n_tokens = e.N;
    out.width = model.config.projector.output_size;
    return true;
}

} // namespace starling::ggml::voxtral
