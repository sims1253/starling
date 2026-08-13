// audio_encoder.cpp — higgs-audio-v3-stt Whisper encoder + avg_pool + MLP
// projector on the Starling ggml runtime.
//
// The audio path (src/starling/higgs/vendor/modeling/modeling_higgs_audio.py,
// HiggsAudioEncoder.forward + HiggsAudioFeatureProjector.forward):
//   Whisper Conv1d front-end (conv1 K3/s1/p1, conv2 K3/s2/p1, both + exact GELU)
//   -> permute [T,D]->[D,T] (here: keep [d_model, T_enc] column-per-token)
//   -> ADD embed_positions[absolute, applied by frame index 0..T_enc) (NOT RoPE)
//   -> 32 WhisperEncoderLayers (global bidirectional attention, exact GELU FFN)
//   -> permute [B,T,D]->[B,D,T]
//   -> AvgPool1d(kernel=2, stride=2) over the time dim
//   -> permute [B,D,T/2]->[B,T/2,D]
//   -> ln_post (LayerNorm)                       <-- NOTE: AFTER avg_pool, not before
//   -> projector: permute [B,T,D]->[B,D,T]
//                -> depthwise temporal Conv1d(K3,s2,p1,groups=1280)
//                -> permute [B,D,T]->[B,T,D]
//                -> Linear(1280->2048)+bias -> ReLU -> Linear(2048->2048)+bias
//
// KEY byte-exactness subtlety: modeling applies ln_post AFTER avg_pool (the task
// brief's "ln_post -> AvgPool1d" ordering is wrong; the vendor code is
// layers -> permute -> avg_pool -> permute -> layer_norm). This file follows the
// vendor code.
//
// On GPU this is ONE captured ReplayGraph keyed on mel_T (global attention => one
// graph per mel_T, LRU-bounded). On CPU / debug it is the one-shot pair. Mirrors
// ark/audio_encoder.cpp's fused encode + LruCache structure; the difference is
// absolute positional embeddings (not RoPE), the post-layer avg_pool, and the
// depthwise-conv MLP projector (not a merge-by-4 linear adapter).
#include "audio_encoder.hpp"
#include "runtime/backend.hpp"
#include "runtime/graph.hpp"
#include "runtime/graph_builder.hpp"
#include "runtime/lru_cache.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace starling::ggml::higgs {
namespace {

ggml_tensor* weight(ggml_context* c, const ModelLoader& ml, const std::string& n) {
    return clone_weight(c, ml, n.c_str());
}
ggml_tensor* bf16(ggml_context* c, ggml_tensor* x) {
    return x->type == GGML_TYPE_BF16 ? x : ggml_cast(c, x, GGML_TYPE_BF16);
}
ggml_tensor* f32(ggml_context* c, ggml_tensor* x) {
    return x->type == GGML_TYPE_F32 ? x : ggml_cast(c, x, GGML_TYPE_F32);
}
// nn.Linear in the BF16 oracle: GEMM (+ optional bias) exposes F32, round at the
// BF16 boundary.
ggml_tensor* linear(ggml_context* c, const ModelLoader& ml, ggml_tensor* x,
                    const std::string& n, bool bias) {
    ggml_tensor* y = ggml_mul_mat(c, weight(c, ml, n + ".weight"), bf16(c, x));
    if (bias) y = ggml_add(c, f32(c, y), f32(c, weight(c, ml, n + ".bias")));
    return bf16(c, y);
}
// exact GELU (erf), the activation_fn for both conv front-end and FFN
// (config.activation_function="gelu", approximate="none").
ggml_tensor* exact_gelu(ggml_context* c, ggml_tensor* x) {
    return bf16(c, ggml_gelu_erf(c, f32(c, x)));
}
ggml_tensor* add_bf16(ggml_context* c, ggml_tensor* a, ggml_tensor* b) {
    return bf16(c, ggml_add(c, f32(c, a), f32(c, b)));
}
// PyTorch LayerNorm: F32 reduction + affine, one BF16 store. eps from config.
ggml_tensor* layer_norm(ggml_context* c, const ModelLoader& ml, ggml_tensor* x,
                        const std::string& n, float eps) {
    ggml_tensor* y = ggml_norm(c, f32(c, x), eps);
    y = ggml_mul(c, y, f32(c, weight(c, ml, n + ".weight")));
    y = ggml_add(c, y, f32(c, weight(c, ml, n + ".bias")));
    return bf16(c, y);
}

bool debug_enabled() {
    const char* p = std::getenv("STARLING_HIGGS_DEBUG");
    return p && std::strcmp(p, "1") == 0;
}

// Global bidirectional self-attention (WhisperSdpaAttention, absolute positional
// -> NO RoPE here). q/k/v are [d_model, T] bf16. scale = 1/sqrt(head_dim).
// Returns [d_model, T] bf16. is_causal=False (bidirectional); no mask (single
// clip, no padding -> all keys valid).
//
// Two paths share the same head-split inputs (qh/kh/vh = [D, T, H]):
//  - GPU fused path: ggml_flash_attn_ext (the #1 encoder optimization; the full
//    [T,T,H] score tensor is never materialized). Higgs is plain MHA
//    (n_head == n_head_kv == H) and bidirectional -> mask = nullptr.
//  - CPU / byte-exact fallback: the original materialized mul_mat+softmax path,
//    selected via STARLING_HIGGS_NO_FATTN=1 (or non-GPU). Mirrors ark's dual-path.
ggml_tensor* global_attention(ggml_context* ctx, const HiggsModel& model,
                              ggml_tensor* q, ggml_tensor* k, ggml_tensor* v,
                              int64_t T) {
    const auto& ec = model.config.encoder;
    const int H = (int) ec.n_heads, D = (int) ec.head_dim;
    const float scale = 1.0f / std::sqrt((float) D);
    // NO RoPE (higgs uses absolute positional embeddings, already added).

    // Batched attention over all heads. Reshape q/k/v [d_model, T] -> [D, H, T]
    // -> permute to [D, T, H] so the head dim is ne[2] (ggml mul_mat batches over
    // ne[2]). These [D, T, H] tensors are the inputs to BOTH paths: for
    // flash_attn_ext the expected q/k/v layout is [n_embd, n_batch, n_head, ne3]
    // = [D, T, H, 1] (matches exactly).
    auto to_heads = [&](ggml_tensor* z) {
        z = ggml_reshape_3d(ctx, z, D, H, T);
        return ggml_cont(ctx, ggml_permute(ctx, z, 0, 2, 1, 3));  // [D, T, H]
    };
    ggml_tensor* qh = to_heads(q), * kh = to_heads(k), * vh = to_heads(v);

    const char* no_fattn = std::getenv("STARLING_HIGGS_NO_FATTN");
    const bool use_flash = global_backend().is_gpu() &&
                           !(no_fattn && std::string(no_fattn) == "1");
    if (use_flash) {
        // Feed F32 q/k/v to flash (its MMA kernel asserts Q->type==F32, and feeding
        // BF16 forces an in-graph BF16->F16 pool conversion that ggml's CUDA-graph
        // capture rejects). Bidirectional -> mask = nullptr. fa is [D, H, T, 1].
        ggml_tensor* fa = ggml_flash_attn_ext(ctx, f32(ctx, qh), f32(ctx, kh),
                                              f32(ctx, vh), /*mask=*/nullptr,
                                              scale, 0.0f, 0.0f);
        return ggml_reshape_2d(ctx, fa, (int64_t) D * H, T);  // [d_model, T]
    }

    // Manual path (byte-exact reference / CPU / STARLING_HIGGS_NO_FATTN=1).
    ggml_tensor* scores = bf16(ctx, ggml_mul_mat(ctx, kh, qh));         // [T, T, H]
    scores = bf16(ctx, ggml_scale(ctx, f32(ctx, scores), scale));
    ggml_tensor* prob = ggml_soft_max_ext(ctx, f32(ctx, scores), nullptr, 1.0f, 0.0f);
    prob = bf16(ctx, prob);
    ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, vh, 1, 0, 2, 3));  // [T, D, H]
    ggml_tensor* co = bf16(ctx, ggml_mul_mat(ctx, vt, prob));             // [D, T, H]
    co = ggml_cont(ctx, ggml_permute(ctx, co, 0, 2, 1, 3));              // [D, H, T]
    return ggml_reshape_2d(ctx, co, (int64_t) D * H, T);                 // [d_model, T]
}

// Whisper Conv1d front-end (conv1 K3/s1/p1 -> GELU -> conv2 K3/s2/p1 -> GELU),
// all in-graph with ggml_conv_1d, feeding it explicit F32 inputs. `mel_in` is the
// raw mel as a [L=mel_T, IC] F32 graph input (feat-major: element (c,t) at c*L+t
// => ggml 2D ne0=L, ne1=IC, which is exactly ggml_conv_1d's data convention).
// Returns the [d_model, T_enc] F32 layers input (d_model-contiguous: element
// (d,t) at t*d_model+d), matching what build_encoder_layers consumes. Identical
// to ark's build_conv_front_end (the two front-ends are the same Whisper convs).
ggml_tensor* build_conv_front_end(ggml_context* ctx, const HiggsModel& model,
                                  int64_t mel_T, ggml_tensor* mel_in) {
    const auto& ec = model.config.encoder;
    // conv1: weight [K,IC,OC], data [mel_T,IC] -> [mel_T, OC] (OC-contig flat).
    ggml_tensor* w1 = f32(ctx, weight(ctx, model.loader, "enc.conv1.weight"));
    ggml_tensor* conv1 = ggml_conv_1d(ctx, w1, f32(ctx, mel_in), /*s=*/1, /*p=*/1, /*d=*/1);
    ggml_tensor* b1 = f32(ctx, weight(ctx, model.loader, "enc.conv1.bias"));
    conv1 = ggml_add(ctx, f32(ctx, conv1), ggml_reshape_2d(ctx, b1, 1, ec.d_model));
    ggml_tensor* g1 = exact_gelu(ctx, conv1);  // bf16 (oracle boundary conv1->conv2)
    // conv2: g1 is [mel_T, OC=1280] (ne0=mel_T=L, ne1=1280=IC) -> feed directly.
    ggml_tensor* w2 = f32(ctx, weight(ctx, model.loader, "enc.conv2.weight"));
    ggml_tensor* conv2 = ggml_conv_1d(ctx, w2, f32(ctx, g1), /*s=*/2, /*p=*/1, /*d=*/1);
    ggml_tensor* b2 = f32(ctx, weight(ctx, model.loader, "enc.conv2.bias"));
    conv2 = ggml_add(ctx, f32(ctx, conv2), ggml_reshape_2d(ctx, b2, 1, ec.d_model));
    ggml_tensor* g2 = exact_gelu(ctx, conv2);  // bf16, [T_enc, OC] (OC-contig flat)
    // transpose [T_enc,OC] (ne0=T_enc) -> [OC,T_enc] (ne0=OC=d_model), contiguous
    // -> d_model-contig flat (element (oc,t) at t*OC+oc), the layers' input.
    // Feed the layers F32 (ggml_norm on a bf16 graph input cast misbehaves in this
    // build, matching ark's host-conv path which fed an f32 input directly).
    ggml_tensor* layers_in = ggml_cont(ctx, ggml_permute(ctx, g2, 1, 0, 2, 3));
    return f32(ctx, layers_in);
}

// Encoder body: ADD absolute positional embeddings, then the 32 global-attention
// layers. `x` is the [d_model, T_enc] conv output. Returns the post-layer hidden
// state [d_model, T_enc] bf16 (ln_post is applied later, after avg_pool, matching
// the vendor modeling order).
ggml_tensor* build_encoder_layers(ggml_context* ctx, const HiggsModel& model,
                                  int64_t T_enc, ggml_tensor* x) {
    const auto& ec = model.config.encoder;
    const ModelLoader& ml = model.loader;
    // Add absolute positional embeddings: embed_positions.weight is
    // [max_source_positions, d_model] = [1500, 1280]. modeling slices [:T, :] and
    // adds to [B, T, D]. In our [d_model, T_enc] (ne0=d_model, ne1=T_enc) layout
    // each column t gets row t of the table. The table stored feat-major as a 2D
    // tensor [1500, 1280] has ne0=1280=d_model, ne1=1500; a view of the first T_enc
    // columns [d_model, T_enc] lines up element (d,t) with table[t,d] exactly.
    ggml_tensor* pos = weight(ctx, ml, "enc.positional_emb.weight");  // [d_model, 1500]
    ggml_tensor* pos_view = ggml_view_2d(ctx, pos, (int64_t) ec.d_model, T_enc,
                                         pos->nb[1], 0);  // [d_model, T_enc]
    x = add_bf16(ctx, x, pos_view);
    for (uint32_t li = 0; li < ec.n_layers; ++li) {
        const std::string pre = "enc.blk." + std::to_string(li) + ".";
        ggml_tensor* r = x;
        ggml_tensor* n = layer_norm(ctx, ml, x, pre + "attn_norm", ec.layer_norm_eps);
        // q/v/o have bias, k has NO bias (WhisperAttention).
        ggml_tensor* q = linear(ctx, ml, n, pre + "attn.q", true);
        ggml_tensor* k = linear(ctx, ml, n, pre + "attn.k", false);
        ggml_tensor* vv = linear(ctx, ml, n, pre + "attn.v", true);
        ggml_tensor* joined = global_attention(ctx, model, q, k, vv, T_enc);
        ggml_tensor* a = linear(ctx, ml, joined, pre + "attn.o", true);
        x = add_bf16(ctx, r, a);
        r = x;
        n = layer_norm(ctx, ml, x, pre + "ffn_norm", ec.layer_norm_eps);
        ggml_tensor* h = linear(ctx, ml, n, pre + "ffn.fc1", true);
        h = exact_gelu(ctx, h);
        h = linear(ctx, ml, h, pre + "ffn.fc2", true);
        x = add_bf16(ctx, r, h);
    }
    return x;  // [d_model, T_enc] bf16
}

// AvgPool1d(kernel=2, stride=2) over the time dim, then ln_post. `x` is the
// [d_model, T_enc] layer output (ne0=d_model, ne1=T_enc). modeling permutes to
// [B, D, T], avg-pools the last dim, permutes back, then layer_norm. In our
// [d_model, T_enc] layout, the time axis is ne[1]; permute to [T_enc, d_model]
// (ne0=T_enc) so the time axis is the innermost, pool, permute back to
// [d_model, T_avg], then layer_norm. T_avg = floor((T_enc-2)/2)+1 (PyTorch
// AvgPool1d with k=2,s=2,p=0). Returns [d_model, T_avg] bf16 (post-ln_post).
//
// NOTE: this uses ggml_pool_2d (NOT ggml_pool_1d). The 1D pool has no CUDA kernel
// in this ggml build (only POOL_2D is GPU-supported), and a CPU-only op in the
// middle of the GPU encoder graph forces ggml's multi-backend scheduler to take
// over, where it then mis-sizes an internal buffer and aborts trying to allocate
// ~232 GB (the original crash). ggml_pool_2d with k1=1,s1=1,p1=0 over the channel
// axis (ne[1]) is a no-op along that axis and a k0=k,s0=k window over ne[0] —
// byte-identical to AvgPool1d(k,s) over the time axis. The CUDA pool2d kernel
// requires F32 input, so cast the (bf16) encoder output to F32 first.
ggml_tensor* build_avg_pool_and_ln(ggml_context* ctx, const HiggsModel& model,
                                   int64_t T_enc, int64_t* T_avg_out, ggml_tensor* x) {
    const auto& ec = model.config.encoder;
    // [d_model, T_enc] -> [T_enc, d_model] so ne0 is the time axis ggml pools.
    ggml_tensor* xt = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));  // [T_enc, d_model]
    // AvgPool1d(k, s) over ne[0]=time via pool_2d with a 1x-equivalent window
    // (k0=k, s0=s on time; k1=1, s1=1, p1=0 -> channel axis untouched). F32 in:
    // the CUDA pool2d kernel asserts src0->type == F32.
    ggml_tensor* pooled = ggml_pool_2d(ctx, f32(ctx, xt), GGML_OP_POOL_AVG,
                                       (int) ec.avg_pool_kernel, /*k1=*/1,
                                       (int) ec.avg_pool_kernel, /*s1=*/1,
                                       /*p0=*/0.0f, /*p1=*/0.0f);
    const int64_t T_avg = pooled->ne[0];
    *T_avg_out = T_avg;
    // [T_avg, d_model] -> [d_model, T_avg] (the projector + layers' column layout).
    ggml_tensor* back = ggml_cont(ctx, ggml_permute(ctx, pooled, 1, 0, 2, 3));  // [d_model, T_avg]
    // ln_post applied AFTER avg_pool (vendor modeling order).
    return layer_norm(ctx, model.loader, back, "enc.ln_post", ec.layer_norm_eps);
}

// MLP projector: depthwise temporal Conv1d (groups=1280, K3, s2, p1) -> Linear ->
// ReLU -> Linear. `x` is [d_model, T_avg] bf16. modeling: x [B, T, C] -> permute
// [B, C, T] -> temporal conv -> permute [B, T, C] -> linear1 -> relu -> linear2.
// In our [d_model, T_avg] (ne0=d_model=C, ne1=T_avg) layout the time axis is ne[1];
// ggml_conv_1d_dw pools ne[0], so permute to [T_avg, d_model] (ne0=T_avg) first,
// conv, then permute back to [d_model, T_proj]. Returns the F32 projector output
// [output, T_proj].
ggml_tensor* build_projector(ggml_context* ctx, const HiggsModel& model,
                             int64_t T_avg, int64_t* T_proj_out, ggml_tensor* x) {
    const auto& pc = model.config.projector;
    // [d_model, T_avg] -> [T_avg, d_model] (ne0=time so depthwise conv strides it).
    ggml_tensor* xt = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));  // [T_avg, d_model]
    // depthwise conv: weight [K, 1, C] (HF stores [C, 1, K] -> ne0=K,ne1=1,ne2=C,
    // exactly ggml_conv_1d_dw's expected a layout). data b [T_avg, d_model].
    // conv_1d_dw decomposes into reshape+im2col+mul_mat+reshape, all GPU-supported,
    // so it stays on the GPU fast path (unlike pool_1d).
    ggml_tensor* dw_w = f32(ctx, weight(ctx, model.loader, "proj.temporal.weight"));
    ggml_tensor* dw = ggml_conv_1d_dw(ctx, dw_w, f32(ctx, xt),
                                      (int) pc.temporal_stride, /*p=*/1, /*d=*/1);
    // depthwise bias [C], broadcast-add over the channel axis.
    ggml_tensor* dw_b = f32(ctx, weight(ctx, model.loader, "proj.temporal.bias"));
    dw = ggml_add(ctx, dw, ggml_reshape_2d(ctx, dw_b, 1, pc.input_size));
    const int64_t T_proj = dw->ne[0];
    *T_proj_out = T_proj;
    // [T_proj, d_model] -> [d_model, T_proj] (column-per-token, linear input).
    ggml_tensor* back = bf16(ctx, ggml_cont(ctx, ggml_permute(ctx, dw, 1, 0, 2, 3)));
    // Linear(d_model -> hidden) + bias -> ReLU -> Linear(hidden -> output) + bias.
    ggml_tensor* h = linear(ctx, model.loader, back, "proj.linear1", true);
    h = bf16(ctx, ggml_relu(ctx, f32(ctx, h)));  // ReLU activation (config)
    h = linear(ctx, model.loader, h, "proj.linear2", true);
    return f32(ctx, h);  // F32 projector output [output, T_proj]
}

} // namespace

// ---------------------------------------------------------------------------
// Host-side fallback for the ops that are risky under CUDA-graph capture (the
// conv front-end, avg_pool, depthwise conv). ggml's conv im2col produced wrong
// values under CUDA-graph capture in the ark port (docs/ggml-ark-port-status.md);
// the depthwise conv + avg_pool have not been validated under capture either, so
// the CPU / debug path runs them as host scalar loops (byte-exact vs torch) and
// only the encoder layers + projector linears run as graphs. Mirrors ark's
// host_conv1d_gelu.
// ---------------------------------------------------------------------------

std::vector<float> read_tensor_to_f32(const ModelLoader& ml, const char* name) {
    ggml_tensor* t = ml.tensor(name);
    if (!t) return {};
    ensure_weights_realized(ml);
    size_t n = (size_t) ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> raw(n);
        ggml_backend_tensor_get(t, raw.data(), 0, n * sizeof(ggml_bf16_t));
        for (size_t i = 0; i < n; ++i) out[i] = ggml_bf16_to_fp32(raw[i]);
    } else if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    }
    return out;
}

// Host-side Conv1d (K=3, pad=1, depthwise flag) + optional GELU, in f32/ double.
// `in` is [IC, L] feat-major (element (c,t) at c*L+t). weight is [OC, IC, K] (HF
// order) for a regular conv; for depthwise OC==IC and the IC axis is 1 (weight
// [C, 1, K]). Returns [OC, OL] f32 (oc-contiguous: element (oc,t) at oc*OL+t),
// matching the next op's expected [IC, L] feat-major input. GELU uses exact erf.
namespace {
std::vector<float> host_conv1d(const ModelLoader& ml, const std::vector<float>& in,
                               int64_t IC, int64_t L, const std::string& wname,
                               int stride, bool depthwise, bool gelu) {
    std::vector<float> wf = read_tensor_to_f32(ml, (wname + ".weight").c_str());
    std::vector<float> bf = read_tensor_to_f32(ml, (wname + ".bias").c_str());
    const int64_t OC = depthwise ? IC : (int64_t) bf.size();
    const int64_t K = 3, p = 1;
    const int64_t OL = (L + 2 * p - K) / stride + 1;
    std::vector<float> x((size_t) IC * (L + 2 * p), 0.0f);
    for (int64_t c = 0; c < IC; ++c)
        for (int64_t t = 0; t < L; ++t)
            x[(size_t) c * (L + 2 * p) + (t + p)] = in[(size_t) c * L + t];
    std::vector<float> y((size_t) OC * OL, 0.0f);
    for (int64_t oc = 0; oc < OC; ++oc) {
        for (int64_t t = 0; t < OL; ++t) {
            double acc = (double) bf[(size_t) oc];
            if (depthwise) {
                // weight [C, 1, K]: oc==channel c, IC axis is 1.
                for (int64_t k = 0; k < K; ++k) {
                    int64_t src = t * stride + k;
                    acc += (double) wf[((size_t) oc * 1 + 0) * K + k] *
                           (double) x[(size_t) oc * (L + 2 * p) + src];
                }
            } else {
                for (int64_t c = 0; c < IC; ++c) {
                    for (int64_t k = 0; k < K; ++k) {
                        int64_t src = t * stride + k;
                        acc += (double) wf[((size_t) oc * IC + c) * K + k] *
                               (double) x[(size_t) c * (L + 2 * p) + src];
                    }
                }
            }
            float v = (float) acc;
            if (gelu) v = 0.5f * v * (1.0f + std::erf(v / (float) M_SQRT2));
            y[(size_t) oc * OL + t] = v;
        }
    }
    return y;  // [OC, OL] f32 (oc-contiguous)
}
} // namespace

// Host-side AvgPool1d(k=2, s=2, p=0) over the time axis. `in` is [C, L]
// feat-major; returns [C, floor((L-2)/2)+1] feat-major.
namespace {
std::vector<float> host_avg_pool1d(const std::vector<float>& in, int64_t C, int64_t L) {
    const int64_t k = 2, s = 2, p = 0;
    const int64_t OL = (L + 2 * p - k) / s + 1;
    std::vector<float> y((size_t) C * OL, 0.0f);
    for (int64_t c = 0; c < C; ++c)
        for (int64_t t = 0; t < OL; ++t)
            y[(size_t) c * OL + t] = 0.5f * (in[(size_t) c * L + (t * s)] +
                                             in[(size_t) c * L + (t * s) + 1]);
    return y;
}
} // namespace

// ---------------------------------------------------------------------------
// Fused conv + encode + avg_pool + ln_post + project with a per-mel_T bounded-LRU
// ReplayGraph cache (GPU). The full Whisper Conv1d front-end + avg_pool + depthwise
// conv run IN-GRAPH on GPU; on CPU / debug they run as the host scalar fallbacks.
// Keyed on mel_T. Mirrors ark's encode_audio_and_adapt + bounded LruCache.
// ---------------------------------------------------------------------------
struct EncoderReplayEntry {
    int64_t mel_T = 0, T_enc = 0, T_avg = 0, T_proj = 0;
    GraphInputPool pool;
    std::unique_ptr<ReplayGraph> graph;
    float* mel_f32_buf = nullptr;  // stable pool backing for the mel graph input
    std::vector<float> enc_capture;  // optional ln_post dump (STARLING_HIGGS_DUMP_ENC)
};

namespace {
struct ShapeKey {
    int64_t mel_T;
    bool operator==(const ShapeKey& o) const { return mel_T == o.mel_T; }
};
struct ShapeKeyHash {
    size_t operator()(const ShapeKey& k) const noexcept { return (size_t) k.mel_T; }
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

bool encode_audio_and_project(const HiggsModel& model, const MelFeatures& mel,
                              AudioEncoding& out, std::string& err) {
    ensure_weights_realized(model.loader);
    const auto& ec = model.config.encoder;
    if (mel.n_mels != (int64_t) ec.num_mel_bins || mel.n_frames <= 0 ||
        mel.data.size() != (size_t) mel.n_mels * mel.n_frames) {
        err = "invalid Higgs mel shape/data";
        return false;
    }
    const int64_t mel_T = mel.n_frames;
    // conv2 stride-2 downsamples time: T_enc = (mel_T + 1) // 2.
    const int64_t T_enc = (mel_T + 1) / 2;
    const float* mel_f32_ptr = mel.f32.data();

    // CPU + debug diagnostic path: host conv front-end + host avg_pool + host
    // depthwise conv, with only the encoder layers + projector linears as graphs.
    // Keeps the validated host loops (byte-exact vs torch) for the CPU fallback
    // where CUDA-graph capture is unavailable / risky.
    if (!global_backend().is_gpu() || debug_enabled()) {
        const auto& pc = model.config.projector;
        // conv1 (s1) + GELU.
        std::vector<float> c1 = host_conv1d(model.loader, mel.f32, ec.num_mel_bins,
                                            mel_T, "enc.conv1", /*stride=*/1,
                                            /*depthwise=*/false, /*gelu=*/true);
        // conv2 (s2) + GELU.
        std::vector<float> c2 = host_conv1d(model.loader, c1, ec.d_model,
                                            mel_T, "enc.conv2", /*stride=*/2,
                                            /*depthwise=*/false, /*gelu=*/true);
        // c2 is [d_model=1280, T_enc] feat-major (oc-contig). The encoder layers
        // consume [d_model, T_enc] (d-contig, element (d,t) at t*D+d): transpose.
        std::vector<float> layers_in((size_t) ec.d_model * T_enc);
        for (int64_t t = 0; t < T_enc; ++t)
            for (int64_t d = 0; d < (int64_t) ec.d_model; ++d)
                layers_in[(size_t) t * ec.d_model + d] =
                    c2[(size_t) d * T_enc + t];
        int64_t T_avg = 0, T_proj = 0;
        std::vector<float> body_out, enc_out;
        const char* dump_enc = std::getenv("STARLING_HIGGS_DUMP_ENC");
        // Run the encoder layers (add pos emb + 32 layers) + avg_pool + ln_post as
        // one graph, feeding the [d_model, T_enc] f32 layers input. ln_post feeds
        // the projector graph.
        std::vector<float> post_pool;
        bool ok = run_graph([&](ggml_context* ctx) -> ggml_tensor* {
            int64_t cne[2] = {ec.d_model, T_enc};
            ggml_tensor* x = graph_input_tensor(ctx, GGML_TYPE_F32, 2, cne,
                                                layers_in.data(),
                                                layers_in.size() * sizeof(float));
            ggml_tensor* enc = build_encoder_layers(ctx, model, T_enc, x);
            ggml_tensor* pooled = build_avg_pool_and_ln(ctx, model, T_enc, &T_avg, enc);
            if (dump_enc) capture_graph_output(f32(ctx, pooled), &enc_out);
            return pooled;
        }, post_pool);
        if (!ok) { err = "Higgs encoder+avgpool graph execution failed"; return false; }
        if (dump_enc) {
            if (FILE* f = std::fopen(dump_enc, "wb")) {
                std::fwrite(enc_out.data(), sizeof(float), enc_out.size(), f);
                std::fclose(f);
            }
        }
        // post_pool is the [d_model, T_avg] post-ln_post f32 output. Feed the
        // projector: depthwise conv + 2 linears.
        ok = run_graph([&](ggml_context* ctx) -> ggml_tensor* {
            int64_t pne[2] = {ec.d_model, T_avg};
            ggml_tensor* p = graph_input_tensor(ctx, GGML_TYPE_F32, 2, pne,
                                                post_pool.data(),
                                                post_pool.size() * sizeof(float));
            ggml_tensor* pb = bf16(ctx, p);
            return build_projector(ctx, model, T_avg, &T_proj, pb);
        }, body_out);
        if (!ok) { err = "Higgs projector graph execution failed"; return false; }
        out.data = std::move(body_out);
        out.n_tokens = T_proj;
        out.width = model.config.projector.output_size;
        return true;
    }

    // --- GPU: captured per-mel_T graph running conv1/gelu/conv2/gelu/layers/
    // avgpool/ln_post/projector end-to-end. The mel is the varying input.
    register_encoder_cache_clearer_once();
    if (!g_encoder_cache)
        g_encoder_cache = std::unique_ptr<LruCache<ShapeKey, EncoderReplayEntry, ShapeKeyHash>>(
            new LruCache<ShapeKey, EncoderReplayEntry, ShapeKeyHash>(replay_cache_size()));

    const char* dump_enc = std::getenv("STARLING_HIGGS_DUMP_ENC");
    ShapeKey key{mel_T};
    EncoderReplayEntry& e = *g_encoder_cache->get_or_init(key,
        [&](EncoderReplayEntry& entry) {
            entry.mel_T = mel_T;
            entry.T_enc = T_enc;
            entry.mel_f32_buf = reinterpret_cast<float*>(entry.pool.alloc_bytes(
                (size_t) ec.num_mel_bins * mel_T * sizeof(float)));
            std::memcpy(entry.mel_f32_buf, mel_f32_ptr,
                        (size_t) ec.num_mel_bins * mel_T * sizeof(float));
            int64_t T_avg = 0, T_proj = 0;
            entry.graph = std::make_unique<ReplayGraph>(global_backend(),
                [&](ggml_context* ctx) -> ggml_tensor* {
                    int64_t mne[2] = {entry.mel_T, (int64_t) ec.num_mel_bins};
                    ggml_tensor* mel_in = graph_input_tensor(ctx, GGML_TYPE_F32, 2, mne,
                        entry.mel_f32_buf,
                        (size_t) ec.num_mel_bins * entry.mel_T * sizeof(float));
                    ggml_tensor* layers_in = build_conv_front_end(ctx, model, entry.mel_T, mel_in);
                    ggml_tensor* enc = build_encoder_layers(ctx, model, entry.T_enc, layers_in);
                    ggml_tensor* pooled = build_avg_pool_and_ln(ctx, model, entry.T_enc,
                                                                &T_avg, enc);
                    if (dump_enc) capture_graph_output(f32(ctx, pooled), &entry.enc_capture);
                    ggml_tensor* pb = pooled;  // already bf16 (post-ln_post)
                    return build_projector(ctx, model, T_avg, &T_proj, pb);
                });
            entry.T_avg = T_avg;
            entry.T_proj = T_proj;
        });
    // Refresh the mel input in its stable pool buffer, then re-upload.
    std::memcpy(e.mel_f32_buf, mel_f32_ptr,
                (size_t) ec.num_mel_bins * e.mel_T * sizeof(float));
    for (size_t i = 0; i < e.graph->n_inputs(); ++i)
        e.graph->set_input(i, e.graph->input_host(i), e.graph->input_nbytes(i));

    std::vector<float> tmp;
    if (!e.graph->compute_with_captures(tmp)) {
        err = "Higgs fused encoder+projector replay failed";
        return false;
    }
    if (dump_enc && !e.enc_capture.empty()) {
        if (FILE* f = std::fopen(dump_enc, "wb")) {
            std::fwrite(e.enc_capture.data(), sizeof(float), e.enc_capture.size(), f);
            std::fclose(f);
        }
    }
    out.data = std::move(tmp);
    out.n_tokens = e.T_proj;
    out.width = model.config.projector.output_size;
    return true;
}
} // namespace starling::ggml::higgs
