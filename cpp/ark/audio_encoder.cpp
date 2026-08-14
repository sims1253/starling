// audio_encoder.cpp — ARK-ASR-3B Whisper encoder + MLP adapter on the Starling
// ggml runtime.
//
// The audio path: Whisper Conv1d front-end (conv1 kernel3 stride1, conv2 kernel3
// stride2 -> halves mel_T) -> GELU -> permute -> 32 global-attention encoder
// layers (RoPE, NOT absolute positional) -> ARK LayerNorm -> adapter (reshape
// merge-by-4 -> Linear(5120->4096) GELU Linear(4096->2048)).
//
// On GPU this is ONE captured ReplayGraph keyed on the (post-conv) encoder
// length T_enc (global attention => one graph per T_enc; the cache is
// LRU-bounded). On CPU / debug it is the one-shot pair.
#include "audio_encoder.hpp"
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
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace starling::ggml::ark {
namespace {

// Shared graph-builder helpers (lib/graph_helpers.hpp); the audio encoder is
// bf16-oracle discipline.
using lib::weight;
using lib::bf16;
using lib::f32;
ggml_tensor* linear(ggml_context* c, const ModelLoader& ml, ggml_tensor* x,
                    const std::string& n, bool bias) {
    return lib::linear_bf16(c, ml, x, n, bias);
}
// exact GELU (erf), the modeling_audio activation_fn for both conv front-end and
// adapter (config.activation_function="gelu", approximate="none").
ggml_tensor* exact_gelu(ggml_context* c, ggml_tensor* x) {
    return lib::gelu_erf_bf16(c, x);
}
ggml_tensor* add_bf16(ggml_context* c, ggml_tensor* a, ggml_tensor* b) {
    return lib::addb(c, a, b);
}
// PyTorch LayerNorm: F32 reduction + affine, one BF16 store. eps from config.
// (ggml_norm and an explicit mean/var/div formulation gave byte-identical output;
// the encoder's residual divergence is upstream — the conv graph input is
// scrambled on the device in this ggml build, see docs/ggml-ark-port-status.md.)
ggml_tensor* layer_norm(ggml_context* c, const ModelLoader& ml, ggml_tensor* x,
                        const std::string& n, float eps) {
    return lib::layer_norm_bf16(c, ml, x, n, eps);
}

bool debug_enabled() {
    return lib::debug_enabled("STARLING_ARK_DEBUG");
}

// Apply RoPE to q/k for global self-attention. The HF path (modeling_audio)
// rotates only the first rope_dim=32 of each 64-dim head, interleaving
// (a,b)*(cos,sin) = (a*cos-b*sin, b*cos+a+b*sin) via the stacked-pair layout.
// cos/sin are [T_enc, rope_dim/2] f32 tables baked in the GGUF.
// q/k are [d_model, T_enc] bf16 (heads folded into d_model = H*head_dim).
// Returns [d_model, T_enc] bf16 with the first rope_dim of every head rotated.
ggml_tensor* apply_enc_rope(ggml_context* c, ggml_tensor* x, ggml_tensor* cos_t,
                            ggml_tensor* sin_t, int head_dim, int rope_dim,
                            int n_heads, int64_t T) {
    // HF ARK RoPE is INTERLEAVED (GPT-J style): pairs (x[2i], x[2i+1]) = (a,b),
    // out_a = a*cos - b*sin, out_b = b*cos + a*sin (modeling_audio).
    // View the rope_dim slice of each head's [head_dim] vector as pairs: a 4D
    // view [2, half, H, T] with ne0=2 (the pair) and stride 2 along the head_dim
    // axis. element (p,i,h,t) = x[p + 2*i + head_dim*h + head_dim*H*t].
    const int half = rope_dim / 2;
    x = ggml_reshape_3d(c, x, head_dim, n_heads, T);  // [head_dim, H, T]
    // HF ARK RoPE is INTERLEAVED (GPT-J style): pairs (x[2i], x[2i+1]) = (a,b),
    // out_a = a*cos - b*sin, out_b = b*cos + a*sin. Take the even (a) and odd (b)
    // elements of the rope_dim slice via two strided 4D views (stride 2 in dim 0).
    ggml_tensor* xf = f32(c, x);  // [head_dim, H, T], memory (d,h,t) at d + D*h + D*H*t
    // Pick the even (a) and odd (b) elements of each head's rope_dim slice. A
    // head's D dims are contiguous (d=0..D-1 at D*h + d); pair i is at d=2i.
    // NOTE: for large T_enc the odd view's offset can trip ggml's strict view
    // bounds check; short/medium-length audio works, very-long currently aborts
    // (see docs/ggml-ark-port-status.md).
    ggml_tensor* a = ggml_view_4d(c, xf, 1, half, n_heads, T,
                                  2 * xf->nb[0], xf->nb[1], xf->nb[2], 0);          // x[2i]
    ggml_tensor* b = ggml_view_4d(c, xf, 1, half, n_heads, T,
                                  2 * xf->nb[0], xf->nb[1], xf->nb[2], xf->nb[0]);  // x[2i+1]
    // cos/sin tables are stored BF16; cast to f32. View the first T rows then
    // reshape to [1, half, 1, T] (broadcast over the pair axis and heads).
    ggml_tensor* cos_row = ggml_view_2d(c, cos_t, half, T, cos_t->nb[1], 0);
    ggml_tensor* sin_row = ggml_view_2d(c, sin_t, half, T, sin_t->nb[1], 0);
    ggml_tensor* cos4 = ggml_reshape_4d(c, f32(c, cos_row), 1, half, 1, T);  // [1,half,1,T]
    ggml_tensor* sin4 = ggml_reshape_4d(c, f32(c, sin_row), 1, half, 1, T);
    ggml_tensor* oa = ggml_sub(c, ggml_mul(c, a, cos4), ggml_mul(c, b, sin4));  // [1,half,H,T]
    ggml_tensor* ob = ggml_add(c, ggml_mul(c, b, cos4), ggml_mul(c, a, sin4));
    // Re-interleave out[2i]=oa[i], out[2i+1]=ob[i]: stack oa,ob as [half,2,H,T]
    // (pair in dim 1), transpose dims 0<->1 -> [2,half,H,T], contiguous-reshape to
    // [rope_dim,H,T] (ne0=2 innermost => the flattened memory is interleaved).
    ggml_tensor* oa3 = ggml_reshape_3d(c, oa, half, n_heads, T);
    ggml_tensor* ob3 = ggml_reshape_3d(c, ob, half, n_heads, T);
    ggml_tensor* s1 = ggml_reshape_4d(c, oa3, half, 1, n_heads, T);  // [half,1,H,T]
    ggml_tensor* s2 = ggml_reshape_4d(c, ob3, half, 1, n_heads, T);
    ggml_tensor* cat2 = ggml_concat(c, s1, s2, 1);                          // [half,2,H,T]
    ggml_tensor* inter = ggml_cont(c, ggml_permute(c, cat2, 1, 0, 2, 3));   // [2,half,H,T]
    ggml_tensor* rope_out = ggml_reshape_3d(c, inter, rope_dim, n_heads, T); // [rope_dim,H,T] interleaved
    // Append the unrotated tail [head_dim-rope_dim, H, T].
    ggml_tensor* rot;
    if (rope_dim < head_dim) {
        ggml_tensor* tail = ggml_view_3d(c, x, head_dim - rope_dim, n_heads, T,
                                         x->nb[1], x->nb[2],
                                         (size_t) rope_dim * x->nb[0]);
        rot = ggml_concat(c, rope_out, f32(c, tail), 0);  // [head_dim,H,T]
    } else {
        rot = rope_out;
    }
    rot = bf16(c, rot);
    return ggml_reshape_2d(c, rot, (int64_t) head_dim * n_heads, T);  // [d_model, T]
}

// Global bidirectional self-attention (WhisperRoPESdpaAttention). q/k/v are
// [d_model, T] bf16. RoPE applied to q/k. scale = 1/sqrt(head_dim). Returns
// [d_model, T] bf16. is_causal=False (bidirectional); no mask.
//
// Two paths share the same RoPE'd, head-split inputs (qh/kh/vh = [D, T, H]):
//  - GPU fused path: ggml_flash_attn_ext (the #1 encoder optimization; the full
//    [T,T,H] score tensor is never materialized). ARK is plain MHA
//    (n_head == n_head_kv == H) and bidirectional -> mask = nullptr, so all the
//    flash_attn_ext broadcast preconditions hold trivially.
//  - CPU / byte-exact fallback: the original materialized mul_mat+softmax path.
//    Kept verbatim and selected via STARLING_ARK_NO_FATTN=1 (or non-GPU), exactly
//    mirroring the proven parakeet dual-path (relpos_attention.cpp).
//
// Both paths return head-major [d_model, T]: the manual path permutes [D,T,H] ->
// [D,H,T] then reshape-2d to [D*H, T]; flash_attn_ext already returns
// [n_embd_v, n_head, n_batch, ne3] = [D, H, T, 1] (head-major), so a plain
// reshape_2d(fa, D*H, T) yields the SAME byte ordering.
ggml_tensor* global_attention(ggml_context* ctx, const ArkModel& model,
                              ggml_tensor* q, ggml_tensor* k, ggml_tensor* v,
                              int64_t T) {
    const auto& ec = model.config.encoder;
    const int H = (int) ec.n_heads, D = (int) ec.head_dim;
    const int rope_dim = (int) ec.rope_dim;
    const float scale = 1.0f / std::sqrt((float) D);
    // Apply RoPE to q/k (rotate the first rope_dim of each head).
    ggml_tensor* cos_t = weight(ctx, model.loader, "enc.rope_cos");  // [T, rope_dim/2] f32
    ggml_tensor* sin_t = weight(ctx, model.loader, "enc.rope_sin");
    // Trim/verify the baked tables cover T (they are sized for the max encoder
    // length; a shorter T just uses the leading rows via a view).
    q = apply_enc_rope(ctx, q, cos_t, sin_t, D, rope_dim, H, T);
    k = apply_enc_rope(ctx, k, cos_t, sin_t, D, rope_dim, H, T);

    // Batched attention over all heads (verified to match the per-head path).
    // Reshape q/k/v [d_model, T] -> [D, H, T] -> permute to [D, T, H] so the head
    // dim is ne[2] (ggml mul_mat batches over ne[2]). These [D, T, H] tensors are
    // the inputs to BOTH paths: for flash_attn_ext the expected q/k/v layout is
    // [n_embd, n_batch, n_head, ne3] = [D, T, H, 1] (matches exactly).
    auto to_heads = [&](ggml_tensor* z) {
        z = ggml_reshape_3d(ctx, z, D, H, T);
        return ggml_cont(ctx, ggml_permute(ctx, z, 0, 2, 1, 3));  // [D, T, H]
    };
    ggml_tensor* qh = to_heads(q), * kh = to_heads(k), * vh = to_heads(v);

    // Kill-switch / CPU selection, mirroring parakeet's STARLING_PARAKEET_NO_FATTN.
    const char* no_fattn = std::getenv("STARLING_ARK_NO_FATTN");
    const bool use_flash = global_backend().is_gpu() &&
                           !(no_fattn && std::string(no_fattn) == "1");
    if (use_flash) {
        // Feed F32 q/k/v to flash, mirroring parakeet (its qh/kh/vh are the raw
        // F32 mul_mat output). The MMA flash kernel asserts Q->type==F32, and
        // feeding BF16 forces an in-graph BF16->F16 pool conversion that ggml's
        // CUDA-graph capture rejects (parakeet sidesteps this by staying F32).
        // Bidirectional -> mask = nullptr. fa is [D, H, T, 1] (head-major).
        ggml_tensor* fa = ggml_flash_attn_ext(ctx, f32(ctx, qh), f32(ctx, kh),
                                              f32(ctx, vh), /*mask=*/nullptr,
                                              scale, 0.0f, 0.0f);
        return ggml_reshape_2d(ctx, fa, (int64_t) D * H, T);  // [d_model, T]
    }

    // Manual path (byte-exact reference / CPU / STARLING_ARK_NO_FATTN=1 fallback).
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
// (d,t) at t*d_model+d), matching what build_encoder_layers consumes.
//
// ggml_conv_1d layout (verified in ggml.c + ops): weight a=[K,IC,OC] (the GGUF
// stores HF [OC,IC,K] innermost-first -> ne0=K,ne1=IC,ne2=OC, exactly what conv
// wants); data b=[L,IC,N] (ne0=L,ne1=IC); result reshape_3d -> [OW,OC,N] (ne0=OW).
// readback flattens ne0-innermost so element (ow,oc) is at oc*OW+ow (OC-contiguous).
// CUDA im2col asserts src1 (data) is F32 (the flash-attention lesson: feed F32 so
// no in-graph bf16->f16 pool conversion trips CUDA-graph capture). conv1's output
// must be cast bf16 (the oracle boundary conv1-gelu->conv2), then conv2's output
// is transposed [T_enc,OC]->[OC,T_enc] (ne0=OC=d_model) to the layers' layout.
ggml_tensor* build_conv_front_end(ggml_context* ctx, const ArkModel& model,
                                  int64_t mel_T, int64_t T_enc, ggml_tensor* mel_in) {
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
    // build, matching the prior host-conv path which fed an f32 input directly).
    ggml_tensor* layers_in = ggml_cont(ctx, ggml_permute(ctx, g2, 1, 0, 2, 3));
    return f32(ctx, layers_in);
}

// Encoder body: the 32 global-attention layers + ARK LayerNorm. `x` is the
// [d_model, T_enc] conv output. Returns the post-ln_post hidden state. This is
// the part that benefits from CUDA-graph capture (it dominates the cost).
ggml_tensor* build_encoder_layers(ggml_context* ctx, const ArkModel& model,
                                  int64_t T_enc, ggml_tensor* x) {
    const auto& ec = model.config.encoder;
    const ModelLoader& ml = model.loader;
    for (uint32_t li = 0; li < ec.n_layers; ++li) {
        const std::string pre = "enc.blk." + std::to_string(li) + ".";
        ggml_tensor* r = x;
        ggml_tensor* n = layer_norm(ctx, ml, x, pre + "attn_norm", ec.layer_norm_eps);
        // q/v/o have bias, k has NO bias (WhisperRoPESdpaAttention).
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
    // ARK post-encoder LayerNorm (the Whisper encoder.layer_norm is Identity).
    return layer_norm(ctx, ml, x, "enc.ln_post", ec.layer_norm_eps);
}

// Build the adapter: truncate to a multiple of merge_factor, reshape
// merge-by-4 -> Linear(input->hidden) GELU Linear(hidden->output). Returns the
// F32 adapter output [output, N].
ggml_tensor* build_adapter(ggml_context* ctx, const ArkModel& model,
                           ggml_tensor* x, int64_t T_enc, int64_t* N_out) {
    const auto& ac = model.config.adapter;
    const uint32_t mf = ac.merge_factor;
    int64_t target = (T_enc / mf) * mf;
    if (target <= 0) target = mf;  // very short audio: pad to merge_factor
    if (target < T_enc) {
        // Truncate the trailing frames so T is a multiple of merge_factor.
        x = ggml_view_2d(ctx, x, (int64_t) model.config.encoder.d_model, target,
                         x->nb[1], 0);
    }
    const int64_t N = target / mf;
    *N_out = N;
    // reshape merge-by-4: [d_model, target] -> [d_model*mf, N].
    // ggml reshape: the bytes are row-major [target, d_model] (target rows of
    // d_model). Grouping `mf` consecutive frames -> [N, d_model*mf].
    x = ggml_reshape_2d(ctx, x, (int64_t) ac.input_size, N);  // [input, N]
    auto lin = [&](const std::string& n, ggml_tensor* z) {
        return ggml_cast(ctx, ggml_mul_mat(ctx, weight(ctx, model.loader, n + ".weight"), z),
                         GGML_TYPE_BF16);
    };
    ggml_tensor* h = lin("adapter.fc0", x);
    h = exact_gelu(ctx, h);
    h = lin("adapter.fc2", h);
    // Add the fc2 bias (the adapting MLP's second Linear has a bias).
    h = ggml_add(ctx, f32(ctx, h), f32(ctx, weight(ctx, model.loader, "adapter.fc2.bias")));
    return f32(ctx, ggml_cast(ctx, h, GGML_TYPE_BF16));  // F32 adapter output
}

} // namespace

// ---------------------------------------------------------------------------
// Fused conv + encode + adapt with a per-mel_T bounded-LRU ReplayGraph cache
// (GPU). The full Whisper Conv1d front-end now runs IN-GRAPH (fast GEMMs via
// ggml_conv_1d with F32 inputs) instead of as a host scalar loop, so the
// encoder graph takes the mel as its varying input and runs conv1/gelu/conv2/
// gelu/layers/adapter end-to-end. Keyed on mel_T (conv1's internal sizes depend
// on mel_T directly, not just T_enc; the cache is LRU-bounded).
// ---------------------------------------------------------------------------
struct EncoderReplayEntry {
    int64_t mel_T = 0, T_enc = 0, N = 0;
    GraphInputPool pool;
    std::unique_ptr<ReplayGraph> graph;
    float* mel_f32_buf = nullptr;     // stable pool backing for the mel graph input (f32)
    std::vector<float> enc_capture;   // optional ln_post dump (STARLING_ARK_DUMP_ENC)
};

// Host-side Whisper Conv1d (K=3, pad=1) + exact GELU, in f32. Done on the CPU
// because ggml's conv im2col produces wrong values under CUDA-graph capture AND
// in larger one-shot graphs in this ggml build. The conv is a tiny share of the
// encoder cost (~1-2%), and the bf16 boundary (mel is bf16, the layer input is
// bf16) is preserved by the f32->bf16 round at the end. `in` is [IC, L] bf16
// (feat-major); weight is [OC, IC, K] (HF order); returns [OC, OL] f32.
// im2col must match torch.nn.functional.conv1d bit-exactly.
std::vector<float> read_bf16_to_f32(const ModelLoader& ml, const char* name) {
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
std::vector<float> host_conv1d_gelu(const ModelLoader& ml, const std::vector<ggml_bf16_t>& in,
                                    int64_t IC, int64_t L, const std::string& wname,
                                    int stride) {
    std::vector<float> wf = read_bf16_to_f32(ml, (wname + ".weight").c_str());  // [OC, IC, K]
    std::vector<float> bf = read_bf16_to_f32(ml, (wname + ".bias").c_str());    // [OC]
    const int64_t OC = bf.size(), K = 3, p = 1;
    const int64_t OL = (L + 2 * p - K) / stride + 1;
    // f32 input, zero-padded to [IC, L+2].
    std::vector<float> x((size_t) IC * (L + 2 * p), 0.0f);
    for (int64_t c = 0; c < IC; ++c)
        for (int64_t t = 0; t < L; ++t)
            x[(size_t)c * (L + 2 * p) + (t + p)] = ggml_bf16_to_fp32(in[(size_t)c * L + t]);
    // Output in natural [OC, OL] C-order (oc-contiguous: element (oc,t) at
    // oc*OL + t), matching the next conv's expected [IC, L] feat-major input.
    std::vector<float> y((size_t) OC * OL, 0.0f);
    for (int64_t oc = 0; oc < OC; ++oc) {
        for (int64_t t = 0; t < OL; ++t) {
            double acc = (double) bf[(size_t) oc];
            for (int64_t c = 0; c < IC; ++c) {
                for (int64_t k = 0; k < K; ++k) {
                    int64_t src = t * stride + k;  // index into padded time
                    acc += (double) wf[((size_t) oc * IC + c) * K + k] *
                           (double) x[(size_t) c * (L + 2 * p) + src];
                }
            }
            // exact GELU (erf). PyTorch approximate="none".
            float v = (float) acc;
            v = 0.5f * v * (1.0f + std::erf(v / (float) M_SQRT2));
            y[(size_t) oc * OL + t] = v;
        }
    }
    return y;  // [OC, OL] f32 (oc-contiguous), GELU applied
}

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

bool encode_audio_and_adapt(const ArkModel& model, const MelFeatures& mel,
                            AudioEncoding& out, std::string& err) {
    ensure_weights_realized(model.loader);
    const auto& ec = model.config.encoder;
    if (mel.n_mels != (int64_t) ec.num_mel_bins || mel.n_frames <= 0 ||
        mel.data.size() != (size_t) mel.n_mels * mel.n_frames) {
        err = "invalid ARK mel shape/data";
        return false;
    }
    const int64_t mel_T = mel.n_frames;
    // conv2 stride-2 downsamples time: T_enc = (mel_T + 1) // 2.
    const int64_t T_enc = (mel_T + 1) / 2;
    // The mel is feat-major [IC, L] (element (c,t) at c*L+t). As a ggml 2D input
    // with ne0=L, ne1=IC this is EXACTLY ggml_conv_1d's data convention; mel.f32
    // (populated by compute_log_mel) is the same feat-major layout in f32.
    const float* mel_f32_ptr = mel.f32.data();

    // CPU + debug diagnostic path: one-shot conv (host) + layers + adapter.
    // Keeps the validated host_conv1d_gelu (byte-exact vs torch) for the CPU
    // fallback where CUDA-graph capture is unavailable and the host scalar loop
    // is the correct, portable path.
    if (!global_backend().is_gpu() || debug_enabled()) {
        std::vector<float> c1 = host_conv1d_gelu(model.loader, mel.data, ec.num_mel_bins,
                                                 mel_T, "enc.conv1", /*stride=*/1);
        std::vector<ggml_bf16_t> c1_bf16(c1.size());
        for (size_t i = 0; i < c1.size(); ++i) c1_bf16[i] = ggml_fp32_to_bf16(c1[i]);
        std::vector<float> conv_oc = host_conv1d_gelu(model.loader, c1_bf16, ec.d_model,
                                                      mel_T, "enc.conv2", /*stride=*/2);
        if (const char* dcv = std::getenv("STARLING_ARK_DUMP_CONV")) {
            if (FILE* f = std::fopen(dcv, "wb")) {
                std::fwrite(conv_oc.data(), sizeof(float), conv_oc.size(), f);
                std::fclose(f);
            }
        }
        std::vector<ggml_bf16_t> conv_bf16((size_t) ec.d_model * T_enc);
        for (int64_t t = 0; t < T_enc; ++t)
            for (int64_t d = 0; d < (int64_t) ec.d_model; ++d)
                conv_bf16[(size_t) t * ec.d_model + d] =
                    ggml_fp32_to_bf16(conv_oc[(size_t) d * T_enc + t]);
        int64_t N = 0;
        std::vector<float> body_out, enc_out, conv_f32_host(conv_bf16.size());
        for (size_t i = 0; i < conv_bf16.size(); ++i) conv_f32_host[i] = ggml_bf16_to_fp32(conv_bf16[i]);
        const char* dump_enc = std::getenv("STARLING_ARK_DUMP_ENC");
        bool ok = run_graph([&](ggml_context* ctx) -> ggml_tensor* {
            int64_t cne[2] = {ec.d_model, T_enc};
            ggml_tensor* x = graph_input_tensor(ctx, GGML_TYPE_F32, 2, cne, conv_f32_host.data(),
                                                conv_f32_host.size() * sizeof(float));
            ggml_tensor* enc = build_encoder_layers(ctx, model, T_enc, x);
            if (dump_enc) capture_graph_output(f32(ctx, enc), &enc_out);
            return build_adapter(ctx, model, enc, T_enc, &N);
        }, body_out);
        if (!ok) { err = "ARK encoder graph execution failed"; return false; }
        if (dump_enc) {
            if (FILE* f = std::fopen(dump_enc, "wb")) {
                std::fwrite(enc_out.data(), sizeof(float), enc_out.size(), f);
                std::fclose(f);
            }
        }
        out.data = std::move(body_out);
        out.n_tokens = N;
        out.width = model.config.adapter.output_size;
        return true;
    }

    // --- GPU: captured per-mel_T graph running conv1/gelu/conv2/gelu/layers/
    // adapter end-to-end. The mel is the varying input; the conv front-end runs
    // as fast in-graph GEMMs (ggml_conv_1d with F32 inputs, the flash-attention
    // lesson) instead of the ~7s host scalar loop it replaced.
    register_encoder_cache_clearer_once();
    if (!g_encoder_cache)
        g_encoder_cache = std::unique_ptr<LruCache<ShapeKey, EncoderReplayEntry, ShapeKeyHash>>(
            new LruCache<ShapeKey, EncoderReplayEntry, ShapeKeyHash>(replay_cache_size()));

    const char* dump_enc = std::getenv("STARLING_ARK_DUMP_ENC");
    ShapeKey key{mel_T};
    EncoderReplayEntry& e = *g_encoder_cache->get_or_init(key,
        [&](EncoderReplayEntry& entry) {
            entry.mel_T = mel_T;
            entry.T_enc = T_enc;
            entry.mel_f32_buf = reinterpret_cast<float*>(entry.pool.alloc_bytes(
                (size_t) ec.num_mel_bins * mel_T * sizeof(float)));
            std::memcpy(entry.mel_f32_buf, mel_f32_ptr,
                        (size_t) ec.num_mel_bins * mel_T * sizeof(float));
            int64_t N = 0;
            entry.graph = std::make_unique<ReplayGraph>(global_backend(),
                [&](ggml_context* ctx) -> ggml_tensor* {
                    // mel as [L=mel_T, IC] f32 graph input (feat-major flat = ne0=L).
                    int64_t mne[2] = {entry.mel_T, (int64_t) ec.num_mel_bins};
                    ggml_tensor* mel_in = graph_input_tensor(ctx, GGML_TYPE_F32, 2, mne,
                        entry.mel_f32_buf,
                        (size_t) ec.num_mel_bins * entry.mel_T * sizeof(float));
                    ggml_tensor* layers_in = build_conv_front_end(ctx, model, entry.mel_T,
                                                                  entry.T_enc, mel_in);
                    ggml_tensor* enc = build_encoder_layers(ctx, model, entry.T_enc, layers_in);
                    if (dump_enc) capture_graph_output(f32(ctx, enc), &entry.enc_capture);
                    return build_adapter(ctx, model, enc, entry.T_enc, &N);
                });
            entry.N = N;
        });
    // Refresh the mel input in its stable pool buffer, then re-upload.
    std::memcpy(e.mel_f32_buf, mel_f32_ptr,
                (size_t) ec.num_mel_bins * e.mel_T * sizeof(float));
    for (size_t i = 0; i < e.graph->n_inputs(); ++i)
        e.graph->set_input(i, e.graph->input_host(i), e.graph->input_nbytes(i));

    std::vector<float> tmp;
    if (!e.graph->compute_with_captures(tmp)) { err = "ARK fused encoder+adapter replay failed"; return false; }
    if (dump_enc && !e.enc_capture.empty()) {
        if (FILE* f = std::fopen(dump_enc, "wb")) {
            std::fwrite(e.enc_capture.data(), sizeof(float), e.enc_capture.size(), f);
            std::fclose(f);
        }
    }
    out.data = std::move(tmp);
    out.n_tokens = e.N;
    out.width = model.config.adapter.output_size;
    return true;
}
} // namespace starling::ggml::ark
