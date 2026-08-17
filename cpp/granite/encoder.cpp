// encoder.cpp — granite-speech-4.1-2b CTC conformer encoder on the Starling
// ggml runtime.
//
// The audio path: Linear(160 -> 1024) then 16 conformer blocks, each
//   x += 0.5 * ff1(x);  x += attn(x);  x += conv(x);  x += 0.5 * ff2(x);
//   x = post_norm(x)
// with the self-conditioned mid CTC (x += out_mid(softmax(out(x)))) firing
// after 1-indexed block 8. ff = LayerNorm -> Linear(1024 -> 4096) -> SiLU ->
// Linear(4096 -> 1024) (both biased). Attention is BLOCK-LOCAL (windows of
// context_size 200 frames) with Shaw relative-position bias; the per-layer
// (200, 200, 128) bias table is PREBAKED in the GGUF (an exact embedding
// gather, converter-side). The conv module is LayerNorm -> pointwise up
// (1024 -> 4096) -> GLU -> depthwise conv k15 (zero pad (7,7), no bias) ->
// eval BatchNorm -> SiLU -> pointwise down (2048 -> 1024). The projector then
// windows the (1, T, 1024) output into 15-frame blocks and emits 3 decoder
// tokens per block (projector.cpp).
//
// Shaw attention in one batched matmul: the bias term needs, per query row c,
// the dot of q[:, :, c] with rep[c, :, :] — not a plain batched GEMM in the
// natural layout. Reordering q to [head_dim, heads*blocks, c] makes the
// c-axis the BATCH dim of ggml_mul_mat against the [head_dim, r, c] bias
// table, so the whole term is ONE matmul + one permute. Padded tail blocks
// are masked additively on the last block only, mirroring the reference.
//
// On GPU this is ONE captured ReplayGraph keyed on the stacked-mel length
// (LRU-bounded); on CPU / debug it is the one-shot build. All rounding
// boundaries follow the bf16 oracle (f32 elementwise, bf16 between ops).
//
// Divergence probe: STARLING_GRANITE_ONLY=<stage> truncates the build at the
// named stage and makes that node the graph's REAL output (the moss L0_STAGE
// pattern — intermediate capture readback is NOT a numerical oracle, it
// produced garbage during the port). Stages: melin, in, ffn, ffr, ffu, ffs,
// ffm (ff1's LN / raw norm / up / silu / down), ff1, attnm (raw attention),
// attn, cn, cu, cg, cd, cb, convm (conv LN / up / GLU / depthwise / BN /
// raw module), conv, ff2, post. The stage values land in the
// STARLING_GRANITE_DUMP_ENC file and transcription stops with an error.
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
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace starling::ggml::granite {
namespace {

using lib::bf16;
using lib::f32;
using lib::weight;

// The mask value the reference uses for padded attention (-bf16 max, NOT -inf,
// so a fully-masked row softmaxes to a finite uniform).
constexpr float kMaskNeg = -3.3895313892515355e38f;
// PyTorch BatchNorm1d default eps (the granite encoder config does not
// override it).
constexpr float kBnEps = 1e-5f;

// STARLING_GRANITE_ONLY probe state. `hit` marks the stage that truncated the
// build; stored in the caller-owned slot (the LRU entry / stack frame that
// outlives the graph build).
struct StageStop {
    const char* name = nullptr;
    bool hit = false;
};
inline bool stage_wants(const char* only, const char* stage) {
    return only && std::strcmp(only, stage) == 0;
}

// Build the per-T host constants backing the graph's constant inputs.
EncScratch make_scratch(const Config& c, int64_t T) {
    const auto& ec = c.encoder;
    EncScratch s;
    s.zeros7.assign((size_t) ec.conv_inner * (ec.conv_kernel / 2), 0.0f);
    s.nblocks = (T + ec.context_size - 1) / ec.context_size;
    const int64_t T_pad = s.nblocks * ec.context_size;
    s.pad = T_pad - T;
    if (s.pad > 0) {
        s.zeros_q.assign((size_t) ec.hidden * s.pad, 0.0f);
        s.zeros_kv.assign((size_t) ec.hidden * 2 * s.pad, 0.0f);
        // Additive [CS(r), CS(c), 1, nblk] f32 mask: all zeros except the LAST
        // block's plane, which carries -bf16max where c >= rem or r >= rem
        // (the reference masks only the padded tail block's pos_attn).
        const int64_t rem = T % ec.context_size;
        s.blk_mask.assign((size_t) ec.context_size * ec.context_size * s.nblocks, 0.0f);
        float* last = s.blk_mask.data() +
                      (size_t) ec.context_size * ec.context_size * (s.nblocks - 1);
        for (int64_t cq = 0; cq < ec.context_size; ++cq)
            for (int64_t r = 0; r < ec.context_size; ++r)
                if (cq >= rem || r >= rem)
                    last[(size_t) cq * ec.context_size + r] = kMaskNeg;
    }
    s.ones_bn.assign(ec.conv_inner, 1.0f);
    s.eps_bn.assign(ec.conv_inner, kBnEps);
    const int64_t nblk15 = (T + c.projector.window_size - 1) / c.projector.window_size;
    const int64_t pad15 = nblk15 * c.projector.window_size - T;
    if (pad15 > 0)
        s.zeros_proj.assign((size_t) c.projector.hidden * pad15, 0.0f);
    return s;
}

// ---- block pieces (all take/return [hidden, T] bf16) -----------------------

// ff = LN -> up -> SiLU -> down (all biased).
ggml_tensor* conformer_ff(ggml_context* c, const ModelLoader& ml,
                          const std::string& p, ggml_tensor* x, float eps,
                          StageStop* stop) {
    ggml_tensor* h = lib::layer_norm_bf16(c, ml, x, p + "_norm", eps);
    if (stage_wants(stop->name, "ffn")) { stop->hit = true; return h; }
    if (stage_wants(stop->name, "ffr")) { stop->hit = true; return ggml_norm(c, f32(c, x), eps); }
    h = lib::linear_bf16(c, ml, h, p + "_up", true);
    if (stage_wants(stop->name, "ffu")) { stop->hit = true; return h; }
    h = bf16(c, ggml_silu(c, f32(c, h)));
    if (stage_wants(stop->name, "ffs")) { stop->hit = true; return h; }
    h = lib::linear_bf16(c, ml, h, p + "_down", true);
    if (stage_wants(stop->name, "ffm")) { stop->hit = true; return h; }
    return h;
}

// Block-local Shaw relative-position attention (pre-norm inside).
ggml_tensor* shaw_attention(ggml_context* c, const GraniteModel& m, int li,
                            ggml_tensor* x, const EncScratch& s) {
    const auto& ec = m.config.encoder;
    const ModelLoader& ml = m.loader;
    const int64_t D = ec.head_dim, H = ec.n_heads, CS = ec.context_size;
    const int64_t hidden = ec.hidden;
    const int64_t T = x->ne[1];
    const int64_t nblk = s.nblocks;
    const int64_t T_pad = nblk * CS;
    const float scale = std::pow((float) ec.head_dim, -0.5f);
    const std::string p = "enc.blk." + std::to_string(li) + ".";

    ggml_tensor* n = lib::layer_norm_bf16(c, ml, x, p + "attn_norm",
                                          ec.layer_norm_eps);
    ggml_tensor* q = lib::linear_bf16(c, ml, n, p + "attn_q", false);   // [hidden, T]
    ggml_tensor* kv = lib::linear_bf16(c, ml, n, p + "attn_kv", false); // [2*hidden, T]
    // Zero-pad to whole blocks (the projections are bias-free, so padding the
    // projected q/kv with zero rows equals the reference's pad-then-project).
    if (s.pad > 0) {
        int64_t qne[2] = {hidden, s.pad};
        ggml_tensor* zq = graph_input_tensor(c, GGML_TYPE_F32, 2, qne,
                                             s.zeros_q.data(),
                                             s.zeros_q.size() * sizeof(float));
        int64_t kvne[2] = {hidden * 2, s.pad};
        ggml_tensor* zkv = graph_input_tensor(c, GGML_TYPE_F32, 2, kvne,
                                              s.zeros_kv.data(),
                                              s.zeros_kv.size() * sizeof(float));
        q = bf16(c, ggml_concat(c, f32(c, q), zq, 1));
        kv = bf16(c, ggml_concat(c, f32(c, kv), zkv, 1));
    }
    // Split kv: k = rows [0, hidden), v = rows [hidden, 2*hidden). The strided
    // half-views go through cont() (a value-exact copy) — reshape_4d below
    // asserts contiguity.
    ggml_tensor* k = ggml_cont(c, ggml_view_2d(c, kv, hidden, T_pad, kv->nb[1], 0));
    ggml_tensor* v = ggml_cont(c, ggml_view_2d(c, kv, hidden, T_pad, kv->nb[1],
                                               (size_t) hidden * kv->nb[0]));
    // [hidden, T_pad] -> [D, H, CS, nblk].
    auto to_blocks = [&](ggml_tensor* z) {
        return ggml_reshape_4d(c, z, D, H, CS, nblk);
    };
    ggml_tensor* q4 = to_blocks(q), * k4 = to_blocks(k), * v4 = to_blocks(v);

    // Content scores: [D, r, H, nblk] x [D, c, H, nblk] -> [r, c, H, nblk].
    ggml_tensor* k_ = ggml_cont(c, ggml_permute(c, k4, 0, 2, 1, 3));
    ggml_tensor* q_ = ggml_cont(c, ggml_permute(c, q4, 0, 2, 1, 3));
    ggml_tensor* sc = bf16(c, ggml_mul_mat(c, k_, q_));
    sc = bf16(c, ggml_scale(c, f32(c, sc), scale));

    // Shaw bias in one batched matmul: q as [D, H*nblk, c] against the baked
    // bias table viewed [D, r, c] (contraction over ne0, batch over c).
    ggml_tensor* qc = ggml_cont(c, ggml_permute(c, q4, 0, 1, 3, 2));  // (d, h, b, c)
    qc = ggml_reshape_3d(c, qc, D, H * nblk, CS);
    ggml_tensor* rep = weight(c, ml, p + "rel_pos_bias");             // (c, r, d) storage
    rep = ggml_reshape_3d(c, rep, D, CS, CS);                         // [D, r, c]
    ggml_tensor* pos = bf16(c, ggml_mul_mat(c, rep, qc));             // [r, H*nblk, c]
    pos = bf16(c, ggml_scale(c, f32(c, pos), scale));
    // (r, H*nblk, c) -> (r, H, nblk, c) -> (r, c, H, nblk), matching sc.
    // (ggml_permute is scatter-style: old dim i lands at position i_k.)
    pos = ggml_reshape_4d(c, pos, CS, H, nblk, CS);
    pos = ggml_cont(c, ggml_permute(c, pos, 0, 2, 3, 1));

    ggml_tensor* tot = lib::addb(c, sc, pos);
    if (s.pad > 0) {
        int64_t mne[4] = {CS, CS, 1, nblk};
        ggml_tensor* mask = graph_input_tensor(c, GGML_TYPE_F32, 4, mne,
                                               s.blk_mask.data(),
                                               s.blk_mask.size() * sizeof(float));
        tot = bf16(c, ggml_add(c, f32(c, tot), mask));
    }
    ggml_tensor* pr = bf16(c, ggml_soft_max_ext(c, f32(c, tot), nullptr, 1.0f, 0.0f));
    ggml_tensor* v_ = ggml_cont(c, ggml_permute(c, v4, 0, 2, 1, 3));  // [D, r, H, nblk]
    ggml_tensor* vt = ggml_cont(c, ggml_permute(c, v_, 1, 0, 2, 3));  // [r, D, H, nblk]
    ggml_tensor* co = bf16(c, ggml_mul_mat(c, vt, pr));               // [D, c, H, nblk]
    // heads -> features and drop the pad: [D, c, H, nblk] -> [hidden, T_pad] -> [:, :T].
    co = ggml_cont(c, ggml_permute(c, co, 0, 2, 1, 3));               // (d, h, c, b)
    co = ggml_reshape_2d(c, co, hidden, T_pad);
    ggml_tensor* out = ggml_view_2d(c, co, hidden, T, co->nb[1], 0);
    return lib::linear_bf16(c, ml, out, p + "attn_o", true);
}

// Conv module: LN -> pw up -> GLU -> depthwise k15 -> eval BatchNorm -> SiLU
// -> pw down. The depthwise conv is a 15-tap shift-multiply-accumulate (ggml's
// im2col conv is unvalidated under CUDA-graph capture in this build; pure
// elementwise ops are not), accumulated in f32 and rounded to bf16 once —
// the nn.Conv1d bf16 rounding boundary. NOTE: the pad is (K/2, K/2) — left
// AND right (a right-only pad shifts the whole signal and was the port's
// transcript-breaking bug).
ggml_tensor* conv_module(ggml_context* c, const GraniteModel& m, int li,
                         ggml_tensor* x, const EncScratch& s, StageStop* stop) {
    const auto& ec = m.config.encoder;
    const ModelLoader& ml = m.loader;
    const int64_t C = ec.conv_inner, T = x->ne[1], K = ec.conv_kernel;
    const std::string p = "enc.blk." + std::to_string(li) + ".";

    ggml_tensor* h = lib::layer_norm_bf16(c, ml, x, p + "conv_norm",
                                          ec.layer_norm_eps);
    if (stage_wants(stop->name, "cn")) { stop->hit = true; return h; }
    ggml_tensor* up = lib::linear_bf16(c, ml, h, p + "conv_up", true);  // [2C, T]
    if (stage_wants(stop->name, "cu")) { stop->hit = true; return up; }
    // GLU over channels: a = rows [0, C), b = rows [C, 2C); a * sigmoid(b).
    // The strided half-views go through cont() (value-exact) so the
    // elementwise ops below get contiguous inputs.
    ggml_tensor* a = ggml_cont(c, ggml_view_2d(c, up, C, T, up->nb[1], 0));
    ggml_tensor* b = ggml_cont(c, ggml_view_2d(c, up, C, T, up->nb[1],
                                               (size_t) C * up->nb[0]));
    ggml_tensor* g = lib::mulb(c, a, bf16(c, ggml_sigmoid(c, f32(c, b))));  // [C, T]
    if (stage_wants(stop->name, "cg")) { stop->hit = true; return g; }

    // Depthwise: zero-pad (K/2, K/2) in f32, then sum_k w[k, :] .* shifted rows.
    int64_t zne[2] = {C, K / 2};
    ggml_tensor* z7 = graph_input_tensor(c, GGML_TYPE_F32, 2, zne,
                                         s.zeros7.data(),
                                         s.zeros7.size() * sizeof(float));
    ggml_tensor* gp = ggml_concat(c, z7, f32(c, g), 1);                // left pad
    gp = ggml_concat(c, gp, z7, 1);                                    // right pad → [C, T+K-1]
    ggml_tensor* w = weight(c, ml, p + "conv_depth.weight");           // [C, K] k-major
    ggml_tensor* acc = nullptr;
    for (int64_t k = 0; k < K; ++k) {
        ggml_tensor* col = ggml_view_2d(c, w, C, 1, w->nb[1],
                                        (size_t) k * w->nb[1]);        // [C, 1]
        ggml_tensor* shifted = ggml_view_2d(c, gp, C, T, gp->nb[1],
                                            (size_t) k * gp->nb[1]);   // [C, T]
        ggml_tensor* term = ggml_mul(c, f32(c, shifted), f32(c, col));
        acc = acc ? ggml_add(c, acc, term) : term;
    }
    ggml_tensor* dw = bf16(c, acc);
    if (stage_wants(stop->name, "cd")) { stop->hit = true; return dw; }

    // Eval BatchNorm: y = ((x - mean) * invstd) * weight + bias with
    // invstd = 1 / sqrt(var + eps) per channel (f32 math, bf16 store).
    int64_t cne[2] = {C, 1};
    ggml_tensor* mean = ggml_reshape_2d(c, weight(c, ml, p + "bn_mean"), C, 1);
    ggml_tensor* var = ggml_reshape_2d(c, weight(c, ml, p + "bn_var"), C, 1);
    ggml_tensor* bnw = ggml_reshape_2d(c, weight(c, ml, p + "bn_weight"), C, 1);
    ggml_tensor* bnb = ggml_reshape_2d(c, weight(c, ml, p + "bn_bias"), C, 1);
    ggml_tensor* ones = graph_input_tensor(c, GGML_TYPE_F32, 2, cne,
                                           s.ones_bn.data(),
                                           s.ones_bn.size() * sizeof(float));
    ggml_tensor* epsv = graph_input_tensor(c, GGML_TYPE_F32, 2, cne,
                                           s.eps_bn.data(),
                                           s.eps_bn.size() * sizeof(float));
    ggml_tensor* stdv = ggml_sqrt(c, ggml_add(c, f32(c, var), epsv));
    ggml_tensor* invstd = ggml_div(c, ones, stdv);
    ggml_tensor* t = ggml_sub(c, f32(c, dw), f32(c, mean));
    t = ggml_mul(c, t, invstd);
    t = ggml_mul(c, t, f32(c, bnw));
    t = ggml_add(c, t, f32(c, bnb));
    ggml_tensor* sb = bf16(c, t);
    if (stage_wants(stop->name, "cb")) { stop->hit = true; return sb; }

    sb = bf16(c, ggml_silu(c, f32(c, sb)));
    return lib::linear_bf16(c, ml, sb, p + "conv_down", true);
}

// One conformer block + the mid-CTC hook.
ggml_tensor* conformer_block(ggml_context* c, const GraniteModel& m, int li,
                             ggml_tensor* x, const EncScratch& s,
                             StageStop* stop) {
    const auto& ec = m.config.encoder;
    const ModelLoader& ml = m.loader;
    const std::string p = "enc.blk." + std::to_string(li) + ".";
    // x += 0.5 * ff(x): 0.5 is an exact power of two.
    auto half_ff = [&](const char* ff) {
        ggml_tensor* y = conformer_ff(c, ml, p + ff, x, ec.layer_norm_eps, stop);
        if (stop->hit) return y;
        y = bf16(c, ggml_scale(c, f32(c, y), 0.5f));
        return lib::addb(c, x, y);
    };
    x = half_ff("ff1");
    if (stop->hit) return x;
    if (stage_wants(stop->name, "ff1")) { stop->hit = true; return x; }
    {
        ggml_tensor* a = shaw_attention(c, m, li, x, s);
        if (stage_wants(stop->name, "attnm")) { stop->hit = true; return a; }
        x = lib::addb(c, x, a);
    }
    if (stage_wants(stop->name, "attn")) { stop->hit = true; return x; }
    {
        ggml_tensor* v = conv_module(c, m, li, x, s, stop);
        if (stop->hit) return v;
        x = lib::addb(c, x, v);
    }
    if (stage_wants(stop->name, "conv")) { stop->hit = true; return x; }
    x = half_ff("ff2");
    if (stop->hit) return x;
    if (stage_wants(stop->name, "ff2")) { stop->hit = true; return x; }
    x = lib::layer_norm_bf16(c, ml, x, p + "post_norm", ec.layer_norm_eps);
    if (stage_wants(stop->name, "post")) { stop->hit = true; return x; }
    if ((int) li + 1 == (int) ec.mid_layer) {
        // Self-conditioned mid CTC: x += out_mid(softmax(out(x))).
        ggml_tensor* mid = lib::linear_bf16(c, ml, x, "enc.out", true);   // [348, T]
        ggml_tensor* pr = bf16(c, ggml_soft_max_ext(c, f32(c, mid), nullptr, 1.0f, 0.0f));
        ggml_tensor* fb = lib::linear_bf16(c, ml, pr, "enc.out_mid", true);
        x = lib::addb(c, x, fb);
    }
    return x;
}

// The full fused body: input linear -> 16 blocks -> projector. `mel_in` is
// [160, T] bf16; returns [output_dim, N] f32. When `enc_capture` is non-null
// the encoder's last hidden state is additionally read back (divergence dump).
ggml_tensor* build_fused(ggml_context* c, const GraniteModel& m, ggml_tensor* mel_in,
                         const EncScratch& s, std::vector<float>* enc_capture,
                         StageStop* stop) {
    const auto& ec = m.config.encoder;
    ggml_tensor* x = lib::linear_bf16(c, m.loader, mel_in, "enc.input_linear", true);
    if (stage_wants(stop->name, "melin")) { stop->hit = true; return f32(c, mel_in); }
    if (stage_wants(stop->name, "in")) { stop->hit = true; return f32(c, x); }
    for (uint32_t li = 0; li < ec.n_layers; ++li) {
        x = conformer_block(c, m, (int) li, x, s, stop);
        if (stop->hit) return f32(c, x);
    }
    if (enc_capture) capture_graph_output(f32(c, x), enc_capture);
    return f32(c, build_projector(c, m, x, s));
}

} // namespace

// ---------------------------------------------------------------------------
// Fused encode + project with a per-T bounded-LRU ReplayGraph cache (GPU),
// one-shot run_graph otherwise (the ark audio-encoder pattern).
// ---------------------------------------------------------------------------
namespace {
struct ShapeKey {
    int64_t T;
    bool operator==(const ShapeKey& o) const { return T == o.T; }
};
struct ShapeKeyHash {
    size_t operator()(const ShapeKey& k) const noexcept { return (size_t) k.T; }
};
struct EncoderReplayEntry {
    int64_t T = 0;
    GraphInputPool pool;
    EncScratch scratch;  // host constants backing graph inputs; stable for the
                         // ReplayGraph's lifetime (never resized after build)
    StageStop stop;      // STARLING_GRANITE_ONLY probe state (stable storage)
    std::unique_ptr<ReplayGraph> graph;
    ggml_bf16_t* mel_buf = nullptr;
    std::vector<float> enc_capture;
};
std::unique_ptr<LruCache<ShapeKey, EncoderReplayEntry, ShapeKeyHash>> g_encoder_cache;
std::once_flag g_encoder_once;
} // namespace

size_t encoder_replay_cache_size() {
    return g_encoder_cache ? g_encoder_cache->size() : 0;
}

bool encode_audio_and_project(const GraniteModel& model, const MelFeatures& mel,
                              AudioEmbeds& out, std::string& err) {
    ensure_weights_realized(model.loader);
    const auto& ec = model.config.encoder;
    const auto& pc = model.config.projector;
    if (mel.n_mels != (int64_t) ec.input_dim || mel.n_frames <= 0 ||
        mel.data.size() != (size_t) mel.n_mels * mel.n_frames) {
        err = "invalid GRANITE mel shape/data";
        return false;
    }
    const int64_t T = mel.n_frames;
    const int64_t nblk15 = (T + pc.window_size - 1) / pc.window_size;
    const int64_t N = nblk15 * pc.num_queries;
    const char* dump_env = std::getenv("STARLING_GRANITE_DUMP_ENC");
    if (const char* ms = std::getenv("STARLING_GRANITE_DUMP_MELSTK")) {
        if (FILE* f = std::fopen(ms, "wb")) {
            std::fwrite(mel.data.data(), sizeof(ggml_bf16_t), mel.data.size(), f);
            std::fclose(f);
        }
    }
    auto write_stage_dump = [&](const std::vector<float>& v) {
        if (dump_env) {
            if (FILE* f = std::fopen(dump_env, "wb")) {
                std::fwrite(v.data(), sizeof(float), v.size(), f);
                std::fclose(f);
            }
        }
    };

    // CPU + debug diagnostic path: one-shot build.
    if (!global_backend().is_gpu() || lib::debug_enabled("STARLING_GRANITE_DEBUG")) {
        EncScratch s = make_scratch(model.config, T);
        std::vector<float> enc_out;
        StageStop stop{std::getenv("STARLING_GRANITE_ONLY")};
        bool ok = run_graph([&](ggml_context* c) -> ggml_tensor* {
            int64_t mne[2] = {ec.input_dim, T};
            ggml_tensor* mel_in = graph_input_tensor(c, GGML_TYPE_BF16, 2, mne,
                                                     mel.data.data(),
                                                     mel.data.size() * sizeof(mel.data[0]));
            return build_fused(c, model, mel_in, s, dump_env ? &enc_out : nullptr, &stop);
        }, out.data);
        if (!ok) { err = "GRANITE encoder graph execution failed"; return false; }
        write_stage_dump(enc_out);
        if (stop.hit) {
            // Stage-only probe: the graph output IS the probed stage; dump it
            // host-side and stop the transcription here.
            write_stage_dump(out.data);
            err = "GRANITE stage-only probe (STARLING_GRANITE_ONLY)";
            return false;
        }
        out.n_tokens = N;
        out.width = pc.output_dim;
        return true;
    }

    // GPU: captured per-T graph; the mel is the only varying input.
    std::call_once(g_encoder_once, [] {
        register_decode_cache_clearer([] { g_encoder_cache.reset(); });
    });
    if (!g_encoder_cache)
        g_encoder_cache = std::unique_ptr<LruCache<ShapeKey, EncoderReplayEntry, ShapeKeyHash>>(
            new LruCache<ShapeKey, EncoderReplayEntry, ShapeKeyHash>(replay_cache_size()));

    ShapeKey key{T};
    EncoderReplayEntry& e = *g_encoder_cache->get_or_init(key,
        [&](EncoderReplayEntry& entry) {
            entry.T = T;
            entry.scratch = make_scratch(model.config, T);
            entry.stop.name = std::getenv("STARLING_GRANITE_ONLY");
            entry.mel_buf = reinterpret_cast<ggml_bf16_t*>(entry.pool.alloc_bytes(
                (size_t) ec.input_dim * T * sizeof(ggml_bf16_t)));
            std::memcpy(entry.mel_buf, mel.data.data(),
                        (size_t) ec.input_dim * T * sizeof(ggml_bf16_t));
            entry.graph = std::make_unique<ReplayGraph>(global_backend(),
                [&](ggml_context* c) -> ggml_tensor* {
                    int64_t mne[2] = {ec.input_dim, entry.T};
                    ggml_tensor* mel_in = graph_input_tensor(c, GGML_TYPE_BF16, 2, mne,
                        entry.mel_buf,
                        (size_t) ec.input_dim * entry.T * sizeof(ggml_bf16_t));
                    return build_fused(c, model, mel_in, entry.scratch,
                                       dump_env ? &entry.enc_capture : nullptr,
                                       &entry.stop);
                });
        });
    std::memcpy(e.mel_buf, mel.data.data(),
                (size_t) ec.input_dim * T * sizeof(ggml_bf16_t));
    for (size_t i = 0; i < e.graph->n_inputs(); ++i)
        e.graph->set_input(i, e.graph->input_host(i), e.graph->input_nbytes(i));
    std::vector<float> tmp;
    if (!e.graph->compute_with_captures(tmp)) {
        err = "GRANITE fused encoder+projector replay failed";
        return false;
    }
    write_stage_dump(e.enc_capture);
    if (e.stop.hit) {
        write_stage_dump(tmp);
        err = "GRANITE stage-only probe (STARLING_GRANITE_ONLY)";
        return false;
    }
    out.data = std::move(tmp);
    out.n_tokens = N;
    out.width = pc.output_dim;
    return true;
}

} // namespace starling::ggml::granite
