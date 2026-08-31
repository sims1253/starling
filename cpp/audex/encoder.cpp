// encoder.cpp — Nemotron-Labs-Audex-2B audio encoder (stock Qwen2AudioEncoder,
// whisper-large-v3 shaped) + projector on the Starling ggml runtime.
//
// Fixed-shape path per 30 s clip (every clip is zero-padded to 3000 mel
// frames, and Qwen2AudioEncoder.forward hard-asserts that length): the
// time-major [3000, 128] mel runs through two GELU Conv1d k3/p1 layers over
// time (stride 1 then 2: 3000 -> 1500), is permuted to feature-major
// [1280, 1500], and gets the LEARNED (1500, 1280) positional table added.
// 32 pre-norm layers run FULL bidirectional attention over all 1500 frames
// (no attention mask — the reference attends padded tail frames like any
// other): biased LayerNorm, biased q/v/out with a BIAS-FREE k projection,
// the query pre-scaled by head_dim^-0.5 = 0.125 at projection time (exact in
// bf16; the eager reference then runs softmax(QK^T) V with scaling 1.0), 20
// heads x 64, biased erf-GELU FFN 1280 -> 5120 -> 1280. The avg-pooler
// halves 1500 -> 750 (pairs averaged in f32, one bf16 round — torch
// avg_pool1d's acc-type kernel) and the final biased LayerNorm closes the
// encoder. The projector's norm is the hand-written
// NemotronDenseAudexRMSNorm (f32 normalize + f32 affine, ONE bf16 round at
// the return — numerically the F.rms_norm single-round discipline; only the
// decoder trunk's NemotronDenseRMSNorm literally calls F.rms_norm) ->
// bias-free fc1 1280 -> 4096 -> relu(x)^2 -> bias-free fc2 -> 2048.
//
// The conv stack is an explicit F32 im2col + F32 GEMM (see conv1d_step),
// built over ggml_conv_1d for the same reason as qwen3's conv_step: ggml's
// packaged conv paths land the GEMM on the batched F16 cuBLAS path, which
// accumulates in F16 — a systematic ~1-ulp-of-bf16 error. With F32 operands
// the GEMM accumulates in F32 (bf16 inputs are exact in both). As with
// qwen3's conv frontend, a GEMM formulation cannot bitwise-match cuDNN conv
// in general; parity holds on the gated fixtures.
//
// On GPU this is ONE captured ReplayGraph (the shapes are fixed, so a
// single-entry LRU keyed on the mel frame count); on CPU / debug it is the
// one-shot build. All rounding boundaries follow the bf16 oracle (f32
// elementwise, bf16 between ops).
//
// Divergence probe: STARLING_AUDEX_ONLY=<stage> truncates the build at the
// named stage and makes that node the graph's REAL output (the granite/qwen3
// stage-truncation pattern — intermediate capture readback is NOT a
// numerical oracle). Stages: melin, c1, c2 (post-GELU convs), pos
// (+positional table), anorm / q / sc / pr / attnm / attn / ffn (layer-0 LN /
// scaled query / scores / softmax / attention output / residuals), lay<i>
// (layer-i output), pool (avg-pooled), enc (post-ln_post hidden state — also
// the STARLING_AUDEX_DUMP_ENC capture), proj (projector output). The stage
// values land in the STARLING_AUDEX_DUMP_ENC file and transcription stops
// with an error.
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
#include <vector>

namespace starling::ggml::audex {
namespace {

using lib::bf16;
using lib::f32;
using lib::weight;

// STARLING_AUDEX_ONLY probe state. `hit` marks the stage that truncated the
// build; stored in the caller-owned slot (the LRU entry / stack frame that
// outlives the graph build).
struct StageStop {
    const char* name = nullptr;
    bool hit = false;
};
inline bool stage_wants(const char* only, const char* stage) {
    return only && std::strcmp(only, stage) == 0;
}

// One GELU conv1d k3/p1 step over the time axis (stride 1 or 2): im2col +
// GEMM, all F32, + bias, bf16 round, erf GELU. Takes/returns [time, 1, C, N]
// bf16. Built explicitly over ggml's packaged conv for the F32-accumulation
// reason documented in the file comment (the qwen3 conv_step pattern with a
// degenerate H axis: Conv1d k3 == Conv2d (1, k3), so H=1, KH=1, p1=0).
ggml_tensor* conv1d_step(ggml_context* c, const ModelLoader& ml, const char* wname,
                         const char* bname, ggml_tensor* y, int stride) {
    // The GGUF carries the checkpoint's 3D [OC, IC, K] layout, which ggml
    // reads as ne [K, IC, OC]; im2col wants the 4D kernel [K, KH=1, IC, OC]
    // (the degenerate axis insertion is byte-compatible).
    ggml_tensor* w = f32(c, weight(c, ml, wname));
    w = ggml_reshape_4d(c, w, w->ne[0], 1, w->ne[1], w->ne[2]);  // [K, 1, IC, OC]
    ggml_tensor* x = f32(c, y);                                // [W, 1, IC, N]
    ggml_tensor* im2 = ggml_im2col(c, w, x, stride, 1, 1, 0, 1, 1, true,
                                   GGML_TYPE_F32);             // [K, OW, OH, N]
    const int64_t K = im2->ne[0];
    ggml_tensor* im2_2d = ggml_reshape_2d(c, im2, K,
                                          im2->ne[3] * im2->ne[2] * im2->ne[1]);
    ggml_tensor* w2 = ggml_reshape_2d(c, w, K, w->ne[3]);
    ggml_tensor* conv = ggml_mul_mat(c, im2_2d, w2);           // [spatial, OC]
    conv = ggml_reshape_4d(c, conv, im2->ne[1], im2->ne[2], im2->ne[3], w->ne[3]);
    conv = ggml_cont(c, ggml_permute(c, conv, 0, 1, 3, 2));   // [OW, OH, OC, N]
    ggml_tensor* b = f32(c, weight(c, ml, bname));
    int64_t OC = conv->ne[2];
    int64_t bne[4] = {1, 1, OC, 1};
    ggml_tensor* bias = ggml_reshape_4d(c, b, bne[0], bne[1], bne[2], bne[3]);
    ggml_tensor* t = bf16(c, ggml_add(c, f32(c, conv), bias));
    return lib::gelu_erf_bf16(c, t);
}

// ---- one pre-norm full-attention layer (all take/return [hidden, T] bf16) --

ggml_tensor* encoder_layer(ggml_context* c, const AudexModel& m, int li,
                           ggml_tensor* x, int64_t T, StageStop* stop) {
    const auto& ec = m.config.encoder;
    const ModelLoader& ml = m.loader;
    const int64_t D = ec.head_dim, H = ec.n_heads;
    const int64_t hidden = ec.hidden;
    // The reference pre-scales the query at projection time (head_dim^-0.5 =
    // 0.125, exactly representable — the bf16 round-trip is lossless) and
    // passes scaling=1.0 to the eager attention.
    const float qscale = 1.0f / std::sqrt((float) D);
    const std::string p = "enc.blk." + std::to_string(li) + ".";

    ggml_tensor* n = lib::layer_norm_bf16(c, ml, x, p + "attn_norm",
                                          ec.layer_norm_eps);
    if (stage_wants(stop->name, "anorm")) { stop->hit = true; return f32(c, n); }
    ggml_tensor* q = lib::linear_bf16(c, ml, n, p + "attn_q", true);
    q = bf16(c, ggml_scale(c, f32(c, q), qscale));
    if (stage_wants(stop->name, "q")) { stop->hit = true; return f32(c, q); }
    ggml_tensor* k = lib::lin(c, ml, n, p + "attn_k.weight");  // bias-free keys
    ggml_tensor* v = lib::linear_bf16(c, ml, n, p + "attn_v", true);
    // [hidden, T] -> [D, T, H]: the within-sequence position must be ne1 so
    // the score matmul contracts D and yields [T(k), T(q), H] (heads ride the
    // batch dim). Feature = h*D + d. ggml_permute is scatter-style: old dim j
    // lands at position axis_j.
    auto to_heads = [&](ggml_tensor* z) {
        return ggml_cont(c, ggml_permute(c,
            ggml_reshape_3d(c, z, D, H, T), 0, 2, 1, 3));
    };
    ggml_tensor* q4 = to_heads(q), * k4 = to_heads(k), * v4 = to_heads(v);
    // Scores with KEYS innermost (softmax runs over ne0): [T(k), T(q), H].
    // NO additive mask — the reference attends every frame bidirectionally.
    ggml_tensor* sc = bf16(c, ggml_mul_mat(c, k4, q4));
    if (stage_wants(stop->name, "sc")) { stop->hit = true; return f32(c, sc); }
    ggml_tensor* pr = bf16(c, ggml_soft_max_ext(c, f32(c, sc), nullptr,
                                                1.0f, 0.0f));
    if (stage_wants(stop->name, "pr")) { stop->hit = true; return f32(c, pr); }
    // ggml_permute is scatter-style (old dim j lands at position axis_j):
    // a 0<->1 axis swap gives [T, D, H] from [D, T, H].
    ggml_tensor* vt = ggml_cont(c, ggml_permute(c, v4, 1, 0, 2, 3));  // [T, D, H]
    ggml_tensor* co = bf16(c, ggml_mul_mat(c, vt, pr));               // [D, T, H]
    co = ggml_cont(c, ggml_permute(c, co, 0, 2, 1, 3));               // [D, H, T]
    co = ggml_reshape_2d(c, co, hidden, T);
    ggml_tensor* a = lib::linear_bf16(c, ml, co, p + "attn_o", true);
    if (stage_wants(stop->name, "attnm")) { stop->hit = true; return a; }
    x = lib::addb(c, x, a);
    if (stage_wants(stop->name, "attn")) { stop->hit = true; return x; }

    ggml_tensor* n2 = lib::layer_norm_bf16(c, ml, x, p + "ffn_norm",
                                           ec.layer_norm_eps);
    ggml_tensor* u = lib::linear_bf16(c, ml, n2, p + "ff_up", true);
    u = lib::gelu_erf_bf16(c, u);
    ggml_tensor* d = lib::linear_bf16(c, ml, u, p + "ff_down", true);
    x = lib::addb(c, x, d);
    if (stage_wants(stop->name, "ffn")) { stop->hit = true; return x; }
    return x;
}

// The full fused body: convs -> +pos -> 32 layers -> avg-pool -> ln_post ->
// projector. `mel_in` is [3000, 128] bf16 (feat-major host layout, time
// innermost); returns [output_dim, 750] f32. When `enc_capture` is non-null
// the post-ln_post hidden state is additionally read back (divergence dump).
ggml_tensor* build_fused(ggml_context* c, const AudexModel& m, ggml_tensor* mel_in,
                         std::vector<float>* enc_capture, StageStop* stop) {
    const auto& ec = m.config.encoder;
    const ModelLoader& ml = m.loader;
    const int64_t hidden = ec.hidden;
    const int64_t T = mel_in->ne[0];  // 3000 (time innermost)
    if ((int64_t) mel_in->ne[1] != ec.n_mel || T != 2 * (int64_t) ec.max_pos_emb) {
        // Shape contract is enforced host-side; a mismatch here is a graph
        // construction bug, not a user input.
        return nullptr;
    }
    if (stage_wants(stop->name, "melin")) { stop->hit = true; return f32(c, mel_in); }
    // Conv data [W=time, H=1, IC=128, N=1] from the time-major mel.
    ggml_tensor* x = f32(c, mel_in);
    x = ggml_reshape_4d(c, x, T, 1, ec.n_mel, 1);
    ggml_tensor* g = conv1d_step(c, ml, "enc.conv1.weight", "enc.conv1.bias",
                                 x, /*stride=*/1);
    if (stage_wants(stop->name, "c1")) { stop->hit = true; return f32(c, g); }
    g = conv1d_step(c, ml, "enc.conv2.weight", "enc.conv2.bias", g, /*stride=*/2);
    if (stage_wants(stop->name, "c2")) { stop->hit = true; return f32(c, g); }
    // [t, 1, ch, 1] -> [ch, t]: permute (1, 2, 0, 3) puts old dim 2 (ch) at
    // position 0 and old dim 0 (t) at position 1 (scatter-style permute).
    ggml_tensor* h = ggml_cont(c, ggml_permute(c, g, 1, 2, 0, 3));
    h = ggml_reshape_2d(c, h, hidden, ec.max_pos_emb);
    // + learned positional table ([hidden, 1500], same shape — plain add).
    ggml_tensor* pos = f32(c, weight(c, ml, "enc.pos_embed"));
    h = bf16(c, ggml_add(c, f32(c, h), pos));
    if (stage_wants(stop->name, "pos")) { stop->hit = true; return f32(c, h); }
    for (uint32_t li = 0; li < ec.n_layers; ++li) {
        h = encoder_layer(c, m, (int) li, h, ec.max_pos_emb, stop);
        if (stop->hit) return f32(c, h);
        {
            char lay[16];
            std::snprintf(lay, sizeof lay, "lay%u", li);
            if (stage_wants(stop->name, lay)) { stop->hit = true; return f32(c, h); }
        }
    }
    // Avg-pooler (kernel 2, stride 2) over consecutive time pairs: even/odd
    // strided views, f32 add + *0.5, one bf16 round (torch's acc-type
    // avg_pool1d kernel). ggml_reshape asserts contiguity, so the pair split
    // goes through views, not a [hidden, 2, 750] reshape.
    ggml_tensor* xe = ggml_view_2d(c, h, hidden, ec.out_frames,
                                   2 * h->nb[1], 0);
    ggml_tensor* xo = ggml_view_2d(c, h, hidden, ec.out_frames,
                                   2 * h->nb[1], h->nb[1]);
    ggml_tensor* pooled = bf16(c, ggml_scale(c,
        ggml_add(c, f32(c, xe), f32(c, xo)), 0.5f));
    if (stage_wants(stop->name, "pool")) { stop->hit = true; return f32(c, pooled); }
    ggml_tensor* enc = lib::layer_norm_bf16(c, ml, pooled, "enc.ln_post",
                                            ec.layer_norm_eps);
    if (stage_wants(stop->name, "enc")) { stop->hit = true; return f32(c, enc); }
    if (enc_capture) capture_graph_output(f32(c, enc), enc_capture);
    // Projector: single-round RMSNorm -> fc1 -> relu^2 -> fc2 (bias-free).
    ggml_tensor* pn = lib::rms_single(c, ml, enc, "proj.norm.weight",
                                      m.config.projector.norm_eps);
    ggml_tensor* u = lib::lin(c, ml, pn, "proj.fc1.weight");
    ggml_tensor* r = ggml_relu(c, f32(c, u));
    ggml_tensor* pj = lib::lin(c, ml, bf16(c, ggml_mul(c, r, r)),
                               "proj.fc2.weight");
    if (stage_wants(stop->name, "proj")) { stop->hit = true; return f32(c, pj); }
    return f32(c, pj);
}

} // namespace

// ---------------------------------------------------------------------------
// Fused encode + project with a bounded-LRU ReplayGraph cache (GPU). The
// mel frame count is pinned to 3000 by the mel adapter, so in practice a
// single entry serves every clip; the LRU keeps the structure identical to
// the qwen3 encoder cache. One-shot run_graph otherwise (CPU / debug).
// ---------------------------------------------------------------------------
namespace {
struct EncoderReplayEntry {
    int64_t T_pad = 0;
    GraphInputPool pool;
    std::unique_ptr<ReplayGraph> graph;
    ggml_bf16_t* mel_buf = nullptr;
    std::vector<float> enc_capture;
};
// Deliberately leaked pointer, mirroring qwen3's encoder cache (and
// qwen_decode's spec_states()): the ONLY teardown path is the decode-cache
// clearer registered at first use — a namespace-scope destructor would race
// the shutdown handler instead.
LruCache<int64_t, EncoderReplayEntry>* g_encoder_cache = nullptr;
std::once_flag g_encoder_once;
} // namespace

size_t encoder_replay_cache_size() {
    return g_encoder_cache ? g_encoder_cache->size() : 0;}

bool encode_audio_and_project(const AudexModel& model, const MelFeatures& mel,
                              AudioEmbeds& out, std::string& err) {
    ensure_weights_realized(model.loader);
    const auto& ec = model.config.encoder;
    if (mel.n_mels != (int64_t) ec.n_mel ||
        mel.n_frames != 2 * (int64_t) ec.max_pos_emb ||
        mel.valid_frames != mel.n_frames ||
        mel.data.size() != (size_t) mel.n_mels * mel.n_frames) {
        err = "invalid AUDEX mel shape/data";
        return false;
    }
    const char* dump_env = std::getenv("STARLING_AUDEX_DUMP_ENC");
    if (const char* ms = std::getenv("STARLING_AUDEX_DUMP_MELSTK")) {
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

    // CPU + debug diagnostic path: one-shot build. Stage probes also force
    // this path — a truncated probe graph must never enter the ReplayGraph
    // LRU, where it would poison later normal transcriptions (granite
    // pullfrog review).
    if (!global_backend().is_gpu() || lib::debug_enabled("STARLING_AUDEX_DEBUG") ||
        std::getenv("STARLING_AUDEX_ONLY")) {
        std::vector<float> enc_out;
        StageStop stop{std::getenv("STARLING_AUDEX_ONLY")};
        bool ok = run_graph([&](ggml_context* c) -> ggml_tensor* {
            int64_t mne[2] = {mel.n_frames, mel.n_mels};
            ggml_tensor* mel_in = graph_input_tensor(c, GGML_TYPE_BF16, 2, mne,
                                                     mel.data.data(),
                                                     mel.data.size() * sizeof(mel.data[0]));
            return build_fused(c, model, mel_in,
                               dump_env ? &enc_out : nullptr, &stop);
        }, out.data);
        if (!ok) { err = "AUDEX encoder graph execution failed"; return false; }
        write_stage_dump(enc_out);
        if (stop.hit) {
            // Stage-only probe: the graph output IS the probed stage; dump it
            // host-side and stop the transcription here.
            write_stage_dump(out.data);
            err = "AUDEX stage-only probe (STARLING_AUDEX_ONLY)";
            return false;
        }
        out.n_tokens = ec.out_frames;
        out.width = model.config.projector.output_dim;
        return true;
    }

    // GPU: captured graph (single shape in practice); the mel is the only
    // varying input.
    std::call_once(g_encoder_once, [] {
        register_decode_cache_clearer([] { delete g_encoder_cache; g_encoder_cache = nullptr; });
    });
    if (!g_encoder_cache)
        g_encoder_cache = new LruCache<int64_t, EncoderReplayEntry>(
            replay_cache_size());

    EncoderReplayEntry& e = *g_encoder_cache->get_or_init(mel.n_frames,
        [&](EncoderReplayEntry& entry) {
            entry.T_pad = mel.n_frames;
            entry.mel_buf = reinterpret_cast<ggml_bf16_t*>(entry.pool.alloc_bytes(
                (size_t) mel.n_mels * mel.n_frames * sizeof(ggml_bf16_t)));
            std::memcpy(entry.mel_buf, mel.data.data(),
                        (size_t) mel.n_mels * mel.n_frames * sizeof(ggml_bf16_t));
            StageStop probe_stop{nullptr};  // probes never reach this path
            entry.graph = std::make_unique<ReplayGraph>(global_backend(),
                [&](ggml_context* c) -> ggml_tensor* {
                    int64_t mne[2] = {entry.T_pad, mel.n_mels};
                    ggml_tensor* mel_in = graph_input_tensor(c, GGML_TYPE_BF16, 2, mne,
                        entry.mel_buf,
                        (size_t) mel.n_mels * entry.T_pad * sizeof(mel.data[0]));
                    return build_fused(c, model, mel_in,
                                       dump_env ? &entry.enc_capture : nullptr,
                                       &probe_stop);
                });
        });
    std::memcpy(e.mel_buf, mel.data.data(),
                (size_t) mel.n_mels * mel.n_frames * sizeof(ggml_bf16_t));
    for (size_t i = 0; i < e.graph->n_inputs(); ++i)
        e.graph->set_input(i, e.graph->input_host(i), e.graph->input_nbytes(i));
    std::vector<float> tmp;
    if (!e.graph->compute_with_captures(tmp)) {
        err = "AUDEX fused encoder+projector replay failed";
        return false;
    }
    write_stage_dump(e.enc_capture);
    out.data = std::move(tmp);
    out.n_tokens = ec.out_frames;
    out.width = model.config.projector.output_dim;
    return true;
}

} // namespace starling::ggml::audex
