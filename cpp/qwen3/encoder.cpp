// encoder.cpp — Qwen3-ASR-1.7B windowed-attention conv encoder on the Starling
// ggml runtime.
//
// The audio path per 100-frame mel chunk: three GELU Conv2d k3/s2/p1 layers
// (480 channels; freq 128->64->32->16, time 100->50->25->13), then a
// bias-free Linear(7680 -> 1024) over the (channel-major, freq-inner)
// features, + a (13, 1024) sinusoidal position table. The first post-CNN
// rows per chunk (triple ceil-halving of the chunk's VALID frame count: 13
// per full chunk) are gathered into a packed sequence of length L, padded to
// whole 104-row attention windows (n_window_infer 800 = 8 chunks), then 24
// pre-norm layers run FULL (non-causal) attention within each window — biased
// MHA (16 heads x 64, scale 1/8) + biased FFN (1024 -> 4096 -> 1024, erf
// GELU). ln_post (biased LayerNorm) closes the encoder; the projector is
// Linear(1024 -> 1024) + erf GELU + Linear(1024 -> 2048).
//
// The conv stack is an explicit F32 im2col + F32 GEMM (see conv_step):
// ggml_conv_2d's F16 im2col lands the GEMM on ggml's batched F16 cuBLAS
// path, which accumulates in F16 — a systematic ~1-ulp-of-bf16 error on a
// quarter of the outputs. With F32 operands the GEMM accumulates in F32
// (bf16 inputs are exact in both).
// The windowed attention is batched over (heads, windows) in three matmuls:
// the reference splits the packed sequence per window and attends within;
// padded tail rows of the last window are masked additively (finite -bf16max
// so softmax stays finite) and trimmed after ln_post — row-local ops keep
// the garbage out of valid rows.
//
// On GPU this is ONE captured ReplayGraph keyed on (T_pad, L) (LRU-bounded);
// on CPU / debug it is the one-shot build. All rounding boundaries follow
// the bf16 oracle (f32 elementwise, bf16 between ops).
//
// Divergence probe: STARLING_QWEN3_ONLY=<stage> truncates the build at the
// named stage and makes that node the graph's REAL output (the granite
// stage-truncation pattern — intermediate capture readback is NOT a
// numerical oracle). Stages: melin, c1, c2, c3 (post-GELU convs), flat (the
// reshaped conv_out feature matrix), lin (conv_out linear), pos
// (+positional table), pack (window-padded valid-row gather), anorm / sc /
// pr / attnm / attn / ffn (layer-0 LN / scaled scores+mask / softmax /
// attention output / residuals), lay<i> (layer-i output), enc (post-ln_post
// packed hidden state). The stage values land in the STARLING_QWEN3_DUMP_ENC
// file and transcription stops with an error.
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

namespace starling::ggml::qwen3 {
namespace {

using lib::bf16;
using lib::f32;
using lib::weight;

// The mask value the padded-window attention uses (-bf16 max, NOT -inf, so a
// fully-masked row softmaxes to a finite uniform).
constexpr float kMaskNeg = -3.3895313892515355e38f;

// STARLING_QWEN3_ONLY probe state. `hit` marks the stage that truncated the
// build; stored in the caller-owned slot (the LRU entry / stack frame that
// outlives the graph build).
struct StageStop {
    const char* name = nullptr;
    bool hit = false;
};
inline bool stage_wants(const char* only, const char* stage) {
    return only && std::strcmp(only, stage) == 0;
}

// Triple ceil-halving ((x-1)/2+1 thrice); zero stays zero
// (Qwen3ASREncoder._post_cnn_length).
inline int64_t post_cnn(int64_t x) {
    for (int i = 0; i < 3 && x > 0; ++i) x = (x - 1) / 2 + 1;
    return x;
}

// Build the per-shape host constants backing the graph's constant inputs.
struct EncScratch {
    int64_t nc = 0;        // chunks (T_pad / 100)
    int64_t L = 0;         // packed valid length
    int64_t W = 0;         // attention window size (packed rows)
    int64_t nW = 0;        // number of windows
    int64_t rem = 0;       // valid rows in the last window (0 -> full)
    int64_t pad = 0;       // packed tail pad (nW*W - L)
    std::vector<int32_t> valid_indices;          // [L (+ pad tail)]
    std::vector<float> wmask;                    // [W*W*nW], empty when rem==0
};
EncScratch make_scratch(const Config& c, int64_t T) {
    const auto& ec = c.encoder;
    EncScratch s;
    const int64_t chunk = 2 * (int64_t) c.frontend.n_window;   // 100
    const int64_t full_rows = ec.max_pos_emb;                  // 13
    const int64_t r = T % chunk;
    const int64_t nc = (T + chunk - 1) / chunk;
    s.nc = nc;
    // Packed length: 13 rows per full chunk plus the triple-halved tail of a
    // PARTIAL last chunk (r == 0 means every chunk is full — the last chunk
    // is not an extra one).
    s.L = r == 0 ? nc * full_rows : (nc - 1) * full_rows + post_cnn(r);
    // Window size: n_window_ratio = n_window_infer / chunk = 8 chunks worth
    // of the LARGEST per-chunk post-CNN length (get_audio_cu_seqlens): 13
    // whenever a full chunk exists, else the lone partial chunk's length.
    const int64_t max_rows = (r == 0 || nc > 1) ? full_rows : post_cnn(r);
    s.W = max_rows * (ec.n_window_infer / chunk);
    s.nW = s.L / s.W + (s.L % s.W != 0 ? 1 : 0);
    s.rem = s.L % s.W;
    s.pad = s.nW * s.W - s.L;
    s.valid_indices.reserve((size_t) (s.L + s.pad));
    for (int64_t ch = 0; ch < nc; ++ch) {
        const int64_t rows = (r != 0 && ch + 1 == nc) ? post_cnn(r) : full_rows;
        for (int64_t t = 0; t < rows; ++t)
            s.valid_indices.push_back((int32_t) (ch * full_rows + t));
    }
    // Tail rows padding out the last attention window duplicate row 0: their
    // values never enter the valid rows (attention masks them as keys; every
    // layer op is row-local) and the trim after ln_post drops them — so no
    // zero-pad concat is needed anywhere.
    for (int64_t i = 0; i < s.pad; ++i) s.valid_indices.push_back(0);
    if (s.rem != 0) {
        // Additive [W(k), W(q), 1, nW] f32 mask: all zeros except the LAST
        // window's plane, which carries -bf16max where row >= rem or
        // col >= rem (the reference's per-window split has no tail at all;
        // the mask reproduces exact-length attention for the valid rows).
        s.wmask.assign((size_t) s.W * s.W * s.nW, 0.0f);
        float* last = s.wmask.data() + (size_t) s.W * s.W * (s.nW - 1);
        for (int64_t q = 0; q < s.W; ++q)
            for (int64_t k = 0; k < s.W; ++k)
                if (q >= s.rem || k >= s.rem)
                    last[(size_t) q * s.W + k] = kMaskNeg;
    }
    return s;
}

// ---- layer pieces (all take/return [hidden, P] bf16) -----------------------

// One windowed-attention layer: pre-LN, batched MHA over (heads, windows),
// residual, pre-LN, biased FFN (erf GELU), residual.
ggml_tensor* windowed_layer(ggml_context* c, const Qwen3Model& m, int li,
                            ggml_tensor* x, const EncScratch& s,
                            StageStop* stop) {
    const auto& ec = m.config.encoder;
    const ModelLoader& ml = m.loader;
    const int64_t D = ec.head_dim, H = ec.n_heads;
    const int64_t hidden = ec.hidden;
    const int64_t P = s.nW * s.W;
    const float scale = 1.0f / std::sqrt((float) D);   // 64^-0.5 = 0.125
    const std::string p = "enc.blk." + std::to_string(li) + ".";

    ggml_tensor* n = lib::layer_norm_bf16(c, ml, x, p + "attn_norm",
                                          ec.layer_norm_eps);
    if (stage_wants(stop->name, "anorm")) { stop->hit = true; return f32(c, n); }
    ggml_tensor* q = lib::linear_bf16(c, ml, n, p + "attn_q", true);
    ggml_tensor* k = lib::linear_bf16(c, ml, n, p + "attn_k", true);
    ggml_tensor* v = lib::linear_bf16(c, ml, n, p + "attn_v", true);
    // [hidden, P] -> [D, W, H, nW]: the WITHIN-WINDOW position must be ne1 so
    // the score matmul contracts D and yields [W(k), W(q), H, nW] (heads and
    // windows ride the batch dims). Feature = h*D + d, position = w_in + W*w,
    // so the 4D view [D, H, W, nW] swaps its middle axes (ggml_permute is
    // scatter-style: old dim j lands at position axis_j).
    auto to_heads = [&](ggml_tensor* z) {
        return ggml_cont(c, ggml_permute(c,
            ggml_reshape_4d(c, z, D, H, s.W, s.nW), 0, 2, 1, 3));
    };
    ggml_tensor* q4 = to_heads(q), * k4 = to_heads(k), * v4 = to_heads(v);
    // Scores with KEYS innermost (softmax runs over ne0): [W(k), W(q), H, nW].
    ggml_tensor* sc = bf16(c, ggml_mul_mat(c, k4, q4));
    sc = bf16(c, ggml_scale(c, f32(c, sc), scale));
    if (!s.wmask.empty()) {
        int64_t mne[4] = {s.W, s.W, 1, s.nW};
        ggml_tensor* mask = graph_input_tensor(c, GGML_TYPE_F32, 4, mne,
                                               s.wmask.data(),
                                               s.wmask.size() * sizeof(float));
        sc = bf16(c, ggml_add(c, f32(c, sc), mask));
    }
    if (stage_wants(stop->name, "sc")) { stop->hit = true; return f32(c, sc); }
    ggml_tensor* pr = bf16(c, ggml_soft_max_ext(c, f32(c, sc), nullptr,
                                                1.0f, 0.0f));
    if (stage_wants(stop->name, "pr")) { stop->hit = true; return f32(c, pr); }
    // ggml_permute is scatter-style (old dim j lands at position axis_j):
    // a 0<->1 axis swap gives [W, D, H, nW] from [D, W, H, nW].
    ggml_tensor* vt = ggml_cont(c, ggml_permute(c, v4, 1, 0, 2, 3)); // [W, D, H, nW]
    ggml_tensor* co = bf16(c, ggml_mul_mat(c, vt, pr));              // [D, W, H, nW]
    co = ggml_cont(c, ggml_permute(c, co, 0, 2, 1, 3));              // [D, H, W, nW]
    co = ggml_reshape_2d(c, co, hidden, P);
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

// One GELU conv2d k3/s2/p1 step: im2col + GEMM, all F32, + bias, bf16 round,
// erf GELU. Takes/returns [time, freq, 480, nc]; bf16 in, bf16 out. Built
// explicitly over ggml_conv_2d because the latter's F16 im2col puts the GEMM
// on ggml's batched F16 cuBLAS path, which accumulates in F16 (COMPUTE_16F)
// — a ~1-ulp-of-bf16 systematic error on a quarter of the outputs. With an
// F32 im2col and F32 weights the GEMM accumulates in F32 (bf16 inputs are
// exact in both), leaving only reduction-order noise far below the bf16
// rounding boundary.
ggml_tensor* conv_step(ggml_context* c, const ModelLoader& ml, const char* wname,
                       const char* bname, ggml_tensor* y) {
    ggml_tensor* w = f32(c, weight(c, ml, wname));            // [KW,KH,IC,OC]
    ggml_tensor* x = f32(c, y);                                // [W,H,IC,N]
    ggml_tensor* im2 = ggml_im2col(c, w, x, 2, 2, 1, 1, 1, 1, true,
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

// The full fused body: chunked convs -> conv_out -> +pos -> gather -> 24
// windowed layers -> ln_post -> projector. `mel_in` is [T_pad, 128] bf16
// (feat-major host layout, time innermost); returns [output_dim, L] f32.
// When `enc_capture` is non-null the encoder's last hidden state is
// additionally read back (divergence dump).
ggml_tensor* build_fused(ggml_context* c, const Qwen3Model& m, ggml_tensor* mel_in,
                         const EncScratch& s, std::vector<float>* enc_capture,
                         StageStop* stop) {
    const auto& ec = m.config.encoder;
    const ModelLoader& ml = m.loader;
    const int64_t hidden = ec.hidden;
    const int64_t chunk = 2 * (int64_t) m.config.frontend.n_window;
    const int64_t T_pad = mel_in->ne[1];

    if (stage_wants(stop->name, "melin")) { stop->hit = true; return f32(c, mel_in); }
    // Chunked conv data [W=chunk, H=128, C=1, N=nc] from the time-major mel:
    // (t, m) flat t + T*m -> (t', chk, m) -> permute -> (t', m, chk).
    ggml_tensor* x = f32(c, mel_in);
    x = ggml_reshape_3d(c, x, chunk, s.nc, ec.n_mel);
    x = ggml_cont(c, ggml_permute(c, x, 0, 2, 1, 3));
    x = ggml_reshape_4d(c, x, chunk, ec.n_mel, 1, s.nc);

    ggml_tensor* g = conv_step(c, ml, "enc.conv1.weight", "enc.conv1.bias", x);
    if (stage_wants(stop->name, "c1")) { stop->hit = true; return f32(c, g); }
    g = conv_step(c, ml, "enc.conv2.weight", "enc.conv2.bias", g);
    if (stage_wants(stop->name, "c2")) { stop->hit = true; return f32(c, g); }
    g = conv_step(c, ml, "enc.conv3.weight", "enc.conv3.bias", g);
    if (stage_wants(stop->name, "c3")) { stop->hit = true; return f32(c, g); }
    // (t, f, ch, n) -> features (ch-major, f-inner) x positions (n-major,
    // t-inner): the stock conv_out.permute(0, 3, 1, 2).view layout.
    // ggml_permute is scatter-style (old dim j lands at position axis_j):
    // t->2, f->0, ch->1, n->3 gives (f, ch, t, n).
    ggml_tensor* z = ggml_cont(c, ggml_permute(c, g, 2, 0, 1, 3));
    const int64_t feat = ec.downsample_hidden * (ec.n_mel / 8);  // 480*16
    z = ggml_reshape_2d(c, z, feat, ec.max_pos_emb * s.nc);
    if (stage_wants(stop->name, "flat")) { stop->hit = true; return f32(c, z); }
    ggml_tensor* h = lib::lin(c, ml, z, "enc.out.weight");       // [hidden, 13*nc]
    if (stage_wants(stop->name, "lin")) { stop->hit = true; return f32(c, h); }
    // + positional table, broadcast per chunk ([hidden, 13, nc] +
    // [hidden, 13, 1]; bf16 in-place-add boundary).
    ggml_tensor* h3 = ggml_reshape_3d(c, h, hidden, ec.max_pos_emb, s.nc);
    ggml_tensor* pos = f32(c, weight(c, ml, "enc.pos_embed"));   // [hidden, 13]
    ggml_tensor* pos3 = ggml_reshape_3d(c, pos, hidden, ec.max_pos_emb, 1);
    h3 = bf16(c, ggml_add(c, f32(c, h3), pos3));
    h = ggml_reshape_2d(c, h3, hidden, ec.max_pos_emb * s.nc);
    if (stage_wants(stop->name, "pos")) { stop->hit = true; return f32(c, h); }
    // Gather the valid rows into the window-padded packed sequence. The pad
    // tail duplicates row 0 (values irrelevant — masked + trimmed later).
    // The gather runs on an F32 copy of the table: this ggml build's CPU
    // get_rows bf16 kernel writes f32 rows into the bf16 destination (a 2x
    // overwrite), and bf16 -> f32 -> gather -> bf16 is an exact round trip.
    int64_t ine[1] = {(int64_t) s.valid_indices.size()};
    ggml_tensor* idx = graph_input_tensor(c, GGML_TYPE_I32, 1, ine,
                                          s.valid_indices.data(),
                                          s.valid_indices.size() * sizeof(int32_t));
    ggml_tensor* body = bf16(c, ggml_get_rows(c, f32(c, h), idx));   // [hidden, P]
    if (stage_wants(stop->name, "pack")) { stop->hit = true; return f32(c, body); }
    for (uint32_t li = 0; li < ec.n_layers; ++li) {
        body = windowed_layer(c, m, (int) li, body, s, stop);
        if (stop->hit) return f32(c, body);
        {
            char lay[16];
            std::snprintf(lay, sizeof lay, "lay%u", li);
            if (stage_wants(stop->name, lay)) { stop->hit = true; return f32(c, body); }
        }
    }
    ggml_tensor* ln = lib::layer_norm_bf16(c, ml, body, "enc.ln_post",
                                           ec.layer_norm_eps);
    ggml_tensor* enc = ggml_view_2d(c, ln, hidden, s.L, ln->nb[1], 0);
    if (stage_wants(stop->name, "enc")) { stop->hit = true; return f32(c, enc); }
    if (enc_capture) capture_graph_output(f32(c, enc), enc_capture);
    // Projector: Linear + erf GELU + Linear (all biased).
    ggml_tensor* pj = lib::linear_bf16(c, ml, enc, "proj.linear_1", true);
    pj = lib::gelu_erf_bf16(c, pj);
    pj = lib::linear_bf16(c, ml, pj, "proj.linear_2", true);
    return f32(c, pj);
}

} // namespace

// ---------------------------------------------------------------------------
// Fused encode + project with a per-shape bounded-LRU ReplayGraph cache
// (GPU), one-shot run_graph otherwise (the granite encoder pattern).
// ---------------------------------------------------------------------------
namespace {
struct ShapeKey {
    int64_t T_pad, L;
    bool operator==(const ShapeKey& o) const { return T_pad == o.T_pad && L == o.L; }
};
struct ShapeKeyHash {
    size_t operator()(const ShapeKey& k) const noexcept {
        return (size_t) (k.T_pad * 1000003u ^ k.L);
    }
};
struct EncoderReplayEntry {
    int64_t T_pad = 0, L = 0;
    GraphInputPool pool;
    EncScratch scratch;  // host constants backing graph inputs; stable for the
                         // ReplayGraph's lifetime (never resized after build)
    std::unique_ptr<ReplayGraph> graph;
    ggml_bf16_t* mel_buf = nullptr;
    std::vector<float> enc_capture;
};
// Deliberately leaked pointer, mirroring granite's encoder cache (and
// qwen_decode's spec_states()): the ONLY teardown path is the decode-cache
// clearer registered at first use. This namespace-scope pointer is
// const-initialized at load, before any atexit call, so a destructor would
// run after the shutdown handler anyway; the leak keeps a single teardown
// route instead of racing one.
using EncoderCache = LruCache<ShapeKey, EncoderReplayEntry, ShapeKeyHash>;
} // namespace

size_t encoder_replay_cache_size(const Qwen3Model& model) {
    const auto* cache = model.loader.find_cache<EncoderCache>();
    return cache ? cache->size() : 0;}

bool encode_audio_and_project(const Qwen3Model& model, const MelFeatures& mel,
                              AudioEmbeds& out, std::string& err) {
    ensure_weights_realized(model.loader);
    const auto& ec = model.config.encoder;
    if (mel.n_mels != (int64_t) ec.n_mel || mel.n_frames <= 0 ||
        mel.valid_frames <= 0 || mel.valid_frames > mel.n_frames ||
        mel.data.size() != (size_t) mel.n_mels * mel.n_frames) {
        err = "invalid QWEN3 mel shape/data";
        return false;
    }
    const int64_t T = mel.valid_frames;
    EncScratch s = make_scratch(model.config, T);
    if (s.nc * 2 * (int64_t) model.config.frontend.n_window != mel.n_frames) {
        err = "QWEN3 mel frame count does not match the chunked layout";
        return false;
    }
    const char* dump_env = std::getenv("STARLING_QWEN3_DUMP_ENC");
    if (const char* ms = std::getenv("STARLING_QWEN3_DUMP_MELSTK")) {
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
    // LRU, where it would poison later normal transcriptions at the same
    // mel length (granite pullfrog review).
    if (!global_backend().is_gpu() || lib::debug_enabled("STARLING_QWEN3_DEBUG") ||
        std::getenv("STARLING_QWEN3_ONLY")) {
        std::vector<float> enc_out;
        StageStop stop{std::getenv("STARLING_QWEN3_ONLY")};
        bool ok = run_graph([&](ggml_context* c) -> ggml_tensor* {
            int64_t mne[2] = {mel.n_frames, mel.n_mels};
            ggml_tensor* mel_in = graph_input_tensor(c, GGML_TYPE_BF16, 2, mne,
                                                     mel.data.data(),
                                                     mel.data.size() * sizeof(mel.data[0]));
            return build_fused(c, model, mel_in, s, dump_env ? &enc_out : nullptr, &stop);
        }, out.data);
        if (!ok) { err = "QWEN3 encoder graph execution failed"; return false; }
        write_stage_dump(enc_out);
        if (stop.hit) {
            // Stage-only probe: the graph output IS the probed stage; dump it
            // host-side and stop the transcription here.
            write_stage_dump(out.data);
            err = "QWEN3 stage-only probe (STARLING_QWEN3_ONLY)";
            return false;
        }
        out.n_tokens = s.L;
        out.width = ec.output_dim;
        return true;
    }

    // GPU: captured per-shape graph; the mel is the only varying input.
    auto& encoder_cache = model.loader.cache<EncoderCache>();
    if (!encoder_cache) encoder_cache = std::make_unique<EncoderCache>(replay_cache_size());

    ShapeKey key{mel.n_frames, s.L};
    EncoderReplayEntry& e = *encoder_cache->get_or_init(key,
        [&](EncoderReplayEntry& entry) {
            entry.T_pad = mel.n_frames;
            entry.L = s.L;
            entry.scratch = make_scratch(model.config, T);
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
                    return build_fused(c, model, mel_in, entry.scratch,
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
        err = "QWEN3 fused encoder+projector replay failed";
        return false;
    }
    write_stage_dump(e.enc_capture);
    out.data = std::move(tmp);
    out.n_tokens = s.L;
    out.width = ec.output_dim;
    return true;
}

} // namespace starling::ggml::qwen3
