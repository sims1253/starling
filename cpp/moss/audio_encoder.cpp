#include "audio_encoder.hpp"
#include "adapter.hpp"
#include "runtime/backend.hpp"
#include "runtime/graph.hpp"
#include "runtime/graph_builder.hpp"
#include "runtime/lru_cache.hpp"
#include "lib/graph_helpers.hpp"
#include "ggml.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace starling::ggml::moss {
namespace {

// Shared graph-builder helpers (lib/graph_helpers.hpp); the audio encoder is
// bf16-oracle discipline.
using lib::weight;
using lib::bf16;
using lib::f32;
// nn.Linear in the BF16 oracle: the GEMM and bias constitute one operation and
// expose a BF16 tensor. ggml GEMM exposes F32, so round at that boundary.
ggml_tensor* linear(ggml_context* c, const ModelLoader& ml, ggml_tensor* x,
                    const std::string& n, bool bias) {
    return lib::linear_bf16(c, ml, x, n, bias);
}
ggml_tensor* conv2d_bf16(ggml_context* c, ggml_tensor* kernel, ggml_tensor* input) {
    // Use ggml's canonical Conv2d builder. Its im2col kernels do not accept a
    // BF16 destination on CUDA, so present the BF16 values as F32 and round the
    // convolution output immediately back to the oracle's BF16 boundary.
    return bf16(c, ggml_conv_2d(c, f32(c, kernel), f32(c, input),
                                2, 2, 1, 1, 1, 1));
}

ggml_tensor* exact_gelu(ggml_context* c, ggml_tensor* x) {
    // ggml_gelu is the tanh approximation. GELU_ERF is the required
    // approximate="none" path. Generic elementwise kernels are F32, then we
    // immediately restore the ATen BF16 output boundary.
    return lib::gelu_erf_bf16(c, x);
}
ggml_tensor* add_bf16(ggml_context* c, ggml_tensor* a, ggml_tensor* b) {
    return lib::addb(c, a, b);
}
ggml_tensor* layer_norm(ggml_context* c, const ModelLoader& ml, ggml_tensor* x,
                        const std::string& n, float eps) {
    // PyTorch ordinary LayerNorm: F32 reduction and affine, one final BF16
    // store. Do not use the fused NORM+MUL+ADD patch: retaining its F32 result
    // through the following GEMM violates the explicit cast policy.
    return lib::layer_norm_bf16(c, ml, x, n, eps);
}

bool debug_enabled() {
    return lib::debug_enabled("STARLING_MOSS_DEBUG");
}

// Per-call mel packing shape. (C, tail) fully determines every graph shape
// (P, M, A, W); the captured graph is keyed on (C, tail).
struct MelShape {
    int C = 0, tail = 0, P = 0, M = 0, A = 0;
};
MelShape mel_shape(int64_t T) {
    MelShape s;
    s.C = (int)((T + 99) / 100);
    s.tail = (int)(T % 100 ? T % 100 : 100);
    s.P = (s.C == 1) ? s.tail : 100;   // longest piece
    s.M = (int)audio_token_length(s.P);
    s.A = (int)audio_token_length(T);
    return s;
}

// Pack mel into the chunked/zero-packed layout the conv stack reads (ggml Conv2d
// input [P,128,1,C] bf16) and the subsample `valid` index vector ([A] i32).
// `chunks` must hold C*128*P bf16; `valid` must hold A int32. The chunk padding
// (last chunk's [tail,P) time slots) is zeroed so reused buffers (the captured
// graph's stable pool) stay correct across calls of the same shape.
void pack_mel_into(const MelFeatures& mel, const MelShape& s,
                   ggml_bf16_t* chunks, int32_t* valid) {
    const int64_t T = mel.n_frames;
    const int C = s.C, P = s.P, M = s.M;
    std::memset(chunks, 0, (size_t)C * 128 * P * sizeof(ggml_bf16_t));
    int off = 0;
    for (int ci = 0; ci < C; ++ci) {
        const int len = (ci == C - 1) ? s.tail : 100;
        const int ai = (int)audio_token_length(len);
        for (int f = 0; f < 128; ++f)
            for (int t = 0; t < len; ++t)
                chunks[((size_t)ci * 128 + f) * P + t] =
                    mel.data[(size_t)f * T + ci * 100 + t];
        for (int t = 0; t < ai; ++t) valid[off + t] = ci * M + t;
        off += ai;
    }
}

// The windowed attention for one encoder layer. B2 restructured the serial
// per-window loop into ONE batched attention over all full windows
// ([D, W, H, n_full_windows] batched mul_mat) plus a single tail-window pass
// when A % W != 0. Per-window math and output token order are unchanged: the
// full-window batch is bit-identical to concatenating the individual windows
// (ggml mul_mat reduces independently per batch index, so each window's GEMM is
// the same reduction in the same order), and the original's f32 round-trips at
// the concat boundaries are no-ops on the bf16-representable values. This cuts
// the node count from 32 layers x n_windows sub-graphs to 32 x 2.
// `q/k/v` are [d_model, A] bf16. Returns [d_model, A] bf16.
ggml_tensor* windowed_attention(ggml_context* ctx, const MelShape& s,
                                const EncoderConfig& ec,
                                ggml_tensor* q, ggml_tensor* k, ggml_tensor* v) {
    const int A = s.A, M = s.M;
    const int H = (int)ec.n_heads, D = (int)ec.head_dim;
    const int W = M * ((int)ec.n_window_infer / 100);
    const float scale = 1.0f / std::sqrt((float)D);
    const int n_full = (W > 0) ? (A / W) : 0;   // number of full-size windows
    const int tail_S = A - n_full * W;          // 0, or the trailing partial window

    // Batched attention over the n_full full windows (each size W).
    ggml_tensor* joined_full = nullptr;
    if (n_full > 0) {
        // [d_model, W*n_full] -> [D, H, W, n_full] -> [D, W, H, n_full].
        auto batch = [&](ggml_tensor* z) {
            ggml_tensor* vw = ggml_view_2d(ctx, z, ec.d_model, (int64_t)W * n_full,
                                           z->nb[1], 0);
            vw = ggml_reshape_4d(ctx, vw, D, H, W, n_full);
            return ggml_cont(ctx, ggml_permute(ctx, vw, 0, 2, 1, 3));
        };
        ggml_tensor* qw = batch(q), * kw = batch(k), * vw = batch(v);
        // BF16 QK GEMM result and BF16 scalar multiply boundaries, followed by
        // F32 softmax and BF16 probabilities (same boundaries as the per-window path).
        ggml_tensor* scores = bf16(ctx, ggml_mul_mat(ctx, kw, qw));        // [W,W,H,n_full]
        scores = bf16(ctx, ggml_scale(ctx, f32(ctx, scores), scale));
        ggml_tensor* prob = ggml_soft_max_ext(ctx, f32(ctx, scores), nullptr, 1.0f, 0.0f);
        prob = bf16(ctx, prob);
        ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, vw, 1, 0, 2, 3));  // [W,D,H,n_full]
        ggml_tensor* co = bf16(ctx, ggml_mul_mat(ctx, vt, prob));            // [D,W,H,n_full]
        co = ggml_cont(ctx, ggml_permute(ctx, co, 0, 2, 1, 3));              // [D,H,W,n_full]
        // [D,H,W,n_full] contiguous -> [d_model=D*H, W*n_full], token order
        // (window j, pos w) -> column w + j*W == concat(window_0..window_{n_full-1}).
        joined_full = ggml_reshape_2d(ctx, co, ec.d_model, (int64_t)W * n_full);
    }

    // Tail window (size tail_S): identical op sequence to one original per-window.
    ggml_tensor* joined_tail = nullptr;
    if (tail_S > 0) {
        const int begin = n_full * W;
        auto window = [&](ggml_tensor* z) {
            ggml_tensor* vw = ggml_view_2d(ctx, z, ec.d_model, tail_S, z->nb[1],
                                           (size_t)begin * z->nb[1]);
            vw = ggml_reshape_3d(ctx, vw, D, H, tail_S);
            return ggml_cont(ctx, ggml_permute(ctx, vw, 0, 2, 1, 3));   // [D,tail_S,H]
        };
        ggml_tensor* qw = window(q), * kw = window(k), * vw = window(v);
        ggml_tensor* scores = bf16(ctx, ggml_mul_mat(ctx, kw, qw));
        scores = bf16(ctx, ggml_scale(ctx, f32(ctx, scores), scale));
        ggml_tensor* prob = ggml_soft_max_ext(ctx, f32(ctx, scores), nullptr, 1.0f, 0.0f);
        prob = bf16(ctx, prob);
        ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, vw, 1, 0, 2, 3));  // [tail_S,D,H]
        ggml_tensor* co = bf16(ctx, ggml_mul_mat(ctx, vt, prob));            // [D,tail_S,H]
        co = ggml_cont(ctx, ggml_permute(ctx, co, 0, 2, 1, 3));
        joined_tail = ggml_reshape_2d(ctx, co, ec.d_model, tail_S);
    }

    if (joined_full && joined_tail)
        return bf16(ctx, ggml_concat(ctx, f32(ctx, joined_full), f32(ctx, joined_tail), 1));
    return joined_full ? joined_full : joined_tail;
}

// Build the encoder graph (conv stack -> 32 windowed-attention layers -> ln_post
// -> proj1/gelu/proj2). Returns the proj2 output (BF16) -- the encoder hidden
// state that the adapter consumes. `chunks_host`/`valid_host` back the two graph
// inputs (registered in this order -> input indices 0 and 1). Debug capture dst
// pointers are optional (nullptr = skip); only the one-shot path sets them.
ggml_tensor* build_encoder_body(ggml_context* ctx, const MossModel& model,
                                const MelShape& s,
                                const ggml_bf16_t* chunks_host,
                                const int32_t* valid_host, bool debug,
                                std::vector<float>* dbg_conv,
                                std::vector<float>* dbg_l0,
                                std::vector<float>* dbg_l31,
                                std::vector<float>* dbg_post) {
    const auto& ec = model.config.encoder;
    const ModelLoader& ml = model.loader;
    const int C = s.C, P = s.P, M = s.M, A = s.A;

    int64_t ine[4] = {P, 128, 1, C};
    ggml_tensor* x = graph_input_tensor(ctx, GGML_TYPE_BF16, 4, ine,
        chunks_host, (size_t)C * 128 * P * sizeof(ggml_bf16_t));
    int channels[3] = {480, 480, 480};
    for (int i = 0; i < 3; ++i) {
        const std::string n = "enc.conv" + std::to_string(i + 1);
        ggml_tensor* cw = weight(ctx, ml, n + ".weight");
        x = conv2d_bf16(ctx, cw, x);
        x = ggml_add(ctx, f32(ctx, x),
                     ggml_reshape_4d(ctx, f32(ctx, weight(ctx, ml, n + ".bias")),
                                     1, 1, channels[i], 1));
        x = exact_gelu(ctx, bf16(ctx, x));   // conv boundary, then exact GELU boundary
    }
    // [M,16,480,C] -> contiguous [16,480,M,C], flatten each time row.
    x = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3));
    x = ggml_reshape_2d(ctx, x, 16 * 480, (int64_t)M * C);
    x = linear(ctx, ml, x, "enc.conv_out", false);

    ggml_tensor* pet = weight(ctx, ml, "enc.positional_embedding");
    ggml_tensor* pe = ggml_view_2d(ctx, pet, ec.d_model, M, pet->nb[1], 0);
    x = add_bf16(ctx, x, bf16(ctx, pe));
    int64_t vne[1] = {A};
    ggml_tensor* vi = graph_input_tensor(ctx, GGML_TYPE_I32, 1, vne,
        valid_host, (size_t)A * sizeof(int32_t));
    x = bf16(ctx, ggml_get_rows(ctx, x, vi));
    if (debug && dbg_conv) capture_graph_output(f32(ctx, x), dbg_conv);

    for (int li = 0; li < (int)ec.n_layers; ++li) {
        const std::string pre = "enc.blk." + std::to_string(li) + ".";
        ggml_tensor* r = x;
        ggml_tensor* n = layer_norm(ctx, ml, x, pre + "attn_norm", ec.layer_norm_eps);
        ggml_tensor* q = linear(ctx, ml, n, pre + "attn.q", true);
        ggml_tensor* k = linear(ctx, ml, n, pre + "attn.k", true);
        ggml_tensor* v = linear(ctx, ml, n, pre + "attn.v", true);
        ggml_tensor* joined = windowed_attention(ctx, s, ec, q, k, v);
        ggml_tensor* a = linear(ctx, ml, joined, pre + "attn.o", true);
        x = add_bf16(ctx, r, a);
        r = x;
        n = layer_norm(ctx, ml, x, pre + "ffn_norm", ec.layer_norm_eps);
        ggml_tensor* h = linear(ctx, ml, n, pre + "ffn.fc1", true);
        h = exact_gelu(ctx, h);
        h = linear(ctx, ml, h, pre + "ffn.fc2", true);
        x = add_bf16(ctx, r, h);
        if (debug && dbg_l0 && li == 0) capture_graph_output(f32(ctx, x), dbg_l0);
        if (debug && dbg_l31 && li == 31) capture_graph_output(f32(ctx, x), dbg_l31);
    }
    x = layer_norm(ctx, ml, x, "enc.ln_post", ec.layer_norm_eps);
    if (debug && dbg_post) capture_graph_output(f32(ctx, x), dbg_post);
    x = linear(ctx, ml, x, "enc.proj1", true);
    x = exact_gelu(ctx, x);
    return linear(ctx, ml, x, "enc.proj2", true);   // BF16 proj2 output
}

// Append the adapter (gate/up SiLU-mul down) to the encoder body output. `x` is
// the BF16 proj2 output, which is bit-identical to the host f32->bf16 round-trip
// the standalone apply_adapter performs (those values are BF16-representable, so
// ggml_fp32_to_bf16 is the identity). Returns the F32 adapter output.
ggml_tensor* build_adapter(ggml_context* ctx, const MossModel& model, ggml_tensor* x) {
    const ModelLoader& ml = model.loader;
    auto lin = [&](const char* n, ggml_tensor* z) {
        return ggml_cast(ctx, ggml_mul_mat(ctx, clone_weight(ctx, ml, n), z), GGML_TYPE_BF16);
    };
    ggml_tensor* g = lin("adapter.gate.weight", x);
    ggml_tensor* u = lin("adapter.up.weight", x);
    // ATen boundaries: BF16 gate -> F32 SiLU -> BF16, then one BF16 multiply
    // result before the down projection. Generic ggml elementwise is F32-only.
    ggml_tensor* a = ggml_cast(ctx, ggml_silu(ctx, ggml_cast(ctx, g, GGML_TYPE_F32)), GGML_TYPE_BF16);
    ggml_tensor* z = ggml_cast(ctx, ggml_mul(ctx, ggml_cast(ctx, a, GGML_TYPE_F32),
                                             ggml_cast(ctx, u, GGML_TYPE_F32)), GGML_TYPE_BF16);
    return ggml_cast(ctx, lin("adapter.down.weight", z), GGML_TYPE_F32);
}

} // namespace

bool encode_audio(const MossModel& model, const MelFeatures& mel,
                  AudioEncoding& out, std::string& err) {
    // Weight realization may call global_backend(); do it before run_graph(),
    // which holds the non-recursive global backend mutex while building.
    // Realizing lazily from clone_weight() inside the build lambda deadlocks.
    ensure_weights_realized(model.loader);
    const auto& ec = model.config.encoder;
    if (mel.n_mels != 128 || mel.n_frames <= 0 ||
        mel.data.size() != (size_t)mel.n_mels * mel.n_frames) {
        err = "invalid MOSS mel shape/data"; return false;
    }
    // Unit assertions demanded by the contract; these also protect the subtle
    // remainder-zero behavior from future simplification.
    assert(audio_token_length(743) == 97);
    assert(audio_token_length(2230) == 290);
    assert(audio_token_length(7435) == 967);

    const MelShape s = mel_shape(mel.n_frames);
    std::vector<ggml_bf16_t> chunks((size_t)s.C * 128 * s.P, ggml_fp32_to_bf16(0));
    std::vector<int32_t> valid((size_t)s.A);
    pack_mel_into(mel, s, chunks.data(), valid.data());
    if ((int)valid.size() != s.A) { err = "MOSS packed length invariant failed"; return false; }

    std::vector<float> dbg_conv, dbg_l0, dbg_l31, dbg_post;
    const bool debug = debug_enabled();
    bool ok = run_graph([&](ggml_context* ctx) -> ggml_tensor* {
        ggml_tensor* body = build_encoder_body(ctx, model, s, chunks.data(),
                                               valid.data(), debug,
                                               &dbg_conv, &dbg_l0, &dbg_l31, &dbg_post);
        return f32(ctx, body);
    }, out.data);
    if (!ok) { err = "MOSS encoder graph execution failed"; return false; }
    out.n_tokens = s.A; out.width = ec.output_dim;
    if (debug) {
        auto report = [](const char* n, const std::vector<float>& v) {
            double mx = 0; for (float x : v) mx = std::max(mx, std::abs((double)x));
            std::fprintf(stderr, "MOSS_DEBUG %-12s elements=%zu max_abs_value=%.9g (stage golden unavailable)\n",
                         n, v.size(), mx); };
        report("post-conv", dbg_conv); report("post-layer0", dbg_l0);
        report("post-layer31", dbg_l31); report("post-ln_post", dbg_post);
    }
    return true;
}

// ---------------------------------------------------------------------------
// B1: persistent per-shape ReplayGraph fusing encode_audio + apply_adapter.
//
// The adapter is 3 linears + SiLU; the f32 host round-trip of the encoder
// output between them (apply_adapter's ggml_fp32_to_bf16 of the whole tensor)
// is pure overhead and is elided by fusing both into ONE captured graph. The
// cast boundary is preserved exactly inside the graph: the encoder proj2 output
// is BF16-representable, so in-graph it IS the host f32->bf16 conversion. The
// graph is cached per (C, tail) mel shape (which fully determines every graph
// shape) in a bounded LRU cache (runtime/lru_cache.hpp). The bound is
// STARLING_REPLAY_CACHE_SIZE (default 16): real audio spans a near-continuous
// length distribution, so an unbounded map would permanently pin one private
// gallocr + captured CUDA graph per distinct shape until exit (the Wave H OOM
// bug); LRU evicts the least-recently-used shape (freeing its device buffer) at
// capacity. The parakeet encoder ReplayCache (cpp/parakeet/encoder.{hpp,cpp})
// uses the same bounded helper. Freed with the owning model.
// Capture is GPU-only; CPU and the STARLING_MOSS_DEBUG diagnostic path keep the
// one-shot encode_audio + apply_adapter pair.
// ---------------------------------------------------------------------------

struct EncoderReplayEntry {
    MelShape shape;
    GraphInputPool pool;
    std::unique_ptr<ReplayGraph> graph;
    ggml_bf16_t* chunks_buf = nullptr;  // stable pool backing for input #0
    int32_t* valid_buf = nullptr;       // stable pool backing for input #1
};

namespace {
struct ShapeKey { int C, tail;
    bool operator==(const ShapeKey& o) const { return C == o.C && tail == o.tail; } };
struct ShapeKeyHash { size_t operator()(const ShapeKey& k) const noexcept {
    return (size_t)k.C * 1009u + (size_t)k.tail; } };

// Bounded LRU (runtime/lru_cache.hpp). Lazily created on first use so the
// STARLING_REPLAY_CACHE_SIZE env var is read at first encode, not at process
// start. reset() (the decode-cache clearer) frees every cached ReplayGraph
// while the backend is still alive.
using EncoderCache = LruCache<ShapeKey, EncoderReplayEntry, ShapeKeyHash>;
} // namespace

// Current number of cached encoder graphs (diagnostic / regression-test hook).
size_t encoder_replay_cache_size(const MossModel& model) {
    const auto* cache = model.loader.find_cache<EncoderCache>();
    return cache ? cache->size() : 0;
}

bool encode_audio_and_adapt(const MossModel& model, const MelFeatures& mel,
                            AudioEncoding& out, std::string& err) {
    ensure_weights_realized(model.loader);
    // CPU backend + debug diagnostic path keep the one-shot pair (capture is
    // GPU-only, like everywhere else).
    if (!global_backend().is_gpu() || debug_enabled()) {
        AudioEncoding enc;
        if (!encode_audio(model, mel, enc, err)) return false;
        return apply_adapter(model, enc, out, err);
    }
    if (mel.n_mels != 128 || mel.n_frames <= 0 ||
        mel.data.size() != (size_t)mel.n_mels * mel.n_frames) {
        err = "invalid MOSS mel shape/data"; return false;
    }

    auto& encoder_cache = model.loader.cache<EncoderCache>();
    if (!encoder_cache) encoder_cache = std::make_unique<EncoderCache>(replay_cache_size());

    const MelShape s = mel_shape(mel.n_frames);
    ShapeKey key{s.C, s.tail};
    // get_or_init places the entry in the map (stable address) first, then fills
    // it: the ReplayGraph build lambda captures the stable pool pointers. On a
    // miss at capacity the LRU shape is evicted (its ReplayGraph freed) before
    // this entry is inserted.
    EncoderReplayEntry& e = *encoder_cache->get_or_init(key,
        [&](EncoderReplayEntry& entry) {
            entry.shape = s;
            entry.chunks_buf = reinterpret_cast<ggml_bf16_t*>(
                entry.pool.alloc_bytes((size_t)s.C * 128 * s.P * sizeof(ggml_bf16_t)));
            entry.valid_buf = entry.pool.alloc_i32(s.A);
            pack_mel_into(mel, s, entry.chunks_buf, entry.valid_buf);  // valid data for the build
            entry.graph = std::make_unique<ReplayGraph>(global_backend(),
                [&](ggml_context* ctx) -> ggml_tensor* {
                    ggml_tensor* body = build_encoder_body(ctx, model, s,
                                                           entry.chunks_buf, entry.valid_buf,
                                                           /*debug=*/false,
                                                           nullptr, nullptr, nullptr, nullptr);
                    return build_adapter(ctx, model, body);
                });
        });
    // Refresh the two inputs in their stable pool buffers, then re-upload
    // (ReplayGraph does not promise input persistence across replays).
    pack_mel_into(mel, e.shape, e.chunks_buf, e.valid_buf);
    for (size_t i = 0; i < e.graph->n_inputs(); ++i)
        e.graph->set_input(i, e.graph->input_host(i), e.graph->input_nbytes(i));

    std::vector<float> tmp;
    if (!e.graph->compute(tmp)) {
        err = "MOSS fused encoder+adapter replay failed"; return false;
    }
    out.data = std::move(tmp);
    out.n_tokens = e.shape.A;
    out.width = model.config.adapter_output;
    return true;
}
} // namespace starling::ggml::moss
