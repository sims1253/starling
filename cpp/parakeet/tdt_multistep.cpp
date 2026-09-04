// tdt_multistep.cpp — K-step CUDA-graph-captured TDT greedy decode (Wave C).
//
// Starling port of parakeet.cpp's tdt_multistep.cpp (branch dev). Captures K
// consecutive (pred.step -> joint -> argmax -> TDT frame-advance) decode steps
// into ONE ReplayGraph and syncs to host ONCE per K steps instead of once per
// step. The serial loop is launch/sync-bound (~150-200us host stall per forced
// sync on a mostly-idle GPU); collapsing K steps into one replay + one sync cuts
// the per-replay host-sync tax that dominates decode wall time.
//
// Why this is sound (the in-graph chaining) -- every per-step state tensor is a
// graph-internal node advanced in-graph, so step j+1 of ONE replay reads step
// j's output tensors directly (no host round-trip between captured steps):
//   * last_token  -- get_rows(embed, last_tok) ON THE DEVICE.
//   * frame_idx   -- advanced in-graph by the TDT duration rule (blank && dur==0
//     -> dur=1); the enc_proj row for the (clamped) frame is gathered in-graph.
//   * LSTM h/c     -- advanced in-graph; FROZEN on a blank step (matching
//     tdt_greedy's "commit only on non-blank") via where(advance, h_new, h_old).
//   * cc           -- the frozen decoder-output cache (blank-skip reuse).
//
// STARLING ADAPTATION: the reference drops blanks (returns the non-blank hyp).
// Starling's golden id stream INCLUDES blanks, so this port emits EVERY token
// (the host scatter always appends the step's token, including blanks). This is
// byte-exact with the serial loop on the short/medium/long goldens: the one
// divergent case (blank+dur0 -> the serial loop emits max_symbols blanks then
// advances 1; the graph forces dur=1 and emits 1 blank) does NOT occur on the
// goldens (max blank run is 2). See the header + plans/wave-c-parakeet-gpu-decode.md.
//
// Byte-exactness vs the serial loop: the in-graph blank-skip freeze is provably
// equivalent to tdt_greedy's "discard out_state on blank, feed last_emitted"
// rule. See parakeet.cpp's tdt_multistep.cpp design comment for the full argument.
//
// K is T-aware: 16 for T<=512 (lowest per-replay latency), 64 for T>512 (halves
// the replay count on long). Both byte-exact. Override with STARLING_GGML_TDT_KSTEP.

#include "tdt_multistep.hpp"

#include "runtime/backend.hpp"   // ReplayGraph, graph_input_tensor, clone_weight,
                                // capture_graph_output, add_graph_root
#include "runtime/graph.hpp"     // global_backend, register_decode_cache_clearer

#include "ggml.h"
#include "ggml-backend.h"        // ggml_backend_alloc_ctx_tensors / tensor_set (DecodeDevCache)

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace starling::ggml::parakeet {
namespace {

// Read STARLING_GGML_TDT_KSTEP (>=1). Default is T-dependent (see below). K is
// the number of TDT decode steps captured into one CUDA-graph replay (one host<->
// device sync per K steps instead of per step). ggml's CUDA-graph capture has a
// topology-dependent defect that corrupts the in-graph state chain for certain
// node counts, so NOT every K is byte-exact. The defaults below are the proven
// byte-exact values from the Wave D K-sweep (plans/wave-d-parakeet-long-closure.md):
//   - K=16: byte-exact on short (T_enc=93) / medium (279).
//   - K=96: byte-exact + fastest on long (T=930); 5 replays w/ only ~7 wasted
//     steps vs K=64's 8 replays / ~39 wasted. (K=48 and K=128 are topology-
//     defect-inexact on long; K=32 is exact but slower.)
int tdt_kstep_config(int T) {
    if (const char* e = std::getenv("STARLING_GGML_TDT_KSTEP")) {
        int v = std::atoi(e);
        if (v >= 1) return v;
    }
    return (T <= 512) ? 16 : 96;
}

bool tdt_kstep_debug() {
    const char* e = std::getenv("STARLING_GGML_TDT_KSTEP_DEBUG");
    return e && e[0] == '1';
}

} // namespace

// ---------------------------------------------------------------------------
// DecodeDevCache: persistent device-resident K-step decode state (Wave D, D1).
// Mirrors the Wave A DeviceCache pattern in cpp/moss/llm.cpp: fixed backend-
// buffer tensors referenced as graph leaves, written in-graph at the END of
// each K-step replay (via view + cpy registered as add_graph_root side effects)
// and read in-graph at the START of the next replay. Host no longer round-trips
// the LSTM h/c + cc + frame + last_token between replays; it reads back ONLY
// the emitted-id ring (tokens + post-step frames) for the termination check.
//
// Shape is model-level (Hp, L), independent of (T, K), so ONE process-global
// cache backs every (T, K) graph. Single-threaded decode, so no aliasing. Each
// utterance RE-SEEDS the buffers before its first replay (the eager step-0
// state). Freed by a registered decode-cache-clearer before backend teardown.
// ---------------------------------------------------------------------------
struct DecodeDevCache {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor* frame = nullptr;      // i32 [1]  (matches the step-0 cast topology)
    ggml_tensor* last_tok = nullptr;   // i32 [1]
    ggml_tensor* cc = nullptr;         // f32 [Hp]
    std::vector<ggml_tensor*> h;       // [L] f32 [Hp]
    std::vector<ggml_tensor*> c;       // [L] f32 [Hp]
    int Hp = 0, L = 0;

    bool init(int Hp_, int L_, ggml_backend_t backend, std::string& e);
    ~DecodeDevCache() {
        if (shutting_down()) return;  // driver gone -> leak (fine at exit)
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
    }
};

bool DecodeDevCache::init(int Hp_, int L_, ggml_backend_t backend, std::string& e) {
    Hp = Hp_; L = L_;
    const size_t n_tensors = (size_t)(2 * L + 3);
    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (n_tensors + 8),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx = ggml_init(params);
    if (!ctx) { e = "DecodeDevCache: ggml_init failed"; return false; }

    frame    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    last_tok = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    cc       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, Hp);
    h.assign((size_t)L, nullptr);
    c.assign((size_t)L, nullptr);
    for (int l = 0; l < L; ++l) {
        h[(size_t)l] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, Hp);
        c[(size_t)l] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, Hp);
    }
    buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { e = "DecodeDevCache: backend alloc failed"; return false; }
    return true;
}

// Process-global decode device cache (one per process; Hp/L are model-level
// constants, so a single cache backs every (T, K) K-step graph).
std::unique_ptr<DecodeDevCache> g_decode_dev_cache;
std::once_flag g_decode_dev_cache_once;
void ensure_decode_dev_cache_clearer_registered() {
    std::call_once(g_decode_dev_cache_once, [] {
        // Register BEFORE the K-step graph clearer's effect matters: both just
        // drop process-globals before the backend resets. Order between them is
        // safe (both leak-at-exit under shutting_down(); no cross-deref).
        register_decode_cache_clearer([] { g_decode_dev_cache.reset(); });
    });
}
// Lazily create the process-global decode device cache. Hp/L come from the
// prediction net (model-level). Returns nullptr on alloc failure.
DecodeDevCache* get_decode_dev_cache(int Hp, int L, std::string& e) {
    ensure_decode_dev_cache_clearer_registered();
    if (g_decode_dev_cache) return g_decode_dev_cache.get();
    g_decode_dev_cache = std::unique_ptr<DecodeDevCache>(new DecodeDevCache());
    if (!g_decode_dev_cache->init(Hp, L, global_backend().handle(), e)) {
        g_decode_dev_cache.reset();
        return nullptr;
    }
    return g_decode_dev_cache.get();
}

// ---------------------------------------------------------------------------
// K-step decode graph: one ReplayGraph capturing K decode steps, with the
// per-step state (frame_idx, last_token, h/c, cc) chained in-graph. One graph
// per (T, K); cached process-globally and reused across utterances of the same
// encoder length (building the graph = ggml build + gallocr alloc + CUDA-graph
// warmup, expensive enough to amortise).
// ---------------------------------------------------------------------------
struct KStepGraph {
    KStepGraph() = default;
    KStepGraph(const KStepGraph&) = delete;
    KStepGraph& operator=(const KStepGraph&) = delete;

    int K = 0, T = 0;
    int Hj = 0, Hp = 0, L = 0;
    int token_count = 0, num_dur = 0, blank_id = 0;

    // Wave D (D1): the per-replay chained state (frame, last_token, cc, h/c) no
    // longer round-trips through host inputs; it lives in this persistent
    // device cache, read at step 0 of each replay and written back in-graph at
    // the end (add_graph_root side-effect cpys). One cache backs every (T, K)
    // graph; the caller seeds it once per utterance before the first replay.
    //
    // device_resident gates the path: true for the proven-byte-exact small-K
    // graphs (K<=16, i.e. short/medium) where the in-graph write-back is stable;
    // false for K>16 (long), where the write-back cpy into persistent device
    // buffers trips ggml's CUDA-graph topology defect (illegal memory access)
    // -- that path keeps the original host round-trip (state inputs + captures).
    // See tdt_multistep.cpp's D1 notes + plans/wave-d-parakeet-long-closure.md.
    bool device_resident = false;
    DecodeDevCache* dc = nullptr;

    std::unique_ptr<ReplayGraph> rg;

    // Input-tensor indices (registration order from the build). enc_proj is set
    // once per utterance; the constant tables are set once per utterance too.
    // When device_resident, the chained state is NOT an input (it flows through
    // dc's device leaves); otherwise the state inputs (frame/last_tok/cc/h/c)
    // precede the constants.
    size_t in_enc_proj = 0;       // [Hj, T] f32
    size_t in_adv_mask = 0;       // f32 [vocab_p1]  (1.0 except 0.0 at blank)
    size_t in_dur_tbl  = 0;       // i32 [num_dur]
    size_t in_zero_dur = 0;       // f32 [num_dur]   (1.0 at idx 0 else 0.0)
    size_t in_one      = 0;       // f32 [1] = {1.0}
    // State inputs (baseline path only; device_resident=false).
    size_t in_frame    = 0;       // i32 [1]
    size_t in_last_tok = 0;       // i32 [1]
    size_t in_cc       = 0;       // f32 [Hp]
    std::vector<size_t> in_h;     // [L] f32 [Hp]
    std::vector<size_t> in_c;     // [L] f32 [Hp]

    // Constant-table host backing (filled once).
    std::vector<float>   adv_mask_table;    // [vocab_p1]
    std::vector<int32_t> dur_table_int;     // [num_dur]
    std::vector<float>   zero_dur_table;    // [num_dur]
    std::vector<float>   one_scalar;        // [1] = {1.0}
    // Per-replay state host backing (baseline path only).
    std::vector<int32_t> host_frame;        // [1]
    std::vector<int32_t> host_last_tok;     // [1]
    std::vector<float>   host_cc;           // [Hp]
    std::vector<std::vector<float>> host_h; // [L][Hp]
    std::vector<std::vector<float>> host_c; // [L][Hp]

    // Capture destinations (read back each replay via the single sync).
    // tokens[K]/frames[K] are i32; the capture reads nelements*sizeof(float) ==
    // nelements*sizeof(int32_t) bytes into a float vector (byte-correct, same
    // trick as the fused step's duration-argmax capture). These are the host
    // reads for the termination check + emitted-id stream in BOTH paths.
    std::vector<float> cap_tokens_f;        // [K]  (i32 reinterpreted)
    std::vector<float> cap_frames_f;        // [K]  (i32 reinterpreted)
    // Final-state captures (baseline path only; device_resident=false). Seed the
    // next replay when the utterance is not yet finished.
    std::vector<std::vector<float>> cap_h;  // [L][Hp]
    std::vector<std::vector<float>> cap_c;  // [L][Hp]
    std::vector<float> cap_cc_f;            // [Hp]
    // Frame-advance echo captures (dur_idx / dur_final / one / dur_i32 of the
    // LAST captured step). RETAINED DELIBERATELY: they perturb the gallocr
    // allocation enough that removing them previously destabilised the captured
    // graph's buffer layout (a symptom of the ggml CUDA-graph topology defect
    // documented at tdt_kstep_config()). 4 single-float captures -- negligible.
    std::vector<float> cap_dbg_dur_idx_f;   // [1] (i32 reinterpreted, last step)
    std::vector<float> cap_dbg_dur_final_f; // [1] (f32, last step)
    std::vector<float> cap_dbg_one_f;       // [1] (echo of one_t constant)
    std::vector<float> cap_dbg_dur_i32_f;   // [1] (echo of dur_i32, i32 reinterpreted)
};

// Process-global cache keyed on (T, K).
namespace {
struct KKey { int T, K; bool operator==(const KKey& o) const { return T==o.T && K==o.K; } };
struct KKeyHash { size_t operator()(const KKey& k) const noexcept {
    return (size_t)k.T * 257u + (size_t)k.K; } };
std::unordered_map<KKey, std::unique_ptr<KStepGraph>, KKeyHash> g_kstep_cache;

// Register the cache clearer exactly once (after main starts, so the clearer
// vector in graph.cpp is already constructed). shutdown_backend() calls it
// BEFORE resetting the Backend, freeing the cached graphs' device buffers +
// captured CUDA graphs while the driver is alive.
std::once_flag g_register_clearer_once;
void ensure_clearer_registered() {
    std::call_once(g_register_clearer_once, [] {
        register_decode_cache_clearer([] { g_kstep_cache.clear(); });
    });
}
} // namespace

// Free every cached K-step graph (its ReplayGraph owns a device buffer + a
// captured CUDA graph). Driven by shutdown_backend() via the registered clearer.
void clear_kstep_cache() {
    g_kstep_cache.clear();
}

// where(m, A, B) = m*A + (1-m)*B for scalar f32 mask m (f32 [1]) and equal-shape
// f32 operands A,B. ggml has no where/comparison op, so the mask is gathered
// from a precomputed table (adv_mask_table) and the select is expressed with
// repeat + arithmetic (ggml mul/add do not auto-broadcast a scalar).
static ggml_tensor* where_scalar(ggml_context* ctx, ggml_tensor* mask1,
                                 ggml_tensor* one1, ggml_tensor* A,
                                 ggml_tensor* B) {
    ggml_tensor* maskA = ggml_repeat(ctx, mask1, A);     // [N]
    ggml_tensor* mA    = ggml_mul(ctx, maskA, A);
    ggml_tensor* nmask = ggml_sub(ctx, one1, mask1);     // [1]
    ggml_tensor* nmaskB= ggml_repeat(ctx, nmask, B);     // [N]
    ggml_tensor* nmB   = ggml_mul(ctx, nmaskB, B);
    return ggml_add(ctx, mA, nmB);
}
// Scalar where (mask + both operands are f32 [1]).
static ggml_tensor* where_scalar1(ggml_context* ctx, ggml_tensor* mask1,
                                  ggml_tensor* one1, ggml_tensor* A,
                                  ggml_tensor* B) {
    ggml_tensor* mA   = ggml_mul(ctx, mask1, A);
    ggml_tensor* nmask= ggml_sub(ctx, one1, mask1);
    ggml_tensor* nmB  = ggml_mul(ctx, nmask, B);
    return ggml_add(ctx, mA, nmB);
}

// Build (or fetch) the K-step graph for (T, K). Returns a ready KStepGraph with
// its ReplayGraph allocated + the input/capture indices recorded, or nullptr on
// failure.
static KStepGraph* get_or_build_kstep(const PredictionNet& pred, const Joint& joint,
                                      int T, int K, int blank_id,
                                      const std::vector<int32_t>& durations,
                                      DecodeDevCache* dc,
                                      bool device_resident) {
    ensure_clearer_registered();

    KKey key{ T, K };
    auto it = g_kstep_cache.find(key);
    if (it != g_kstep_cache.end()) return it->second.get();

    const int Hj = joint.joint_hidden();
    const int Hp = pred.hidden_size();
    const int L  = pred.num_layers();
    const int token_count = joint.vocab_size() + 1;
    const int num_dur = (int)durations.size();
    const int vocab_p1 = pred.vocab_p1();
    const ModelLoader& pml = pred.model_loader();
    const ModelLoader& jml = joint.model_loader();

    auto kg = std::unique_ptr<KStepGraph>(new KStepGraph());
    kg->K = K; kg->T = T; kg->Hj = Hj; kg->Hp = Hp; kg->L = L;
    kg->token_count = token_count; kg->num_dur = num_dur; kg->blank_id = blank_id;
    kg->device_resident = device_resident;
    kg->dc = dc;   // persistent device-resident state (D1); used iff device_resident

    // Constant-table host backing (filled once, set once per utterance -- tiny).
    kg->adv_mask_table.assign((size_t)vocab_p1, 1.0f);
    kg->adv_mask_table[blank_id] = 0.0f;        // advance = (token != blank)
    kg->dur_table_int = durations;              // verbatim [num_dur] i32
    kg->zero_dur_table.assign((size_t)num_dur, 0.0f);
    if (num_dur > 0) kg->zero_dur_table[0] = 1.0f;  // 1.0 where dur==0
    kg->one_scalar.assign(1, 1.0f);
    // Ring captures (both paths) + debug echoes (load-bearing for the gallocr
    // layout, see cap_dbg_* comment in the struct).
    kg->cap_tokens_f.assign((size_t)K, 0.0f);
    kg->cap_frames_f.assign((size_t)K, 0.0f);
    kg->cap_dbg_dur_idx_f.assign(1, 0.0f);
    kg->cap_dbg_dur_final_f.assign(1, 0.0f);
    kg->cap_dbg_one_f.assign(1, 0.0f);
    kg->cap_dbg_dur_i32_f.assign(1, 0.0f);
    // Baseline path only: per-replay state host backing + final-state captures.
    if (!device_resident) {
        kg->host_frame.assign(1, 0);
        kg->host_last_tok.assign(1, 0);
        kg->host_cc.assign((size_t)Hp, 0.0f);
        kg->host_h.assign((size_t)L, std::vector<float>((size_t)Hp, 0.0f));
        kg->host_c.assign((size_t)L, std::vector<float>((size_t)Hp, 0.0f));
        kg->cap_h.assign((size_t)L, std::vector<float>((size_t)Hp, 0.0f));
        kg->cap_c.assign((size_t)L, std::vector<float>((size_t)Hp, 0.0f));
        kg->cap_cc_f.assign((size_t)Hp, 0.0f);
    }

    KStepGraph* raw = kg.get();
    // Sizes used in multiple places.
    int64_t HjT_ne[2]  = { Hj, T };            // enc_proj [Hj, T]
    int64_t vp1_ne[2]  = { 1, vocab_p1 };      // adv_mask as [1, vocab_p1] for gather
    int64_t ndur_ne[2] = { 1, num_dur };

    raw->rg = std::unique_ptr<ReplayGraph>(new ReplayGraph(
        global_backend(),
        [&](ggml_context* ctx) -> ggml_tensor* {
            // ---- Inputs (registration order is the set_input index). ----
            // 0: enc_proj [Hj, T] (set once per utterance). The host backing
            // pointer here is a placeholder -- ReplayGraph does NOT push it in
            // the ctor; the real data arrives via set_input before each
            // utterance's first compute. one_scalar is a stable valid pointer.
            int64_t one1[1] = {1};
            int64_t Hp_n[1] = { Hp };
            ggml_tensor* enc_proj_t = graph_input_tensor(
                ctx, GGML_TYPE_F32, 2, HjT_ne, raw->one_scalar.data(), (size_t)T*Hj*sizeof(float));
            raw->in_enc_proj = 0;
            size_t idx = 1;

            // Running in-graph state tensors (read at step 0, advanced in-graph).
            // device_resident (D1): seeded from the persistent device-cache
            // leaves (external buffers, like loader weights); written back at
            // the end. Otherwise: per-replay host-backed INPUT tensors.
            ggml_tensor* frame;
            ggml_tensor* last_tok;
            ggml_tensor* cc_cur;
            std::vector<ggml_tensor*> h_cur(L), c_cur(L);
            if (raw->device_resident) {
                frame    = raw->dc->frame;       // i32 [1]
                last_tok = raw->dc->last_tok;    // i32 [1]
                cc_cur   = raw->dc->cc;          // f32 [Hp]
                for (int l = 0; l < L; ++l) { h_cur[(size_t)l] = raw->dc->h[(size_t)l];
                                             c_cur[(size_t)l] = raw->dc->c[(size_t)l]; }
            } else {
                ggml_tensor* frame_in = graph_input_tensor(
                    ctx, GGML_TYPE_I32, 1, one1, raw->host_frame.data(), sizeof(int32_t));
                raw->in_frame = idx++;
                ggml_tensor* last_tok_in = graph_input_tensor(
                    ctx, GGML_TYPE_I32, 1, one1, raw->host_last_tok.data(), sizeof(int32_t));
                raw->in_last_tok = idx++;
                ggml_tensor* cc_in = graph_input_tensor(
                    ctx, GGML_TYPE_F32, 1, Hp_n, raw->host_cc.data(), (size_t)Hp*sizeof(float));
                raw->in_cc = idx++;
                raw->in_h.assign((size_t)L, 0);
                raw->in_c.assign((size_t)L, 0);
                std::vector<ggml_tensor*> h_in(L), c_in(L);
                for (int l = 0; l < L; ++l) {
                    h_in[(size_t)l] = graph_input_tensor(
                        ctx, GGML_TYPE_F32, 1, Hp_n, raw->host_h[(size_t)l].data(), (size_t)Hp*sizeof(float));
                    raw->in_h[(size_t)l] = idx++;
                    c_in[(size_t)l] = graph_input_tensor(
                        ctx, GGML_TYPE_F32, 1, Hp_n, raw->host_c[(size_t)l].data(), (size_t)Hp*sizeof(float));
                    raw->in_c[(size_t)l] = idx++;
                }
                frame = frame_in; last_tok = last_tok_in; cc_cur = cc_in;
                h_cur = h_in; c_cur = c_in;
            }

            // Constant tables (registered after the state inputs in both paths;
            // set once per utterance).
            ggml_tensor* adv_mask = graph_input_tensor(
                ctx, GGML_TYPE_F32, 2, vp1_ne, raw->adv_mask_table.data(),
                (size_t)vocab_p1*sizeof(float));
            raw->in_adv_mask = idx++;
            ggml_tensor* dur_tbl = graph_input_tensor(
                ctx, GGML_TYPE_I32, 2, ndur_ne, raw->dur_table_int.data(),
                (size_t)num_dur*sizeof(int32_t));
            raw->in_dur_tbl = idx++;
            ggml_tensor* zero_dur = graph_input_tensor(
                ctx, GGML_TYPE_F32, 2, ndur_ne, raw->zero_dur_table.data(),
                (size_t)num_dur*sizeof(float));
            raw->in_zero_dur = idx++;
            ggml_tensor* one_t = graph_input_tensor(
                ctx, GGML_TYPE_F32, 1, one1, raw->one_scalar.data(), sizeof(float));
            raw->in_one = idx++;
            // The embedding table + LSTM/joint weights: zero-copy loader leaves.
            ggml_tensor* embed_w = clone_weight(ctx, pml, "decoder.prediction.embed.weight");

            // Ring: accumulate each step's (token, post-step frame) so the host
            // can read all K at once. Built via concat at the end.
            std::vector<ggml_tensor*> tok_nodes;
            std::vector<ggml_tensor*> frame_nodes;
            tok_nodes.reserve((size_t)K);
            frame_nodes.reserve((size_t)K);

            // DEBUG: track the last step's dur_idx / dur_final for capture.
            ggml_tensor* dbg_dur_idx = nullptr;
            ggml_tensor* dbg_dur_final = nullptr;
            ggml_tensor* dbg_one = nullptr;
            ggml_tensor* dbg_dur_i32 = nullptr;

            for (int j = 0; j < K; ++j) {
                // --- Prediction LSTM (mirrors step_fused_argmax's LSTM) ---
                // Layer-0 input: embedding of last_tok (ON DEVICE gather).
                //   get_rows(embed_w[Hp, vocab_p1], last_tok[i32,1]) -> [Hp, 1].
                //   cont_1d -> [Hp].
                ggml_tensor* emb_row = ggml_cont_1d(
                    ctx, ggml_get_rows(ctx, embed_w, last_tok), Hp);
                ggml_tensor* layer_in = emb_row;
                ggml_tensor* g_proj = nullptr;     // top-layer h' == decoder output g
                std::vector<ggml_tensor*> h_new(L), c_new(L);
                for (int l = 0; l < L; ++l) {
                    const std::string s = "_l" + std::to_string(l);
                    ggml_tensor* Wih = clone_weight(ctx, pml,
                        ("decoder.prediction.dec_rnn.lstm.weight_ih" + s).c_str());
                    ggml_tensor* Whh = clone_weight(ctx, pml,
                        ("decoder.prediction.dec_rnn.lstm.weight_hh" + s).c_str());
                    ggml_tensor* bih = clone_weight(ctx, pml,
                        ("decoder.prediction.dec_rnn.lstm.bias_ih" + s).c_str());
                    ggml_tensor* bhh = clone_weight(ctx, pml,
                        ("decoder.prediction.dec_rnn.lstm.bias_hh" + s).c_str());
                    ggml_tensor* z = ggml_add(ctx,
                        ggml_add(ctx, ggml_mul_mat(ctx, Wih, layer_in), bih),
                        ggml_add(ctx, ggml_mul_mat(ctx, Whh, h_cur[l]), bhh));
                    ggml_tensor* ig = ggml_sigmoid(ctx, ggml_cont(ctx, ggml_view_1d(ctx, z, Hp, 0)));
                    ggml_tensor* fg = ggml_sigmoid(ctx, ggml_cont(ctx, ggml_view_1d(ctx, z, Hp, (size_t)Hp*sizeof(float))));
                    ggml_tensor* cg = ggml_tanh   (ctx, ggml_cont(ctx, ggml_view_1d(ctx, z, Hp, (size_t)2*Hp*sizeof(float))));
                    ggml_tensor* og = ggml_sigmoid(ctx, ggml_cont(ctx, ggml_view_1d(ctx, z, Hp, (size_t)3*Hp*sizeof(float))));
                    ggml_tensor* c_fresh = ggml_add(ctx, ggml_mul(ctx, fg, c_cur[l]),
                                                    ggml_mul(ctx, ig, cg));
                    ggml_tensor* h_fresh = ggml_mul(ctx, og, ggml_tanh(ctx, c_fresh));
                    h_new[l] = h_fresh;
                    c_new[l] = c_fresh;
                    layer_in = h_fresh;
                    g_proj   = h_fresh;
                }
                // g_proj is the prediction output g [Hp] (top-layer h').

                // --- blank-skip freeze (device-side where on advance flag). ---
                // advance = (last_tok != blank) as f32 [1] via the adv_mask gather.
                ggml_tensor* adv = ggml_cont_1d(
                    ctx, ggml_get_rows(ctx, adv_mask, last_tok), 1);   // f32 [1]
                // decoder_out = where(advance, g_proj, cc_cur).
                ggml_tensor* decoder_out = where_scalar(ctx, adv, one_t, g_proj, cc_cur);
                // Committed state: freeze on blank.
                std::vector<ggml_tensor*> h_next(L), c_next(L);
                for (int l = 0; l < L; ++l) {
                    h_next[l] = where_scalar(ctx, adv, one_t, h_new[l], h_cur[l]);
                    c_next[l] = where_scalar(ctx, adv, one_t, c_new[l], c_cur[l]);
                }
                ggml_tensor* cc_next = decoder_out;  // cc always := this step's decoder_out

                // --- Joint (enc_proj[frame] + decoder_out -> logits). ---
                // Carry frame as f32 in-graph (cast from the i32 input on step 0,
                // advanced by dur_final each step). Clamp to [0, T-1] for the
                // gather on a cast copy (ggml_clamp is in-place, so cast first to
                // avoid corrupting the chained frame node), then cast to i32.
                ggml_tensor* frame_f = ggml_cast(ctx, frame, GGML_TYPE_F32);     // f32 [1]
                ggml_tensor* frame_f_clamped = ggml_clamp(ctx, frame_f, 0.0f, (float)(T-1));
                ggml_tensor* frame_clamped = ggml_cast(ctx, frame_f_clamped, GGML_TYPE_I32); // i32 [1]
                // enc_proj row for frame_clamped: get_rows(enc_proj[Hj,T], idx) -> [Hj,1]
                ggml_tensor* ep_row = ggml_cont_1d(
                    ctx, ggml_get_rows(ctx, enc_proj_t, frame_clamped), Hj);     // f32 [Hj]
                ggml_tensor* Wp = clone_weight(ctx, jml, "joint.pred.weight");
                ggml_tensor* pp = ggml_mul_mat(ctx, Wp, decoder_out);            // [Hj]
                ggml_tensor* bp = clone_weight(ctx, jml, "joint.pred.bias");
                pp = ggml_add(ctx, pp, bp);
                ggml_tensor* fr = ggml_relu(ctx, ggml_add(ctx, ep_row, pp));    // [Hj]
                ggml_tensor* Wo = clone_weight(ctx, jml, "joint.joint_net.2.weight");
                ggml_tensor* y  = ggml_mul_mat(ctx, Wo, fr);                    // [V]
                ggml_tensor* bo = clone_weight(ctx, jml, "joint.joint_net.2.bias");
                y = ggml_add(ctx, y, bo);                                       // [V_plus]

                // --- Argmax (token slice + duration slice) ON DEVICE. ---
                ggml_tensor* tok_view = ggml_view_1d(ctx, y, token_count, 0);
                ggml_tensor* dur_view = ggml_view_1d(ctx, y, num_dur,
                                        (size_t)token_count * sizeof(float));
                ggml_tensor* tok = ggml_argmax(ctx, tok_view);   // i32 [1]
                ggml_tensor* dur_idx = ggml_argmax(ctx, dur_view); // i32 [1]

                // --- TDT frame-advance (in-graph). ---
                // dur = dur_table[dur_idx]  (i32 gather).
                ggml_tensor* dur_i32 = ggml_cont_1d(
                    ctx, ggml_get_rows(ctx, dur_tbl, dur_idx), 1);              // i32 [1]
                // blank_flag = (tok == blank) = 1 - advance_tok, where advance_tok
                //   = get_rows(adv_mask, tok).  dur0_flag = get_rows(zero_dur, dur).
                ggml_tensor* adv_tok = ggml_cont_1d(
                    ctx, ggml_get_rows(ctx, adv_mask, tok), 1);                 // f32 [1]
                ggml_tensor* blank_flag = ggml_sub(ctx, one_t, adv_tok);        // f32 [1]
                ggml_tensor* dur0_flag  = ggml_cont_1d(
                    ctx, ggml_get_rows(ctx, zero_dur, dur_i32), 1);             // f32 [1]
                // force = blank_flag AND dur0_flag = blank_flag * dur0_flag.
                ggml_tensor* force = ggml_mul(ctx, blank_flag, dur0_flag);      // f32 [1]
                // dur_final = where(force, 1, dur_i32).
                ggml_tensor* dur_f = ggml_cast(ctx, dur_i32, GGML_TYPE_F32);
                ggml_tensor* dur_final = where_scalar1(ctx, force, one_t, one_t, dur_f);
                dbg_dur_idx   = dur_idx;
                dbg_dur_final = dur_final;
                dbg_one = one_t;
                dbg_dur_i32 = dur_i32;
                // frame_next = frame + dur_final.  Carry frame as f32 across steps.
                ggml_tensor* frame_next_f = ggml_add(ctx, frame_f, dur_final);    // f32 [1]

                // --- Ring writes: this step's token + post-step frame. ---
                // Cast the i32 token to f32 BEFORE pushing: the ring is built via
                // ggml_concat, and ggml-cuda's CONCAT rejects I32 operands (only
                // F32/F16 supported). Token ids are <= blank_id (8192) << 2^23,
                // so (float)tok is bit-exact and reversible via (int32_t)f.
                tok_nodes.push_back(ggml_cast(ctx, tok, GGML_TYPE_F32));   // f32 [1]
                frame_nodes.push_back(frame_next_f);

                // --- Chain state for step j+1. ---
                last_tok = tok;                 // i32 [1]
                cc_cur   = cc_next;             // f32 [Hp]
                h_cur    = h_next;
                // Cell state must chain through c_next (the freeze-selected value),
                // NOT c_new. On a blank-input step adv=0 -> c_next = c_cur (frozen);
                // chaining c_new would leak the discarded LSTM c' and corrupt the
                // decoder state across blank steps (the reference's termination fix).
                c_cur    = c_next;
                frame    = frame_next_f;        // f32 [1] from here on
            }

            // ---- Captures: ring (tokens, frames) ONLY. The chained state
            // (h/c/cc/frame/last_token) is NOT captured -- D1 keeps it in the
            // persistent device cache, written back below via add_graph_root.
            ggml_tensor* ring_tok = tok_nodes[0];
            for (int j = 1; j < K; ++j)
                ring_tok = ggml_concat(ctx, ring_tok, tok_nodes[j], 0);   // f32 [K]
            ggml_tensor* ring_frame = frame_nodes[0];
            for (int j = 1; j < K; ++j)
                ring_frame = ggml_concat(ctx, ring_frame, frame_nodes[j], 0); // f32 [K]
            capture_graph_output(ring_tok,   &raw->cap_tokens_f);
            capture_graph_output(ring_frame, &raw->cap_frames_f);
            // NOTE: the final frame_idx and last_token are NOT captured separately
            // -- they are the ring's LAST element (frame_next_f / cast(tok) of the
            // final step), so the host reads them from cap_frames_f[K-1] and
            // cap_tokens_f[K-1].
            capture_graph_output(dbg_dur_idx,   &raw->cap_dbg_dur_idx_f);
            capture_graph_output(dbg_dur_final, &raw->cap_dbg_dur_final_f);
            capture_graph_output(dbg_one,       &raw->cap_dbg_one_f);
            capture_graph_output(dbg_dur_i32,   &raw->cap_dbg_dur_i32_f);

            // ---- State sink: write the final K-step state back so the next
            // replay resumes from it.
            //   device_resident (D1): in-graph cpy into the persistent device
            //     cache leaves (add_graph_root side effects; NO host readback).
            //   baseline: capture h/c/cc for the host to seed the next replay.
            if (raw->device_resident) {
                // frame is f32 here; dc->frame is i32 (matching the step-0 cast
                // topology), so cast f32->i32 first. Frame indices are small
                // integers, exact in f32 -> the round-trip is bit-identical.
                ggml_tensor* frame_i32 = ggml_cast(ctx, frame, GGML_TYPE_I32);
                add_graph_root(ggml_cpy(ctx, frame_i32,
                    ggml_view_1d(ctx, raw->dc->frame, 1, 0)));
                add_graph_root(ggml_cpy(ctx, last_tok,
                    ggml_view_1d(ctx, raw->dc->last_tok, 1, 0)));
                add_graph_root(ggml_cpy(ctx, cc_cur,
                    ggml_view_1d(ctx, raw->dc->cc, Hp, 0)));
                for (int l = 0; l < L; ++l) {
                    add_graph_root(ggml_cpy(ctx, h_cur[(size_t)l],
                        ggml_view_1d(ctx, raw->dc->h[(size_t)l], Hp, 0)));
                    add_graph_root(ggml_cpy(ctx, c_cur[(size_t)l],
                        ggml_view_1d(ctx, raw->dc->c[(size_t)l], Hp, 0)));
                }
            } else {
                for (int l = 0; l < L; ++l) {
                    capture_graph_output(h_cur[(size_t)l], &raw->cap_h[(size_t)l]);
                    capture_graph_output(c_cur[(size_t)l], &raw->cap_c[(size_t)l]);
                }
                capture_graph_output(cc_cur, &raw->cap_cc_f);
            }

            // The graph output is arbitrary (ReplayGraph requires one); use the
            // tokens ring (also captured). Mark it the output.
            return ring_tok;
        }));

    if (raw->rg == nullptr) return nullptr;
    it = g_kstep_cache.emplace(key, std::move(kg)).first;
    return it->second.get();
}

std::optional<std::vector<int32_t>> tdt_greedy_multistep(
    const PredictionNet& pred, const Joint& joint,
    const std::vector<float>& enc_proj, int T,
    const std::vector<int32_t>& durations,
    int blank_id, int max_symbols) {
    if (!global_backend().is_gpu()) return std::nullopt;      // caller falls back to serial
    if (T <= 0 || durations.empty()) return std::nullopt;
    const bool dbg = tdt_kstep_debug();
    const int K = tdt_kstep_config(T);
    if (dbg) std::fprintf(stderr, "[tdt_multistep] entry K=%d T=%d gpu=%d\n",
                          K, T, (int)global_backend().is_gpu());

    const int Hj = joint.joint_hidden();
    const int Hp = pred.hidden_size();
    const int L  = pred.num_layers();
    const int token_count = joint.vocab_size() + 1;

    // ---- 1. Step 0 EAGER (init path: zero cache, NO freeze). Mirrors Starling's
    // decode_mega.py _step0_eager: feed embedding(blank) (== the zero SOS vector,
    // since the embedding has padding_idx=blank), run the LSTM, and COMMIT the
    // output state UNCONDITIONALLY -- even when step 0 emits a blank. This keeps
    // h/c aligned with the "LSTM ran on the most recent step" invariant the
    // in-graph freeze assumes.
    //   * cc (frozen decoder-out cache) := step-0 prediction output g (no freeze).
    //   * last_tok fed into the first captured step := k0 (the emitted token; if
    //     blank, embedding(blank) == SOS zeros, so feeding it is a no-op skip).
    int k0 = 0, d0 = 0;
    PredState committed;
    joint.step_fused_argmax(pred, enc_proj.data(), token_count,
                            /*token_id=*/blank_id, /*is_sos=*/true,
                            pred.zero_state(), committed, k0, d0);
    // step_fused_argmax populated `committed` with the step-0 LSTM (h', c'). Fetch
    // the step-0 prediction output g (== the joint's decoder input, == cc) via
    // pred.step (deterministic recompute; bit-identical to the g step_fused_argmax
    // used internally).
    std::vector<float> cc((size_t)Hp, 0.0f);
    {
        std::vector<float> g0;
        PredState zero = pred.zero_state();
        PredState tmp;   // == committed (LSTM(SOS=zeros, zero)); discarded.
        pred.step(blank_id, true, zero, g0, tmp);
        std::memcpy(cc.data(), g0.data(), (size_t)Hp * sizeof(float));
    }

    if (dbg) {
        // Is embedding(blank) == zeros? (the in-graph chaining feeds
        // get_rows(embed, last_tok) with last_tok=blank after a blank step,
        // which must equal the serial loop's SOS=zeros input.)
        const std::vector<float>& eh = pred.embed_host();
        if (!eh.empty()) {
            double sum = 0.0; double mx = 0.0;
            for (int i = 0; i < Hp; ++i) { float v = eh[(size_t)blank_id * Hp + i]; sum += v; if (std::fabs(v) > mx) mx = std::fabs(v); }
            std::fprintf(stderr, "[tdt_multistep] embed[blank] sum=%.4f maxabs=%.4f (0 => SOS equiv holds)\n", sum, mx);
        }
    }

    std::vector<int32_t> hyp;
    hyp.push_back(k0);          // STARLING: emit EVERY token, including blank.
    int frame0 = durations[(size_t)d0];
    if (dbg) std::fprintf(stderr, "[tdt_multistep] step0 k0=%d d0=%d frame0=%d T=%d\n", k0, d0, frame0, T);
    if (frame0 >= T) return hyp;       // step 0 finished the utterance (success)

    // ---- 2. Pick the decode path. device_resident (D1: persistent device-
    // resident state, no host round-trip between replays) is used for the
    // proven-byte-exact small-K graphs (K<=16, i.e. short/medium). The K>16 long
    // graph keeps the baseline host round-trip: the D1 in-graph state write-back
    // cpy into persistent device buffers trips ggml's CUDA-graph topology defect
    // (illegal memory access for the K=64 graph; see the D1 notes in the file
    // header), and the device-resident path is gated to K<=16 for safety.
    // (D2 K-sweep: plans/wave-d-parakeet-long-closure.md.)
    const bool d1 = (K <= 16);

    // D1 only: get (or create) the persistent decode device cache and SEED it
    // once with the eager step-0 state. The K-step graph reads these device
    // leaves at step 0 of each replay and writes the final state back in-graph.
    DecodeDevCache* dc = nullptr;
    if (d1) {
        std::string ddc_err;
        dc = get_decode_dev_cache(Hp, L, ddc_err);
        if (!dc) {
            if (dbg) std::fprintf(stderr, "[tdt_multistep] DecodeDevCache alloc failed: %s\n", ddc_err.c_str());
            return std::nullopt;       // caller falls back to the serial loop
        }
        int32_t seed_frame[1] = { frame0 };
        int32_t seed_last[1]  = { k0 };
        ggml_backend_tensor_set(dc->frame,    seed_frame, 0, sizeof(int32_t));
        ggml_backend_tensor_set(dc->last_tok, seed_last,  0, sizeof(int32_t));
        ggml_backend_tensor_set(dc->cc,       cc.data(),  0, (size_t)Hp * sizeof(float));
        for (int l = 0; l < L; ++l) {
            ggml_backend_tensor_set(dc->h[(size_t)l], committed.h[(size_t)l].data(), 0, (size_t)Hp * sizeof(float));
            ggml_backend_tensor_set(dc->c[(size_t)l], committed.c[(size_t)l].data(), 0, (size_t)Hp * sizeof(float));
        }
    }

    // ---- 3. Get (or build) the K-step graph for (T, K), bound to dc (D1).
    KStepGraph* kg = get_or_build_kstep(pred, joint, T, K, blank_id, durations, dc, d1);
    if (!kg) return std::nullopt;       // graph build failed; caller falls back
    if (dbg) std::fprintf(stderr, "[tdt_multistep] K=%d T=%d d1=%d (graph built)\n", K, T, (int)d1);
    // Seed enc_proj once (persists across replays in the input tensor).
    kg->rg->set_input(kg->in_enc_proj, enc_proj.data(), (size_t)T * Hj * sizeof(float));

    // ---- 4. Seed the constant-table inputs once (they never change). Small.
    kg->rg->set_input(kg->in_adv_mask, kg->adv_mask_table.data(),
                      (size_t)pred.vocab_p1() * sizeof(float));
    kg->rg->set_input(kg->in_dur_tbl, kg->dur_table_int.data(),
                      (size_t)kg->num_dur * sizeof(int32_t));
    kg->rg->set_input(kg->in_zero_dur, kg->zero_dur_table.data(),
                      (size_t)kg->num_dur * sizeof(float));
    kg->rg->set_input(kg->in_one, kg->one_scalar.data(), sizeof(float));

    // Baseline path only: host-resident running state (re-uploaded each replay).
    int32_t cur_frame = frame0, cur_last_tok = k0;
    std::vector<float> cur_cc = cc;
    std::vector<std::vector<float>> cur_h = committed.h, cur_c = committed.c;

    // ---- 5. Replay loop: ONE replay, ONE sync, scatter the emitted-id ring.
    const int max_steps = max_symbols * T + 16;
    int steps_done = 1;   // step 0 done

    while (steps_done < max_steps) {
        // Baseline path: re-upload the per-replay state (host -> device).
        if (!kg->device_resident) {
            kg->host_frame[0]    = cur_frame;
            kg->host_last_tok[0] = cur_last_tok;
            std::memcpy(kg->host_cc.data(), cur_cc.data(), (size_t)Hp*sizeof(float));
            kg->rg->set_input(kg->in_frame,    kg->host_frame.data(),    sizeof(int32_t));
            kg->rg->set_input(kg->in_last_tok, kg->host_last_tok.data(), sizeof(int32_t));
            kg->rg->set_input(kg->in_cc,       kg->host_cc.data(),       (size_t)Hp*sizeof(float));
            for (int l = 0; l < L; ++l) {
                kg->rg->set_input(kg->in_h[(size_t)l], cur_h[(size_t)l].data(), (size_t)Hp*sizeof(float));
                kg->rg->set_input(kg->in_c[(size_t)l], cur_c[(size_t)l].data(), (size_t)Hp*sizeof(float));
            }
        }
        // (device_resident: no set_input -- state is seeded/chained on-device.)

        // ONE replay + ONE sync.
        std::vector<float> out_unused;
        if (!kg->rg->compute_with_captures(out_unused)) return std::nullopt;  // caller falls back

        std::vector<int32_t> tokens((size_t)K);
        std::vector<float>   frames_f((size_t)K);
        // Tokens were stored as f32 (float)tok; convert back via truncation.
        for (int j = 0; j < K; ++j) tokens[j] = (int32_t)kg->cap_tokens_f[j];
        std::memcpy(frames_f.data(), kg->cap_frames_f.data(), (size_t)K*sizeof(float));
        if (dbg) {
            int32_t di; std::memcpy(&di, kg->cap_dbg_dur_idx_f.data(), sizeof(int32_t));
            int32_t duriv; std::memcpy(&duriv, kg->cap_dbg_dur_i32_f.data(), sizeof(int32_t));
            std::fprintf(stderr, "[tdt_multistep] replay: tok[0]=%d dur_idx=%d dur_i32=%d one=%.3f dur_final=%.3f frame[0]=%.1f\n",
                         tokens[0], di, duriv, kg->cap_dbg_one_f[0], kg->cap_dbg_dur_final_f[0], frames_f[0]);
            // Full K-step ring + the boundary state (D1: the device-cache
            // leaves the add_graph_root cpys wrote). The K-step-vs-serial
            // divergence on Vulkan shows up here as the ring's frames
            // jumping to f32-bit-pattern garbage (e.g. 1073741824.0) on the
            // SECOND+ replay of the same graph while the dc leaves
            // themselves read back correct — see the diagnostics README.
            std::fprintf(stderr, "[tdt_multistep]   ring tok:");
            for (int j = 0; j < K; ++j) std::fprintf(stderr, " %d", tokens[j]);
            std::fprintf(stderr, "  frame:");
            for (int j = 0; j < K; ++j) std::fprintf(stderr, " %.1f", frames_f[j]);
            std::fprintf(stderr, "\n");
            if (kg->device_resident && kg->dc) {
                int32_t fr = -1, lt = -1;
                ggml_backend_tensor_get(kg->dc->frame, &fr, 0, sizeof(int32_t));
                ggml_backend_tensor_get(kg->dc->last_tok, &lt, 0, sizeof(int32_t));
                float cc0 = 0.0f;
                ggml_backend_tensor_get(kg->dc->cc, &cc0, 0, sizeof(float));
                std::fprintf(stderr, "[tdt_multistep]   post-replay dc: frame=%d last_tok=%d cc[0]=%.6f\n",
                             fr, lt, cc0);
            }
        }

        // STARLING: emit EVERY token (including blanks); stop at the first step
        // whose frame >= T. (The reference dropped blanks; Starling keeps them.)
        bool finished = false;
        for (int j = 0; j < K; ++j) {
            hyp.push_back(tokens[j]);
            if ((int)frames_f[j] >= T) { finished = true; break; }
        }
        if (finished) break;

        if (kg->device_resident) {
            // Not finished: the device cache now holds this replay's final state
            // (written in-graph); the next replay reads it at step 0. No host work.
        } else {
            // Baseline: read the final-state capture and seed the next replay.
            cur_frame    = (int32_t)frames_f[K - 1];
            cur_last_tok = tokens[K - 1];
            std::memcpy(cur_cc.data(), kg->cap_cc_f.data(), (size_t)Hp*sizeof(float));
            for (int l = 0; l < L; ++l) {
                std::memcpy(cur_h[(size_t)l].data(), kg->cap_h[(size_t)l].data(), (size_t)Hp*sizeof(float));
                std::memcpy(cur_c[(size_t)l].data(), kg->cap_c[(size_t)l].data(), (size_t)Hp*sizeof(float));
            }
        }
        steps_done += K;
    }
    return hyp;
}

} // namespace starling::ggml::parakeet
