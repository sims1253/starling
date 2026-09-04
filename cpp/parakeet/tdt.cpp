// tdt.cpp — parakeet-tdt serial greedy TDT decode loop (Phase 1c).
//
// Starling-authored port of parakeet.cpp's tdt.cpp:16-194 serial path. Mirrors
// the greedy decode bit-for-bit, EXCEPT that — to match the golden
// parakeet_tdt_{short,medium,long}_ids.pt tensors — it emits EVERY token id
// (including blanks) to the output stream. (parakeet.cpp's `hyp` drops blanks;
// the golden ids here keep them. The blank id 8192 is out of the tokenizer's
// [0, vocab_size) range, so detokenize naturally skips it.)
//
// enc_proj is row-major [T, H] (enc_proj[t*H + h]) — i.e. frame t's projected
// encoder vector is contiguous. This is exactly the encoder Phase 1b output
// layout (verified byte-exact vs golden parakeet_tdt_*_enc.pt).

#include "tdt.hpp"
#include "tdt_multistep.hpp"

#include "runtime/backend.hpp"  // Backend::is_gpu (full def)
#include "runtime/graph.hpp"    // global_backend

#include <cassert>
#include <cstdlib>
#include <string>
#include <vector>

namespace starling::ggml::parakeet {

// STARLING_GGML_TDT_SERIAL=1 forces the byte-identical serial loop (no fused
// replay, no K-step multistep). Used as the reference path and the non-GPU
// fallback; mirrors parakeet.cpp's gating.
static bool tdt_serial_forced() {
    const char* e = std::getenv("STARLING_GGML_TDT_SERIAL");
    return e && e[0] == '1';
}

// The K-step multistep graph seeds its constant inputs (enc_proj, duration
// table, masks) ONCE per utterance and replays the same cgraph per K decode
// steps. That contract is what ggml patch 0011 makes sound: the gallocr's
// live-range reuse used to recycle input storage after the last in-graph
// read, so replay #2+ re-read the late intermediates that compute #1's tail
// wrote over the constant tables (the early-termination bug PR #45 gated
// off on Vulkan — full story in scripts/diagnostics/vulkan/README.md).
// With 0011, inputs are pinned like outputs and the multistep path is exact
// on every backend (validated on Vulkan with the multistep path forced at
// K=2 and K=16).

std::vector<int32_t> tdt_greedy(const PredictionNet& pred, const Joint& joint,
                                const std::vector<float>& enc_proj,
                                int T, int H,
                                const std::vector<int32_t>& durations,
                                int blank_id, int max_symbols) {
    assert((int)enc_proj.size() == (size_t)T * H);
    assert(!durations.empty());
    (void)H;

    // GPU K-step multistep fast path: capture K consecutive decode steps into
    // ONE CUDA graph and sync once per K steps instead of once per step. The
    // serial loop below is launch/sync-bound; this collapses ~T syncs to ~T/K.
    // Byte-exact with the serial loop (emits EVERY token, including blanks, to
    // match the golden id stream). On CPU, when the serial path is forced, or
    // if capture fails, tdt_greedy_multistep returns nullopt and we fall
    // through to the serial loop. Mirrors parakeet.cpp's tdt.cpp dispatch.
    if (global_backend().is_gpu() && !tdt_serial_forced()) {
        // Safety ceiling on encoder length (protects against an unbounded-length
        // capture on extraordinarily long audio); override with
        // STARLING_GGML_TDT_KSTEP_MAX_T (a value < T forces the serial path).
        int kstep_max_t = 4096;
        if (const char* e = std::getenv("STARLING_GGML_TDT_KSTEP_MAX_T")) {
            int v = std::atoi(e); if (v > 0) kstep_max_t = v;
        }
        if (T <= kstep_max_t) {
            auto ms = tdt_greedy_multistep(pred, joint, enc_proj, T, durations,
                                           blank_id, max_symbols);
            if (ms.has_value()) return std::move(*ms);
            // nullopt -> fall through to the byte-exact serial loop.
        }
    }

    const int token_count = joint.vocab_size() + 1;

    // The committed (non-blank) decoding state + last-emitted token. SOS until
    // the first non-blank emit (last_label = blank_id is ignored when is_sos).
    PredState committed = pred.zero_state();
    int32_t last_token = blank_id;   // placeholder; only meaningful post-emit
    bool emitted_any = false;

    // Scratch reused across inner steps.
    std::vector<float> g;            // prediction output (top LSTM layer h')
    PredState out_state;             // new (h', c') from each prediction step
    bool g_valid = false;            // cached-g reuse across non-emit steps

    std::vector<int32_t> hyp;        // emitted id stream (INCLUDING blanks)

    // GPU fast path: drive each inner step with ONE fused prediction-LSTM +
    // joint + argmax graph (one host<-device sync per step) instead of two
    // separate run_graph calls (pred, then joint). The fused graph keeps the
    // prediction output g on the device (pred -> joint, no host round-trip). It
    // drops the g_valid cache (it recomputes the LSTM every step): a non-emit
    // step's (last_label, committed_state) are unchanged, so the recompute
    // yields bit-identical g/logits/argmax -> byte-exact hyp.
    const bool fused_gpu = global_backend().is_gpu() && !tdt_serial_forced();

    int t = 0;
    while (t < T) {
        int symbols_added = 0;
        bool need_loop = true;
        int skip = 0;

        while (need_loop && symbols_added < max_symbols) {
            assert(t >= 0 && t < T && "enc_proj row out of range");
            const float* enc_proj_t = enc_proj.data() + (size_t)t * H;

            int k = 0, d_k = 0;
            if (fused_gpu) {
                // ONE graph = ONE sync per step: prediction LSTM + joint + argmax.
                // SOS until the first emit; otherwise feed the last EMITTED token.
                const bool is_sos = !emitted_any;
                const int32_t last_label = emitted_any ? last_token : blank_id;
                joint.step_fused_argmax(pred, enc_proj_t, token_count,
                                        last_label, is_sos,
                                        committed, out_state, k, d_k);
            } else {
                // Unfused path (CPU, or STARLING_GGML_TDT_SERIAL=1): pred.step
                // then joint, with the prediction output carried through the host.
                if (!g_valid) {
                    const bool is_sos = !emitted_any;
                    const int32_t last_label = emitted_any ? last_token : blank_id;
                    pred.step(last_label, is_sos, committed, g, out_state);
                    g_valid = true;
                }
                joint.step_argmax(enc_proj_t, token_count,
                                  g.data(), (int)g.size(), k, d_k);
            }
            skip = durations[(size_t)d_k];

            // Emit k EVERY step, including blank — matches the golden _ids.pt.
            hyp.push_back((int32_t)k);

            // Commit state + last_token ONLY when k != blank.
            if (k != blank_id) {
                last_token = (int32_t)k;
                committed = out_state;   // carry the step's new (h', c')
                emitted_any = true;
                g_valid = false;         // committed state advanced -> recompute g
            }
            // else: discard out_state; committed/last_token unchanged (g stays
            // valid for the next inner step — blank-skip reuse).

            symbols_added += 1;
            t += skip;
            need_loop = (skip == 0);
        }

        // Infinite-loop guard: if we exited with duration 0 (blank + dur 0),
        // step forward by one frame anyway.
        if (skip == 0) skip = 1;

        // If we stopped because max_symbols was hit (not because of a positive
        // duration), advance the frame by one to make progress.
        if (symbols_added == max_symbols) t += 1;
    }

    return hyp;
}

} // namespace starling::ggml::parakeet
