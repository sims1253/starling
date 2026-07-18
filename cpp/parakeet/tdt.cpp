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

#include <cassert>
#include <vector>

namespace starling::ggml::parakeet {

std::vector<int32_t> tdt_greedy(const PredictionNet& pred, const Joint& joint,
                                const std::vector<float>& enc_proj,
                                int T, int H,
                                const std::vector<int32_t>& durations,
                                int blank_id, int max_symbols) {
    assert((int)enc_proj.size() == (size_t)T * H);
    assert(!durations.empty());
    (void)H;

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

    int t = 0;
    while (t < T) {
        int symbols_added = 0;
        bool need_loop = true;
        int skip = 0;

        while (need_loop && symbols_added < max_symbols) {
            assert(t >= 0 && t < T && "enc_proj row out of range");
            const float* enc_proj_t = enc_proj.data() + (size_t)t * H;

            // Prediction step (cached unless the committed state advanced).
            if (!g_valid) {
                const bool is_sos = !emitted_any;
                const int32_t last_label = emitted_any ? last_token : blank_id;
                pred.step(last_label, is_sos, committed, g, out_state);
                g_valid = true;
            }

            // Joint step + argmax.
            int k = 0, d_k = 0;
            joint.step_argmax(enc_proj_t, joint.vocab_size() + 1,
                              g.data(), (int)g.size(), k, d_k);
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
