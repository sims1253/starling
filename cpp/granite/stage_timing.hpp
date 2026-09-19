// stage_timing.hpp — the STARLING_GRANITE_TIMING stage clock shared by
// capi_granite.cpp and the stage-attribution regression test
// (cpp/tests/granite_stage_test.cpp).
//
// transcribe_piece() records three stage durations per chunk (mel+encode+
// project, prompt+embeds, generate). The long-audio loop used to reuse one
// array across chunks, so the final GRANITE_STAGE line printed the LAST
// chunk's durations next to a whole-request total and chunk count (issue
// #170). StageTiming accumulates every chunk instead, and the two render
// helpers below are the single source of truth for the log lines — the
// emission in capi_granite.cpp and the test's parser share them:
//
//   GRANITE_STAGE chunk=<i> mel+enc+proj=..ms prompt+embeds=..ms gen=..ms piece=..ms
//   GRANITE_STAGE request chunks=<n> audio=..s mel+enc+proj=..ms prompt+embeds=..ms \
//                  gen=..ms stages=..ms bookkeeping=..ms total=..ms
//
// Reconciliation contract: `stages` sums the per-chunk stage durations over
// ALL chunks, `total` is the whole-request wall time measured around the
// chunk loop AND the final text join, and `bookkeeping = total - stages` is
// the unattributed remainder (loop memcpy/padding, per-chunk detokenize, the
// text join, and the timing prints themselves — the output-buffer malloc and
// copy are response emission and stay outside the window). The three agree by
// construction within the %.1f rendering rounding; the regression test pins
// that.
#pragma once

#include <cstdio>
#include <string>

namespace starling::ggml::granite {

constexpr int kStageCount = 3;  // mel+enc+proj, prompt+embeds, generate

// Per-chunk stage durations plus their whole-request aggregation for one
// starling_ggml_granite_decode call. Durations are milliseconds.
struct StageTiming {
    double chunk_ms[kStageCount] = {0, 0, 0};  // last completed chunk
    double total_ms[kStageCount] = {0, 0, 0};  // sum over ALL chunks
    int64_t chunks = 0;                        // completed chunks

    // Record one completed chunk (the per-piece array transcribe_piece
    // filled). Latches the per-chunk values and adds them to the totals.
    void add_chunk(const double piece_ms[kStageCount]) {
        for (int i = 0; i < kStageCount; ++i) {
            chunk_ms[i] = piece_ms[i];
            total_ms[i] += piece_ms[i];
        }
        ++chunks;
    }

    // Sum of the last chunk's stages (its wall-clock extent inside the loop).
    double chunk_total_ms() const {
        double sum = 0;
        for (int i = 0; i < kStageCount; ++i) sum += chunk_ms[i];
        return sum;
    }

    // Sum of the per-chunk stage durations over every chunk.
    double stages_total_ms() const {
        double sum = 0;
        for (int i = 0; i < kStageCount; ++i) sum += total_ms[i];
        return sum;
    }

    // Whole-request wall time minus the summed stages: the loop/join/
    // detokenize bookkeeping around the stage clocks. Never negative beyond
    // rendering rounding.
    double bookkeeping_ms(double request_total_ms) const {
        return request_total_ms - stages_total_ms();
    }
};

// The per-chunk summary line, for the chunk with 1-based index `chunk_index`
// (StageTiming holds that chunk's values in chunk_ms).
inline std::string format_stage_chunk_line(const StageTiming& s, int64_t chunk_index) {
    char buf[256];
    std::snprintf(buf, sizeof buf,
        "GRANITE_STAGE chunk=%lld mel+enc+proj=%.1fms prompt+embeds=%.1fms "
        "gen=%.1fms piece=%.1fms",
        (long long) chunk_index, s.chunk_ms[0], s.chunk_ms[1], s.chunk_ms[2],
        s.chunk_total_ms());
    return buf;
}

// The whole-request summary line. `request_total_ms` is the wall time around
// the whole chunk loop plus the final text join (the caller's t_start..t_end
// span; the output-buffer malloc/copy stay outside it).
inline std::string format_stage_request_line(const StageTiming& s, double audio_seconds,
                                             double request_total_ms) {
    char buf[320];
    std::snprintf(buf, sizeof buf,
        "GRANITE_STAGE request chunks=%lld audio=%.2fs mel+enc+proj=%.1fms "
        "prompt+embeds=%.1fms gen=%.1fms stages=%.1fms bookkeeping=%.1fms "
        "total=%.1fms",
        (long long) s.chunks, audio_seconds, s.total_ms[0], s.total_ms[1],
        s.total_ms[2], s.stages_total_ms(), s.bookkeeping_ms(request_total_ms),
        request_total_ms);
    return buf;
}

} // namespace starling::ggml::granite
