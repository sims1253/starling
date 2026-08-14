// threads.hpp — shared host-side threading for the model engines.
//
// Extracted verbatim from the four identical mel_thread_count/mel_parallel
// copies in moss/ark/higgs/hojo mel.cpp (this branch's "starling lib" layer).
// parallel_for splits [0, total) into contiguous per-thread ranges; bodies
// see disjoint regions, so results do not depend on the thread count.
#pragma once

#include <algorithm>
#include <cstdlib>
#include <thread>
#include <vector>

namespace starling::ggml::lib {

// Thread count for the mel front-ends: STARLING_MEL_THREADS override
// (>= 1 honored verbatim), else hardware concurrency capped at 16.
inline size_t mel_thread_count() {
    if (const char* p = std::getenv("STARLING_MEL_THREADS")) {
        char* end = nullptr;
        long v = std::strtol(p, &end, 10);
        if (end != p && v >= 1) return static_cast<size_t>(v);
    }
    unsigned hc = std::thread::hardware_concurrency();
    if (hc == 0) hc = 1;
    if (hc > 16) hc = 16;
    return static_cast<size_t>(hc);
}

// Thread count without the env override (e.g. hojo's host conv stack).
inline size_t default_thread_count() {
    unsigned hc = std::thread::hardware_concurrency();
    if (hc == 0) hc = 1;
    if (hc > 16) hc = 16;
    return static_cast<size_t>(hc);
}

// Run body(tid, lo, hi) over a contiguous split of [0, total).
template <typename Body>
void parallel_for(size_t nthr, size_t total, Body&& body) {
    if (total == 0) return;
    if (nthr <= 1) { body((size_t)0, (size_t)0, total); return; }
    if (nthr > total) nthr = total;
    std::vector<std::thread> ths;
    ths.reserve(nthr);
    const size_t chunk = (total + nthr - 1) / nthr;
    for (size_t i = 0; i < nthr; ++i) {
        const size_t lo = i * chunk;
        if (lo >= total) break;
        const size_t hi = std::min(lo + chunk, total);
        ths.emplace_back([&, i, lo, hi]() { body(i, lo, hi); });
    }
    for (auto& t : ths) t.join();
}

} // namespace starling::ggml::lib
