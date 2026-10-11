// encoder.hpp — parakeet-tdt Conformer encoder.
#pragma once

#include "conformer.hpp"
#include "config.hpp"
#include "loader.hpp"
#include "pos_enc.hpp"
#include "subsampling.hpp"
#include "runtime/backend.hpp"
#include "runtime/graph_builder.hpp"
#include "runtime/lru_cache.hpp"

#include <memory>
#include <unordered_map>
#include <vector>

namespace starling::ggml::parakeet {

class Encoder {
public:
    explicit Encoder(const ParakeetModel& model);
    ~Encoder();

    bool encode(const std::vector<float>& mel, int n_mels, int T,
                std::vector<float>& out, int& out_Tp) const;

    int proj_dim() const { return (int)config_.joint_hidden; }

    // Current number of cached per-T encoder graphs (diagnostic + the Wave H
    // bounded-LRU regression-test hook). Zero before first GPU encode.
    size_t cache_size() const;

private:
    // Host data of a bucketed graph's LengthMasks, sized for the bucket at
    // build time (stable addresses: the graph's inputs point here) and
    // refilled for the true mel length before every replay.
    struct PadMasks {
        std::vector<float> stage1, stage2, time, key;
    };
    struct ReplayEntry {
        int T = 0;
        int Tp = 0;
        int valid_len = 0;
        PadMasks masks;
        GraphInputPool pool;
        std::unique_ptr<ReplayGraph> graph;
        // [starling pos-cache] per-layer positional projections [dk, P, H],
        // constant per T': each layer is computed once (small one-shot device
        // compute), uploaded into its PERSISTENT replay input, then the
        // scratch is reused -> transient host footprint is one ph layer.
        std::vector<float> ph_scratch;   // stable-address scratch (entry-owned)
    };
    // Bounded by the graphs' tracked device bytes, not their count: an entry
    // costs ~0.09 MB per mel frame (RADV, parakeet-0.6b: 117 MB at 1280
    // frames, 610 MB at 6656), so a live stream's window buckets (~0.6 GB)
    // stay resident on a shared-memory device whose count cap of 2 made
    // every partial a miss, while one long take still evicts the rest.
    struct ReplayCache {
        ByteBudgetLruCache<int, ReplayEntry> by_T;
        // Largest tracked bytes per mel frame seen so far: sizes the room a
        // new bucket's graph needs before it is built (ByteBudgetLruCache::reserve).
        double bytes_per_frame = 0.0;
        explicit ReplayCache(size_t budget) : by_T(budget, "parakeet.encoder") {}
        void clear() { by_T.clear(); }
    };

    ggml_tensor* build_graph(ggml_context* ctx, const std::vector<float>& mel,
                             int n_mels, int T, GraphInputPool& pool,
                             int& Tp, int& valid_len,
                             const float* ph_scratch = nullptr,
                             PadMasks* pad = nullptr) const;

    // Size a bucketed graph's masks for bucket Tb and fill them for an input
    // of T <= Tb mel frames.
    void fill_pad_masks(PadMasks& m, int Tb, int T) const;

    // One-shot per-layer pos compute: layer L's [dk, P, H] projection of the
    // sinusoidal table for this T' (device compute, host readback).
    bool compute_pos_layer(int Tp, int layer, std::vector<float>& ph_out) const;

    const ParakeetModel& model_;
    Subsampling sub_;
    const Config& config_;
};

} // namespace starling::ggml::parakeet
