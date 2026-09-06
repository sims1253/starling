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
    struct ReplayEntry {
        int T = 0;
        int Tp = 0;
        int valid_len = 0;
        GraphInputPool pool;
        std::unique_ptr<ReplayGraph> graph;
    };
    struct ReplayCache {
        LruCache<int, ReplayEntry> by_T;
        explicit ReplayCache(size_t cap) : by_T(cap) {}
        void clear() { by_T.clear(); }
    };

    ggml_tensor* build_graph(ggml_context* ctx, const std::vector<float>& mel,
                             int n_mels, int T, GraphInputPool& pool,
                             int& Tp, int& valid_len) const;

    const ParakeetModel& model_;
    Subsampling sub_;
    const Config& config_;
};

} // namespace starling::ggml::parakeet
