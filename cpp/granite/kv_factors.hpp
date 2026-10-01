// kv_factors.hpp — optional low-rank K/V projection factors for the granite
// encoder attention (issue #59 research follow-up).
//
// When the env var STARLING_GRANITE_KVFACT points at a factor file exported
// by benchmarks/export_kv_lowrank.py, each attention layer's fused
// attn_kv GEMM is replaced, per head h, by
//
//     z_h = n @ C_h^T          (rank-r down-projection)
//     k_h = z_h @ B_h^T        (expand back to head_dim)
//
// (and likewise for V) — a weight-space SVD factorization W_h ~= B_h C_h.
// Rank 0 for a half leaves that half on the original path. This is a
// NUMERICS-CHANGING research path: every kept result must pass the
// exact-transcript + WER gates (AUTORESEARCH.md). The factor file is the
// pinned artifact; do not regenerate it between measurements of the same
// experiment.
//
// File format (little-endian), documented here and in the exporter:
//   char magic[8] = "STLGKVF1"
//   u32 version = 1, u32 n_layers, u32 hidden, u32 n_heads
//   u32 rank_k, u32 rank_v            (0 = half unchanged)
//   per layer: f32 f1k[n_heads*rank_k*hidden], f32 f2k[hidden*n_heads*rank_k],
//              f32 f1v[...], f32 f2v[...]    (only when that rank > 0)
// f1 rows are per-head C_h stacked ([n_heads*rank, hidden], row-major);
// f2 is the block-diagonal expansion ([hidden, n_heads*rank], row-major).
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <vector>

namespace starling::ggml::granite {

struct KVFactors {
    int n_layers = 0, hidden = 0, n_heads = 0;
    int rank_k = 0, rank_v = 0;  // per-head ranks; 0 = unchanged path
    // Per-layer ranks (v3 "selective" files; empty for uniform v1 files).
    std::vector<int> rank_k_l, rank_v_l;
    // Per layer, row-major float32 (layouts in the header comment above).
    // A layer with per-layer rank 0 stores EMPTY vectors (no payload).
    std::vector<std::vector<float>> f1k, f2k, f1v, f2v;
    // f2k transposed ([n_heads*rank, hidden], row-major) for the optional
    // in-r-space score path (STARLING_GRANITE_KVINR): qtilde = f2T @ q.
    std::vector<std::vector<float>> f2tk;

    bool enabled() const { return rank_k > 0 || rank_v > 0; }
    int rank_layer_k(int li) const {
        return rank_k_l.empty() ? rank_k : rank_k_l[(size_t) li];
    }
    int rank_layer_v(int li) const {
        return rank_v_l.empty() ? rank_v : rank_v_l[(size_t) li];
    }

    // `path` is read fully into host memory; buffers must outlive every graph
    // that references them (they live in GraniteModel, which does).
    bool load(const char* path, int exp_layers, int exp_hidden, int exp_heads,
              std::string& err);
};

inline bool KVFactors::load(const char* path, int exp_layers, int exp_hidden,
                            int exp_heads, std::string& err) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { err = std::string("cannot open KV factor file: ") + path; return false; }
    auto rd = [&](void* dst, size_t n) -> bool {
        return std::fread(dst, 1, n, f) == n;
    };
    char magic[8];
    if (!rd(magic, 8)) {
        err = "KV factor file: truncated header"; std::fclose(f); return false;
    }
    const bool v3 = std::string(magic, 8) == "STLGKVF3";
    if (!v3 && std::string(magic, 8) != "STLGKVF1") {
        err = "KV factor file: bad magic"; std::fclose(f); return false;
    }
    unsigned int common[4];
    if (!rd(common, sizeof(common))) {
        err = "KV factor file: truncated header"; std::fclose(f); return false;
    }
    n_layers = (int) common[1]; hidden = (int) common[2]; n_heads = (int) common[3];
    if (v3) {
        if (common[0] != 3) {
            err = "KV factor file: bad version"; std::fclose(f); return false;
        }
        // v3: [ver, n_layers, hidden, n_heads] + per-layer rank tables.
        rank_k_l.resize(n_layers); rank_v_l.resize(n_layers);
        if (!rd(rank_k_l.data(), sizeof(int) * (size_t) n_layers) ||
            !rd(rank_v_l.data(), sizeof(int) * (size_t) n_layers)) {
            err = "KV factor file: truncated rank tables"; std::fclose(f); return false;
        }
        rank_k = rank_v = 0;
        for (int i = 0; i < n_layers; ++i) {
            rank_k = std::max(rank_k, rank_k_l[(size_t) i]);
            rank_v = std::max(rank_v, rank_v_l[(size_t) i]);
        }
    } else {
        if (common[0] != 1) {
            err = "KV factor file: bad version"; std::fclose(f); return false;
        }
        // v1: [ver, n_layers, hidden, n_heads, rank_k, rank_v].
        unsigned int rk_rv[2];
        if (!rd(rk_rv, sizeof(rk_rv))) {
            err = "KV factor file: truncated header"; std::fclose(f); return false;
        }
        rank_k = (int) rk_rv[0]; rank_v = (int) rk_rv[1];
    }
    if (n_layers != exp_layers || hidden != exp_hidden || n_heads != exp_heads) {
        err = "KV factor file: model dims mismatch"; std::fclose(f); return false;
    }
    // `ranks` is empty for uniform files (fall back to the global rank of
    // the half being read, identified by which f1s vector was passed).
    auto read_half = [&](const std::vector<int>& ranks, int uniform_rank,
                         std::vector<std::vector<float>>& f1s,
                         std::vector<std::vector<float>>& f2s) -> bool {
        f1s.resize(n_layers); f2s.resize(n_layers);
        for (int i = 0; i < n_layers; ++i) {
            const int r_eff = ranks.empty() ? uniform_rank : ranks[(size_t) i];
            if (r_eff <= 0) continue;  // layer keeps the original fused rows
            const size_t r_tot = (size_t) n_heads * r_eff;
            f1s[(size_t) i].resize(r_tot * hidden);
            f2s[(size_t) i].resize((size_t) hidden * r_tot);
            if (!rd(f1s[(size_t) i].data(), f1s[(size_t) i].size() * sizeof(float)) ||
                !rd(f2s[(size_t) i].data(), f2s[(size_t) i].size() * sizeof(float))) {
                err = "KV factor file: truncated payload"; return false;
            }
        }
        return true;
    };
    bool ok = true;
    if (rank_k > 0) ok = read_half(rank_k_l, rank_k, f1k, f2k);
    if (ok && rank_v > 0) ok = read_half(rank_v_l, rank_v, f1v, f2v);
    std::fclose(f);
    if (!ok) return false;
    // Materialize f2tk = f2k^T for the K half — only when the in-r-space
    // score path is active (STARLING_GRANITE_KVINR); otherwise the ~15 MB
    // transpose would sit unused in memory.
    if (rank_k > 0 && std::getenv("STARLING_GRANITE_KVINR")) {
        f2tk.resize(f2k.size());
        for (size_t i = 0; i < f2k.size(); ++i) {
            if (f2k[i].empty()) continue;
            const size_t cols = f2k[i].size() / (size_t) hidden;  // n_heads*rank
            f2tk[i].resize(f2k[i].size());
            for (size_t d = 0; d < (size_t) hidden; ++d)
                for (size_t j = 0; j < cols; ++j)
                    f2tk[i][j * (size_t) hidden + d] = f2k[i][d * cols + j];
        }
    }
    return true;
}

}  // namespace starling::ggml::granite
