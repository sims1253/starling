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
#include <string>
#include <vector>

namespace starling::ggml::granite {

struct KVFactors {
    int n_layers = 0, hidden = 0, n_heads = 0;
    int rank_k = 0, rank_v = 0;  // per-head ranks; 0 = unchanged path
    // Per layer, row-major float32 (layouts in the header comment above).
    std::vector<std::vector<float>> f1k, f2k, f1v, f2v;

    bool enabled() const { return rank_k > 0 || rank_v > 0; }

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
    unsigned int hdr[6];
    if (!rd(magic, 8) || !rd(hdr, sizeof(hdr))) {
        err = "KV factor file: truncated header"; std::fclose(f); return false;
    }
    if (std::string(magic, 8) != "STLGKVF1" || hdr[0] != 1) {
        err = "KV factor file: bad magic/version"; std::fclose(f); return false;
    }
    n_layers = (int) hdr[1]; hidden = (int) hdr[2]; n_heads = (int) hdr[3];
    rank_k = (int) hdr[4]; rank_v = (int) hdr[5];
    if (n_layers != exp_layers || hidden != exp_hidden || n_heads != exp_heads) {
        err = "KV factor file: model dims mismatch"; std::fclose(f); return false;
    }
    auto read_half = [&](int rank, std::vector<std::vector<float>>& f1s,
                         std::vector<std::vector<float>>& f2s) -> bool {
        const size_t r_tot = (size_t) n_heads * rank;
        f1s.resize(n_layers); f2s.resize(n_layers);
        for (int i = 0; i < n_layers; ++i) {
            f1s[i].resize(r_tot * hidden);
            f2s[i].resize((size_t) hidden * r_tot);
            if (!rd(f1s[i].data(), f1s[i].size() * sizeof(float)) ||
                !rd(f2s[i].data(), f2s[i].size() * sizeof(float))) {
                err = "KV factor file: truncated payload"; return false;
            }
        }
        return true;
    };
    bool ok = true;
    if (rank_k > 0) ok = read_half(rank_k, f1k, f2k);
    if (ok && rank_v > 0) ok = read_half(rank_v, f1v, f2v);
    std::fclose(f);
    return ok;
}

}  // namespace starling::ggml::granite
