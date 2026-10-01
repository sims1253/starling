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
// File formats (little-endian), documented here and in the exporters:
//   v1 (uniform ranks, export_kv_lowrank*.py):
//     char magic[8] = "STLGKVF1"
//     u32 version = 1, u32 n_layers, u32 hidden, u32 n_heads
//     u32 rank_k, u32 rank_v            (0 = half unchanged)
//   v3 (per-layer ranks, export_kv_lowrank_selective.py):
//     char magic[8] = "STLGKVF3"
//     u32 version = 3, u32 n_layers, u32 hidden, u32 n_heads
//     u32 rank_k[n_layers], u32 rank_v[n_layers]   (0 = layer half unchanged)
//   payload, both versions, layer-major:
//     per layer: f32 f1k[n_heads*rank_k*hidden], f32 f2k[hidden*n_heads*rank_k],
//                f32 f1v[...], f32 f2v[...]    (each half only when its rank > 0)
// Every rank must lie in [0, head_dim] (head_dim = hidden / n_heads); the
// file must end exactly after the last payload.
// f1 rows are per-head C_h stacked ([n_heads*rank, hidden], row-major);
// f2 is the block-diagonal expansion ([hidden, n_heads*rank], row-major).
#pragma once

#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <memory>
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
    // In-r-space scores, decided once at load (f2tk is materialized iff
    // set) so the encoder can never take the in-r branch without f2tk.
    bool in_r = false;

    bool enabled() const { return rank_k > 0 || rank_v > 0; }
    int rank_layer_k(int li) const {
        return rank_k_l.empty() ? rank_k : rank_k_l[(size_t) li];
    }
    int rank_layer_v(int li) const {
        return rank_v_l.empty() ? rank_v : rank_v_l[(size_t) li];
    }

    // `path` is read fully into host memory; buffers must outlive every graph
    // that references them (they live in GraniteModel, which does).
    // `want_in_r` requests the in-r-space score path (STARLING_GRANITE_KVINR).
    bool load(const char* path, int exp_layers, int exp_hidden, int exp_heads,
              bool want_in_r, std::string& err);
};

inline bool KVFactors::load(const char* path, int exp_layers, int exp_hidden,
                            int exp_heads, bool want_in_r, std::string& err) {
    *this = KVFactors{};
    const std::string where = std::string(" (") + path + ")";
    std::unique_ptr<FILE, int (*)(FILE*)> fp(std::fopen(path, "rb"), &std::fclose);
    if (!fp) { err = "cannot open KV factor file" + where; return false; }
    FILE* f = fp.get();
    auto rd = [&](void* dst, size_t n) -> bool {
        return std::fread(dst, 1, n, f) == n;
    };
    auto fail = [&](const std::string& what) {
        err = "KV factor file: " + what + where;
        *this = KVFactors{};
        return false;
    };
    char magic[8];
    if (!rd(magic, 8)) return fail("truncated header");
    const bool v3 = std::string(magic, 8) == "STLGKVF3";
    if (!v3 && std::string(magic, 8) != "STLGKVF1") return fail("bad magic");
    uint32_t common[4];
    if (!rd(common, sizeof(common))) return fail("truncated header");
    if (common[0] != (v3 ? 3u : 1u)) return fail("bad version");
    if (common[1] != (uint32_t) exp_layers || common[2] != (uint32_t) exp_hidden ||
        common[3] != (uint32_t) exp_heads || exp_heads <= 0 || exp_hidden % exp_heads)
        return fail("model dims mismatch");
    n_layers = exp_layers; hidden = exp_hidden; n_heads = exp_heads;
    const uint32_t head_dim = (uint32_t) (hidden / n_heads);
    // Ranks are u32 on disk; anything above head_dim (including values that
    // would wrap negative as int) is rejected before any payload is sized.
    auto rank_ok = [&](uint32_t r) { return r <= head_dim; };
    if (v3) {
        std::vector<uint32_t> rk((size_t) n_layers), rv((size_t) n_layers);
        if (!rd(rk.data(), sizeof(uint32_t) * rk.size()) ||
            !rd(rv.data(), sizeof(uint32_t) * rv.size()))
            return fail("truncated rank tables");
        for (int i = 0; i < n_layers; ++i) {
            if (!rank_ok(rk[(size_t) i]) || !rank_ok(rv[(size_t) i]))
                return fail("layer " + std::to_string(i) + " rank exceeds head_dim " +
                            std::to_string(head_dim));
            rank_k_l.push_back((int) rk[(size_t) i]);
            rank_v_l.push_back((int) rv[(size_t) i]);
            rank_k = std::max(rank_k, rank_k_l.back());
            rank_v = std::max(rank_v, rank_v_l.back());
        }
    } else {
        uint32_t rk_rv[2];
        if (!rd(rk_rv, sizeof(rk_rv))) return fail("truncated header");
        if (!rank_ok(rk_rv[0]) || !rank_ok(rk_rv[1]))
            return fail("rank exceeds head_dim " + std::to_string(head_dim));
        rank_k = (int) rk_rv[0]; rank_v = (int) rk_rv[1];
    }
    // Payload is layer-major: each layer's K pair, then its V pair (the
    // order every exporter writes).
    f1k.resize((size_t) n_layers); f2k.resize((size_t) n_layers);
    f1v.resize((size_t) n_layers); f2v.resize((size_t) n_layers);
    auto read_pair = [&](int li, int rank, std::vector<float>& f1,
                         std::vector<float>& f2) -> bool {
        if (rank <= 0) return true;  // layer keeps the original fused rows
        const size_t r_tot = (size_t) n_heads * (size_t) rank;
        f1.resize(r_tot * (size_t) hidden);
        f2.resize((size_t) hidden * r_tot);
        return rd(f1.data(), f1.size() * sizeof(float)) &&
               rd(f2.data(), f2.size() * sizeof(float));
    };
    for (int i = 0; i < n_layers; ++i) {
        const size_t li = (size_t) i;
        if (!read_pair(i, rank_layer_k(i), f1k[li], f2k[li]))
            return fail("truncated K payload in layer " + std::to_string(i));
        if (!read_pair(i, rank_layer_v(i), f1v[li], f2v[li]))
            return fail("truncated V payload in layer " + std::to_string(i));
    }
    if (std::fgetc(f) != EOF) return fail("trailing bytes after the payload");
    // Materialize f2tk = f2k^T for the K half — only when the in-r-space
    // score path is requested; otherwise the ~15 MB transpose would sit
    // unused in memory.
    in_r = want_in_r && rank_k > 0;
    if (in_r) {
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
