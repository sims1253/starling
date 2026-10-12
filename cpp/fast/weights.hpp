// weights.hpp — repack GGUF tensors into the fast engines' native layouts.
//
// The GGUF file stays the distribution format; at load every weight matrix
// is rewritten once into a layout chosen for the kernels that read it:
//
//   GPU W4   4-bit, groups of 32 along K, one (f16 scale, f16 offset) pair
//            per group: w = s*q + o. Nibbles are sequential inside each
//            32-bit word and stored separately from the scales, so a
//            workgroup's weight tile is a few contiguous 16-byte loads.
//            Lossless for Q4_0; Q4_K keeps its 6-bit sub-block scale/min
//            products (rounded once to f16).
//   GPU W8   int8, groups of 32 with an f16 scale per 16 weights. Lossless
//            for Q8_0 and Q6_K.
//   GPU F16  plain f16 rows (F16 weights, BF16/F32 converted).
//   CPU Q8   int8 rows with an f32 scale per 32 — the decoder GEMV format
//            (lossless for Q8_0 / Q4_0).
//
// Types without a native mapping are dequantized with ggml's type traits and
// requantized to W8 / Q8 (near-lossless; reported once at load).

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "layout.hpp"

struct ggml_tensor;

namespace starling::fast {

enum class GpuFmt : uint32_t { W4 = 0, W8 = 1, F16 = 2 };

const char* fmt_name(GpuFmt f);

// A repacked matrix [N rows][K cols] in host memory, ready for upload.
struct HostMatrix {
    GpuFmt fmt = GpuFmt::F16;
    uint32_t N = 0, K = 0;
    std::vector<uint32_t> q;   // W4: N*K/8 words; W8: N*K/4 words; F16: N*K/2 words
    std::vector<uint32_t> s;   // W4/W8: N*K/32 words (two f16); F16: empty
    std::vector<uint32_t> x;   // packed-file layouts: per-row super scales (words)
    LayoutDesc layout;         // packed-file descriptor; GGUF repacks carry the
                               // legacy one (w4g32asym / w8g16sym; F16 unused)
    bool lossless = true;
    size_t bytes() const { return (q.size() + s.size() + x.size()) * 4; }
};

// ---- Row-level repacking (the load path) -----------------------------------
// The engines repack straight into device memory: the format of every
// matrix is planned from its source type before any byte is touched, then
// rows are converted in parallel into their final place.

// 32-bit words per row of K weights: codes (q) and group scales (s).
uint32_t q_row_words(GpuFmt f, uint32_t K);
uint32_t s_row_words(GpuFmt f, uint32_t K);

// The format rows of `ggml_type` repack to at row length K: W4 from
// Q4_0/Q4_1/Q4_K, W8 from Q8_0/Q6_K, F16 from F16/F32/BF16, W8 requantized
// from anything else (force_w8: W8 whatever the type). `lossless` is false
// where values round. Fails for shapes or types the formats cannot hold.
bool plan_gpu_rows(int ggml_type, uint32_t K, bool force_w8, GpuFmt& fmt, bool& lossless,
                   std::string& err);

// Destination row of source row i: first + i * step, or table[i].
struct RowMap {
    uint64_t first = 0, step = 1;
    const uint32_t* table = nullptr;
    uint64_t operator()(uint64_t i) const { return table ? table[i] : first + i * step; }
};

// Repack n rows (ggml row-major, ggml_row_size apart) to `fmt` rows of the
// matrix whose codes start at q and scales at s, source row i landing at
// row map(i). `scale` != 1 folds a multiplier into the group scales (W4/W8)
// or the weights (F16), each re-rounded once to f16 exactly as scale_matrix
// does. F16 needs even K here (odd K: pack_gpu_matrix_raw). Thread-safe;
// returns false only for a type ggml cannot dequantize.
bool pack_gpu_rows(int ggml_type, const void* src, uint32_t n, uint32_t K, GpuFmt fmt, float scale,
                   uint32_t* q, uint32_t* s, const RowMap& map, std::string& err);

// Words of f16 pairs (W4/W8 group scales, F16 weights) times f, each half
// re-rounded to f16.
void scale_f16_words(uint32_t* w, size_t n, float f);

// Repack a 2-D ggml tensor (ne0 = K, ne1 = N) into `out` (float tensors
// become F16).
bool pack_gpu_matrix(const ggml_tensor* t, HostMatrix& out, std::string& err);

// Same for raw data of a given ggml type (row-major, N rows of K).
bool pack_gpu_matrix_raw(int ggml_type, const void* data, uint32_t N, uint32_t K,
                         HostMatrix& out, std::string& err);

// Repack into W8 whatever the source type (lossless for Q8_0 / Q6_K,
// requantized otherwise): the common format when the two halves of a fused
// row pair come in different types (Granite: Q5_K gate, Q4_K up).
bool pack_gpu_matrix_w8(const ggml_tensor* t, HostMatrix& out, std::string& err);

// Multiply every weight by `f` through the per-group scales (W4 scale and
// offset, W8 scales; F16 weights directly), each re-rounded once to f16:
// folds a constant output multiplier into the matrix. Legacy layouts only.
bool scale_matrix(HostMatrix& m, float f, std::string& err);

// Row permutation: out row r = in row src[r] (exact for every format).
void permute_rows(HostMatrix& m, const std::vector<uint32_t>& src);

// Append the rows of `b` below `a` (formats and K must match).
bool concat_rows(HostMatrix& a, const HostMatrix& b, std::string& err);

// Dequantize any 2-D/1-D tensor to f32 (row-major, ne0 fastest).
bool tensor_to_f32(const ggml_tensor* t, std::vector<float>& out, std::string& err);

// f32 -> packed f16 words, (n + 1) / 2 of them (odd n: the last high half is 0).
std::vector<uint32_t> pack_f16(const float* x, size_t n);

// CPU decoder weights: int8 [N][K] + f32 scale per 32.
struct CpuQ8 {
    uint32_t N = 0, K = 0;
    std::vector<int8_t> q;
    std::vector<float> s;
    bool lossless = true;
};
bool pack_cpu_q8(const ggml_tensor* t, CpuQ8& out, std::string& err);

} // namespace starling::fast
