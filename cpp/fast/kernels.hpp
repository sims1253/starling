// kernels.hpp — typed dispatch helpers over the fast-engine shaders, plus
// the weight arena that places repacked matrices in device buffers.

#pragma once

#include "vk_runtime.hpp"
#include "weights.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace starling::fast {

// ---------------------------------------------------------------------------
// Weight arena: every weight of an engine, placed into a few device buffers
// (each <= maxStorageBufferRange and <= 256 MiB) at aligned offsets.
//
// Small blobs (norms, biases, tables computed at load) are added with their
// bytes. Matrices are added as plans (MatSpec): their format and size follow
// from the source types alone, so finalize() can allocate the device memory
// first and then repack every row in parallel straight into it — into the
// mapped memory itself on UMA devices, through one staging buffer at a time
// elsewhere. The host never holds a repacked copy of the weights, and the
// GGUF pages a finished matrix was read from are dropped again when they come
// from a file mapping (release_sources).
// ---------------------------------------------------------------------------
struct PackedTensor;

// Rows of one source landing in a planned matrix.
struct MatPart {
    int type = 0;                       // ggml type of the source rows
    const void* data = nullptr;         // first source row
    uint32_t rows = 0, K = 0;           // K: source row length (must match the spec's)
    RowMap dst;                         // destination row of source row i
    std::shared_ptr<const std::vector<uint32_t>> table;   // owns dst.table when set
    // STARLING_FAST_PACKED: rows [packed_row0, + rows) of a pre-packed tensor
    // (legacy W4/W8 layouts) are copied instead of repacked.
    const PackedTensor* packed = nullptr;
    uint32_t packed_row0 = 0;
};

struct MatSpec {
    std::string name;                   // for messages
    uint32_t N = 0, K = 0;              // destination shape (the parts cover [0, N))
    std::vector<MatPart> parts;
    float scale = 1.0f;                 // folded into the group scales (see pack_gpu_rows)
};

// Builders. tensor_rows: a 2-D ggml tensor (ne0 = K) as [N][K];
// raw_rows: N rows of K of `type` at `data` (conv weights with leading
// singleton dims); row_range: rows [r0, r0 + n) of a one-part spec.
MatSpec tensor_rows(const ggml_tensor* t, const std::string& name);
MatSpec raw_rows(int type, const void* data, uint32_t N, uint32_t K, const std::string& name);
MatSpec row_range(const MatSpec& m, uint32_t r0, uint32_t n);
// b's rows below a's (K must match).
MatSpec spec_concat(MatSpec a, const MatSpec& b);
// a0 b0 a1 b1 ...: the paired GEMM/GEMV epilogues own both halves of each
// gated unit (a.N == b.N).
MatSpec spec_interleave(MatSpec a, const MatSpec& b);
// Output row r = input row order[r] (one-part specs).
MatSpec spec_permute(MatSpec m, const std::vector<uint32_t>& order);

// The format a MatSpec repacks to (parts of different formats all go W8),
// with the per-part decisions applied to `m` (packed overrides dropped for
// parts that must be requantized).
bool plan_matrix(MatSpec& m, GpuFmt& fmt, LayoutDesc& layout, std::string& err);
// Repack rows [row0, row0 + rows) of part `pi` of a planned spec into the
// matrix whose codes start at q and scales at s.
bool fill_spec_rows(const MatSpec& m, GpuFmt fmt, size_t pi, uint32_t row0, uint32_t rows,
                    uint32_t* q, uint32_t* s, std::string& err);
// Plan and repack a spec into host memory (tests; the engines go through
// Arena, straight into device memory).
bool materialize(MatSpec m, HostMatrix& out, std::string& err);

struct GMat;

class Arena {
public:
    using Id = size_t;
    Id add(std::vector<uint32_t> words);
    Id add_f32(const std::vector<float>& v);
    // Plan a matrix; its rows are repacked during finalize().
    bool add_matrix(MatSpec m, GMat& out, std::string& err);
    // A blob of `words` whose bytes `fn` writes during finalize(), in
    // parallel with the repacks and straight into device memory (load-time
    // transforms too irregular for a MatSpec: conv weight reorders).
    using Producer = std::function<bool(uint32_t* dst, std::string& err)>;
    Id add_deferred(size_t words, Producer fn);
    // Drop the source pages of each matrix once it is repacked (madvise;
    // only for sources in a private file mapping, which refaults them from
    // the file if anything reads them again).
    void release_sources(bool on) { release_sources_ = on; }
    bool finalize(vk::Context& ctx, std::string& err);
    vk::Ref ref(Id id) const;
    size_t total_bytes() const { return total_; }

private:
    struct Blob {
        std::vector<uint32_t> words;    // host bytes (empty for planned matrices)
        Producer fn;                    // deferred blobs
        VkDeviceSize bytes = 0;
        size_t group = 0;               // > 0: blobs placed in one buffer from here (codes + scales)
        size_t buffer = 0;
        VkDeviceSize off = 0;
    };
    struct Planned { MatSpec spec; GpuFmt fmt; Id q, s; };
    struct Unit { size_t mat, part; uint32_t row0, rows; };
    bool fill_unit(const Unit& u, const std::vector<uint8_t*>& base, std::string& err) const;
    std::vector<Blob> blobs_;
    std::vector<Planned> mats_;
    std::vector<std::unique_ptr<vk::Buffer>> buffers_;
    size_t total_ = 0;
    bool release_sources_ = false;
};

// A weight matrix resident on the device.
struct GMat {
    GpuFmt fmt = GpuFmt::F16;
    uint32_t N = 0, K = 0;
    Arena::Id q = 0, s = 0, x = 0;   // x: per-row super scales (u8super layouts)
    bool has_s = false, has_x = false;
    LayoutDesc layout;               // packed-file descriptor (Phase 1 dispatch)
};
GMat arena_matrix(Arena& a, HostMatrix&& m);
// An F16 [N][K] matrix whose f32 values (N*K, row-major) `make` computes
// during finalize (Arena::add_deferred); converted like pack_f16.
GMat deferred_f16(Arena& a, uint32_t N, uint32_t K,
                  std::function<bool(std::vector<float>&, std::string&)> make);

// Run f(i) for i in [0, n) on up to hardware_concurrency threads.
void parallel_for(size_t n, const std::function<void(size_t)>& f);

// ---------------------------------------------------------------------------
// GEMM (gemm.comp). Field names follow the shader's push-constant block.
// ---------------------------------------------------------------------------
enum class Epi : uint32_t { F32 = 0, F16 = 1, Residual = 2, Glu = 3, SwiGlu = 4, Quv = 5 };
enum class Act : uint32_t { None = 0, Silu = 1, Relu = 2, Gelu = 3 };

struct GemmArgs {
    uint32_t M = 0, N = 0, K = 0;
    uint32_t lda = 0, ldb = 0, ldc = 0;
    uint32_t a_off = 0, b_off = 0, c_off = 0;
    uint32_t nb_lo = 1;
    uint32_t sa_hi = 0, sa_lo = 0, sb_hi = 0, sb_lo = 0, sc_hi = 0, sc_lo = 0;
    uint32_t gqa = 1;
    uint32_t row_valid = 0xffffffffu;
    float alpha = 1.0f;
    uint32_t mdiv = 0xffffffffu, ldc_mhi = 0;
    uint32_t ldc_n = 1;
    uint32_t bias_mod = 1;
    uint32_t cv_cin = 0, cv_ti = 0, cv_fi = 0, cv_to = 0, cv_fo = 0;
    uint32_t causal = 0xffffffffu;   // attention: query m sees keys <= m + causal (gemm.comp)
};
static_assert(sizeof(GemmArgs) <= vk::kPushBytes, "gemm push constants");

// Operand kind of B (selects the shader variant).
enum class BKind { W4, W8, F16, F16T };

struct GemmCall {
    GemmArgs a;
    BKind b = BKind::F16;
    Epi epi = Epi::F32;
    Act act = Act::None;
    uint32_t bias_mode = 0;      // 0 none, 1 per column, 2 per row, 3 row-periodic table
    bool a_conv = false;         // A is an implicit 3x3/s2 conv im2col (see gemm.comp)
    uint32_t batch = 1;
    vk::Ref A, Bq, Bs, C, bias, bias2, C2;
};

// BK: k per tile (32 = one quant group; 16 / 8 trade barriers for
// shared memory, i.e. resident workgroups).
struct TileCfg { uint32_t BM = 64, BN = 128, TM = 4, TN = 8, BK = 32; };

class Kernels {
public:
    bool init(vk::Context& ctx, std::string& err);

    // Pick GEMM tile and GEMV row counts for this device by timing the
    // candidates on synthetic weights at the engines' real shapes. Results
    // are cached in $STARLING_FAST_CACHE_DIR (per device + driver) and the
    // search runs when that directory is set or STARLING_FAST_TUNE=1.
    // Explicit STARLING_FAST_TILE / STARLING_FAST_GEMV_ROWS win.
    bool autotune(std::string& err);
    // Isolated single-kernel probes (STARLING_FAST_MICRO=...; see kernels.cpp):
    // print a throughput figure to stderr, touch no engine state.
    bool micro(const char* mi, std::string& err);
    vk::Context& ctx() { return *ctx_; }

    bool gemm(vk::Recording& rec, const GemmCall& c, std::string& err);
    // Whether gemm() would take the subgroup-broadcast kernel for `c`.
    bool sg_eligible(const GemmCall& c) const;
    // Convenience: activation [M, K] (f16) times a resident weight matrix.
    bool gemm_w(vk::Recording& rec, const Arena& ar, const GMat& w, vk::Ref A, uint32_t M,
                uint32_t lda, vk::Ref C, uint32_t ldc, Epi epi, Act act, vk::Ref bias,
                float alpha, uint32_t row_valid, std::string& err, vk::Ref bias2 = {},
                vk::Ref C2 = {});

    // norm.comp: rows x D.
    bool norm(vk::Recording& rec, uint32_t mode, uint32_t rows, uint32_t D, vk::Ref x,
              uint32_t ld_x, vk::Ref g, vk::Ref b, vk::Ref g2, vk::Ref b2, vk::Ref out,
              uint32_t ld_o, float eps, float eps2, std::string& err);

    struct SoftmaxArgs {
        uint32_t T, ld_s, ld_b, ld_p, sh_s, sh_b, sh_p;
        float scale;
        uint32_t key_valid, causal_off, T_rel;
        // Sliding window over query blocks (softmax.comp); win 0 = off.
        uint32_t win = 0, blk_q = 0, nb_h = 1, blk0 = 0;
    };
    bool softmax(vk::Recording& rec, bool relpos, uint32_t rows, uint32_t heads,
                 const SoftmaxArgs& a, vk::Ref S, vk::Ref BD, vk::Ref P, std::string& err);

    struct PkConvArgs { uint32_t C, Ti, Fi, To, Fo, K; };
    bool pk_conv(vk::Recording& rec, uint32_t op, const PkConvArgs& a, uint32_t gy, uint32_t gz,
                 vk::Ref in, vk::Ref w, vk::Ref b, vk::Ref shift, vk::Ref out, std::string& err);

    // gemv.comp (decode). `g` non-null fuses RMSNorm(x)·g; `state` non-null
    // makes the dispatch a no-op once generation is done.
    struct GemvArgs { uint32_t N = 0, K = 0, x_off = 0, y_off = 0; float eps = 0; uint32_t row0 = 0, has_bias = 0;
                      // M-token variants (GEMV_M): token-1 x/y offsets, in
                      // floats/elements (x_off2 multiple of 4). gemv.comp's
                      // push-constant block declares them for every variant,
                      // so the block size is the same for all; others ignore them.
                      uint32_t x_off2 = 0, y_off2 = 0; };
    bool gemv(vk::Recording& rec, const Arena& ar, const GMat& w, vk::Ref x, vk::Ref y, vk::Ref g,
              vk::Ref state, vk::Ref bias, uint32_t epi, GemvArgs a, std::string& err);
    uint32_t gemv_rows_max = 32, gemv_min_wgs = 256;
    uint32_t gemv_tgt_wgs = 384;   // grow rows while N/rows stays above this
    uint32_t gemv_rows(uint32_t N, uint32_t K);    // rows per workgroup for an N x K GEMV
    uint32_t gemv_rsplit_target(uint32_t lanes) const;   // row slots padding lanes to a subgroup
    uint32_t gemv_rpp_min = 0;   // min rows per row slot (0: off); STARLING_FAST_GEMV_RPP

    const vk::Buffer& dummy() const { return dummy_; }
    vk::Ref or_dummy(const vk::Ref& r) const { return r.buf ? r : vk::Ref(dummy_); }
    TileCfg tile;
    bool f16_math = false;
    bool w4_unpack_ = false;    // W4 GEMV via unpackUnorm4x8 (gemv_w4u)
    bool w8_unpack_ = false;    // W8 GEMV via xor + unpackUnorm4x8 (gemv_w8u)
    const char* gemv_shader(GpuFmt f) const;
    bool tile_pinned_ = false;  // STARLING_FAST_TILE given: no per-op tile heuristics
    bool rsplit_on_ = true;  // RSPLIT row slots to pad GEMV workgroups
    // Encoder/prefill GEMMs through gemm_sg.comp (subgroup-broadcast weights,
    // no shared memory) instead of gemm.comp's shared tiles; sg_tex_: its A
    // operand through the texture path where the view limits allow.
    bool sg_gemm_ = false;
    bool sg_tex_ = true;
    uint32_t sg_texel_max_ = 0;   // texels per A view (maxTexelBufferElements; tests may lower it)

private:
    bool gemm_sg(vk::Recording& rec, const GemmCall& c, std::string& err);
    vk::Context* ctx_ = nullptr;
    vk::Buffer dummy_;
};

inline uint32_t round_up(uint32_t v, uint32_t m) { return (v + m - 1) / m * m; }
inline uint32_t ceil_div(uint32_t v, uint32_t m) { return (v + m - 1) / m; }

} // namespace starling::fast
