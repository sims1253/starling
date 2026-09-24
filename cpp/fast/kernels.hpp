// kernels.hpp — typed dispatch helpers over the fast-engine shaders, plus
// the weight arena that places repacked matrices in device buffers.

#pragma once

#include "vk_runtime.hpp"
#include "weights.hpp"

#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace starling::fast {

// ---------------------------------------------------------------------------
// Weight arena: collect host blobs, then place them into a few device
// buffers (each <= maxStorageBufferRange and <= 256 MiB) with aligned
// offsets and upload them once.
// ---------------------------------------------------------------------------
class Arena {
public:
    using Id = size_t;
    Id add(std::vector<uint32_t> words);
    Id add_f32(const std::vector<float>& v);
    bool finalize(vk::Context& ctx, std::string& err);
    vk::Ref ref(Id id) const;
    size_t total_bytes() const { return total_; }

private:
    struct Blob { std::vector<uint32_t> words; size_t buffer = 0; VkDeviceSize off = 0, bytes = 0; };
    std::vector<Blob> blobs_;
    std::vector<std::unique_ptr<vk::Buffer>> buffers_;
    size_t total_ = 0;
};

// A weight matrix resident on the device.
struct GMat {
    GpuFmt fmt = GpuFmt::F16;
    uint32_t N = 0, K = 0;
    Arena::Id q = 0, s = 0;
    bool has_s = false;
};
GMat arena_matrix(Arena& a, HostMatrix&& m);

// Run f(i) for i in [0, n) on up to hardware_concurrency threads.
void parallel_for(size_t n, const std::function<void(size_t)>& f);

// Weight repacking jobs run in parallel at load, then land in the arena in
// the order they were added (deterministic layout).
class PackJobs {
public:
    using Fn = std::function<bool(HostMatrix&, std::string&)>;
    // Result becomes a resident matrix written to *dst.
    void add(GMat* dst, Fn fn) { jobs_.push_back({dst, nullptr, std::move(fn)}); }
    // Result is left in *sink for the caller (e.g. to split into chunks).
    void add_host(HostMatrix* sink, Fn fn) { jobs_.push_back({nullptr, sink, std::move(fn)}); }
    bool run(Arena& ar, std::string& err);

private:
    struct Job { GMat* dst; HostMatrix* sink; Fn fn; HostMatrix out; std::string err; bool ok = false; };
    std::deque<Job> jobs_;
};

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

struct TileCfg { uint32_t BM = 64, BN = 128, TM = 4, TN = 8; };

class Kernels {
public:
    bool init(vk::Context& ctx, std::string& err);

    // Pick GEMM tile and GEMV row counts for this device by timing the
    // candidates on synthetic weights at the engines' real shapes. Results
    // are cached in $STARLING_FAST_CACHE_DIR (per device + driver) and the
    // search runs when that directory is set or STARLING_FAST_TUNE=1.
    // Explicit STARLING_FAST_TILE / STARLING_FAST_GEMV_ROWS win.
    bool autotune(std::string& err);
    vk::Context& ctx() { return *ctx_; }

    bool gemm(vk::Recording& rec, const GemmCall& c, std::string& err);
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
    };
    bool softmax(vk::Recording& rec, bool relpos, uint32_t rows, uint32_t heads,
                 const SoftmaxArgs& a, vk::Ref S, vk::Ref BD, vk::Ref P, std::string& err);

    struct PkConvArgs { uint32_t C, Ti, Fi, To, Fo, K; };
    bool pk_conv(vk::Recording& rec, uint32_t op, const PkConvArgs& a, uint32_t gy, uint32_t gz,
                 vk::Ref in, vk::Ref w, vk::Ref b, vk::Ref shift, vk::Ref out, std::string& err);

    // gemv.comp (decode). `g` non-null fuses RMSNorm(x)·g; `state` non-null
    // makes the dispatch a no-op once generation is done.
    struct GemvArgs { uint32_t N = 0, K = 0, x_off = 0, y_off = 0; float eps = 0; uint32_t row0 = 0, has_bias = 0; };
    bool gemv(vk::Recording& rec, const Arena& ar, const GMat& w, vk::Ref x, vk::Ref y, vk::Ref g,
              vk::Ref state, vk::Ref bias, uint32_t epi, GemvArgs a, std::string& err);
    uint32_t gemv_rows_max = 32, gemv_min_wgs = 256;
    uint32_t gemv_rows(uint32_t N);    // rows per workgroup for an N-row GEMV

    const vk::Buffer& dummy() const { return dummy_; }
    vk::Ref or_dummy(const vk::Ref& r) const { return r.buf ? r : vk::Ref(dummy_); }
    TileCfg tile;
    bool f16_math = false;
    bool coopmat_ = false;   // dispatch GEMMs to gemm_coop shaders
    bool rsplit_on_ = true;  // RSPLIT row slots to pad GEMV workgroups

private:
    vk::Context* ctx_ = nullptr;
    vk::Buffer dummy_;
};

inline uint32_t round_up(uint32_t v, uint32_t m) { return (v + m - 1) / m * m; }
inline uint32_t ceil_div(uint32_t v, uint32_t m) { return (v + m - 1) / m; }

} // namespace starling::fast
