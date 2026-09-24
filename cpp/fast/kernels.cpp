// kernels.cpp — see kernels.hpp.

#include "kernels.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace starling::fast {

// ---------------------------------------------------------------------------
// Arena
// ---------------------------------------------------------------------------

Arena::Id Arena::add(std::vector<uint32_t> words) {
    Blob b;
    b.bytes = std::max<size_t>(words.size() * 4, 4);
    b.words = std::move(words);
    blobs_.push_back(std::move(b));
    return blobs_.size() - 1;
}

Arena::Id Arena::add_f32(const std::vector<float>& v) {
    std::vector<uint32_t> w(v.size());
    std::memcpy(w.data(), v.data(), v.size() * 4);
    return add(std::move(w));
}

bool Arena::finalize(vk::Context& ctx, std::string& err) {
    const auto& info = ctx.info();
    VkDeviceSize cap = 256ull << 20;
    cap = std::min<VkDeviceSize>(cap, info.max_storage_range);
    if (info.max_alloc) cap = std::min<VkDeviceSize>(cap, info.max_alloc);
    const VkDeviceSize align = std::max<VkDeviceSize>(info.min_storage_align, 256);
    // First-fit placement in insertion order.
    std::vector<VkDeviceSize> sizes;
    for (Blob& b : blobs_) {
        if (b.bytes > cap) {
            err = "fast engine: tensor of " + std::to_string(b.bytes >> 20) +
                  " MiB exceeds the device's storage-buffer range";
            return false;
        }
        if (sizes.empty() || (sizes.back() + align - 1) / align * align + b.bytes > cap) sizes.push_back(0);
        VkDeviceSize off = (sizes.back() + align - 1) / align * align;
        b.buffer = sizes.size() - 1;
        b.off = off;
        sizes.back() = off + b.bytes;
    }
    buffers_.clear();
    for (VkDeviceSize sz : sizes) {
        auto buf = std::make_unique<vk::Buffer>();
        if (!ctx.create_buffer(*buf, sz, vk::Mem::Device, err)) return false;
        buffers_.push_back(std::move(buf));
        total_ += sz;
    }
    for (Blob& b : blobs_) {
        if (!ctx.upload(*buffers_[b.buffer], b.off, b.words.data(), b.words.size() * 4, err))
            return false;
        std::vector<uint32_t>().swap(b.words);   // release the host copy
    }
    return true;
}

vk::Ref Arena::ref(Id id) const {
    const Blob& b = blobs_[id];
    return vk::Ref(*buffers_[b.buffer], b.off, b.bytes);
}

GMat arena_matrix(Arena& a, HostMatrix&& m) {
    GMat g;
    g.fmt = m.fmt;
    g.N = m.N;
    g.K = m.K;
    g.q = a.add(std::move(m.q));
    if (!m.s.empty()) { g.s = a.add(std::move(m.s)); g.has_s = true; }
    return g;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

bool Kernels::init(vk::Context& ctx, std::string& err) {
    ctx_ = &ctx;
    if (!ctx.create_buffer(dummy_, 256, vk::Mem::Device, err)) return false;
    // 256-thread tiles need maxComputeWorkGroupInvocations >= 256 (the spec
    // only guarantees 128); fall back to 8x4 register tiles otherwise.
    if (ctx.info().max_wg_invocations < 256) tile = TileCfg{64, 64, 4, 4};
    // Packed-f16 products (f32 accumulation per 32-deep slice): opt-in until
    // validated per device (STARLING_FAST_F16=1).
    if (const char* e = std::getenv("STARLING_FAST_F16")) f16_math = ctx.info().f16 && e[0] == '1';
    if (const char* e = std::getenv("STARLING_FAST_TILE")) {
        unsigned bm, bn, tm, tn;
        if (std::sscanf(e, "%u,%u,%u,%u", &bm, &bn, &tm, &tn) == 4) tile = TileCfg{bm, bn, tm, tn};
    }
    return true;
}

bool Kernels::gemm(vk::Recording& rec, const GemmCall& c, std::string& err) {
    std::string name = c.b == BKind::W4 ? "gemm_w4" : c.b == BKind::W8 ? "gemm_w8"
                     : c.b == BKind::F16 ? "gemm_f16" : "gemm_f16t";
    if (f16_math) name += "_h";
    const TileCfg t = tile;
    const uint32_t wg = (t.BM / t.TM) * (t.BN / t.TN);
    const vk::Pipeline* p = ctx_->pipeline(
        name.c_str(), {wg, t.BM, t.BN, t.TM, t.TN, (uint32_t)c.epi, (uint32_t)c.act, c.bias_mode}, err);
    if (!p) return false;
    if ((c.b == BKind::W4 || c.b == BKind::W8) && c.a.K % 32) {
        err = "gemm: weight K must be a multiple of 32";
        return false;
    }
    if (c.a.lda % 8 || c.a.a_off % 8 || c.a.sa_hi % 8 || c.a.sa_lo % 8) {
        err = "gemm: A operand must be 8-element aligned";
        return false;
    }
    rec.dispatch(*p, {or_dummy(c.A), or_dummy(c.Bq), or_dummy(c.Bs), or_dummy(c.C),
                      or_dummy(c.bias), or_dummy(c.bias2), or_dummy(c.C2)},
                 &c.a, sizeof(c.a), ceil_div(c.a.M, t.BM), ceil_div(c.a.N, t.BN), c.batch);
    return true;
}

bool Kernels::gemm_w(vk::Recording& rec, const Arena& ar, const GMat& w, vk::Ref A, uint32_t M,
                     uint32_t lda, vk::Ref C, uint32_t ldc, Epi epi, Act act, vk::Ref bias,
                     float alpha, uint32_t row_valid, std::string& err, vk::Ref bias2,
                     vk::Ref C2) {
    GemmCall c;
    c.b = w.fmt == GpuFmt::W4 ? BKind::W4 : w.fmt == GpuFmt::W8 ? BKind::W8 : BKind::F16;
    c.epi = epi;
    c.act = act;
    c.bias_mode = bias.buf ? 1u : 0u;
    c.a.M = M; c.a.N = w.N; c.a.K = w.K;
    c.a.lda = lda; c.a.ldb = w.K; c.a.ldc = ldc;
    c.a.alpha = alpha;
    c.a.row_valid = row_valid;
    c.A = A;
    c.Bq = ar.ref(w.q);
    if (w.has_s) c.Bs = ar.ref(w.s);
    c.C = C;
    c.bias = bias;
    c.bias2 = bias2;
    c.C2 = C2;
    return gemm(rec, c, err);
}

bool Kernels::norm(vk::Recording& rec, uint32_t mode, uint32_t rows, uint32_t D, vk::Ref x,
                   uint32_t ld_x, vk::Ref g, vk::Ref b, vk::Ref g2, vk::Ref b2, vk::Ref out,
                   uint32_t ld_o, float eps, float eps2, std::string& err) {
    const uint32_t wg = 256;
    if (D % 2 || D > 16 * wg) { err = "norm: unsupported row length"; return false; }
    const vk::Pipeline* p = ctx_->pipeline("norm", {wg, mode}, err);
    if (!p) return false;
    struct { uint32_t D, ld_x, ld_o, x_off, o_off; float eps, eps2; } pc{D, ld_x, ld_o, 0, 0, eps, eps2};
    rec.dispatch(*p, {x, or_dummy(g), or_dummy(b), or_dummy(g2), or_dummy(b2), or_dummy(out)},
                 &pc, sizeof(pc), rows);
    return true;
}

bool Kernels::softmax(vk::Recording& rec, bool relpos, uint32_t rows, uint32_t heads,
                      const SoftmaxArgs& a, vk::Ref S, vk::Ref BD, vk::Ref P, std::string& err) {
    const vk::Pipeline* p = ctx_->pipeline("softmax", {256u, relpos ? 1u : 0u}, err);
    if (!p) return false;
    rec.dispatch(*p, {S, or_dummy(BD), P}, &a, sizeof(a), rows, heads);
    return true;
}

bool Kernels::pk_conv(vk::Recording& rec, uint32_t op, const PkConvArgs& a, uint32_t gy,
                      uint32_t gz, vk::Ref in, vk::Ref w, vk::Ref b, vk::Ref shift,
                      vk::Ref out, std::string& err) {
    const uint32_t wg = 64;
    const vk::Pipeline* p = ctx_->pipeline("pk_conv", {wg, op}, err);
    if (!p) return false;
    rec.dispatch(*p, {in, w, or_dummy(b), or_dummy(shift), out}, &a, sizeof(a),
                 ceil_div(a.C / 2, wg), gy, gz);
    return true;
}

} // namespace starling::fast
