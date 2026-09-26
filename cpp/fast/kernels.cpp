// kernels.cpp — see kernels.hpp.

#include "kernels.hpp"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <thread>
#include <chrono>
#include <fstream>
#include <random>
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
    // #325 preflight: refuse the load cleanly when the GPU cannot fit the
    // arena (budget query where the driver exposes it; no-op elsewhere).
    {
        uint64_t total = 0;
        for (const Blob& b : blobs_) total += b.bytes + 4096;
        std::string berr;
        if (!ctx.check_memory_budget(total, berr)) { err = berr; return false; }
    }
    buffers_.clear();
    for (VkDeviceSize sz : sizes) {
        auto buf = std::make_unique<vk::Buffer>();
        // Mapped uploads (plain memcpy) when device memory is CPU-cached
        // (phones: 8x faster than staging); staged copies where mapped device
        // memory is uncached for the CPU (desktop APUs).
        const char* mw = std::getenv("STARLING_FAST_MAPPED_WEIGHTS");
        const bool mapped = mw ? mw[0] == '1' : ctx.info().uma_cached;
        const vk::Mem kind = mapped ? vk::Mem::Device : vk::Mem::DeviceOnly;
        if (!ctx.create_buffer(*buf, sz, kind, err)) return false;
        buffers_.push_back(std::move(buf));
        total_ += sz;
    }
    // Uploads are independent copies (memcpy into mapped memory on UMA
    // devices), so they run in parallel; staged uploads serialize internally.
    std::atomic<bool> ok{true};
    std::string first_err;
    std::mutex mu;
    parallel_for(blobs_.size(), [&](size_t i) {
        Blob& b = blobs_[i];
        std::string e;
        if (ok && !ctx.upload(*buffers_[b.buffer], b.off, b.words.data(), b.words.size() * 4, e)) {
            std::lock_guard<std::mutex> lk(mu);
            if (ok.exchange(false)) first_err = e;
        }
        std::vector<uint32_t>().swap(b.words);   // release the host copy
    });
    if (!ok) { err = first_err; return false; }
    return true;
}

vk::Ref Arena::ref(Id id) const {
    const Blob& b = blobs_[id];
    return vk::Ref(*buffers_[b.buffer], b.off, b.bytes);
}

void parallel_for(size_t n, const std::function<void(size_t)>& f) {
    const size_t hw = std::max(1u, std::thread::hardware_concurrency());
    const size_t nt = std::min(n, std::min<size_t>(hw, 8));
    if (nt <= 1) {
        for (size_t i = 0; i < n; ++i) f(i);
        return;
    }
    std::atomic<size_t> next{0};
    std::vector<std::thread> th;
    for (size_t t = 0; t < nt; ++t)
        th.emplace_back([&] {
            for (size_t i; (i = next.fetch_add(1)) < n;) f(i);
        });
    for (auto& t : th) t.join();
}

bool PackJobs::run(Arena& ar, std::string& err) {
    std::vector<Job*> v;
    for (Job& j : jobs_) v.push_back(&j);
    parallel_for(v.size(), [&](size_t i) { v[i]->ok = v[i]->fn(v[i]->out, v[i]->err); });
    for (Job& j : jobs_) {
        if (!j.ok) { err = j.err.empty() ? "fast engine: weight repack failed" : j.err; return false; }
        if (j.dst) *j.dst = arena_matrix(ar, std::move(j.out));
        else *j.sink = std::move(j.out);
    }
    jobs_.clear();
    return true;
}

GMat arena_matrix(Arena& a, HostMatrix&& m) {
    GMat g;
    g.fmt = m.fmt;
    g.N = m.N;
    g.K = m.K;
    g.q = a.add(std::move(m.q));
    if (!m.s.empty()) { g.s = a.add(std::move(m.s)); g.has_s = true; }
    if (!m.x.empty()) { g.x = a.add(std::move(m.x)); g.has_x = true; }
    g.layout = m.layout;
    return g;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

bool Kernels::init(vk::Context& ctx, std::string& err) {
    ctx_ = &ctx;
    if (!ctx.create_buffer(dummy_, 256, vk::Mem::Device, err)) return false;
    // 256-thread tiles need maxComputeWorkGroupInvocations >= 256 (the spec
    // only guarantees 128); fall back to a 128-thread tile otherwise.
    if (ctx.info().max_wg_invocations < 256) tile = TileCfg{32, 64, 4, 4};
    // Packed-f16 products (f32 accumulation per 32-deep slice) wherever the
    // device has shaderFloat16: same FLEURS WER as f32 on both models.
    // STARLING_FAST_F16=0 forces f32 products.
    f16_math = ctx.info().f16;
    // PowerVR (DXT): f32 products measured faster than the packed-f16 path
    // (the unpack/pack around every product costs more than full-rate f32
    // FMAs save) — Pixel 10 Pro encoder 2246 vs 2315 ms. f32 is also the
    // numerically larger path, so quality gates are unaffected.
    if (ctx.info().vendor_id == 0x1010) f16_math = false;
    if (const char* e = std::getenv("STARLING_FAST_F16")) f16_math = ctx.info().f16 && e[0] == '1';
    const char* tune_env = std::getenv("STARLING_FAST_TUNE");
    const bool tune_forced = tune_env && tune_env[0] == '1';
    // Imagination (PowerVR): the synthetic ranking does not transfer to the
    // real encoders — isolated GEMM pairs rate 64,32,4,4 highest while the
    // Parakeet/MOSS encoders run fastest on 32,128,4,8 (a full 128-thread
    // subgroup with BN=128 halves weight re-reads), and short GEMV bursts
    // rate rows 64 highest while real decode wants 8 (measured on a Pixel
    // 10 Pro / Tensor G5 DXT-48-1536; see RESEARCH_LOG.md). Ship the measured
    // values for this vendor without running the synthetic tuner;
    // STARLING_FAST_TUNE=1 runs it anyway, STARLING_FAST_TILE / _GEMV_ROWS
    // override either. Other vendors keep the tuner's pick.
    if (ctx.info().vendor_id == 0x1010 && !tune_forced) {
        tile = TileCfg{32, 128, 4, 8};
        // Decode GEMV rows=16, pinned: measured best-or-tied at every MOSS
        // decode shape on the Pixel 10 Pro (N 2048..151936, K 2048/6144; the
        // rows=8 default cost 15-31 % on the small-N shapes, growing to 32
        // was neutral-to-worse; see RESEARCH_LOG P1-8). The old adaptive
        // growth (target 384 workgroups) optimizes for workgroup count, but
        // every workgroup re-reads the whole x vector — at rows=8 the x
        // traffic rivals the weights on the small-N GEMVs.
        gemv_rows_max = 16;
        gemv_tgt_wgs = ~0u;   // no adaptive growth
        gemv_min_wgs = 0;     // and no shrink-to-256-workgroups floor: rows=16
                              // measured best at N=2048 with just 128 WGs
    } else if (!autotune(err)) {
        return false;
    }
    // W4 GEMV nibble unpack through unpackUnorm4x8 (~5 ALU ops per 8
    // weights instead of ~16). Decode GEMVs on PowerVR are issue-bound, not
    // bandwidth-bound (W4 and W8 both run at ~33 G weights/s): MOSS decode
    // -12.7 % on the Pixel 10 Pro. Slower on RADV (38 -> 31 GB/s isolated),
    // so PowerVR only; STARLING_FAST_W4U=0/1 overrides.
    w4_unpack_ = ctx.info().vendor_id == 0x1010;
    if (const char* e = std::getenv("STARLING_FAST_W4U")) w4_unpack_ = e[0] == '1';
    // Diagnostics: STARLING_FAST_MICRO runs one isolated kernel probe.
    if (const char* mi = std::getenv("STARLING_FAST_MICRO"))
        if (!micro(mi, err)) return false;
    rsplit_on_ = !(std::getenv("STARLING_FAST_NORSPLIT") && std::getenv("STARLING_FAST_NORSPLIT")[0] == '1');
    if (const char* e = std::getenv("STARLING_FAST_TILE")) {
        unsigned bm, bn, tm, tn;
        if (std::sscanf(e, "%u,%u,%u,%u", &bm, &bn, &tm, &tn) == 4) {
            tile = TileCfg{bm, bn, tm, tn};
            tile_pinned_ = true;
        }
    }
    if (const char* e = std::getenv("STARLING_FAST_GEMV_ROWS")) {
        unsigned r = (unsigned)std::atoi(e);
        if (r >= 8 && r % 8 == 0) {
            gemv_rows_max = r;
            gemv_tgt_wgs = ~0u;   // pinned: no adaptive growth
        }
    }
    return true;
}

bool Kernels::gemm(vk::Recording& rec, const GemmCall& c, std::string& err) {
    std::string name = c.b == BKind::W4 ? "gemm_w4" : c.b == BKind::W8 ? "gemm_w8"
                     : c.b == BKind::F16 ? "gemm_f16" : "gemm_f16t";
    if (c.a_conv) {
        if (c.b != BKind::F16 || c.a.cv_cin % 8) { err = "gemm: conv mode needs f16 weights and cin % 8 == 0"; return false; }
        name = "gemm_conv";
    }
    if (f16_math) name += "_h";
    TileCfg t = tile;
    // Narrow outputs (attention PV at head_dim, tiny subsampling mixes)
    // waste most of a BN=128 tile's columns; a BN=64 tile of the same
    // proven family doubles the useful fraction. Tile shape only partitions
    // outputs — each element's K accumulation order is unchanged.
    if (c.a.N <= 64 && t.BN > 64 && !tile_pinned_) t = TileCfg{32, 64, 4, 4};
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

uint32_t Kernels::gemv_rows(uint32_t N) {
    uint32_t rows = gemv_rows_max;
    // Grow rows for large N: every workgroup re-reads the x vector once, so
    // at rows=8 the x traffic rivals the weights (ff_up: 12.6 MB x vs
    // 15.7 MB weights on MOSS; lm_head reads x 19k times). Keep at least
    // ~384 workgroups for occupancy and cap at 32 (48+ costs registers).
    while (rows < 32 && N / (rows * 2) >= gemv_tgt_wgs) rows *= 2;
    while (rows > 8 && N / rows < gemv_min_wgs) rows -= 8;
    return rows;
}

bool Kernels::gemv(vk::Recording& rec, const Arena& ar, const GMat& w, vk::Ref x, vk::Ref y,
                   vk::Ref g, vk::Ref state, vk::Ref bias, uint32_t epi, GemvArgs a,
                   std::string& err) {
    const char* name = w.fmt == GpuFmt::W4 ? (w4_unpack_ ? "gemv_w4u" : "gemv_w4") : w.fmt == GpuFmt::W8 ? "gemv_w8" : "gemv_f16";
    const uint32_t lanes = w.K / 32;      // one thread per 32-wide K group
    if (w.K % 32 || a.x_off % 4 || a.x_off2 % 4 || lanes > ctx_->info().max_wg_invocations ||
        lanes > 1024) {
        err = "gemv: unsupported K / x offset";
        return false;
    }
    // Enough workgroups to fill the GPU: fewer rows per workgroup for small N.
    uint32_t rows = gemv_rows(w.N);
    // Pad the workgroup to a full subgroup with parallel row slots: GPUs
    // issue whole subgroups, so lanes < subgroup_size strands issue slots
    // (W4 K=1024: ~7 GB/s at 32 threads, ~27 at 128 on PowerVR).
    uint32_t rsplit = 1;
    if (rsplit_on_) {
        const uint32_t target = std::max(32u, ctx_->info().subgroup_size);
        if (const char* e = std::getenv("STARLING_FAST_RSPLIT"))
            rsplit = std::min(8u, (uint32_t)std::atoi(e));
        else
            while (rsplit < 8 && lanes * rsplit < target && rows % (2 * rsplit) == 0u) rsplit *= 2;
    }
    // Each row slot owns ROWS / RSPLIT rows (an env pin may not divide).
    if (rsplit == 0 || rows % rsplit || lanes * rsplit > ctx_->info().max_wg_invocations) rsplit = 1;
    const vk::Pipeline* p = ctx_->pipeline(
        name, {lanes, rows, g.buf ? 1u : 0u, epi, state.buf ? 1u : 0u, rsplit, lanes * rsplit}, err);
    if (!p) return false;
    a.N = w.N;
    a.K = w.K;
    a.has_bias = bias.buf ? 1u : 0u;
    rec.dispatch(*p, {x, ar.ref(w.q), w.has_s ? ar.ref(w.s) : vk::Ref(dummy_), y, or_dummy(g),
                      or_dummy(state), or_dummy(bias)},
                 &a, sizeof(a), ceil_div(w.N, rows));
    return true;
}

namespace {

// Tuning results are process-wide: every engine on the device shares them.
struct TuneResult { bool done = false; TileCfg tile; uint32_t gemv_rows = 32; };
TuneResult& tune_result() { static TuneResult r; return r; }

double time_ms(vk::Recording& rec, std::string& err) {
    double best = 1e30;
    for (int i = 0; i < 3; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        if (!rec.submit_and_wait(err)) return -1.0;
        best = std::min(best, std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t0).count());
    }
    return best;
}

} // namespace

bool Kernels::micro(const char* mi, std::string& err) {
    // Default: isolated GEMV bandwidth probe, STARLING_FAST_MICRO=bits,N,K,reps
    // (bits 4 / 8 / 16) times a pure loop of decode GEMVs at a real shape and
    // prints the achieved GB/s — the honest per-kernel signal on a GPU where
    // per-dispatch timestamps are misattributed. Also idot / norm / alu below.
    if (std::strncmp(mi, "idot", 4) == 0) {
        // Integer-dot probe: packed 4x8 signed dot products (OpSDot), four
        // independent chains per thread. in = [iters, a, b, seed].
        // STARLING_FAST_MICRO=idot[,iters[,groups]]
        if (!ctx_->info().int_dot) {
            err = "idot probe: device lacks shaderIntegerDotProduct (VK_KHR_shader_integer_dot_product)";
            return false;
        }
        uint32_t iters = 65536, groups = 96;
        std::sscanf(mi, "idot,%u,%u", &iters, &groups);
        vk::Buffer ib, ob;
        std::vector<uint32_t> iv{iters, 0x02020202u, 0x03030303u, 0x12345678u};
        std::vector<uint32_t> ov(64 * groups, 0);
        auto upl = [&](vk::Buffer& b, const void* d, size_t bytes) {
            return ctx_->create_buffer(b, bytes, vk::Mem::Device, err) &&
                   ctx_->upload(b, 0, d, bytes, err);
        };
        if (!upl(ib, iv.data(), 16) || !upl(ob, ov.data(), ov.size() * 4)) return false;
        const vk::Pipeline* p = ctx_->pipeline("idot_probe", {}, err);
        if (!p) return false;
        // Correctness: iters = 0 -> out = 4 * (seed ^ SDot(a,b)) = 4 * (seed ^ 24).
        {
            iv[0] = 0;
            if (!ctx_->upload(ib, 0, iv.data(), 16, err)) return false;
            vk::Recording r0(*ctx_);
            r0.begin();
            r0.dispatch(*p, {vk::Ref(ib), vk::Ref(ob)}, nullptr, 0, groups);
            r0.end();
            if (!r0.submit_and_wait(err)) return false;
            std::vector<uint32_t> got(8, 0);
            if (!ctx_->download(ob, 0, got.data(), 32, err)) return false;
            bool ok = true;
            const uint32_t want = 4u * (0x12345678u ^ 24u);
            for (uint32_t g : got) ok = ok && g == want;
            std::fprintf(stderr, "[fast-micro] idot check: %s (want %08x, got %08x)\n",
                         ok ? "OK" : "FAIL", want, got[0]);
            if (!ok) return false;
        }
        if (iters == 0) return true;
        iv[0] = iters;
        if (!ctx_->upload(ib, 0, iv.data(), 16, err)) return false;
        vk::Recording rec(*ctx_);
        rec.begin();
        for (int r = 0; r < 8; ++r) {
            rec.dispatch(*p, {vk::Ref(ib), vk::Ref(ob)}, nullptr, 0, groups);
            rec.barrier();
        }
        rec.end();
        const double ms = time_ms(rec, err);
        if (ms < 0) return false;
        // Four independent chains per thread, 4 MACs per packed dot.
        const double dots = (double)8 * groups * 64 * iters * 4 * 4;
        std::fprintf(stderr, "[fast-micro] idot: %.1f G i8-MAC/s (%.3f ms, 4 ILP chains)\n",
                     dots / (ms * 1e-3) / 1e9, ms);
        return true;
    }
    if (std::strncmp(mi, "gemm", 4) == 0) {
        // Skinny-GEMM probe (#311 speculative decoding): time the tiled GEMM
        // at M = 1,2,4,8,16 over a real decode shape.
        // STARLING_FAST_MICRO=gemm[,K[,N[,reps]]]  (defaults 2048, 6144, 8)
        uint32_t Kd = 2048, Nd = 6144, reps = 8;
        // Bare "gemm" keeps the defaults; sscanf then matches nothing (EOF).
        if ((mi[4] && std::sscanf(mi, "gemm,%u,%u,%u", &Kd, &Nd, &reps) < 1) || Kd == 0 || Kd % 32 ||
            Nd == 0 || reps == 0) {
            err = "STARLING_FAST_MICRO=gemm[,K[,N[,reps]]] (K % 32 == 0, N and reps > 0)";
            return false;
        }
        const uint32_t Mmax = 16;
        std::mt19937 rng(4242);
        auto w16 = [&](size_t n) {
            std::vector<uint32_t> w(n);
            for (auto& v : w) {
                const float f[2] = {(float)(rng() % 2001) / 1000.0f - 1.0f,
                                    (float)(rng() % 2001) / 1000.0f - 1.0f};
                v = pack_f16(f, 2)[0];
            }
            return w;
        };
        auto w32 = [&](size_t n) {
            std::vector<uint32_t> w(n);
            for (auto& v : w) v = (uint32_t)rng();
            return w;
        };
        vk::Buffer A, wq, ws, C;
        const bool ok_up = ctx_->create_buffer(A, (size_t)Mmax * Kd * 2, vk::Mem::Device, err) &&
                           ctx_->upload(A, 0, w16((size_t)Mmax * Kd / 2).data(), (size_t)Mmax * Kd * 2, err);
        const auto qv = w32((size_t)Nd * Kd / 8), svv = [&] {
            auto v = w16((size_t)Nd * Kd / 32);
            for (auto& x : v) x &= 0xbfffbfffu;   // sane f16 scale pairs
            return v;
        }();
        const bool ok_w = ok_up && ctx_->create_buffer(wq, qv.size() * 4, vk::Mem::Device, err) &&
                          ctx_->upload(wq, 0, qv.data(), qv.size() * 4, err) &&
                          ctx_->create_buffer(ws, svv.size() * 4, vk::Mem::Device, err) &&
                          ctx_->upload(ws, 0, svv.data(), svv.size() * 4, err);
        const bool ok_c = ok_w && ctx_->create_buffer(C, (size_t)Mmax * Nd * 4, vk::Mem::Device, err);
        if (!ok_c) return false;
        for (uint32_t M : {1u, 2u, 4u, 8u, 16u}) {
            GemmCall u;
            u.b = BKind::W4;
            u.epi = Epi::F32;
            u.a.M = M; u.a.N = Nd; u.a.K = Kd; u.a.lda = Kd; u.a.ldb = Kd; u.a.ldc = Nd;
            u.A = vk::Ref(A); u.Bq = vk::Ref(wq); u.Bs = vk::Ref(ws); u.C = vk::Ref(C);
            const TileCfg saved_tile = tile;
            vk::Recording rec(*ctx_);
            rec.begin();
            for (uint32_t i = 0; i < reps; ++i) {
                if (!gemm(rec, u, err)) { tile = saved_tile; return false; }
                rec.barrier();
            }
            rec.end();
            tile = saved_tile;
            const double ms = time_ms(rec, err);
            if (ms < 0) return false;
            const double wps = (double)M * Nd * Kd / (ms * 1e-3) / 1e9;
            double m1 = 0;
            if (M == 1) m1 = ms / reps;
            std::fprintf(stderr, "[fast-micro] gemm W4 M=%u N=%u K=%u: %.3f ms = %.1f G w/s (%.2fx M=1)\n",
                         M, Nd, Kd, ms / reps, wps, m1 > 0 ? (ms / reps) / m1 : 1.0);
        }
        return true;
    }
    if (std::strncmp(mi, "f16dot", 6) == 0) {
        // f16 dot issue-rate probe: STARLING_FAST_MICRO=f16dot[,groups[,iters]]
        // — prices dot(f16vec4, f16vec4) against the f32 FMA (alu probe) and
        // the i8 SDot (idot probe); needs shaderFloat16.
        if (!ctx_->info().f16) { err = "f16dot probe: no shaderFloat16"; return false; }
        uint32_t groups = 48, iters = 4096;
        if ((mi[6] && std::sscanf(mi, "f16dot,%u,%u", &groups, &iters) < 1) || groups == 0 ||
            iters == 0) {
            err = "STARLING_FAST_MICRO=f16dot[,groups[,iters]]";
            return false;
        }
        vk::Buffer ob;
        std::vector<float> zeros(128 * groups, 0.0f);
        if (!ctx_->create_buffer(ob, zeros.size() * 4, vk::Mem::Device, err) ||
            !ctx_->upload(ob, 0, zeros.data(), zeros.size() * 4, err))
            return false;
        const vk::Pipeline* p = ctx_->pipeline("f16dot_probe", {}, err);
        if (!p) return false;
        struct { uint32_t iters; } pc{iters};
        vk::Recording rec(*ctx_);
        rec.begin();
        rec.dispatch(*p, {vk::Ref(ob)}, &pc, sizeof(pc), groups);
        rec.end();
        const double ms = time_ms(rec, err);
        if (ms < 0) return false;
        // 32 chains x 1 dot (4 MACs) per iter per thread, 128 threads per
        // group — keep in sync with shaders/f16dot_probe.comp.
        const double macs = (double)groups * 128 * iters * 32 * 4;
        std::fprintf(stderr, "[fast-micro] f16dot: %.1f G f16-MAC/s (%.3f ms)\n",
                     macs / (ms * 1e-3) / 1e9, ms);
        return true;
    }
    if (std::strncmp(mi, "norm", 4) == 0) {
        // Norm-kernel probe: STARLING_FAST_MICRO=norm,rows,D[,reps[,mode]]
        uint32_t rows = 107, D = 2048, reps = 64, mode = 1;
        std::sscanf(mi, "norm,%u,%u,%u,%u", &rows, &D, &reps, &mode);
        vk::Buffer xb, ob, gb;
        std::vector<float> xf(std::max<size_t>(rows * D, 4096), 0.5f);
        auto up = [&](vk::Buffer& b) {
            return ctx_->create_buffer(b, xf.size() * 4, vk::Mem::Device, err) &&
                   ctx_->upload(b, 0, xf.data(), xf.size() * 4, err);
        };
        // xf holds >= D floats, so it doubles as the g/b/g2/b2 vectors.
        if (!up(xb) || !up(ob) || !up(gb))
            return false;
        const vk::Pipeline* p = ctx_->pipeline("norm", {256u, mode}, err);
        if (!p) return false;
        struct { uint32_t D, ld_x, ld_o, x_off, o_off; float eps, eps2; } pc{
            D, D, D, 0, 0, 1e-5f, 1e-5f};
        vk::Recording rec(*ctx_);
        rec.begin();
        for (uint32_t i = 0; i < reps; ++i) {
            rec.dispatch(*p, {vk::Ref(xb), vk::Ref(gb), vk::Ref(gb), vk::Ref(gb), vk::Ref(gb),
                              vk::Ref(ob)},
                         &pc, sizeof(pc), rows);
            rec.barrier();
        }
        rec.end();
        const double ms = time_ms(rec, err);
        if (ms < 0) return false;
        std::fprintf(stderr, "[fast-micro] norm mode=%u rows=%u D=%u: %.3f ms/dispatch = %.1f GB/s\n",
                     mode, rows, D, ms / reps,
                     (double)reps * rows * D * 2 * 4 / ms * 1e-6);
        return true;
    }
    if (std::strncmp(mi, "alu", 3) == 0) {
        // Pure-FMA issue-rate probe: STARLING_FAST_MICRO=alu[,groups[,iters]].
        uint32_t groups = 48, iters = 4096;
        std::sscanf(mi, "alu,%u,%u", &groups, &iters);
        vk::Buffer ob;
        std::vector<float> zeros(128 * groups, 0.0f);
        if (!ctx_->create_buffer(ob, zeros.size() * 4, vk::Mem::Device, err) ||
            !ctx_->upload(ob, 0, zeros.data(), zeros.size() * 4, err))
            return false;
        const vk::Pipeline* p = ctx_->pipeline("alu_probe", {}, err);
        if (!p) return false;
        struct { uint32_t iters; } pc{iters};
        vk::Recording rec(*ctx_);
        rec.begin();
        rec.dispatch(*p, {vk::Ref(ob)}, &pc, sizeof(pc), groups);
        rec.end();
        const double ms = time_ms(rec, err);
        if (ms < 0) return false;
        const double fma = (double)groups * 128 * iters * 32;
        std::fprintf(stderr, "[fast-micro] alu: %.1f GFLOPS (%.3f ms)\n",
                     2.0 * fma / (ms * 1e-3) / 1e9, ms);
        return true;
    }
    uint32_t bits = 4, n = 4096, k = 1024, reps = 64, rows = 0;
    // m2: the GEMV_M (two-token) W4U GEMV — one weight pass, two x vectors,
    // two y rows. Reports per-iteration time covering BOTH tokens' products
    // and the per-token weight rate, against the M=1 gemv_w4u numbers.
    const bool m2 = std::strncmp(mi, "m2,", 3) == 0;   // args required
    if (m2) mi += 3;
    const bool sweep = std::strncmp(mi, "s,", 2) == 0;   // args required
    if (sweep) mi += 2;
    // alt: alternate two pipelines per rep — prices the in-context
    // per-dispatch overhead attribution (pipeline/spec switch cost vs the
    // same pipeline back-to-back). "alt,<bits>,<n>,<k>,<reps>" switches spec
    // constants (rows 16/24); "altx,..." switches shader variants
    // (gemv_w4u/gemv_w4).
    int alt_mode = 0;   // 0 off, 1 spec switch, 2 shader switch
    if (std::strncmp(mi, "altx,", 5) == 0) { alt_mode = 2; mi += 5; }
    else if (std::strncmp(mi, "alt,", 4) == 0) { alt_mode = 1; mi += 4; }
    const int got = std::sscanf(mi, "%u,%u,%u,%u,%u", &bits, &n, &k, &reps, &rows);
    if (got < 3 || (bits != 4 && bits != 8 && bits != 16) || n == 0 || k == 0 || k % 32 ||
        reps == 0) {
        err = "STARLING_FAST_MICRO=bits,N,K[,reps[,rows]] (bits 4/8/16, K % 32 == 0, reps > 0)";
        return false;
    }
    if (m2 && bits != 4) {   // GEMV_M exists only for the W4 shader
        err = "STARLING_FAST_MICRO: m2 (two-token) needs bits=4";
        return false;
    }
    if (alt_mode && sweep) {   // the sweep times one pipeline per rows value
        err = "STARLING_FAST_MICRO: alt/altx cannot be combined with the rows sweep";
        return false;
    }
    if (!rows) rows = gemv_rows(n);
    std::mt19937 rng(99);
    auto rnd = [&](size_t cnt) {
        std::vector<uint32_t> w(cnt);
        for (auto& v : w) v = (uint32_t)rng();
        return w;
    };
    const size_t qw = bits == 4 ? (size_t)n * k / 8 + 1 : bits == 8 ? (size_t)n * k / 4 + 1
                                                                    : (size_t)n * k / 2 + 1;
    const size_t sw = bits == 16 ? 0 : (size_t)n * k / 32 + 1;
    vk::Buffer wq, ws, x, y;
    std::vector<float> xf(m2 ? 2 * k : k);
    for (auto& f : xf) f = (float)(rng() % 2001) / 1000.0f - 1.0f;
    std::vector<float> y0(m2 ? 2 * n : n, 0.0f);
    auto up32 = [&](vk::Buffer& b, const void* d, size_t bytes) {
        return ctx_->create_buffer(b, bytes, vk::Mem::Device, err) &&
               ctx_->upload(b, 0, d, bytes, err);
    };
    auto wv = rnd(qw);
    auto sv = sw ? rnd(sw) : std::vector<uint32_t>{};
    for (auto& v : sv) v &= 0xbfffbfffu;   // f16 scale pairs: exponent < 16, never inf/NaN
    if (!up32(wq, wv.data(), qw * 4) || (sw && !up32(ws, sv.data(), sw * 4)) ||
        !up32(x, xf.data(), xf.size() * 4) || !up32(y, y0.data(), y0.size() * 4))
        return false;
    const char* name = bits == 4 ? (m2 ? "gemv_w4um" : w4_unpack_ ? "gemv_w4u" : "gemv_w4")
                      : bits == 8 ? "gemv_w8" : "gemv_f16";
    const uint32_t lanes = k / 32;
    GemvArgs ga;
    ga.N = n;
    ga.K = k;
    if (m2) { ga.x_off2 = k; ga.y_off2 = n; }   // K % 32 == 0 keeps x_off2 vec4-aligned
    const uint32_t rows_list[] = {8, 16, 24, 32, 48, 64};
    const uint32_t rows_n = sweep ? 6u : 1u;
    uint32_t rsplit_out = 1;
    double plain_mspt = -1.0;
    // (sweep mode: one model load, one rows value per timing recording;
    // timing only — the correctness check runs in the single-rows mode)
    std::vector<double> sweep_ms;
    std::vector<uint32_t> sweep_rsplit;
    for (uint32_t ri = 0; ri < rows_n; ++ri) {
    if (sweep) rows = rows_list[ri];
    uint32_t rsplit = 1;
    {
        const uint32_t target = std::max(32u, ctx_->info().subgroup_size);
        while (rsplit < 8 && lanes * rsplit < target && rows % (2 * rsplit) == 0u) rsplit *= 2;
    }
    const vk::Pipeline* p = ctx_->pipeline(name, {lanes, rows, 0u, 0u, 0u, rsplit, lanes * rsplit}, err);
    if (!p) return false;
    const vk::Pipeline* p2 = p;
    uint32_t rows2 = rows;
    if (alt_mode == 1) {   // same shader, different spec (rows 16 <-> 24)
        rows2 = rows == 16u ? 24u : 16u;
        p2 = ctx_->pipeline(name, {lanes, rows2, 0u, 0u, 0u, rsplit, lanes * rsplit}, err);
        if (!p2) return false;
    } else if (alt_mode == 2) {   // different shader file (w4u <-> w4)
        if (bits != 4 || m2) {
            err = "STARLING_FAST_MICRO: altx (shader switch) needs bits=4 without m2";
            return false;
        }
        p2 = ctx_->pipeline(w4_unpack_ && std::string(name) == "gemv_w4u" ? "gemv_w4" : "gemv_w4u",
                            {lanes, rows, 0u, 0u, 0u, rsplit, lanes * rsplit}, err);
        if (!p2) return false;
    }
    vk::Recording rec(*ctx_);
    rec.begin();
    rsplit_out = rsplit;
    for (uint32_t i = 0; i < reps; ++i) {
        const vk::Pipeline* cur = (alt_mode && (i & 1u)) ? p2 : p;
        const uint32_t r = (alt_mode == 1 && (i & 1u)) ? rows2 : rows;
        rec.dispatch(*cur, {vk::Ref(x), vk::Ref(wq), sw ? vk::Ref(ws) : vk::Ref(dummy_), vk::Ref(y), vk::Ref(dummy_),
                          vk::Ref(dummy_), vk::Ref(dummy_)},
                     &ga, sizeof(ga), ceil_div(n, r));
        rec.barrier();
    }
    rec.end();
    if (sweep) {
        const double ms = time_ms(rec, err);
        if (ms < 0) return false;
        sweep_ms.push_back(ms / reps);
        sweep_rsplit.push_back(rsplit);
        continue;
    }
    {
        // Time while `rec` is in scope (the loop body) — the rows loop below
        // closes its scope before the reporting code runs.
        const double ms = time_ms(rec, err);
        if (ms < 0) return false;
        plain_mspt = ms / reps;
    }
    // Correctness: CPU reference dot products for the first rows, through
    // pipeline `p` (in alt modes the alternate pipeline is timed only).
    if (bits == 4 || bits == 8) {
        vk::Recording rc1(*ctx_);
        rc1.begin();
        rc1.dispatch(*p, {vk::Ref(x), vk::Ref(wq), sw ? vk::Ref(ws) : vk::Ref(dummy_), vk::Ref(y), vk::Ref(dummy_),
                          vk::Ref(dummy_), vk::Ref(dummy_)},
                     &ga, sizeof(ga), ceil_div(n, rows));
        rc1.end();
        if (!rc1.submit_and_wait(err)) return false;
        std::vector<float> got(m2 ? 2 * n : n, 0.0f);
        if (!ctx_->download(y, 0, got.data(), got.size() * 4, err)) return false;
        const uint32_t tok_base = m2 ? n : 0;   // token-1 y rows
        auto h2f = [](uint32_t h) {
            const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
            uint32_t b;
            if (e == 0) b = (s << 31) | (m >> 1);            // subnormal ~
            else b = (s << 31) | ((e + 112) << 23) | (m << 13);
            float f;
            std::memcpy(&f, &b, 4);
            return f;
        };
        double maxerr = 0;
        const uint32_t nchk = std::min(n, 256u);
        for (uint32_t r = 0; r < nchk; ++r) {
            double ref = 0;
            for (uint32_t kk = 0; kk < k; ++kk) {
                const uint32_t grp = kk / 32, in = kk % 32;
                const uint32_t sraw = sv[(size_t)r * (k / 32) + grp];
                if (bits == 4) {
                    const uint32_t w32 = wv[(size_t)r * (k / 8) + grp * 4 + in / 8];
                    const float nib = (float)((w32 >> (4 * (in % 8))) & 15u);
                    // packHalf2x16(scale, offset): scale = low half
                    ref += (h2f(sraw & 0xffff) * nib + h2f(sraw >> 16)) * xf[kk];
                } else {
                    const uint32_t w32 = wv[(size_t)r * (k / 4) + grp * 8 + in / 4];
                    const int i8 = (int8_t)((w32 >> (8 * (in % 4))) & 0xffu);
                    // packHalf2x16(s_lo, s_hi): s_lo covers weights 0..15
                    const float s = h2f((in / 16) == 0 ? (sraw & 0xffff) : (sraw >> 16));
                    ref += s * (float)i8 * xf[kk];
                }
            }
            maxerr = std::max(maxerr, std::abs(ref - got[r]) /
                                          std::max(1.0, std::abs(ref)));
            if (m2) {
                // Token-1 rows: same weights against the second x vector.
                // (Added in the review round — the original probe validated
                // token 0 only; the kernel is not dispatched by the engine
                // yet, so no shipped behavior depended on it.)
                double ref2 = 0;
                for (uint32_t kk = 0; kk < k; ++kk) {
                    const uint32_t grp = kk / 32, in = kk % 32;
                    const uint32_t sraw = sv[(size_t)r * (k / 32) + grp];
                    const uint32_t w32 = wv[(size_t)r * (k / 8) + grp * 4 + in / 8];
                    const float nib = (float)((w32 >> (4 * (in % 8))) & 15u);
                    ref2 += (h2f(sraw & 0xffff) * nib + h2f(sraw >> 16)) * xf[k + kk];
                }
                maxerr = std::max(maxerr, std::abs(ref2 - got[tok_base + r]) /
                                              std::max(1.0, std::abs(ref2)));
            }
        }
        std::fprintf(stderr, "[fast-micro] check rsplit=%u: max rel err = %.5f (first %u rows)\n",
                     rsplit, maxerr, nchk);
    }
    }   // rows loop (sweep)
    if (sweep) {
        double best = 1e30;
        uint32_t best_rows = 0;
        const double toks = m2 ? 2.0 : 1.0;   // m2: one iteration covers two tokens
        for (uint32_t ri = 0; ri < sweep_ms.size(); ++ri) {
            const double gw = toks * n * k / (sweep_ms[ri] * 1e-3) / 1e9;
            std::fprintf(stderr, "[fast-micro] sweep bits=%u N=%u K=%u rows=%u wg=%ux%u: %.4f ms = %.1f G w/s%s\n",
                         bits, n, k, rows_list[ri], lanes, sweep_rsplit[ri], sweep_ms[ri], gw,
                         m2 ? " per token" : "");
            if (sweep_ms[ri] < best) { best = sweep_ms[ri]; best_rows = rows_list[ri]; }
        }
        std::fprintf(stderr, "[fast-micro] sweep best: rows=%u\n", best_rows);
        return true;
    }
    const double ms = plain_mspt * reps;
    const double bytes = (double)reps * n * k * (bits == 4 ? 0.625 : bits == 8 ? 1.125 : 2.0);
    if (m2) {
        // One iteration covers both tokens: per-token weight rate doubles.
        std::fprintf(stderr,
                     "[fast-micro] gemv-m2 bits=%u N=%u K=%u rows=%u wg=%ux%u: %.3f ms/iter "
                     "(2 tokens) = %.1f G w/s per token\n",
                     bits, n, k, rows, lanes, rsplit_out, ms / reps,
                     2.0 * n * k / (ms / reps * 1e-3) / 1e9);
    } else {
        std::fprintf(stderr,
                     "[fast-micro] gemv%s bits=%u N=%u K=%u rows=%u wg=%ux%u: %.3f ms/iter = %.1f GB/s\n",
                     alt_mode == 1 ? "-ALT(spec)" : alt_mode == 2 ? "-ALT(shader)" : "",
                     bits, n, k, rows, lanes, rsplit_out, ms / reps, bytes / ms * 1e-6);
    }
    return true;
}

bool Kernels::autotune(std::string& err) {
    TuneResult& R = tune_result();
    if (R.done) { tile = R.tile; gemv_rows_max = R.gemv_rows; return true; }
    const auto& info = ctx_->info();
    const char* dir = std::getenv("STARLING_FAST_CACHE_DIR");
    const char* force = std::getenv("STARLING_FAST_TUNE");
    const bool forced = force && force[0] == '1';
    if ((!dir || !*dir) && !forced) return true;   // built-in defaults
    char tag[160];
    std::snprintf(tag, sizeof tag, "/starling-fast-tune-%08x-%08x-%08x-%s.txt", info.vendor_id,
                  info.device_id, info.driver_version, f16_math ? "h" : "f");
    const std::string path = std::string(dir && *dir ? dir : ".") + tag;
    if (!forced) {
        std::ifstream in(path);
        TileCfg t;
        uint32_t rows = 0;
        const bool fits = [&] {
            if (!(in >> t.BM >> t.BN >> t.TM >> t.TN >> rows) || rows < 8 || !t.TM || !t.TN) return false;
            const uint32_t wg = (t.BM / t.TM) * (t.BN / t.TN);
            return wg >= 32 && wg <= info.max_wg_invocations &&
                   (size_t)32 * (t.BM + t.BN) * 2 <= info.max_shared_bytes;
        }();
        if (fits) {
            R = TuneResult{true, t, rows};
            tile = t;
            gemv_rows_max = rows;
            return true;
        }
    }

    // Synthetic problem at the encoder's dominant shapes: ~280 frames (odd,
    // like real subsampled lengths — a multiple of every BM hides tail-wave
    // costs and mis-ranks tiles on PowerVR) through a 1024 -> 4096 W4
    // projection and a 4096 -> 1024 W8 projection, plus a 4096 x 1024 W4
    // decode GEMV.
    const uint32_t M = std::getenv("STARLING_FAST_TUNE_M")
                           ? (uint32_t)std::atoi(std::getenv("STARLING_FAST_TUNE_M"))
                           : 293,
                 D = 1024, F = 4096;
    std::mt19937 rng(1234);
    auto rnd_words = [&](size_t n, bool halves) {
        std::vector<uint32_t> w(n);
        for (auto& v : w) {
            if (halves) {
                const float f[2] = {(float)(rng() % 2001) / 1000.0f - 1.0f,
                                    (float)(rng() % 2001) / 1000.0f - 1.0f};
                v = pack_f16(f, 2)[0];
            } else {
                v = (uint32_t)rng();
            }
        }
        return w;
    };
    vk::Buffer a, wq4, ws4, wq8, ws8, c, x, y, ones;
    auto up = [&](vk::Buffer& b, const std::vector<uint32_t>& w) {
        return ctx_->create_buffer(b, w.size() * 4, vk::Mem::Device, err) &&
               ctx_->upload(b, 0, w.data(), w.size() * 4, err);
    };
    // x holds f32 values for the GEMV: small random floats.
    std::vector<uint32_t> xw(D);
    for (auto& v : xw) { const float f = (float)(rng() % 2001) / 1000.0f - 1.0f; std::memcpy(&v, &f, 4); }
    if (!up(a, rnd_words((size_t)M * F / 2, true)) || !up(wq4, rnd_words((size_t)F * D / 8, false)) ||
        !up(ws4, rnd_words((size_t)F * D / 32, true)) || !up(wq8, rnd_words((size_t)D * F / 4, false)) ||
        !up(ws8, rnd_words((size_t)D * F / 32, true)) || !up(x, xw) ||
        !up(ones, std::vector<uint32_t>(F, 0x3f800000u)) ||   // norm g/b (1.0f)
        !ctx_->create_buffer(c, (size_t)M * F * 4, vk::Mem::Device, err) ||
        !ctx_->create_buffer(y, (size_t)F * 4, vk::Mem::Device, err))
        return false;

    const TileCfg cands[] = {{64, 128, 4, 8}, {64, 64, 4, 4}, {128, 128, 8, 8}, {128, 64, 8, 4},
                             {64, 64, 8, 8}, {32, 64, 4, 8}, {64, 128, 8, 8}, {128, 128, 8, 4},
                             {32, 128, 4, 8}, {64, 32, 4, 4}, {32, 64, 4, 4}, {48, 64, 4, 4}};
    TileCfg best_t = tile;
    double best = 1e30;
    const TileCfg saved = tile;
    for (const TileCfg& t : cands) {
        const uint32_t wg = (t.BM / t.TM) * (t.BN / t.TN);
        const size_t shared = (size_t)32 * (t.BM + t.BN) * 2;
        if (wg > info.max_wg_invocations || wg < 32 || shared > info.max_shared_bytes) continue;
        tile = t;
        vk::Recording rec(*ctx_);
        rec.begin();
        for (int rep = 0; rep < 3; ++rep) {
            GemmCall u;
            u.b = BKind::W4; u.epi = Epi::F16; u.act = Act::Silu;
            u.a.M = M; u.a.N = F; u.a.K = D; u.a.lda = D; u.a.ldb = D; u.a.ldc = F;
            u.A = vk::Ref(a); u.Bq = vk::Ref(wq4); u.Bs = vk::Ref(ws4); u.C = vk::Ref(c);
            GemmCall dn;
            dn.b = BKind::W8; dn.epi = Epi::F32;
            dn.a.M = M; dn.a.N = D; dn.a.K = F; dn.a.lda = F; dn.a.ldb = F; dn.a.ldc = D;
            dn.A = vk::Ref(a); dn.Bq = vk::Ref(wq8); dn.Bs = vk::Ref(ws8); dn.C = vk::Ref(c);
            // Match the real layer pattern: a barrier after every kernel (a
            // tile-based GPU flushes at barriers, so back-to-back dispatches
            // without one would rank tiles wrongly).
            if (!gemm(rec, u, err)) { tile = saved; return false; }
            rec.barrier();
            // A norm between the GEMMs, as in a real layer: the pipeline
            // switches and small kernels between large ones are part of what
            // a tile choice costs on a tile-based GPU.
            if (!norm(rec, 0, M, F, vk::Ref(c), F, vk::Ref(ones), vk::Ref(ones), {}, {}, vk::Ref(c), F,
                      1e-5f, 1e-5f, err)) { tile = saved; return false; }
            rec.barrier();
            if (!gemm(rec, dn, err)) { tile = saved; return false; }
            rec.barrier();
            if (!norm(rec, 0, M, D, vk::Ref(c), D, vk::Ref(ones), vk::Ref(ones), {}, {}, vk::Ref(c), D,
                      1e-5f, 1e-5f, err)) { tile = saved; return false; }
            rec.barrier();
        }
        rec.end();
        const double ms = time_ms(rec, err);
        if (ms < 0) { tile = saved; return false; }
        if (std::getenv("STARLING_FAST_VERBOSE"))
            std::fprintf(stderr, "[fast-tune] gemm %u,%u,%u,%u: %.3f ms\n", t.BM, t.BN, t.TM, t.TN, ms);
        if (ms < best) { best = ms; best_t = t; }
    }
    tile = best_t;

    uint32_t best_rows = gemv_rows_max;
    double best_g = 1e30;
    const uint32_t saved_tgt = gemv_tgt_wgs;
    gemv_tgt_wgs = ~0u;   // candidates are pinned; growth would blur the sweep
    for (uint32_t rows : {8u, 16u, 32u, 64u}) {
        gemv_rows_max = rows;
        const uint32_t r = gemv_rows(F);
        const vk::Pipeline* p = ctx_->pipeline("gemv_w4", {D / 32, r, 0u, 0u, 0u, 1u, D / 32}, err);
        if (!p) return false;
        vk::Recording rec(*ctx_);
        rec.begin();
        GemvArgs ga;
        ga.N = F;
        ga.K = D;
        for (int rep = 0; rep < 8; ++rep) {
            rec.dispatch(*p, {vk::Ref(x), vk::Ref(wq4), vk::Ref(ws4), vk::Ref(y), vk::Ref(dummy_),
                              vk::Ref(dummy_), vk::Ref(dummy_)},
                         &ga, sizeof(ga), ceil_div(F, r));
            rec.barrier();
            if (!norm(rec, 1, 1, D, vk::Ref(y), D, vk::Ref(ones), vk::Ref(ones), vk::Ref(ones), vk::Ref(ones), vk::Ref(y), D,
                      1e-5f, 1e-5f, err)) return false;
            rec.barrier();
        }
        rec.end();
        const double ms = time_ms(rec, err);
        if (ms < 0) return false;
        if (std::getenv("STARLING_FAST_VERBOSE"))
            std::fprintf(stderr, "[fast-tune] gemv rows %u: %.3f ms\n", rows, ms);
        if (ms < best_g) { best_g = ms; best_rows = rows; }
    }
    gemv_tgt_wgs = saved_tgt;
    gemv_rows_max = best_rows;
    R = TuneResult{true, tile, gemv_rows_max};
    if (dir && *dir) {
        std::ofstream out(path);
        out << tile.BM << " " << tile.BN << " " << tile.TM << " " << tile.TN << " " << gemv_rows_max << "\n";
    }
    if (std::getenv("STARLING_FAST_VERBOSE"))
        std::fprintf(stderr, "[fast-tune] chose gemm %u,%u,%u,%u gemv rows %u\n", tile.BM, tile.BN,
                     tile.TM, tile.TN, gemv_rows_max);
    return true;
}

} // namespace starling::fast
