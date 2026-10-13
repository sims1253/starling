// kernels.cpp — see kernels.hpp.

#include "kernels.hpp"
#include "packed_file.hpp"

#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <chrono>
#include <fstream>
#include <random>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace starling::fast {

// ---------------------------------------------------------------------------
// Arena
// ---------------------------------------------------------------------------

namespace {

bool env_flag(const char* name, bool dflt) {
    const char* e = std::getenv(name);
    return e && e[0] ? e[0] != '0' : dflt;
}

// Return the whole pages inside [p, p + n) of a private file mapping to the
// kernel (they refault from the file if read again).
void drop_pages(const void* p, size_t n) {
#if defined(__linux__)
    static const uintptr_t page = (uintptr_t)sysconf(_SC_PAGESIZE);
    const uintptr_t b = ((uintptr_t)p + page - 1) / page * page;
    const uintptr_t e = ((uintptr_t)p + n) / page * page;
    if (e > b) madvise((void*)b, e - b, MADV_DONTNEED);
#else
    (void)p;
    (void)n;
#endif
}

} // namespace

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

Arena::Id Arena::add_deferred(size_t words, Producer fn) {
    Blob b;
    b.bytes = std::max<size_t>(words * 4, 4);
    b.fn = std::move(fn);
    blobs_.push_back(std::move(b));
    return blobs_.size() - 1;
}

GMat deferred_f16(Arena& a, uint32_t N, uint32_t K,
                  std::function<bool(std::vector<float>&, std::string&)> make) {
    const size_t n = (size_t)N * K;
    GMat g;
    g.fmt = GpuFmt::F16;
    g.N = N;
    g.K = K;
    g.q = a.add_deferred((n + 1) / 2, [n, make = std::move(make)](uint32_t* dst, std::string& err) {
        std::vector<float> v;
        if (!make(v, err)) return false;
        if (v.size() != n) { err = "fast engine: deferred matrix size mismatch"; return false; }
        const std::vector<uint32_t> w = pack_f16(v.data(), n);
        std::memcpy(dst, w.data(), w.size() * 4);
        return true;
    });
    return g;
}

MatSpec raw_rows(int type, const void* data, uint32_t N, uint32_t K, const std::string& name) {
    MatSpec m;
    m.name = name;
    m.N = N;
    m.K = K;
    MatPart p;
    p.type = type;
    p.data = data;
    p.rows = N;
    p.K = K;
    m.parts.push_back(p);
    return m;
}

MatSpec tensor_rows(const ggml_tensor* t, const std::string& name) {
    const uint32_t K = (uint32_t)t->ne[0];
    return raw_rows((int)t->type, t->data, (uint32_t)(ggml_nelements(t) / t->ne[0]), K, name);
}

MatSpec row_range(const MatSpec& m, uint32_t r0, uint32_t n) {
    MatSpec r = m;
    MatPart& p = r.parts.at(0);
    if (p.packed) p.packed_row0 += r0;
    else p.data = (const uint8_t*)p.data + (size_t)r0 * ggml_row_size((ggml_type)p.type, m.K);
    p.rows = n;
    r.N = n;
    return r;
}

MatSpec spec_concat(MatSpec a, const MatSpec& b) {
    for (MatPart p : b.parts) {
        p.dst.first += a.N;   // (tables are only made by spec_permute, last)
        a.parts.push_back(std::move(p));
    }
    a.N += b.N;
    return a;
}

MatSpec spec_interleave(MatSpec a, const MatSpec& b) {
    for (MatPart& p : a.parts) { p.dst.first *= 2; p.dst.step *= 2; }
    for (MatPart p : b.parts) {
        p.dst.first = 2 * p.dst.first + 1;
        p.dst.step *= 2;
        a.parts.push_back(std::move(p));
    }
    a.N += b.N;
    return a;
}

MatSpec spec_permute(MatSpec m, const std::vector<uint32_t>& order) {
    auto inv = std::make_shared<std::vector<uint32_t>>(order.size());
    for (uint32_t r = 0; r < order.size(); ++r) (*inv)[order[r]] = r;
    MatPart& p = m.parts.at(0);
    p.table = inv;
    p.dst.table = inv->data();
    return m;
}

bool plan_matrix(MatSpec& m, GpuFmt& fmt, LayoutDesc& layout, std::string& err) {
    uint64_t rows = 0;
    bool inside = true;   // every destination row inside [0, N)
    for (const MatPart& p : m.parts) {
        rows += p.rows;
        if (!p.packed && !p.data) {
            err = "fast engine: " + m.name + ": tensor data missing";
            return false;
        }
        if (p.K != m.K) {
            err = "fast engine: " + m.name + ": parts of different row lengths";
            return false;
        }
        if (p.rows == 0) continue;
        if (p.table) {
            inside = inside && p.table->size() == p.rows && p.dst.table == p.table->data();
            for (uint32_t r : *p.table) inside = inside && r < m.N;
        } else {
            inside = inside && !p.dst.table && p.dst.first + (uint64_t)(p.rows - 1) * p.dst.step < m.N;
        }
    }
    if (m.parts.empty() || m.N == 0 || m.K == 0 || rows != m.N || !inside) {
        err = "fast engine: " + m.name + ": parts do not cover the matrix";
        return false;
    }
    std::vector<GpuFmt> f(m.parts.size());
    std::vector<LayoutDesc> lay(m.parts.size());
    auto plan_part = [&](size_t i, bool force_w8) {
        MatPart& p = m.parts[i];
        bool lossless;
        if (!plan_gpu_rows(p.type, m.K, force_w8, f[i], lossless, err)) {
            err = m.name + ": " + err;
            return false;
        }
        lay[i] = f[i] == GpuFmt::W8 ? LayoutDesc{8, 16, true} : LayoutDesc{};   // w8g16sym / w4g32asym
        return true;
    };
    for (size_t i = 0; i < m.parts.size(); ++i) {
        const PackedTensor* pt = m.parts[i].packed;
        if (!pt) {
            if (!plan_part(i, false)) return false;
            continue;
        }
        // Kernel support: exactly the two legacy shapes (see PackedWeights::matrix).
        if (!pt->desc.is_legacy_w4() && !pt->desc.is_legacy_w8()) {
            err = "packed file: " + m.name + " uses layout " + layout_to_string(pt->desc) +
                  ", which has no kernel support in this build";
            return false;
        }
        f[i] = pt->desc.is_legacy_w4() ? GpuFmt::W4 : GpuFmt::W8;
        lay[i] = pt->desc;
    }
    // Fused row groups need one format: when the parts come in different
    // formats (Granite: Q5_K gate -> W8, Q4_K up -> W4), the others are
    // repacked from their GGUF rows to W8.
    for (size_t i = 1; i < f.size(); ++i) {
        if (f[i] == f[0]) continue;
        for (size_t j = 0; j < f.size(); ++j) {
            if (f[j] == GpuFmt::W8) continue;
            m.parts[j].packed = nullptr;
            if (!plan_part(j, true)) return false;
        }
        break;
    }
    fmt = f[0];
    layout = lay[0];
    if (fmt == GpuFmt::F16 && m.K % 2) {
        err = "fast engine: " + m.name + ": f16 rows need an even K";
        return false;
    }
    if (m.scale != 1.0f && fmt != GpuFmt::F16 && !layout.is_legacy_w4() && !layout.is_legacy_w8()) {
        err = "scale_matrix: packed-file layout " + layout_to_string(layout);
        return false;
    }
    return true;
}

bool Arena::add_matrix(MatSpec m, GMat& g, std::string& err) {
    GpuFmt fmt;
    LayoutDesc layout;
    if (!plan_matrix(m, fmt, layout, err)) return false;
    g = GMat{};
    g.fmt = fmt;
    g.N = m.N;
    g.K = m.K;
    g.layout = layout;
    // Codes and scales are placed in one buffer (finalize fills a buffer at
    // a time where it stages).
    Blob q;
    q.bytes = std::max<VkDeviceSize>((VkDeviceSize)m.N * q_row_words(fmt, m.K) * 4, 4);
    q.group = 1;
    blobs_.push_back(q);
    g.q = blobs_.size() - 1;
    Planned pm{std::move(m), fmt, g.q, SIZE_MAX};
    const VkDeviceSize sb = (VkDeviceSize)pm.spec.N * s_row_words(fmt, pm.spec.K) * 4;
    if (sb) {
        Blob s;
        s.bytes = sb;
        blobs_.push_back(s);
        g.s = pm.s = blobs_.size() - 1;
        g.has_s = true;
        blobs_[g.q].group = 2;
    }
    mats_.push_back(std::move(pm));
    return true;
}

bool Arena::fill_unit(const Unit& u, const std::vector<uint8_t*>& base, std::string& err) const {
    if (u.mat == SIZE_MAX) {   // a deferred blob
        const Blob& b = blobs_[u.part];
        return b.fn((uint32_t*)(base[b.buffer] + b.off), err);
    }
    const Planned& pm = mats_[u.mat];
    const Blob& qb = blobs_[pm.q];
    uint32_t* q = (uint32_t*)(base[qb.buffer] + qb.off);
    uint32_t* s = nullptr;
    if (pm.s != SIZE_MAX) s = (uint32_t*)(base[blobs_[pm.s].buffer] + blobs_[pm.s].off);
    if (!fill_spec_rows(pm.spec, pm.fmt, u.part, u.row0, u.rows, q, s, err)) return false;
    const MatPart& p = pm.spec.parts[u.part];
    if (release_sources_ && !p.packed) {
        const size_t rb = ggml_row_size((ggml_type)p.type, pm.spec.K);
        drop_pages((const uint8_t*)p.data + (size_t)u.row0 * rb, (size_t)u.rows * rb);
    }
    return true;
}

bool fill_spec_rows(const MatSpec& m, GpuFmt fmt, size_t pi, uint32_t row0, uint32_t rows,
                    uint32_t* q, uint32_t* s, std::string& err) {
    const MatPart& p = m.parts[pi];
    const uint32_t K = m.K;
    RowMap map = p.dst;   // rows [row0, row0 + rows) of the part
    if (map.table) map.table += row0;
    else map.first += (uint64_t)row0 * map.step;
    if (p.packed) {
        const uint32_t qw = q_row_words(fmt, K), sw = s_row_words(fmt, K);
        const size_t cb = (size_t)layout_code_bytes(p.packed->desc, K);
        const size_t sbytes = (size_t)layout_scale_bytes(p.packed->desc, K);
        for (uint32_t i = 0; i < rows; ++i) {
            const size_t r = (size_t)p.packed_row0 + row0 + i;
            const uint64_t d = map(i);
            std::memcpy(q + d * qw, p.packed->codes.data() + r * cb, cb);
            std::memcpy(s + d * sw, p.packed->scales.data() + r * sbytes, sbytes);
            if (m.scale != 1.0f) scale_f16_words(s + d * sw, sw, m.scale);
        }
        return true;
    }
    const size_t rb = ggml_row_size((ggml_type)p.type, K);
    if (!pack_gpu_rows(p.type, (const uint8_t*)p.data + (size_t)row0 * rb, rows, K, fmt, m.scale, q, s, map, err)) {
        err = m.name + ": " + err;
        return false;
    }
    return true;
}

bool materialize(MatSpec m, HostMatrix& out, std::string& err) {
    out = HostMatrix{};
    if (!plan_matrix(m, out.fmt, out.layout, err)) return false;
    out.N = m.N;
    out.K = m.K;
    out.q.assign((size_t)m.N * q_row_words(out.fmt, m.K), 0);
    out.s.assign((size_t)m.N * s_row_words(out.fmt, m.K), 0);
    for (size_t pi = 0; pi < m.parts.size(); ++pi)
        if (!fill_spec_rows(m, out.fmt, pi, 0, m.parts[pi].rows, out.q.data(),
                            out.s.empty() ? nullptr : out.s.data(), err))
            return false;
    return true;
}

bool Arena::finalize(vk::Context& ctx, std::string& err) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto& info = ctx.info();
    VkDeviceSize cap = 256ull << 20;
    cap = std::min<VkDeviceSize>(cap, info.max_storage_range);
    if (info.max_alloc) cap = std::min<VkDeviceSize>(cap, info.max_alloc);
    const VkDeviceSize align = std::max<VkDeviceSize>(info.min_storage_align, 256);
    auto up = [&](VkDeviceSize x) { return (x + align - 1) / align * align; };
    // First-fit placement in insertion order; a matrix's codes and scales
    // share a buffer.
    std::vector<VkDeviceSize> sizes;
    for (size_t i = 0; i < blobs_.size();) {
        const size_t n = std::max<size_t>(blobs_[i].group, 1);
        VkDeviceSize need = 0;
        for (size_t j = i; j < i + n; ++j) need = up(need) + blobs_[j].bytes;
        if (need > cap) {
            err = "fast engine: tensor of " + std::to_string(need >> 20) +
                  " MiB exceeds the device's storage-buffer range";
            return false;
        }
        if (sizes.empty() || up(sizes.back()) + need > cap) sizes.push_back(0);
        for (size_t j = i; j < i + n; ++j) {
            blobs_[j].buffer = sizes.size() - 1;
            blobs_[j].off = up(sizes.back());
            sizes.back() = blobs_[j].off + blobs_[j].bytes;
        }
        i += n;
    }
    // #325 preflight: refuse the load cleanly when the GPU cannot fit the
    // arena (budget query where the driver exposes it; no-op elsewhere).
    {
        uint64_t total = 0;
        for (const Blob& b : blobs_) total += b.bytes + 4096;
        std::string berr;
        if (!ctx.check_memory_budget(total, berr)) { err = berr; return false; }
    }
    // Mapped device memory where the CPU writes it fast (UMA phones):
    // repacked rows land in place. Elsewhere (desktop APUs map device memory
    // uncached; discrete GPUs) each buffer is filled in host memory and
    // staged. STARLING_FAST_MAPPED_WEIGHTS=0/1 overrides.
    const bool mapped = env_flag("STARLING_FAST_MAPPED_WEIGHTS", info.uma_cached);
    buffers_.clear();
    total_ = 0;

    // Fill units: ~1 MiB of repacked rows each, or one deferred blob, in
    // buffer order (the order the buffers are allocated in).
    std::vector<Unit> units;
    for (size_t i = 0; i < blobs_.size(); ++i)
        if (blobs_[i].fn) units.push_back({SIZE_MAX, i, 0, 0});
    for (size_t mi = 0; mi < mats_.size(); ++mi) {
        const Planned& pm = mats_[mi];
        const size_t row_bytes = ((size_t)q_row_words(pm.fmt, pm.spec.K) + s_row_words(pm.fmt, pm.spec.K)) * 4;
        const uint32_t step = (uint32_t)std::max<size_t>(1, (1u << 20) / std::max<size_t>(row_bytes, 1));
        for (size_t pi = 0; pi < pm.spec.parts.size(); ++pi)
            for (uint32_t r = 0; r < pm.spec.parts[pi].rows; r += step)
                units.push_back({mi, pi, r, std::min(step, pm.spec.parts[pi].rows - r)});
    }
    auto unit_buffer = [&](const Unit& u) { return blobs_[u.mat == SIZE_MAX ? u.part : mats_[u.mat].q].buffer; };
    std::stable_sort(units.begin(), units.end(),
                     [&](const Unit& a, const Unit& b) { return unit_buffer(a) < unit_buffer(b); });
    std::atomic<bool> ok{true};
    std::string first_err;
    std::mutex mu;
    auto fail = [&](const std::string& e) {
        std::lock_guard<std::mutex> lk(mu);
        if (ok.exchange(false)) first_err = e;
    };
    auto run_units = [&](const std::vector<uint8_t*>& base, const std::vector<size_t>& idx) {
        parallel_for(idx.size(), [&](size_t k) {
            std::string e;
            if (ok && !fill_unit(units[idx[k]], base, e)) fail(e);
        });
    };
    std::vector<uint8_t*> base(sizes.size(), nullptr);
    std::chrono::steady_clock::time_point t1;
    if (mapped) {
        // Allocation (the driver zeroes the pages: ~6 GB/s per thread on the
        // Pixel) overlaps the fill: allocator threads create the buffers in
        // order while the workers repack into those already mapped.
        // STARLING_FAST_ALLOC_THREADS=n (default 2).
        const size_t nb = sizes.size();
        std::vector<std::unique_ptr<vk::Buffer>> bufs(nb);
        std::unique_ptr<std::atomic<bool>[]> ready(new std::atomic<bool>[nb]);
        for (size_t i = 0; i < nb; ++i) ready[i] = false;
        std::atomic<bool> abort{false};
        std::condition_variable cv;
        std::mutex cv_mu;
        std::atomic<size_t> next{0}, next_buf{0};
        const size_t nt = std::min<size_t>(std::max(1u, std::thread::hardware_concurrency()), 8);
        std::vector<std::thread> workers;
        for (size_t t = 0; t < nt; ++t)
            workers.emplace_back([&] {
                for (size_t i; (i = next.fetch_add(1)) < units.size();) {
                    const size_t bi = unit_buffer(units[i]);
                    if (!ready[bi].load(std::memory_order_acquire)) {
                        std::unique_lock<std::mutex> lk(cv_mu);
                        cv.wait(lk, [&] { return ready[bi].load(std::memory_order_acquire) || abort.load(); });
                    }
                    std::string e;
                    if (ok && !abort && !fill_unit(units[i], base, e)) fail(e);
                }
            });
        auto allocate = [&] {
            for (size_t i; ok && (i = next_buf.fetch_add(1)) < nb;) {
                auto buf = std::make_unique<vk::Buffer>();
                std::string e;
                if (!ctx.create_buffer(*buf, sizes[i], vk::Mem::Device, e)) fail(e);
                else if (!buf->host) fail("fast engine: weight buffer is not host-mapped");
                if (!ok) break;
                base[i] = (uint8_t*)buf->host;
                bufs[i] = std::move(buf);
                for (Blob& b : blobs_)
                    if (b.buffer == i && !b.words.empty()) std::memcpy(base[i] + b.off, b.words.data(), b.words.size() * 4);
                {
                    std::lock_guard<std::mutex> lk(cv_mu);
                    ready[i].store(true, std::memory_order_release);
                }
                cv.notify_all();
            }
        };
        size_t na = 2;
        if (const char* e = std::getenv("STARLING_FAST_ALLOC_THREADS")) na = (size_t)std::max(1, std::atoi(e));
        std::vector<std::thread> allocators;
        for (size_t a = 1; a < std::min(na, nb); ++a) allocators.emplace_back(allocate);
        allocate();
        for (auto& t : allocators) t.join();
        t1 = std::chrono::steady_clock::now();
        if (!ok) {
            {
                std::lock_guard<std::mutex> lk(cv_mu);
                abort = true;   // release workers waiting on buffers that never came
            }
            cv.notify_all();
        }
        for (auto& t : workers) t.join();
        if (!ok) { err = first_err; return false; }
        for (size_t i = 0; i < nb; ++i) {
            total_ += sizes[i];
            ctx.flush(*bufs[i], 0, bufs[i]->size);
            buffers_.push_back(std::move(bufs[i]));
        }
    } else {
        for (VkDeviceSize sz : sizes) {
            auto buf = std::make_unique<vk::Buffer>();
            if (!ctx.create_buffer(*buf, sz, vk::Mem::DeviceOnly, err)) return false;
            buffers_.push_back(std::move(buf));
            total_ += sz;
        }
        t1 = std::chrono::steady_clock::now();
        // One buffer at a time through host memory: the host holds at most
        // one buffer's worth of repacked rows.
        std::vector<std::vector<size_t>> by_buf(buffers_.size());
        for (size_t i = 0; i < units.size(); ++i) {
            const Unit& u = units[i];
            by_buf[blobs_[u.mat == SIZE_MAX ? u.part : mats_[u.mat].q].buffer].push_back(i);
        }
        std::vector<uint32_t> stage((size_t)(*std::max_element(sizes.begin(), sizes.end()) + 3) / 4);
        for (size_t bi = 0; bi < buffers_.size(); ++bi) {
            std::fill(base.begin(), base.end(), nullptr);
            base[bi] = (uint8_t*)stage.data();
            for (Blob& b : blobs_)
                if (b.buffer == bi && !b.words.empty())
                    std::memcpy(base[bi] + b.off, b.words.data(), b.words.size() * 4);
            run_units(base, by_buf[bi]);
            if (!ok) { err = first_err; return false; }
            if (!ctx.upload(*buffers_[bi], 0, stage.data(), (size_t)sizes[bi], err)) return false;
        }
    }
    // The planned matrices keep only their shapes; small host blobs and the
    // producers (and whatever they captured) go too.
    for (Blob& b : blobs_) {
        std::vector<uint32_t>().swap(b.words);
        b.fn = nullptr;
    }
    std::vector<Planned>().swap(mats_);
    if (env_flag("STARLING_FAST_TIMING", false)) {
        using ms = std::chrono::duration<double, std::milli>;
        // Mapped: the fill runs from the first allocation on; "allocated" is
        // when the last buffer was created, "filled" when the last unit was.
        std::fprintf(stderr, "[fast] arena: %zu buffers %.1f MiB (%s), allocated %.1f ms, filled %.1f ms\n",
                     buffers_.size(), total_ / 1048576.0, mapped ? "mapped" : "staged", ms(t1 - t0).count(),
                     ms(std::chrono::steady_clock::now() - t0).count());
    }
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
        // BK = 16: the half-depth k-tile halves each thread's tile-load
        // registers and the shared tiles (10 -> 5 KiB), and more workgroups
        // stay resident: encoder/prefill GEMMs +20-40 % isolated, Qwen3-ASR
        // 0.6B enc+prefill -12 % short / -15 % medium, Parakeet encoder -20 %
        // (bit-exact; BK = 8 gains less, RESEARCH_LOG P5).
        tile = TileCfg{32, 128, 4, 8, 16};
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
        // ...but at least 16 rows per row slot (rows 64 at K = 1024, 32 at
        // K = 2048), kept at >= 64 workgroups: Qwen3-0.6B decode 39.8 ->
        // 31.1 ms/step, MOSS 74.7 -> 72.6 (RESEARCH_LOG P4-2).
        gemv_rpp_min = 16;
    } else if (ctx.info().vendor_id == 0x1002 && !tune_forced) {
        // AMD (measured on a RADV RENOIR iGPU, MOSS q4e8 decode): cap rows at
        // 16 with no growth — the grow-to-384-WGs rule took the big-N GEMVs
        // (gateup, lm_head) to 32, and 16 measured 6.6 % faster in decode.
        // Small N shrinks deeper instead (min 512 WGs, down to 4-row
        // workgroups: o/down 4, qkv 8); pinning them to 16 cost 4.4 pt.
        // STARLING_FAST_TUNE=1 runs the tuner instead; STARLING_FAST_GEMV_ROWS
        // replaces the cap (the shrink still applies).
        gemv_rows_max = 16;
        gemv_tgt_wgs = ~0u;
        gemv_min_wgs = 512;
    } else if (!autotune(err)) {
        return false;
    }
    // W4 GEMV nibble unpack through unpackUnorm4x8 (~5 ALU ops per 8
    // weights instead of ~16). Decode GEMVs on PowerVR are issue-bound, not
    // bandwidth-bound (W4 and W8 both run at ~33 G weights/s): MOSS decode
    // -12.7 % on the Pixel 10 Pro. On RADV an isolated micro measured it
    // slower (38 -> 31 GB/s), but in context MOSS decode gained 1.8 pt on
    // top of the rows pin above. STARLING_FAST_W4U=0/1 overrides.
    w4_unpack_ = ctx.info().vendor_id == 0x1010 || ctx.info().vendor_id == 0x1002;
    if (const char* e = std::getenv("STARLING_FAST_W4U")) w4_unpack_ = e[0] == '1';
    // The same for W8: xor 0x80 turns the int8 into a biased byte, and one
    // unpackUnorm4x8 replaces four bitfieldExtracts + int-to-float
    // conversions. Neutral where only the lm_head is W8 (Qwen3/MOSS q4e8),
    // -10 % decode on Granite and -11 % on Voxtral, whose Q5_K linears are
    // repacked to W8 (Pixel 10 Pro; RESEARCH_LOG P4-7). STARLING_FAST_W8U
    // overrides.
    w8_unpack_ = ctx.info().vendor_id == 0x1010;
    if (const char* e = std::getenv("STARLING_FAST_W8U")) w8_unpack_ = e[0] == '1';
    rsplit_on_ = !(std::getenv("STARLING_FAST_NORSPLIT") && std::getenv("STARLING_FAST_NORSPLIT")[0] == '1');
    // Subgroup-broadcast GEMM (gemm_sg.comp, the structure of LiteRT's
    // PowerVR matmul): one row of A per lane, 16 output columns per
    // workgroup, the weights loaded and dequantized by 16 lanes and handed
    // to all of them with subgroupBroadcast, A through the texture path, no
    // shared memory. PowerVR (DXT, subgroup 128): ~2x gemm.comp's GFLOPS at
    // the encoder/prefill shapes (benchmarks/litert_gpu, RESEARCH_LOG P6).
    // Needs subgroup broadcast in compute shaders and a subgroup of >= 16
    // lanes that fits one workgroup. STARLING_FAST_SG=0/1 overrides the
    // vendor default; STARLING_FAST_SG_TEX=0 reads A from the storage buffer.
    {
        const vk::DeviceInfo& di = ctx.info();
        const bool sg_ok = di.subgroup_compute && (di.subgroup_ops & VK_SUBGROUP_FEATURE_BALLOT_BIT) &&
                           di.subgroup_size >= 16 && di.subgroup_size <= di.max_wg_invocations;
        sg_gemm_ = sg_ok && di.vendor_id == 0x1010;
        if (const char* e = std::getenv("STARLING_FAST_SG")) sg_gemm_ = sg_ok && e[0] == '1';
        if (const char* e = std::getenv("STARLING_FAST_SG_TEX")) sg_tex_ = e[0] == '1';

        // STARLING_FAST_SG_TEXEL_MAX (tests): a smaller view limit, so hosts
        // with huge limits exercise the row-blocked views PowerVR needs.
        sg_texel_max_ = di.max_texel_elems;
        if (const char* e = std::getenv("STARLING_FAST_SG_TEXEL_MAX"))
            sg_texel_max_ = std::min<uint32_t>(sg_texel_max_, (uint32_t)std::strtoul(e, nullptr, 10));
        if (sg_texel_max_ < 1024) sg_tex_ = false;
    }
    if (const char* e = std::getenv("STARLING_FAST_TILE")) {
        unsigned bm, bn, tm, tn, bk = tile.BK;
        if (std::sscanf(e, "%u,%u,%u,%u,%u", &bm, &bn, &tm, &tn, &bk) >= 4 &&
            (bk == 8 || bk == 16 || bk == 32)) {
            tile = TileCfg{bm, bn, tm, tn, bk};
            tile_pinned_ = true;
        }
    }
    if (const char* e = std::getenv("STARLING_FAST_BK")) {
        const unsigned bk = (unsigned)std::atoi(e);
        if (bk == 8 || bk == 16 || bk == 32) tile.BK = bk;
    }
    if (const char* e = std::getenv("STARLING_FAST_GEMV_RPP")) gemv_rpp_min = (uint32_t)std::atoi(e);
    if (const char* e = std::getenv("STARLING_FAST_GEMV_ROWS")) {
        unsigned r = (unsigned)std::atoi(e);
        if (r >= 8 && r % 8 == 0) {
            gemv_rows_max = r;
            gemv_tgt_wgs = ~0u;   // pinned: no adaptive growth
        }
    }
    // Diagnostics: STARLING_FAST_MICRO runs isolated kernel probes
    // (';'-separated list, one model load).
    if (const char* mi = std::getenv("STARLING_FAST_MICRO")) {
        std::string all(mi), one;
        for (char ch : all + ";") {
            if (ch != ';') { one += ch; continue; }
            if (!one.empty() && !micro(one.c_str(), err)) return false;
            one.clear();
        }
    }
    return true;
}

bool Kernels::sg_eligible(const GemmCall& c) const {
    if (!sg_gemm_ || c.b == BKind::F16T || c.a.K % 8 || !c.a.M || !c.a.N) return false;
    if ((c.b == BKind::W4 || c.b == BKind::W8) && c.a.K % 32) return false;
    if (c.a.lda % 8 || c.a.a_off % 8 || c.a.sa_hi % 8 || c.a.sa_lo % 8) return false;
    if (c.a_conv) return c.b == BKind::F16 && c.a.cv_cin % 8 == 0 && c.a.cv_cin >= 8;
    if (c.b == BKind::F16 && (c.a.ldb % 8 || c.a.b_off % 8 || c.a.sb_hi % 8 || c.a.sb_lo % 8)) return false;
    return true;
}

// gemm_sg.comp's push block: gemm.comp's plus the A view's first texel and
// the first row of the dispatch.
struct SgArgs {
    GemmArgs a;
    uint32_t a_tex0 = 0, m_first = 0;
};
static_assert(sizeof(SgArgs) <= vk::kPushBytes, "gemm_sg push constants");

bool Kernels::gemm_sg(vk::Recording& rec, const GemmCall& c, std::string& err) {
    const vk::DeviceInfo& di = ctx_->info();
    const uint32_t WG = di.subgroup_size;
    const std::string base = c.a_conv ? "gemm_sg_conv" : c.b == BKind::W4 ? "gemm_sg_w4"
                           : c.b == BKind::W8 ? "gemm_sg_w8" : "gemm_sg_f16";
    const uint32_t gx = ceil_div(c.a.N, 16);
    const std::vector<uint32_t> spec{WG, (uint32_t)c.epi, (uint32_t)c.act, c.bias_mode};
    auto bindings = [&](vk::Ref a) {
        return std::vector<vk::Ref>{a, or_dummy(c.Bq), or_dummy(c.Bs), or_dummy(c.C),
                                    or_dummy(c.bias), or_dummy(c.bias2), or_dummy(c.C2)};
    };
    // Elements of A (relative to the Ref) that rows [m0, m1) read: [lo, hi).
    // Conv rows (chunk, to, fo) read input rows 2*to-1 .. 2*to+1 of their chunk.
    auto conv_span = [&](uint32_t m0, uint32_t m1, uint64_t& elo, uint64_t& ehi) {
        const uint32_t per = c.a.cv_to * c.a.cv_fo;
        const uint64_t row_el = (uint64_t)c.a.cv_fi * c.a.cv_cin;
        const uint32_t ca = m0 / per, ta = (m0 % per) / c.a.cv_fo;
        const uint32_t cb = (m1 - 1) / per, tb = ((m1 - 1) % per) / c.a.cv_fo;
        const uint64_t tia = ta ? 2u * ta - 1u : 0u, tib = std::min<uint64_t>(2u * tb + 1u, c.a.cv_ti - 1u);
        elo = c.a.a_off + ((uint64_t)ca * c.a.cv_ti + tia) * row_el;
        ehi = c.a.a_off + ((uint64_t)cb * c.a.cv_ti + tib + 1u) * row_el;
    };
    const uint64_t lo = c.a.a_off;
    uint64_t hi;
    if (c.a_conv) {
        uint64_t l2;
        conv_span(0, c.a.M, l2, hi);
    } else {
        const uint64_t zh = (c.batch - 1) / c.a.nb_lo, zl = std::min(c.batch - 1, c.a.nb_lo - 1);
        hi = lo + zh * c.a.sa_hi + zl * c.a.sa_lo + (uint64_t)(c.a.M - 1) * c.a.lda + c.a.K;
    }
    const uint64_t max_bytes = (uint64_t)sg_texel_max_ * 16;
    const uint64_t align = di.min_texel_align;
    // A texture-path dispatch of rows [m_first, m_first + rows) reading
    // elements [elo, ehi): a view from the aligned byte below elo.
    auto tex_dispatch = [&](uint32_t m_first, uint32_t rows, uint64_t elo, uint64_t ehi) -> bool {
        const VkDeviceSize a0 = c.A.off + elo * 2;
        const VkDeviceSize voff = a0 / align * align;
        const VkDeviceSize need = c.A.off + ehi * 2 - voff;
        const VkDeviceSize avail = c.A.buf->size - voff;
        const VkDeviceSize range = std::min<VkDeviceSize>(std::min<VkDeviceSize>(avail, max_bytes), (need + 15) / 16 * 16);
        const vk::Pipeline* p = ctx_->pipeline((base + "_t").c_str(), spec, err);
        if (!p) return false;
        SgArgs a;
        a.a = c.a;
        // texel = (element >> 3) - a_tex0, with the element counted from the
        // Ref (A.off is 16-byte aligned, checked by the caller).
        a.a_tex0 = (uint32_t)(((int64_t)voff - (int64_t)c.A.off) / 16);
        a.m_first = m_first;
        rec.dispatch(*p, bindings(vk::Ref(*c.A.buf, voff, range)), &a, sizeof(a), gx, ceil_div(rows, WG),
                     c.batch);
        return true;
    };
    if (sg_tex_ && c.A.buf && c.A.off % 16 == 0) {
        const VkDeviceSize a0 = c.A.off + lo * 2;
        if (c.A.off + hi * 2 - a0 / align * align <= max_bytes) return tex_dispatch(0, c.a.M, lo, hi);
        // Conv: blocks of whole workgroups grown while their input window
        // fits one view (Qwen3's conv2 reads 24 MiB in all, ~0.6 MiB per
        // 128 output rows).
        if (c.a_conv && c.batch == 1) {
            bool ok = true;
            std::vector<std::pair<uint32_t, uint32_t>> blocks;
            for (uint32_t m0 = 0; m0 < c.a.M && ok;) {
                uint32_t rows = 0;
                for (uint32_t r = WG; ; r += WG) {
                    const uint32_t m1 = std::min(c.a.M, m0 + r);
                    uint64_t elo, ehi;
                    conv_span(m0, m1, elo, ehi);
                    const VkDeviceSize a0 = c.A.off + elo * 2;
                    if (c.A.off + ehi * 2 - a0 / align * align > max_bytes) break;
                    rows = m1 - m0;
                    if (m1 == c.a.M) break;
                }
                if (!rows) { ok = false; break; }
                blocks.push_back({m0, rows});
                m0 += rows;
            }
            if (ok) {
                for (const auto& b : blocks) {
                    uint64_t elo, ehi;
                    conv_span(b.first, b.first + b.second, elo, ehi);
                    if (!tex_dispatch(b.first, b.second, elo, ehi)) return false;
                }
                return true;
            }
        }
        // Too large for one view: row blocks of whole workgroups, each with
        // its own view (plain row-major A).
        if (!c.a_conv && c.batch == 1) {
            const uint64_t row_bytes = (uint64_t)c.a.lda * 2;
            const uint64_t fit = max_bytes > align + c.a.K * 2u ? (max_bytes - align - c.a.K * 2u) / row_bytes + 1 : 0;
            const uint32_t rb = (uint32_t)std::min<uint64_t>(fit / WG * WG, c.a.M);
            if (rb >= WG) {
                for (uint32_t m0 = 0; m0 < c.a.M; m0 += rb) {
                    const uint32_t rows = std::min(rb, c.a.M - m0);
                    const uint64_t elo = lo + (uint64_t)m0 * c.a.lda;
                    const uint64_t ehi = elo + (uint64_t)(rows - 1) * c.a.lda + c.a.K;
                    if (!tex_dispatch(m0, rows, elo, ehi)) return false;
                }
                return true;
            }
        }
    }
    const vk::Pipeline* p = ctx_->pipeline(base.c_str(), spec, err);
    if (!p) return false;
    SgArgs a;
    a.a = c.a;
    rec.dispatch(*p, bindings(or_dummy(c.A)), &a, sizeof(a), gx, ceil_div(c.a.M, WG), c.batch);
    return true;
}

bool Kernels::gemm(vk::Recording& rec, const GemmCall& c, std::string& err) {
    if (sg_eligible(c)) return gemm_sg(rec, c, err);
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
    if (c.a.N <= 64 && t.BN > 64 && !tile_pinned_) t = TileCfg{32, 64, 4, 4, t.BK};
    const uint32_t wg = (t.BM / t.TM) * (t.BN / t.TN);
    const vk::Pipeline* p = ctx_->pipeline(
        name.c_str(), {wg, t.BM, t.BN, t.TM, t.TN, (uint32_t)c.epi, (uint32_t)c.act, c.bias_mode, t.BK}, err);
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

uint32_t Kernels::gemv_rows(uint32_t N, uint32_t K) {
    uint32_t rows = gemv_rows_max;
    // Grow rows for large N: every workgroup re-reads the x vector once, so
    // at rows=8 the x traffic rivals the weights (ff_up: 12.6 MB x vs
    // 15.7 MB weights on MOSS; lm_head reads x 19k times). Keep at least
    // ~384 workgroups for occupancy and cap at 32 (48+ costs registers).
    while (rows < 32 && N / (rows * 2) >= gemv_tgt_wgs) rows *= 2;
    while (rows > 8 && N / rows < gemv_min_wgs) rows -= 8;
    // Deep shrink (min_wgs > 256, AMD): 4-row workgroups for small N give
    // shorter serial row chains and more workgroups in flight; x re-reads
    // stay in cache. Per-row math is unchanged.
    if (gemv_min_wgs > 256)
        while (rows > 4 && N / rows < gemv_min_wgs) rows -= 4;
    // Rows-per-thread floor: a thread's fixed work (its x slice, the fused
    // RMSNorm statistic, the shared-memory reduction) is amortized over
    // rows / RSPLIT row dots, so small K (more row slots per workgroup)
    // needs more rows. Never below 64 workgroups (the floor would starve
    // the small-N projections of a 0.6B model: o/down at N = 1024).
    if (gemv_rpp_min && K) {
        const uint32_t floor_rows = gemv_rpp_min * gemv_rsplit_target(K / 32);
        uint32_t r = std::max(rows, floor_rows);
        while (r > rows && N / r < 64) r /= 2;
        rows = std::max(rows, r);
    }
    return rows;
}

const char* Kernels::gemv_shader(GpuFmt f) const {
    return f == GpuFmt::W4 ? (w4_unpack_ ? "gemv_w4u" : "gemv_w4")
         : f == GpuFmt::W8 ? (w8_unpack_ ? "gemv_w8u" : "gemv_w8") : "gemv_f16";
}

uint32_t Kernels::gemv_rsplit_target(uint32_t lanes) const {
    const uint32_t target = std::max(32u, ctx_->info().subgroup_size);
    uint32_t rs = 1;
    while (rs < 8 && lanes * rs < target) rs *= 2;
    return rs;
}

bool Kernels::gemv(vk::Recording& rec, const Arena& ar, const GMat& w, vk::Ref x, vk::Ref y,
                   vk::Ref g, vk::Ref state, vk::Ref bias, uint32_t epi, GemvArgs a,
                   std::string& err) {
    const char* name = gemv_shader(w.fmt);
    const uint32_t lanes = w.K / 32;      // one thread per 32-wide K group
    if (w.K % 32 || a.x_off % 4 || a.x_off2 % 4 || lanes > ctx_->info().max_wg_invocations ||
        lanes > 1024) {
        err = "gemv: unsupported K / x offset";
        return false;
    }
    // Enough workgroups to fill the GPU: fewer rows per workgroup for small N.
    uint32_t rows = gemv_rows(w.N, w.K);
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
    // Warm up for >= 100 ms first: GPU governors (PowerVR's polls every
    // 20 ms) start a short probe at an idle clock.
    {
        const auto w0 = std::chrono::steady_clock::now();
        do {
            if (!rec.submit_and_wait(err)) return -1.0;
        } while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count() < 100.0);
    }
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
    if (std::strncmp(mi, "gemm", 4) == 0 && mi[4] != 'x') {
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
    if (std::strncmp(mi, "gemmx,", 6) == 0) {
        // Prefill/encoder GEMM probe at a real shape:
        // STARLING_FAST_MICRO=gemmx,M,N,K,kind[,reps[,variant[,BM,BN,TM,TN[,BK]]]]
        // with kind w4|w8|f16|f16t and variant f (gemm.comp, f32 products) |
        // h (gemm.comp, packed f16) | g (gemm_sg.comp) | - (the device
        // default). Prints GFLOPS, the max relative error of 64 outputs
        // against a CPU reference, and how many outputs differ bitwise from
        // gemm.comp's f32 kernel at the device's default tile.
        unsigned M = 0, N = 0, Kd = 0, reps = 8;
        char kind[8] = {0}, var[8] = {0};
        unsigned tb[5] = {0, 0, 0, 0, 32};
        const int got = std::sscanf(mi, "gemmx,%u,%u,%u,%7[a-z0-9],%u,%7[a-z-],%u,%u,%u,%u,%u", &M, &N, &Kd, kind,
                                    &reps, var, &tb[0], &tb[1], &tb[2], &tb[3], &tb[4]);
        const std::string kd(kind);
        if (got < 4 || !M || !N || !Kd || Kd % 32 || !reps ||
            (kd != "w4" && kd != "w8" && kd != "f16" && kd != "f16t")) {
            err = "STARLING_FAST_MICRO=gemmx,M,N,K,w4|w8|f16|f16t[,reps[,f|h|g|-[,BM,BN,TM,TN[,BK]]]]";
            return false;
        }
        std::mt19937 rng(777);
        auto unif = [&] { return (float)(rng() % 2001) / 1000.0f - 1.0f; };
        std::vector<float> af((size_t)M * Kd);
        for (auto& v : af) v = unif();
        const std::vector<uint32_t> a16 = pack_f16(af.data(), af.size());
        std::vector<uint32_t> bq, bs;
        // Reference weights w[n][k] as the kernel sees them.
        auto wref = std::vector<float>((size_t)N * Kd);
        auto h2f = [](uint32_t h) {
            const uint32_t sg = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
            uint32_t b = e == 0 ? (sg << 31) : (sg << 31) | ((e + 112) << 23) | (m << 13);
            float f;
            std::memcpy(&f, &b, 4);
            return f;
        };
        if (kd == "w4" || kd == "w8") {
            const uint32_t G = Kd / 32;
            bq.resize((size_t)N * G * (kd == "w4" ? 4 : 8));
            bs.resize((size_t)N * G);
            for (auto& v : bq) v = (uint32_t)rng();
            for (uint32_t n = 0; n < N; ++n)
                for (uint32_t g = 0; g < G; ++g) {
                    const float so[2] = {0.01f + 0.02f * (unif() + 1.0f), (kd == "w4" ? -0.15f : 0.01f + 0.02f * (unif() + 1.0f))};
                    const uint32_t w = pack_f16(so, 2)[0];
                    bs[(size_t)n * G + g] = w;
                    for (uint32_t i = 0; i < 32; ++i) {
                        float v;
                        if (kd == "w4") {
                            const uint32_t q = (bq[((size_t)n * G + g) * 4 + i / 8] >> (4 * (i % 8))) & 15u;
                            v = h2f(w & 0xffff) * (float)q + h2f(w >> 16);
                        } else {
                            const int q = (int8_t)((bq[((size_t)n * G + g) * 8 + i / 4] >> (8 * (i % 4))) & 0xffu);
                            v = h2f(i < 16 ? (w & 0xffff) : (w >> 16)) * (float)q;
                        }
                        wref[(size_t)n * Kd + g * 32 + i] = v;
                    }
                }
        } else {
            std::vector<float> b((size_t)N * Kd);
            for (auto& v : b) v = unif();
            if (kd == "f16") {
                bq = pack_f16(b.data(), b.size());
                for (size_t i = 0; i < b.size(); ++i) wref[i] = b[i];
            } else {   // [K, N]
                bq = pack_f16(b.data(), b.size());
                for (uint32_t k = 0; k < Kd; ++k)
                    for (uint32_t n = 0; n < N; ++n) wref[(size_t)n * Kd + k] = b[(size_t)k * N + n];
            }
            for (auto& v : wref) v = h2f(pack_f16(&v, 1)[0] & 0xffff);
        }
        for (auto& v : af) v = h2f(pack_f16(&v, 1)[0] & 0xffff);
        vk::Buffer A, Bq, Bs, C;
        auto up = [&](vk::Buffer& b, const std::vector<uint32_t>& d) {
            return ctx_->create_buffer(b, std::max<size_t>(d.size() * 4, 16), vk::Mem::Device, err) &&
                   (d.empty() || ctx_->upload(b, 0, d.data(), d.size() * 4, err));
        };
        if (!up(A, a16) || !up(Bq, bq) || (!bs.empty() && !up(Bs, bs)) ||
            !ctx_->create_buffer(C, (size_t)M * N * 4, vk::Mem::Device, err))
            return false;
        GemmCall u;
        u.b = kd == "w4" ? BKind::W4 : kd == "w8" ? BKind::W8 : kd == "f16" ? BKind::F16 : BKind::F16T;
        u.epi = Epi::F32;
        u.a.M = M; u.a.N = N; u.a.K = Kd; u.a.lda = Kd; u.a.ldb = kd == "f16t" ? N : Kd; u.a.ldc = N;
        u.A = vk::Ref(A); u.Bq = vk::Ref(Bq); if (!bs.empty()) u.Bs = vk::Ref(Bs); u.C = vk::Ref(C);
        const bool saved_f16 = f16_math, saved_sg = sg_gemm_;
        if (var[0] == 'f') { f16_math = false; sg_gemm_ = false; }
        if (var[0] == 'h') { f16_math = ctx_->info().f16; sg_gemm_ = false; }
        if (var[0] == 'g') {
            const vk::DeviceInfo& di = ctx_->info();
            sg_gemm_ = di.subgroup_compute && (di.subgroup_ops & VK_SUBGROUP_FEATURE_BALLOT_BIT) &&
                       di.subgroup_size >= 16;
            if (!sg_gemm_) { err = "gemmx g: no subgroup broadcast"; f16_math = saved_f16; sg_gemm_ = saved_sg; return false; }
        }
        const bool use_sg = sg_eligible(u);
        const std::string sfx = use_sg ? (sg_tex_ ? "_sg_t" : "_sg") : f16_math ? "_h" : "";
        const TileCfg saved_tile = tile;
        const bool saved_pin = tile_pinned_;
        if (got >= 10) { tile = TileCfg{tb[0], tb[1], tb[2], tb[3], tb[4]}; tile_pinned_ = true; }
        const TileCfg used_tile = tile;
        vk::Recording rec(*ctx_);
        rec.begin();
        bool ok = true;
        for (uint32_t i = 0; i < reps && ok; ++i) {
            ok = gemm(rec, u, err);
            rec.barrier();
        }
        rec.end();
        vk::Recording r1(*ctx_);
        r1.begin();
        ok = ok && gemm(r1, u, err);
        r1.end();
        f16_math = saved_f16;
        sg_gemm_ = saved_sg;
        tile = saved_tile;
        tile_pinned_ = saved_pin;
        if (!ok) return false;
        const double ms = time_ms(rec, err);
        if (ms < 0 || !r1.submit_and_wait(err)) return false;
        std::vector<float> c((size_t)M * N);
        if (!ctx_->download(C, 0, c.data(), c.size() * 4, err)) return false;
        size_t ndiff = 0;
        {
            // Bitwise comparison against gemm.comp's f32-product kernel.
            const bool f2 = f16_math, s2 = sg_gemm_;
            f16_math = false;
            sg_gemm_ = false;
            vk::Recording r2(*ctx_);
            r2.begin();
            const bool ok2 = gemm(r2, u, err);
            r2.end();
            f16_math = f2;
            sg_gemm_ = s2;
            if (!ok2 || !r2.submit_and_wait(err)) return false;
            std::vector<float> c2((size_t)M * N);
            if (!ctx_->download(C, 0, c2.data(), c2.size() * 4, err)) return false;
            for (size_t i = 0; i < c.size(); ++i) ndiff += std::memcmp(&c[i], &c2[i], 4) != 0;
        }
        double maxerr = 0;
        for (int t = 0; t < 64; ++t) {
            const uint32_t m = (uint32_t)(rng() % M), n = (uint32_t)(rng() % N);
            double ref = 0, mag = 0;
            for (uint32_t k = 0; k < Kd; ++k) {
                ref += (double)af[(size_t)m * Kd + k] * wref[(size_t)n * Kd + k];
                mag += std::abs((double)af[(size_t)m * Kd + k] * wref[(size_t)n * Kd + k]);
            }
            maxerr = std::max(maxerr, std::abs(ref - c[(size_t)m * N + n]) / std::max(1e-6, mag));
        }
        std::fprintf(stderr, "[fast-micro] gemmx %s%s M=%u N=%u K=%u tile %u,%u,%u,%u,%u: %.3f ms = %.1f GFLOPS (err %.2e, %zu differ from f)\n",
                     kd.c_str(), sfx.c_str(), M, N, Kd, used_tile.BM, used_tile.BN, used_tile.TM, used_tile.TN, used_tile.BK, ms / reps,
                     2.0 * M * N * Kd * reps / (ms * 1e-3) / 1e9, maxerr, ndiff);
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
    if (std::strncmp(mi, "alu2", 4) == 0) {
        // Register-tiled FMA probe: STARLING_FAST_MICRO=alu2,groups,iters,RA,RB,mode
        // (RA x RB outer product of register operands per iteration).
        uint32_t groups = 480, iters = 1024, ra = 4, rb = 8, mode = 0, sh = 1;
        std::sscanf(mi, "alu2,%u,%u,%u,%u,%u,%u", &groups, &iters, &ra, &rb, &mode, &sh);
        vk::Buffer ob;
        std::vector<float> zeros(128 * (groups + 1), 0.0f);
        if (!ctx_->create_buffer(ob, zeros.size() * 4, vk::Mem::Device, err) ||
            !ctx_->upload(ob, 0, zeros.data(), zeros.size() * 4, err))
            return false;
        const vk::Pipeline* p = ctx_->pipeline("alu2_probe", {ra, rb, mode, std::max(sh, 1u)}, err);
        if (!p) return false;
        struct { uint32_t iters; } pc{iters};
        vk::Recording rec(*ctx_);
        rec.begin();
        rec.dispatch(*p, {vk::Ref(ob)}, &pc, sizeof(pc), groups);
        rec.end();
        const double ms = time_ms(rec, err);
        if (ms < 0) return false;
        const double fma = (double)groups * 128 * iters * ra * rb;
        std::fprintf(stderr, "[fast-micro] alu2 groups=%u RA=%u RB=%u mode=%u shared=%u B: %.1f GFLOPS (%.3f ms)\n",
                     groups, ra, rb, mode, 4 * sh, 2.0 * fma / (ms * 1e-3) / 1e9, ms);
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
    if (!rows) rows = gemv_rows(n, k);
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
    const char* name = m2 ? "gemv_w4um" : gemv_shader(bits == 4 ? GpuFmt::W4 : bits == 8 ? GpuFmt::W8 : GpuFmt::F16);
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
        const uint32_t r = gemv_rows(F, 0);
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
