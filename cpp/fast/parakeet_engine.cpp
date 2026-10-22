// parakeet_engine.cpp — see parakeet_engine.hpp.

#include "parakeet_engine.hpp"

#include "cpu_kernels.hpp"
#include "kernels.hpp"
#include "pk_mel.hpp"
#include "vk_runtime.hpp"
#include "weights.hpp"

#include "parakeet/loader.hpp"
#include "parakeet/mel.hpp"
#include "parakeet/pos_enc.hpp"
#include "parakeet/tdt.hpp"
#include "parakeet/tokenizer.hpp"
#include "runtime/call_abort.hpp"

#include "ggml.h"

#if defined(STARLING_HAVE_NPU)
#include "gguf_fingerprint.hpp"    // cpp/npu: Pixel TPU encoder artifacts
#include "npu_status.hpp"
#include "tpu_runtime.hpp"

#include <unistd.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <map>

namespace starling::fast {

namespace pk = starling::ggml::parakeet;

namespace {

bool env_on(const char* name) {
    const char* v = std::getenv(name);
    return v && v[0] && v[0] != '0';
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int stage_len(int x) { return (x + 2 - 3) / 2 + 1; }

struct Layer {
    GMat ff1_w1, ff1_w2, ff2_w1, ff2_w2, q, k, v, o, pos, pw1, pw2;
    Arena::Id ln_ff1_g, ln_ff1_b, ln_att_g, ln_att_b, ln_conv_g, ln_conv_b;
    Arena::Id ln_ff2_g, ln_ff2_b, ln_out_g, ln_out_b;
    Arena::Id bias_u, bias_v, dw_w, bn_scale, bn_shift;
    // Optional linear biases (absent in parakeet-tdt-0.6b-v3).
    int ff1_b1 = -1, ff1_b2 = -1, ff2_b1 = -1, ff2_b2 = -1;
    int k_b = -1, v_b = -1, o_b = -1, pos_b = -1, pw1_b = -1, pw2_b = -1;
};

#if defined(STARLING_HAVE_NPU)
// The opened TPU pieces: the subsampling chunk and the conformer segments.
struct PkTpu {
    std::unique_ptr<npu::TpuModel> sub;
    std::vector<std::unique_ptr<npu::TpuModel>> segs;
};
using PkTpuResult = std::pair<std::unique_ptr<PkTpu>, std::string>;
#endif

} // namespace

struct ParakeetEngine::Impl {
    // Keeps the shared Vulkan context alive while this engine exists.
    // Declared first, so it is released last — after every buffer and
    // recording below; the last engine's release destroys the device.
    std::shared_ptr<vk::Context> ctx_hold;
    vk::Context* ctx = nullptr;
    Kernels K;
    Arena ar;
    pk::Config cfg;
    pk::MelConstants mel;
    std::unique_ptr<PkMel> fmel;

    uint32_t D = 0, H = 0, dk = 0, FF = 0, L = 0, CK = 0, SC = 0, NM = 0, JH = 0;
    std::vector<Layer> layers;
    // subsampling
    Arena::Id c0_w, c0_b, c2_w, c2_b, c3_b, c5_w, c5_b, c6_b, out_b, jenc_b;
    GMat c3_pw, c6_pw, out_w, jenc;

    // ---- CPU transducer decoder ----
    uint32_t PH = 0, V1 = 0, n_dur = 0;   // pred hidden, vocab+1, durations
    std::vector<float> embed;              // [V1][PH]
    std::vector<CpuQ8> w_ih, w_hh;
    std::vector<std::vector<float>> b_ih, b_hh;
    CpuQ8 jpred, jout;
    std::vector<float> jpred_b, jout_b;
    // Layer-0 input projection W_ih0·embed[tok] + b_ih0 memoized per token id
    // (row V1-1 doubles as the start-of-sequence zero input). Same GEMV, so
    // bit-identical to computing it every step.
    std::vector<std::vector<float>> in0_cache;
    cpu::GemvHelper gemv2;   // decoder's second GEMV thread (parks when idle)

    // ---- runtime state ----
    int cap_T = 0;                         // scratch capacity (mel frames)
    int pos_cap = 0;                       // positional-table capacity (encoder frames)
    vk::Buffer mel_in, sub1, sub2a, sub2b, sub3a, sub3b, x, h, ffh, qu, qv, kk, vv;
    vk::Buffer S, BD, P, att, glu, dw, enc, pe, pos_tab;
    struct Rec { std::unique_ptr<vk::Recording> rec; uint64_t used = 0; int Tp = 0, valid = 0; };
    std::map<int, Rec> recs;
    uint64_t tick = 0;

    bool load(const pk::ParakeetModel& m, std::string& err);
    bool ensure_capacity(int T, std::string& err);
    bool ensure_pos_table(int Tp, std::string& err);
    bool record(int T, Rec& r, std::string& err);
    bool run_encoder(const std::vector<float>& feats, int T, std::vector<float>& enc_out,
                     int& Tp, std::string& err);
    // Sets *aborted (and returns early) when the caller cancels the call
    // (ggml::call_abort_requested, checked once per encoder frame).
    // `timing` receives each step's frame and duration (pk::TdtTiming).
    std::vector<int32_t> tdt_greedy(const std::vector<float>& enc_proj, int T,
                                    bool* aborted = nullptr, pk::TdtTiming* timing = nullptr);

    vk::Ref R(Arena::Id id) const { return ar.ref(id); }
    vk::Ref Ropt(int id) const { return id >= 0 ? ar.ref((Arena::Id)id) : vk::Ref(); }

    // ---- the encoder on the Pixel TPU (cpp/npu, scripts/npu/export_parakeet.py) ----
    // With all parakeet-enc-<model hash>-<piece>.stnpu artifacts of this GGUF
    // in STARLING_NPU_DIR (else STARLING_FAST_CACHE_DIR), inputs of up to one
    // TPU window (kParakeetTpuRows subsampled frames, ~33 s) run the whole
    // encoder on the TPU: the subsampling per 26-frame chunk, then the
    // conformer segments; the GPU encoder stays loaded for longer inputs,
    // STARLING_NPU=0 and any TPU error.
#if defined(STARLING_HAVE_NPU)
    std::unique_ptr<PkTpu> npu;
    std::future<PkTpuResult> npu_pending;
#endif
    std::string npu_note;   // why the TPU path is (not) used, for describe()
    bool npu_ready();
    bool npu_encode(const std::vector<float>& feats, int T, std::vector<float>& enc_out, int& Tp, std::string& err);
    // The encoder on the TPU when it can take T mel frames, else (or after a
    // TPU error) on the GPU.
    bool encode_mel(const std::vector<float>& feats, int T, std::vector<float>& enc_out, int& Tp, std::string& err);
};

ParakeetEngine::ParakeetEngine() : impl_(new Impl) {}
ParakeetEngine::~ParakeetEngine() {
    // Skipped on a hung GPU: vkDeviceWaitIdle has no timeout (#325); the
    // runtime then leaks this engine's objects to the driver instead.
    if (impl_ && impl_->ctx && !impl_->ctx->gpu_hung())
        impl_->ctx->fn().vkDeviceWaitIdle(impl_->ctx->device());
}

namespace {
// ---- Parakeet encoder artifacts on the Pixel TPU (cpp/npu) ----
// Only where a TPU can be used at all (a supported encoder shape, an
// artifact directory, a device whose firmware lists the TPU runtime) does
// anything run, and then on a worker that starts before the GPU weight load:
// it computes the model hash from the GGUF file itself (gguf_fingerprint:
// its own reads, so neither create() nor the caller, which releases the
// model, waits for it), then finds, checks, opens and warms up every piece;
// Impl::npu_ready picks the result up once it is there. Elsewhere nothing is
// hashed and the load is unchanged.
struct PkNpuStart {
    std::string note;
#if defined(STARLING_HAVE_NPU)
    std::future<PkTpuResult> fut;
#endif
};

#if defined(STARLING_HAVE_NPU)
struct PkNpuJob {
    std::string model, gguf, dirs;
    uint32_t nm = 0, d = 0, layers = 0, heads = 0, ff = 0, ck = 0, sc = 0, jh = 0;
    bool verbose = false;
};

PkTpuResult pk_npu_open(const PkNpuJob& j, uint64_t h) {
    PkTpuResult out;
    auto finish = [&](const std::string& s) {
        out.second = s;
        if (j.verbose) std::fprintf(stderr, "[fast-parakeet] npu: %s\n", s.c_str());
        return std::move(out);
    };
    auto report = [&](npu::PieceState st, const std::string& msg) {
        npu::report_piece("parakeet-enc", h, j.gguf, j.model, st, msg);
    };
    const std::vector<std::string> pieces = npu::parakeet_enc_pieces(j.layers);
    std::vector<std::string> paths, missing;
    for (const std::string& piece : pieces) {
        const std::string name = npu::parakeet_enc_artifact_name(h, piece);
        std::string path;
        for (size_t b = 0; b <= j.dirs.size() && path.empty();) {
            size_t e = j.dirs.find(':', b);
            if (e == std::string::npos) e = j.dirs.size();
            const std::string cand = j.dirs.substr(b, e - b) + "/" + name;
            if (e > b && access(cand.c_str(), R_OK) == 0) path = cand;
            b = e + 1;
        }
        if (path.empty()) missing.push_back(name);
        paths.push_back(path);
    }
    if (!missing.empty()) {
        std::string list;
        for (const std::string& n : missing) list += (list.empty() ? "" : ", ") + n;
        report(npu::PieceState::Missing, "no " + list);
        return finish("no artifact " + list + " in " + j.dirs);
    }
    std::string err;
    auto stale = [&](const std::string& why) {
        report(npu::PieceState::Stale, why);
        return finish("TPU file stale, recompile via adb: " + why);
    };
    if (!npu::load_southbound(err)) return finish("TPU runtime unavailable: " + err);
    const std::string vendor = npu::vendor_version();
    auto tpu = std::make_unique<PkTpu>();
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < pieces.size(); ++i) {
        npu::ArtifactInfo info;
        if (!npu::read_artifact_info(paths[i], info, err)) return stale(err);
        std::vector<uint64_t> in_bytes, out_bytes;
        npu::parakeet_enc_io(pieces[i], j.nm, j.d, j.layers, j.jh, in_bytes, out_bytes);
        const std::string key = npu::parakeet_enc_key(pieces[i], j.nm, j.d, j.layers, j.heads, j.ff, j.ck, j.sc, j.jh);
        if (info.key != key || info.model_hash != h || info.in_bytes != in_bytes || info.out_bytes != out_bytes)
            return stale(paths[i] + " was built by another exporter version ('" + info.key + "')");
        if (info.vendor != vendor)
            return stale(paths[i] + " was compiled for TPU firmware " + info.vendor + ", this phone has " + vendor);
        auto m = std::make_unique<npu::TpuModel>();
        if (!m->open(paths[i], err)) {
            report(npu::PieceState::Error, "open " + paths[i] + ": " + err);
            return finish("open " + paths[i] + ": " + err);
        }
        // One zero call: the first invocation pays the weight transfer.
        std::vector<float> z;
        for (size_t k = 0; k < in_bytes.size() && err.empty(); ++k) {
            z.assign(in_bytes[k] / 4, 0.0f);
            if (!m->write_input(k, z.data(), in_bytes[k], err)) break;
        }
        if (!err.empty() || !m->run(err)) {
            report(npu::PieceState::Error, "warm-up " + paths[i] + ": " + err);
            return finish("warm-up " + paths[i] + ": " + err);
        }
        if (!env_on("STARLING_NPU_KEEP_MAP")) m->drop_mapped_pages();
        if (i == 0) tpu->sub = std::move(m);
        else tpu->segs.push_back(std::move(m));
    }
    char b[96];
    std::snprintf(b, sizeof b, " (%zu pieces, load + warm-up %.1f ms)", pieces.size(), ms_since(t0));
    report(npu::PieceState::Ok, paths[0]);
    out.first = std::move(tpu);
    return finish(std::string("encoder on the TPU") + b);
}
#endif

PkNpuStart pk_npu_start(const pk::ParakeetModel& m) {
    PkNpuStart r;
#if defined(STARLING_HAVE_NPU)
    PkNpuJob j;
    j.verbose = env_on("STARLING_FAST_VERBOSE") || env_on("STARLING_FAST_TIMING");
    auto note = [&](const std::string& s) {
        r.note = s;
        if (j.verbose) std::fprintf(stderr, "[fast-parakeet] npu: %s\n", s.c_str());
        return std::move(r);
    };
    if (const char* e = std::getenv("STARLING_NPU"); e && std::strcmp(e, "0") == 0) return note("disabled (STARLING_NPU=0)");
    const pk::Config& c = m.config;
    const ggml_tensor* jenc = m.loader.tensor("joint.enc.weight");
    // The shapes export_parakeet.py builds graphs for.
    if (!jenc || c.conv_norm_type != "batch_norm" || c.xscaling || c.n_mels % 8 || c.d_model % c.n_heads ||
        c.n_layers % npu::kParakeetTpuSegLayers || c.conv_kernel % 2 == 0 || c.ff_dim % 4)
        return r;
    j.dirs = npu::npu_dirs();
    if (j.dirs.empty()) return note("no STARLING_NPU_DIR");
    if (!npu::tpu_present()) return note("no TPU runtime on this device");
    j.nm = c.n_mels;
    j.d = c.d_model;
    j.layers = c.n_layers;
    j.heads = c.n_heads;
    j.ff = c.ff_dim;
    j.ck = c.conv_kernel;
    j.sc = c.subsampling_conv_channels;
    j.jh = (uint32_t)jenc->ne[1];
    j.model = "arch=parakeet";
    j.gguf = m.loader.path();
    r.note = "checking the TPU artifacts";
    auto work = [j]() {
        const auto t0 = std::chrono::steady_clock::now();
        uint64_t h = 0;
        std::string e;
        bool ok = false;
        try {
            ok = npu::gguf_model_hash(j.gguf, h, e);
        } catch (const std::exception& x) {
            e = x.what();
        }
        if (!ok) return PkTpuResult{nullptr, "no model fingerprint: " + e};
        if (j.verbose) std::fprintf(stderr, "[fast-parakeet] npu: encoder digest %.1f ms\n", ms_since(t0));
        return pk_npu_open(j, h);
    };
    try {
        r.fut = std::async(std::launch::async, std::move(work));
    } catch (const std::system_error& e) {   // no thread: the GPU encoder stays in charge
        return note(std::string("no worker thread: ") + e.what());
    }
#else
    (void)m;
#endif
    return r;
}
} // namespace

std::unique_ptr<ParakeetEngine> ParakeetEngine::create(const pk::ParakeetModel& m, std::string& err) {
    const auto t0 = std::chrono::steady_clock::now();
    PkNpuStart ns = pk_npu_start(m);   // TPU artifact hashing + open overlap the GPU load
    std::shared_ptr<vk::Context> hold = vk::Context::acquire(err);
    if (!hold) return nullptr;
    vk::Context* ctx = hold.get();
    std::unique_ptr<ParakeetEngine> e(new ParakeetEngine());
    e->impl_->ctx_hold = std::move(hold);
    e->impl_->ctx = ctx;
    const double t_ctx = ms_since(t0);
    if (!e->impl_->K.init(*ctx, err)) return nullptr;
    const double t_kinit = ms_since(t0);
    if (!e->impl_->load(m, err)) return nullptr;
    if (env_on("STARLING_FAST_TIMING"))
        std::fprintf(stderr, "[fast-parakeet] create: device %.1f ms, kernels %.1f ms, weights %.1f ms\n",
                     t_ctx, t_kinit - t_ctx, ms_since(t0) - t_kinit);
    e->impl_->npu_note = std::move(ns.note);
#if defined(STARLING_HAVE_NPU)
    e->impl_->npu_pending = std::move(ns.fut);
#endif
    return e;
}

bool ParakeetEngine::Impl::npu_ready() {
#if defined(STARLING_HAVE_NPU)
    // Does not wait (unless STARLING_NPU_WAIT=1, for measurements): while the
    // worker still opens the artifacts (~1.5 s for 1.2 GB on a Pixel 10
    // Pro), calls run on the GPU encoder as before.
    static const bool wait = env_on("STARLING_NPU_WAIT");
    if (npu_pending.valid() &&
        (wait || npu_pending.wait_for(std::chrono::seconds(0)) == std::future_status::ready)) {
        try {
            auto r = npu_pending.get();
            npu = std::move(r.first);
            npu_note = std::move(r.second);
        } catch (const std::exception& e) {
            npu.reset();
            npu_note = std::string("open failed: ") + e.what();
        }
    }
    return npu != nullptr;
#else
    return false;
#endif
}

bool ParakeetEngine::Impl::npu_encode(const std::vector<float>& feats, int T, std::vector<float>& enc_out, int& Tp,
                                      std::string& err) {
#if defined(STARLING_HAVE_NPU)
    const uint32_t C = npu::kParakeetTpuChunk, R = npu::kParakeetTpuRows;
    const int T1 = stage_len(T), T2 = stage_len(T1), T3 = stage_len(T2);
    int valid = T - 1;
    for (int s = 0; s < 3; ++s) valid = stage_len(valid);
    valid = std::min(valid, T3);
    if (T3 > (int)R) {
        err = "longer than one TPU window";
        return false;
    }
    const auto t0 = std::chrono::steady_clock::now();
    // Subsampling, per 26-frame chunk: the mel rows it reads (zero outside
    // [0, T)), and which of its stage-1/2 rows lie inside the clip (the GPU
    // convs read zero padding past each stage's length). Chunks past the
    // last frame are not run: their rows are padding, masked below.
    const uint32_t LM = 8 * C + 7, L1 = 4 * C + 3, L2 = 2 * C + 1;
    std::vector<float> x((size_t)R * D, 0.0f), mel((size_t)LM * NM), m1(L1), m2(L2), rmc(C);
    int chunks = 0;
    for (uint32_t k = 0; C * k < (uint32_t)T3; ++k, ++chunks) {
        const int g0 = (int)(8 * C * k) - 7, g1 = (int)(4 * C * k) - 3, g2 = (int)(2 * C * k) - 1;
        for (uint32_t r = 0; r < LM; ++r) {
            const int g = g0 + (int)r;
            float* d = &mel[(size_t)r * NM];
            if (g >= 0 && g < T) std::memcpy(d, &feats[(size_t)g * NM], NM * 4);
            else std::fill(d, d + NM, 0.0f);
        }
        for (uint32_t r = 0; r < L1; ++r) m1[r] = (g1 + (int)r >= 0 && g1 + (int)r < T1) ? 1.0f : 0.0f;
        for (uint32_t r = 0; r < L2; ++r) m2[r] = (g2 + (int)r >= 0 && g2 + (int)r < T2) ? 1.0f : 0.0f;
        for (uint32_t r = 0; r < C; ++r) rmc[r] = (int)(C * k + r) < valid ? 1.0f : 0.0f;
        npu::TpuModel& s = *npu->sub;
        if (!s.write_input(0, mel.data(), mel.size() * 4, err) || !s.write_input(1, m1.data(), m1.size() * 4, err) ||
            !s.write_input(2, m2.data(), m2.size() * 4, err) || !s.write_input(3, rmc.data(), rmc.size() * 4, err) ||
            !s.run(err) || !s.read_output(0, &x[(size_t)C * k * D], (size_t)C * D * 4, err))
            return false;
    }
    const double t_sub = ms_since(t0);
    // Conformer segments over all rows: row mask (valid frames) and additive
    // key mask.
    std::vector<float> rm(R), km(R);
    for (uint32_t r = 0; r < R; ++r) {
        rm[r] = (int)r < valid ? 1.0f : 0.0f;
        km[r] = (int)r < valid ? 0.0f : -1e4f;
    }
    std::vector<float> out((size_t)R * JH);
    for (size_t i = 0; i < npu->segs.size(); ++i) {
        npu::TpuModel& s = *npu->segs[i];
        const bool last = i + 1 == npu->segs.size();
        if (!s.write_input(0, x.data(), x.size() * 4, err) || !s.write_input(1, rm.data(), R * 4, err) ||
            !s.write_input(2, km.data(), R * 4, err) || !s.run(err) ||
            !s.read_output(0, last ? out.data() : x.data(), (last ? out.size() : x.size()) * 4, err))
            return false;
    }
    Tp = T3;
    enc_out.assign(out.begin(), out.begin() + (size_t)T3 * JH);
    if (env_on("STARLING_FAST_TIMING"))
        std::fprintf(stderr, "[fast-parakeet] npu encoder: subsampling %d chunks %.1f ms, layers %.1f ms\n", chunks,
                     t_sub, ms_since(t0) - t_sub);
    return true;
#else
    (void)feats; (void)T; (void)enc_out; (void)Tp;
    err = "built without the TPU path";
    return false;
#endif
}

bool ParakeetEngine::Impl::encode_mel(const std::vector<float>& feats, int T, std::vector<float>& enc_out, int& Tp,
                                      std::string& err) {
#if defined(STARLING_HAVE_NPU)
    if (npu_ready() && stage_len(stage_len(stage_len(T))) <= (int)npu::kParakeetTpuRows) {
        std::string nerr;
        if (npu_encode(feats, T, enc_out, Tp, nerr)) return true;
        // Any TPU error: the GPU encoder from here on.
        npu.reset();
        npu_note = "failed: " + nerr;
        std::fprintf(stderr, "[fast-parakeet] TPU encoder failed (%s); the GPU runs it from now on\n", nerr.c_str());
    }
#endif
    return run_encoder(feats, T, enc_out, Tp, err);
}

std::string ParakeetEngine::describe() const {
    char buf[256];
    std::snprintf(buf, sizeof buf, "fast/vulkan '%s' weights=%.0f MiB cpu=%s",
                  impl_->ctx->info().name.c_str(), impl_->ar.total_bytes() / 1048576.0,
                  cpu::isa_name());
    std::string d = buf;
    if (impl_->npu_ready()) d += " encoder=npu";
    if (impl_->npu_note.find("stale") != std::string::npos) d += " [" + impl_->npu_note + "]";
    return d;
}

bool ParakeetEngine::Impl::load(const pk::ParakeetModel& m, std::string& err) {
    const auto t_load0 = std::chrono::steady_clock::now();
    const auto& ml = m.loader;
    cfg = m.config;
    mel.read_from(ml, cfg);
    if (mel.hop_length == 0 || mel.n_fft < 2 || (mel.n_fft & (mel.n_fft - 1))) {
        err = "fast parakeet: mel needs a power-of-two n_fft and hop > 0";
        return false;
    }
    fmel = std::make_unique<PkMel>(mel);
    D = cfg.d_model; H = cfg.n_heads; dk = D / H; FF = cfg.ff_dim; L = cfg.n_layers;
    CK = cfg.conv_kernel; SC = cfg.subsampling_conv_channels; NM = cfg.n_mels;
    if (cfg.conv_norm_type != "batch_norm" || cfg.xscaling || NM != 128 || SC % 2 || D % 64) {
        err = "fast parakeet: unsupported configuration (needs batch_norm conv, no xscaling, 128 mels)";
        return false;
    }
    auto T = [&](const std::string& n) -> const ggml_tensor* {
        const ggml_tensor* t = ml.tensor(n.c_str());
        if (!t && err.empty()) err = "fast parakeet: missing tensor " + n;
        return t;
    };
    auto has = [&](const std::string& n) { return ml.tensor(n.c_str()) != nullptr; };
    auto f32 = [&](const std::string& n, std::vector<float>& v) {
        const ggml_tensor* t = T(n);
        return t && tensor_to_f32(t, v, err);
    };
    auto vec_id = [&](const std::string& n, Arena::Id& id) {
        std::vector<float> v;
        if (!f32(n, v)) return false;
        id = ar.add_f32(v);
        return true;
    };
    auto opt_vec = [&](const std::string& n, int& id, float scale = 1.0f) {
        if (!has(n)) return true;
        std::vector<float> v;
        if (!f32(n, v)) return false;
        for (float& x : v) x *= scale;
        id = (int)ar.add_f32(v);
        return true;
    };
    // Matrices are planned here and repacked in parallel, straight into
    // device memory, by ar.finalize(); the GGUF pages they come from are
    // dropped as they are consumed when the file is mapped.
    ar.release_sources(ml.weights_mapped());
    auto mat = [&](const std::string& n, GMat& g) {
        const ggml_tensor* t = T(n);
        return t && ar.add_matrix(tensor_rows(t, n), g, err);
    };
    // Conv weights stored with leading singleton dims: view as [N][K].
    auto mat_raw = [&](const std::string& n, uint32_t N_, uint32_t K_, GMat& g) {
        const ggml_tensor* t = T(n);
        if (!t) return false;
        if ((uint64_t)ggml_nelements(t) != (uint64_t)N_ * K_) {
            err = "fast parakeet: unexpected shape for " + n;
            return false;
        }
        return ar.add_matrix(raw_rows((int)t->type, t->data, N_, K_, n), g, err);
    };

    // ---- subsampling ----
    if (!vec_id("encoder.pre_encode.conv.0.weight", c0_w) || !vec_id("encoder.pre_encode.conv.0.bias", c0_b) ||
        !vec_id("encoder.pre_encode.conv.2.weight", c2_w) || !vec_id("encoder.pre_encode.conv.2.bias", c2_b) ||
        !mat_raw("encoder.pre_encode.conv.3.weight", SC, SC, c3_pw) || !vec_id("encoder.pre_encode.conv.3.bias", c3_b) ||
        !vec_id("encoder.pre_encode.conv.5.weight", c5_w) || !vec_id("encoder.pre_encode.conv.5.bias", c5_b) ||
        !mat_raw("encoder.pre_encode.conv.6.weight", SC, SC, c6_pw) || !vec_id("encoder.pre_encode.conv.6.bias", c6_b) ||
        !mat("encoder.pre_encode.out.weight", out_w) || !vec_id("encoder.pre_encode.out.bias", out_b))
        return false;

    // ---- conformer layers ----
    layers.resize(L);
    for (uint32_t l = 0; l < L; ++l) {
        Layer& Y = layers[l];
        const std::string p = "encoder.layers." + std::to_string(l) + ".";
        if (!mat(p + "feed_forward1.linear1.weight", Y.ff1_w1) || !mat(p + "feed_forward1.linear2.weight", Y.ff1_w2) ||
            !mat(p + "feed_forward2.linear1.weight", Y.ff2_w1) || !mat(p + "feed_forward2.linear2.weight", Y.ff2_w2) ||
            !mat(p + "self_attn.linear_q.weight", Y.q) || !mat(p + "self_attn.linear_k.weight", Y.k) ||
            !mat(p + "self_attn.linear_v.weight", Y.v) || !mat(p + "self_attn.linear_out.weight", Y.o) ||
            !mat(p + "self_attn.linear_pos.weight", Y.pos))
            return false;
        if (!opt_vec(p + "feed_forward1.linear1.bias", Y.ff1_b1) ||
            !opt_vec(p + "feed_forward1.linear2.bias", Y.ff1_b2, 0.5f) ||
            !opt_vec(p + "feed_forward2.linear1.bias", Y.ff2_b1) ||
            !opt_vec(p + "feed_forward2.linear2.bias", Y.ff2_b2, 0.5f) ||
            !opt_vec(p + "self_attn.linear_k.bias", Y.k_b) || !opt_vec(p + "self_attn.linear_v.bias", Y.v_b) ||
            !opt_vec(p + "self_attn.linear_out.bias", Y.o_b) || !opt_vec(p + "self_attn.linear_pos.bias", Y.pos_b) ||
            !opt_vec(p + "conv.pointwise_conv2.bias", Y.pw2_b))
            return false;
        // q projection fused with the two positional biases.
        std::vector<float> u, v, qb(D, 0.0f);
        if (!f32(p + "self_attn.pos_bias_u", u) || !f32(p + "self_attn.pos_bias_v", v)) return false;
        if (has(p + "self_attn.linear_q.bias") && !f32(p + "self_attn.linear_q.bias", qb)) return false;
        for (uint32_t i = 0; i < D; ++i) { u[i] += qb[i]; v[i] += qb[i]; }
        Y.bias_u = ar.add_f32(u);
        Y.bias_v = ar.add_f32(v);
        if (!vec_id(p + "norm_feed_forward1.weight", Y.ln_ff1_g) || !vec_id(p + "norm_feed_forward1.bias", Y.ln_ff1_b) ||
            !vec_id(p + "norm_self_att.weight", Y.ln_att_g) || !vec_id(p + "norm_self_att.bias", Y.ln_att_b) ||
            !vec_id(p + "norm_conv.weight", Y.ln_conv_g) || !vec_id(p + "norm_conv.bias", Y.ln_conv_b) ||
            !vec_id(p + "norm_feed_forward2.weight", Y.ln_ff2_g) || !vec_id(p + "norm_feed_forward2.bias", Y.ln_ff2_b) ||
            !vec_id(p + "norm_out.weight", Y.ln_out_g) || !vec_id(p + "norm_out.bias", Y.ln_out_b))
            return false;
        // Pointwise conv 1 with GLU halves interleaved: row 2j = a_j, 2j+1 = gate_j,
        // so the GEMM epilogue owns both halves of every GLU pair.
        {
            const std::string n = p + "conv.pointwise_conv1.weight";
            const ggml_tensor* t = T(n);
            if (!t) return false;
            if ((uint64_t)ggml_nelements(t) != (uint64_t)2 * D * D) {
                err = "fast parakeet: unexpected shape for " + n;
                return false;
            }
            const MatSpec all = raw_rows((int)t->type, t->data, 2 * D, D, n);
            if (!ar.add_matrix(spec_interleave(row_range(all, 0, D), row_range(all, D, D)), Y.pw1, err))
                return false;
            if (has(p + "conv.pointwise_conv1.bias")) {
                std::vector<float> b, bi(2 * D);
                if (!f32(p + "conv.pointwise_conv1.bias", b)) return false;
                for (uint32_t j = 0; j < D; ++j) { bi[2 * j] = b[j]; bi[2 * j + 1] = b[j + D]; }
                Y.pw1_b = (int)ar.add_f32(bi);
            }
        }
        if (!mat_raw(p + "conv.pointwise_conv2.weight", D, D, Y.pw2)) return false;
        // Depthwise conv: [C][K] -> [K][C]; batch-norm (+ conv bias) folded.
        {
            std::vector<float> w, g, b, mean, var, dwb;
            if (!f32(p + "conv.depthwise_conv.weight", w) || !f32(p + "conv.batch_norm.weight", g) ||
                !f32(p + "conv.batch_norm.bias", b) || !f32(p + "conv.batch_norm.running_mean", mean) ||
                !f32(p + "conv.batch_norm.running_var", var))
                return false;
            if (w.size() != (size_t)D * CK) { err = "fast parakeet: depthwise shape"; return false; }
            if (has(p + "conv.depthwise_conv.bias") && !f32(p + "conv.depthwise_conv.bias", dwb)) return false;
            std::vector<float> wt((size_t)CK * D), sc(D), sh(D);
            for (uint32_t c = 0; c < D; ++c)
                for (uint32_t k = 0; k < CK; ++k) wt[(size_t)k * D + c] = w[(size_t)c * CK + k];
            for (uint32_t c = 0; c < D; ++c) {
                sc[c] = g[c] / std::sqrt(var[c] + 1e-5f);
                sh[c] = b[c] - mean[c] * sc[c];
                if (!dwb.empty()) sh[c] += dwb[c] * sc[c];
            }
            Y.dw_w = ar.add_f32(wt);
            Y.bn_scale = ar.add_f32(sc);
            Y.bn_shift = ar.add_f32(sh);
        }
    }
    if (!mat("joint.enc.weight", jenc) || !vec_id("joint.enc.bias", jenc_b)) return false;
    JH = (uint32_t)T("joint.enc.weight")->ne[1];

    // ---- CPU decoder weights ----
    PH = cfg.pred_hidden;
    V1 = cfg.vocab_size + 1;
    n_dur = (uint32_t)cfg.tdt_durations.size();
    if (n_dur == 0) { err = "fast parakeet: no TDT durations"; return false; }
    {
        const ggml_tensor* t = T("decoder.prediction.embed.weight");
        if (!t) return false;
        std::vector<float> e;
        if (!tensor_to_f32(t, e, err)) return false;
        if ((uint32_t)t->ne[0] != PH || (uint32_t)t->ne[1] < V1 - 1) {
            err = "fast parakeet: prediction embedding shape mismatch";
            return false;
        }
        embed.assign((size_t)V1 * PH, 0.0f);   // a table without the blank row leaves it zero
        std::copy_n(e.begin(), std::min(e.size(), embed.size()), embed.begin());
    }
    const uint32_t PL = cfg.pred_rnn_layers ? cfg.pred_rnn_layers : 1;
    w_ih.resize(PL); w_hh.resize(PL); b_ih.resize(PL); b_hh.resize(PL);
    for (uint32_t l = 0; l < PL; ++l) {
        const std::string s = "_l" + std::to_string(l);
        const ggml_tensor* wi = T("decoder.prediction.dec_rnn.lstm.weight_ih" + s);
        const ggml_tensor* wh = T("decoder.prediction.dec_rnn.lstm.weight_hh" + s);
        if (!wi || !wh || !pack_cpu_q8(wi, w_ih[l], err) || !pack_cpu_q8(wh, w_hh[l], err) ||
            !f32("decoder.prediction.dec_rnn.lstm.bias_ih" + s, b_ih[l]) ||
            !f32("decoder.prediction.dec_rnn.lstm.bias_hh" + s, b_hh[l]))
            return false;
    }
    {
        const ggml_tensor* wp = T("joint.pred.weight");
        const ggml_tensor* wo = T("joint.joint_net.2.weight");
        if (!wp || !wo || !pack_cpu_q8(wp, jpred, err) || !pack_cpu_q8(wo, jout, err) ||
            !f32("joint.pred.bias", jpred_b) || !f32("joint.joint_net.2.bias", jout_b))
            return false;
        if (jout.N != V1 + n_dur || jpred.N != JH) { err = "fast parakeet: joint shape mismatch"; return false; }
    }

    // The stage-3 pointwise conv is the A operand of a GEMM: it must be f16.
    if (c6_pw.fmt != GpuFmt::F16) { err = "fast parakeet: pre_encode.conv.6 must be float"; return false; }
    if (out_w.K != SC * (uint32_t)stage_len(stage_len(stage_len((int)NM))) || out_w.N != D) {
        err = "fast parakeet: pre_encode.out shape mismatch";
        return false;
    }
    if (jenc.K != D) { err = "fast parakeet: joint.enc shape mismatch"; return false; }
    const double t_pack = ms_since(t_load0);
    if (!ar.finalize(*ctx, err)) return false;
    if (env_on("STARLING_FAST_TIMING"))
        std::fprintf(stderr, "[fast-parakeet] load: plan %.1f ms, repack+upload %.1f ms\n", t_pack,
                     ms_since(t_load0) - t_pack);
    if (env_on("STARLING_FAST_VERBOSE"))
        std::fprintf(stderr, "[fast] parakeet: %u layers, weights %.1f MiB on '%s', decoder %s\n", L,
                     ar.total_bytes() / 1048576.0, ctx->info().name.c_str(), cpu::isa_name());
    return true;
}

bool ParakeetEngine::Impl::ensure_capacity(int T, std::string& err) {
    if (T <= cap_T) return true;
    const int cT = std::max(T, cap_T + cap_T / 4);
    const int T1 = stage_len(cT), T2 = stage_len(T1), T3 = stage_len(T2);
    const int F1 = stage_len((int)NM), F2 = stage_len(F1), F3 = stage_len(F2);
    const size_t Tp = (size_t)T3;
    recs.clear();
    auto mk = [&](vk::Buffer& b, size_t bytes, vk::Mem kind = vk::Mem::Device) {
        return ctx->create_buffer(b, bytes, kind, err);
    };
    const size_t ldS = Tp, ldB = 2 * Tp, ldP = round_up((uint32_t)Tp, 8);
    const size_t attn_bytes = std::max({(size_t)H * Tp * ldB * 4, (size_t)H * Tp * ldS * 4});
    if (attn_bytes > ctx->info().max_storage_range) {
        err = "fast parakeet: input too long for this device's buffer range";
        return false;
    }
    if (!mk(mel_in, (size_t)cT * NM * 4, ctx->info().uma ? vk::Mem::Device : vk::Mem::Upload) ||
        !mk(sub1, (size_t)T1 * F1 * SC * 2) || !mk(sub2a, (size_t)T2 * F2 * SC * 2) ||
        !mk(sub2b, (size_t)T2 * F2 * SC * 2) || !mk(sub3a, (size_t)T3 * F3 * SC * 2) ||
        !mk(sub3b, (size_t)T3 * F3 * SC * 2) || !mk(x, Tp * D * 4) || !mk(h, Tp * D * 2) ||
        !mk(ffh, Tp * FF * 2) || !mk(qu, Tp * D * 2) || !mk(qv, Tp * D * 2) || !mk(kk, Tp * D * 2) ||
        !mk(vv, Tp * D * 2) || !mk(S, (size_t)H * Tp * ldS * 4) || !mk(BD, (size_t)H * Tp * ldB * 4) ||
        !mk(P, (size_t)H * Tp * ldP * 2) || !mk(att, Tp * D * 2) || !mk(glu, Tp * D * 2) ||
        !mk(dw, Tp * D * 2) || !mk(enc, Tp * JH * 4, vk::Mem::Readback))
        return false;
    cap_T = cT;
    return true;
}

bool ParakeetEngine::Impl::ensure_pos_table(int Tp, std::string& err) {
    if (Tp <= pos_cap) return true;
    const int cap = std::max(round_up((uint32_t)Tp, 64), (uint32_t)std::max(pos_cap + pos_cap / 2, 128));
    const uint32_t Pn = 2 * (uint32_t)cap - 1;
    recs.clear();
    std::vector<float> pe_host;
    pk::rel_pos_encoding(cap, (int)D, pe_host);
    std::vector<uint32_t> pe_words = pack_f16(pe_host.data(), pe_host.size());
    if (!ctx->create_buffer(pe, pe_words.size() * 4, vk::Mem::Device, err) ||
        !ctx->upload(pe, 0, pe_words.data(), pe_words.size() * 4, err))
        return false;
    const size_t per_layer = (size_t)Pn * D;
    if (per_layer * 2 > ctx->info().max_storage_range) { err = "fast parakeet: positional table too large"; return false; }
    if (!ctx->create_buffer(pos_tab, per_layer * 2 * L, vk::Mem::Device, err)) return false;
    vk::Recording rec(*ctx);
    rec.begin();
    for (uint32_t l = 0; l < L; ++l) {
        rec.label("pos_proj");
        // Each layer's table lives in its own binding window (keeps offsets
        // inside maxStorageBufferRange for long inputs).
        vk::Ref out(pos_tab, (VkDeviceSize)l * per_layer * 2, per_layer * 2);
        if (!K.gemm_w(rec, ar, layers[l].pos, vk::Ref(pe), Pn, D, out, D, Epi::F16, Act::None,
                      Ropt(layers[l].pos_b), 1.0f, 0xffffffffu, err))
            return false;
    }
    rec.end();
    if (!rec.submit_and_wait(err)) return false;
    pos_cap = cap;
    return true;
}

bool ParakeetEngine::Impl::record(int T, Rec& r, std::string& err) {
    const int T1 = stage_len(T), T2 = stage_len(T1), T3 = stage_len(T2);
    const int F1 = stage_len((int)NM), F2 = stage_len(F1), F3 = stage_len(F2);
    int valid = T - 1;
    for (int s = 0; s < 3; ++s) valid = stage_len(valid);
    valid = std::min(valid, T3);
    const uint32_t Tp = (uint32_t)T3;
    r.Tp = T3;
    r.valid = valid;
    r.rec = std::make_unique<vk::Recording>(*ctx);
    vk::Recording& rc = *r.rec;
    rc.begin();

    // ---- subsampling ----
    rc.label("sub_conv0");
    if (!K.pk_conv(rc, 0, {SC, (uint32_t)T, NM, (uint32_t)T1, (uint32_t)F1, 0}, F1, T1,
                   vk::Ref(mel_in), R(c0_w), R(c0_b), {}, vk::Ref(sub1), err))
        return false;
    rc.barrier();
    rc.label("sub_dw1");
    if (!K.pk_conv(rc, 1, {SC, (uint32_t)T1, (uint32_t)F1, (uint32_t)T2, (uint32_t)F2, 0}, F2, T2,
                   vk::Ref(sub1), R(c2_w), R(c2_b), {}, vk::Ref(sub2a), err))
        return false;
    rc.barrier();
    rc.label("sub_pw1");
    {
        GemmCall c;
        c.b = BKind::F16; c.epi = Epi::F16; c.act = Act::Relu; c.bias_mode = 1;
        c.a.M = (uint32_t)(T2 * F2); c.a.N = SC; c.a.K = SC;
        c.a.lda = SC; c.a.ldb = SC; c.a.ldc = SC;
        c.A = vk::Ref(sub2a); c.Bq = R(c3_pw.q); c.C = vk::Ref(sub2b); c.bias = R(c3_b);
        if (!K.gemm(rc, c, err)) return false;
    }
    rc.barrier();
    rc.label("sub_dw2");
    if (!K.pk_conv(rc, 1, {SC, (uint32_t)T2, (uint32_t)F2, (uint32_t)T3, (uint32_t)F3, 0}, F3, T3,
                   vk::Ref(sub2b), R(c5_w), R(c5_b), {}, vk::Ref(sub3a), err))
        return false;
    rc.barrier();
    // Stage-3 pointwise conv written channel-major per frame ([t][c][f]) so
    // each frame's flattened vector is NeMo's c*F3 + f order: C_t = W · X_tᵀ.
    rc.label("sub_pw2");
    {
        GemmCall c;
        c.b = BKind::F16; c.epi = Epi::F16; c.act = Act::Relu; c.bias_mode = 2;
        c.a.M = SC; c.a.N = (uint32_t)F3; c.a.K = SC;
        c.a.lda = SC; c.a.ldb = SC; c.a.ldc = (uint32_t)F3;
        c.a.sb_hi = (uint32_t)F3 * SC; c.a.sc_hi = SC * (uint32_t)F3;
        c.batch = Tp;
        c.A = R(c6_pw.q); c.Bq = vk::Ref(sub3a); c.C = vk::Ref(sub3b); c.bias = R(c6_b);
        if (!K.gemm(rc, c, err)) return false;
    }
    rc.barrier();
    rc.label("sub_out");
    if (!K.gemm_w(rc, ar, out_w, vk::Ref(sub3b), Tp, out_w.K, vk::Ref(x), D, Epi::F32, Act::None,
                  R(out_b), 1.0f, (uint32_t)valid, err))
        return false;
    rc.barrier();
    rc.label("norm");
    if (!K.norm(rc, 0, Tp, D, vk::Ref(x), D, R(layers[0].ln_ff1_g), R(layers[0].ln_ff1_b), {}, {},
                vk::Ref(h), D, 1e-5f, 1e-5f, err))
        return false;
    rc.barrier();

    const uint32_t ldS = Tp, ldB = 2 * Tp, ldP = round_up(Tp, 8);
    const uint32_t Pn = 2 * (uint32_t)pos_cap - 1;
    const size_t per_layer = (size_t)Pn * D;
    const uint32_t pos_row0 = (uint32_t)pos_cap - Tp;   // first table row for this length
    const float scale = 1.0f / std::sqrt((float)dk);

    for (uint32_t l = 0; l < L; ++l) {
        const Layer& Y = layers[l];
        rc.split();   // one short GPU job per layer (watchdog / compositor friendly)
        // FFN1 (half-step residual)
        rc.label("ff_up");
        if (!K.gemm_w(rc, ar, Y.ff1_w1, vk::Ref(h), Tp, D, vk::Ref(ffh), FF, Epi::F16, Act::Silu,
                      Ropt(Y.ff1_b1), 1.0f, 0xffffffffu, err))
            return false;
        rc.barrier();
        rc.label("ff_down");
        if (!K.gemm_w(rc, ar, Y.ff1_w2, vk::Ref(ffh), Tp, FF, vk::Ref(x), D, Epi::Residual, Act::None,
                      Ropt(Y.ff1_b2), 0.5f, 0xffffffffu, err))
            return false;
        rc.barrier();
        rc.label("norm");
        if (!K.norm(rc, 0, Tp, D, vk::Ref(x), D, R(Y.ln_att_g), R(Y.ln_att_b), {}, {}, vk::Ref(h), D,
                    1e-5f, 1e-5f, err))
            return false;
        rc.barrier();
        // q (+pos_bias_u / +pos_bias_v), k, v — independent, no barriers between.
        rc.label("qkv");
        if (!K.gemm_w(rc, ar, Y.q, vk::Ref(h), Tp, D, vk::Ref(qu), D, Epi::Quv, Act::None, R(Y.bias_u),
                      1.0f, 0xffffffffu, err, R(Y.bias_v), vk::Ref(qv)))
            return false;
        if (!K.gemm_w(rc, ar, Y.k, vk::Ref(h), Tp, D, vk::Ref(kk), D, Epi::F16, Act::None, Ropt(Y.k_b),
                      1.0f, 0xffffffffu, err))
            return false;
        if (!K.gemm_w(rc, ar, Y.v, vk::Ref(h), Tp, D, vk::Ref(vv), D, Epi::F16, Act::None, Ropt(Y.v_b),
                      1.0f, 0xffffffffu, err))
            return false;
        rc.barrier();
        // ac = (q+u)·kᵀ and raw bd = (q+v)·pᵀ per head.
        rc.label("attn_scores");
        {
            GemmCall c;
            c.b = BKind::F16; c.epi = Epi::F32;
            c.a.M = Tp; c.a.N = Tp; c.a.K = dk;
            c.a.lda = D; c.a.ldb = D; c.a.ldc = ldS;
            c.a.sa_hi = dk; c.a.sb_hi = dk; c.a.sc_hi = Tp * ldS;
            c.batch = H;
            c.A = vk::Ref(qu); c.Bq = vk::Ref(kk); c.C = vk::Ref(S);
            if (!K.gemm(rc, c, err)) return false;
        }
        {
            GemmCall c;
            c.b = BKind::F16; c.epi = Epi::F32;
            c.a.M = Tp; c.a.N = 2 * Tp - 1; c.a.K = dk;
            c.a.lda = D; c.a.ldb = D; c.a.ldc = ldB;
            c.a.b_off = pos_row0 * D;
            c.a.sa_hi = dk; c.a.sb_hi = dk; c.a.sc_hi = Tp * ldB;
            c.batch = H;
            c.A = vk::Ref(qv);
            c.Bq = vk::Ref(pos_tab, (VkDeviceSize)l * per_layer * 2, per_layer * 2);
            c.C = vk::Ref(BD);
            if (!K.gemm(rc, c, err)) return false;
        }
        rc.barrier();
        rc.label("softmax");
        {
            Kernels::SoftmaxArgs a{Tp, ldS, ldB, ldP, Tp * ldS, Tp * ldB, Tp * ldP, scale,
                                   (uint32_t)valid, 0xffffffffu, Tp};
            if (!K.softmax(rc, true, Tp, H, a, vk::Ref(S), vk::Ref(BD), vk::Ref(P), err)) return false;
        }
        rc.barrier();
        rc.label("attn_pv");
        {
            GemmCall c;
            c.b = BKind::F16T; c.epi = Epi::F16;
            c.a.M = Tp; c.a.N = dk; c.a.K = Tp;
            c.a.lda = ldP; c.a.ldb = D; c.a.ldc = D;
            c.a.sa_hi = Tp * ldP; c.a.sb_hi = dk; c.a.sc_hi = dk;
            c.a.row_valid = (uint32_t)valid;
            c.batch = H;
            c.A = vk::Ref(P); c.Bq = vk::Ref(vv); c.C = vk::Ref(att);
            if (!K.gemm(rc, c, err)) return false;
        }
        rc.barrier();
        rc.label("attn_out");
        if (!K.gemm_w(rc, ar, Y.o, vk::Ref(att), Tp, D, vk::Ref(x), D, Epi::Residual, Act::None,
                      Ropt(Y.o_b), 1.0f, 0xffffffffu, err))
            return false;
        rc.barrier();
        // Convolution module.
        rc.label("norm");
        if (!K.norm(rc, 0, Tp, D, vk::Ref(x), D, R(Y.ln_conv_g), R(Y.ln_conv_b), {}, {}, vk::Ref(h), D,
                    1e-5f, 1e-5f, err))
            return false;
        rc.barrier();
        rc.label("conv_pw1_glu");
        if (!K.gemm_w(rc, ar, Y.pw1, vk::Ref(h), Tp, D, vk::Ref(glu), D, Epi::Glu, Act::None,
                      Ropt(Y.pw1_b), 1.0f, (uint32_t)valid, err))
            return false;
        rc.barrier();
        rc.label("conv_dw");
        if (!K.pk_conv(rc, 2, {D, Tp, 0, Tp, 0, CK}, Tp, 1, vk::Ref(glu), R(Y.dw_w), R(Y.bn_scale),
                       R(Y.bn_shift), vk::Ref(dw), err))
            return false;
        rc.barrier();
        rc.label("conv_pw2");
        if (!K.gemm_w(rc, ar, Y.pw2, vk::Ref(dw), Tp, D, vk::Ref(x), D, Epi::Residual, Act::None,
                      Ropt(Y.pw2_b), 1.0f, 0xffffffffu, err))
            return false;
        rc.barrier();
        // FFN2 (half-step residual)
        rc.label("norm");
        if (!K.norm(rc, 0, Tp, D, vk::Ref(x), D, R(Y.ln_ff2_g), R(Y.ln_ff2_b), {}, {}, vk::Ref(h), D,
                    1e-5f, 1e-5f, err))
            return false;
        rc.barrier();
        rc.label("ff_up");
        if (!K.gemm_w(rc, ar, Y.ff2_w1, vk::Ref(h), Tp, D, vk::Ref(ffh), FF, Epi::F16, Act::Silu,
                      Ropt(Y.ff2_b1), 1.0f, 0xffffffffu, err))
            return false;
        rc.barrier();
        rc.label("ff_down");
        if (!K.gemm_w(rc, ar, Y.ff2_w2, vk::Ref(ffh), Tp, FF, vk::Ref(x), D, Epi::Residual, Act::None,
                      Ropt(Y.ff2_b2), 0.5f, 0xffffffffu, err))
            return false;
        rc.barrier();
        // norm_out, fused with the next layer's norm_feed_forward1.
        rc.label("norm");
        if (l + 1 < L) {
            const Layer& Z = layers[l + 1];
            if (!K.norm(rc, 1, Tp, D, vk::Ref(x), D, R(Y.ln_out_g), R(Y.ln_out_b), R(Z.ln_ff1_g),
                        R(Z.ln_ff1_b), vk::Ref(h), D, 1e-5f, 1e-5f, err))
                return false;
        } else {
            if (!K.norm(rc, 0, Tp, D, vk::Ref(x), D, R(Y.ln_out_g), R(Y.ln_out_b), {}, {}, vk::Ref(h), D,
                        1e-5f, 1e-5f, err))
                return false;
        }
        rc.barrier();
    }
    rc.label("joint_enc");
    if (!K.gemm_w(rc, ar, jenc, vk::Ref(h), Tp, D, vk::Ref(enc), JH, Epi::F32, Act::None, R(jenc_b),
                  1.0f, 0xffffffffu, err))
        return false;
    rc.end();
    return true;
}

bool ParakeetEngine::Impl::run_encoder(const std::vector<float>& feats, int T,
                                       std::vector<float>& enc_out, int& Tp, std::string& err) {
    if (T < 2) { err = "fast parakeet: audio too short"; return false; }
    const int Tp_need = stage_len(stage_len(stage_len(T)));
    if (!ensure_capacity(T, err) || !ensure_pos_table(Tp_need, err)) return false;
    auto it = recs.find(T);
    if (it == recs.end()) {
        if (recs.size() >= 8) {
            auto old = recs.begin();
            for (auto i = recs.begin(); i != recs.end(); ++i)
                if (i->second.used < old->second.used) old = i;
            recs.erase(old);
        }
        Rec r;
        if (!record(T, r, err)) return false;
        it = recs.emplace(T, std::move(r)).first;
    }
    it->second.used = ++tick;
    // feats are time-major [T][n_mels] (PkMel), the layout the encoder reads.
    if (!ctx->upload(mel_in, 0, feats.data(), (size_t)T * NM * 4, err)) return false;
    if (!it->second.rec->submit_and_wait(err)) return false;
    it->second.rec->report_profile("parakeet encoder");
    Tp = it->second.Tp;
    enc_out.resize((size_t)Tp * JH);
    return ctx->download(enc, 0, enc_out.data(), enc_out.size() * 4, err);
}

// Serial greedy TDT decode, mirroring pk::tdt_greedy (emits every step's
// token including blanks). The prediction network runs only after a
// non-blank emission; its joint projection is cached with it.
std::vector<int32_t> ParakeetEngine::Impl::tdt_greedy(const std::vector<float>& enc_proj, int T,
                                                      bool* aborted, pk::TdtTiming* timing) {
    const uint32_t PL = (uint32_t)w_ih.size();
    const int blank = (int)cfg.blank_id;
    const int max_symbols = (int)cfg.max_symbols;
    const int token_count = (int)V1;
    std::vector<std::vector<float>> hc(PL, std::vector<float>(PH, 0.0f)), cc = hc;   // committed
    std::vector<std::vector<float>> hn = hc, cn = hc;                                 // candidate
    std::vector<float> xin(PH), z1(4 * PH), z2(4 * PH), pp(JH), f(JH), logits(jout.N);
    cpu::QVec qx;
    int32_t last_token = blank;
    bool emitted_any = false, g_valid = false;
    std::vector<int32_t> hyp;
    if (timing) *timing = pk::TdtTiming{};

    auto sigm = [](float v) { return 1.0f / (1.0f + std::exp(-v)); };
    struct Acc { double pred = 0, joint = 0, arg = 0; } acc;
    if (in0_cache.size() != V1 + 1) in0_cache.assign(V1 + 1, {});
    auto pred_step = [&]() {
        const float* layer_in = nullptr;
        for (uint32_t l = 0; l < PL; ++l) {
            if (l == 0) {
                const size_t key = emitted_any ? (size_t)last_token : V1;
                std::vector<float>& z0 = in0_cache[key];
                if (z0.empty()) {
                    if (emitted_any) std::memcpy(xin.data(), &embed[(size_t)last_token * PH], PH * 4);
                    else std::fill(xin.begin(), xin.end(), 0.0f);
                    cpu::quantize(xin.data(), PH, qx);
                    z0.resize(4 * PH);
                    gemv2.run(w_ih[0], qx, b_ih[0].data(), z0.data());
                }
                std::memcpy(z1.data(), z0.data(), 4 * PH * 4);
            } else {
                cpu::quantize(layer_in, PH, qx);
                gemv2.run(w_ih[l], qx, b_ih[l].data(), z1.data());
            }
            cpu::quantize(hc[l].data(), PH, qx);
            gemv2.run(w_hh[l], qx, b_hh[l].data(), z2.data());
            for (uint32_t i = 0; i < PH; ++i) {
                const float ig = sigm(z1[i] + z2[i]);
                const float fg = sigm(z1[PH + i] + z2[PH + i]);
                const float gg = std::tanh(z1[2 * PH + i] + z2[2 * PH + i]);
                const float og = sigm(z1[3 * PH + i] + z2[3 * PH + i]);
                cn[l][i] = fg * cc[l][i] + ig * gg;
                hn[l][i] = og * std::tanh(cn[l][i]);
            }
            layer_in = hn[l].data();
        }
        cpu::quantize(hn[PL - 1].data(), PH, qx);
        gemv2.run(jpred, qx, jpred_b.data(), pp.data());
    };

    int t = 0;
    auto tpt = std::chrono::steady_clock::now();
    while (t < T) {
        if (aborted && ggml::call_abort_requested()) {
            *aborted = true;
            return {};
        }
        int symbols_added = 0, skip = 0;
        bool need_loop = true;
        while (need_loop && symbols_added < max_symbols) {
            {
                const auto t0 = std::chrono::steady_clock::now();
                if (!g_valid) { pred_step(); g_valid = true; }
                acc.pred += ms_since(t0);
            }
            const auto tj0 = std::chrono::steady_clock::now();
            const float* e = enc_proj.data() + (size_t)t * JH;
            for (uint32_t i = 0; i < JH; ++i) f[i] = std::max(e[i] + pp[i], 0.0f);
            cpu::quantize(f.data(), JH, qx);
            gemv2.run(jout, qx, jout_b.data(), logits.data());
            acc.joint += ms_since(tj0);
            const auto ta0 = std::chrono::steady_clock::now();
            int k = 0;
            for (int i = 1; i < token_count; ++i) if (logits[i] > logits[k]) k = i;
            int dk_ = 0;
            for (int i = 1; i < (int)n_dur; ++i)
                if (logits[token_count + i] > logits[token_count + dk_]) dk_ = i;
            skip = cfg.tdt_durations[(size_t)dk_];
            acc.arg += ms_since(ta0);
            hyp.push_back(k);
            if (timing) {
                timing->frame.push_back(t);
                timing->duration.push_back(skip);
            }
            if (k != blank) {
                last_token = k;
                hc.swap(hn);   // commit the candidate state
                cc.swap(cn);
                emitted_any = true;
                g_valid = false;
            }
            symbols_added += 1;
            t += skip;
            need_loop = (skip == 0);
        }
        if (skip == 0) skip = 1;
        if (symbols_added == max_symbols) t += 1;
    }
    if (std::getenv("STARLING_PARAKEET_TIMING"))
        std::fprintf(stderr, "[fast-parakeet] decode split: pred %.1f ms, joint %.1f ms, argmax %.1f ms (total %.1f)\n",
                     acc.pred, acc.joint, acc.arg, ms_since(tpt));
    return hyp;
}

namespace {
// STARLING_FAST_MEL_CHECK=1: verify the fast mel against the reference
// frontend bit for bit (development aid).
void check_mel(const pk::MelConstants& mc, const float* pcm, size_t n,
               const std::vector<float>& tm, int T) {
    std::vector<float> ref;
    int Tr = 0;
    pk::MelFrontend(mc).compute(pcm, n, ref, Tr);
    size_t bad = 0;
    for (int t = 0; t < T && Tr == T; ++t)
        for (uint32_t f = 0; f < mc.n_mels; ++f)
            if (std::memcmp(&ref[(size_t)f * T + t], &tm[(size_t)t * mc.n_mels + f], 4) != 0) ++bad;
    std::fprintf(stderr, "[fast-parakeet] mel check: T=%d/%d mismatches=%zu\n", T, Tr, bad);
}
} // namespace

bool ParakeetEngine::encode(const float* pcm, size_t n, std::vector<float>& enc, int& Tp, std::string& err) {
    Impl& I = *impl_;
    std::vector<float> feats;
    int T = 0;
    I.fmel->compute(pcm, n, feats, T);
    return I.encode_mel(feats, T, enc, Tp, err);
}

bool ParakeetEngine::decode_ids(const float* pcm, size_t n, std::vector<int32_t>& ids, std::string& err,
                                pk::TdtTiming* timing) {
    Impl& I = *impl_;
    // Keep the decoder's GEMV worker busy (and on a big core) from here on;
    // it parks again once this transcription is done.
    struct Hold {
        cpu::GemvHelper& g;
        explicit Hold(cpu::GemvHelper& h) : g(h) { g.hold(true); }
        ~Hold() { g.hold(false); }
    } hold(I.gemv2);
    const bool print_timing = env_on("STARLING_FAST_TIMING");
    const auto t0 = std::chrono::steady_clock::now();
    // A cancelled call (ggml::call_abort_requested) stops at the next stage
    // boundary or decoder frame; the recorded encoder submission itself is
    // not interruptible.
    auto aborted = [&] {
        if (!ggml::call_abort_requested()) return false;
        err = ggml::kCallAbortedError;
        return true;
    };
    if (aborted()) return false;
    std::vector<float> feats;
    int T = 0;
    I.fmel->compute(pcm, n, feats, T);
    const double t_mel = ms_since(t0);
    if (env_on("STARLING_FAST_MEL_CHECK")) check_mel(I.mel, pcm, n, feats, T);
    if (aborted()) return false;
    std::vector<float> enc;
    int Tp = 0;
    if (!I.encode_mel(feats, T, enc, Tp, err)) return false;
    const double t_enc = ms_since(t0);
    if (aborted()) return false;
    bool decode_aborted = false;
    ids = I.tdt_greedy(enc, Tp, &decode_aborted, timing);
    if (decode_aborted) {
        err = ggml::kCallAbortedError;
        return false;
    }
    if (print_timing)
        std::fprintf(stderr, "[fast-parakeet] audio=%.2fs mel=%.1fms encoder=%.1fms decode=%.1fms total=%.1fms (T=%d Tp=%d)\n",
                     n / 16000.0, t_mel, t_enc - t_mel, ms_since(t0) - t_enc, ms_since(t0), T, Tp);
    return true;
}

bool ParakeetEngine::transcribe(const float* pcm, size_t n, std::string& text, std::string& err) {
    std::vector<int32_t> ids;
    if (!decode_ids(pcm, n, ids, err)) return false;
    text = pk::detokenize(impl_->cfg.tokenizer_pieces, ids);
    return true;
}

} // namespace starling::fast
