// moss_engine.cpp — see moss_engine.hpp.

#include "moss_engine.hpp"

#include "kernels.hpp"
#include "packed_file.hpp"
#include "vk_runtime.hpp"
#include "weights.hpp"

#include "moss/audio_encoder.hpp"
#include "moss/loader.hpp"
#include "moss/mel.hpp"

#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>

namespace starling::fast {

constexpr size_t kMaxPromptIds = 256;   // prompt prefix + suffix ids (device buffer)

namespace ms = starling::ggml::moss;

namespace {

bool env_on(const char* name) {
    const char* v = std::getenv(name);
    return v && v[0] && v[0] != '0';
}
double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}
int half_len(int x) { return (x - 1) / 2 + 1; }

struct EncLayer {
    GMat q, k, v, o, fc1, fc2;
    Arena::Id q_b, k_b, v_b, o_b, fc1_b, fc2_b, ln1_g, ln1_b, ln2_g, ln2_b;
};
struct LlmLayer {
    GMat qkv, o, gateup, down;
    Arena::Id attn_norm, ffn_norm, q_norm, k_norm;
};

// A large matrix split into row chunks (each within one storage-buffer range).
struct ChunkedMat {
    std::vector<GMat> parts;
    uint32_t chunk_rows = 0, N = 0, K = 0;
    GpuFmt fmt = GpuFmt::W8;
};

// Split `m` into chunks of at most `rows` rows.
std::vector<HostMatrix> split_rows(HostMatrix&& m, uint32_t rows) {
    std::vector<HostMatrix> out;
    const size_t qw = m.q.size() / m.N, sw = m.s.empty() ? 0 : m.s.size() / m.N,
                 xw = m.x.empty() ? 0 : m.x.size() / m.N;
    for (uint32_t r0 = 0; r0 < m.N; r0 += rows) {
        const uint32_t n = std::min(rows, m.N - r0);
        HostMatrix c;
        c.fmt = m.fmt; c.N = n; c.K = m.K; c.lossless = m.lossless; c.layout = m.layout;
        c.q.assign(m.q.begin() + (size_t)r0 * qw, m.q.begin() + (size_t)(r0 + n) * qw);
        if (sw) c.s.assign(m.s.begin() + (size_t)r0 * sw, m.s.begin() + (size_t)(r0 + n) * sw);
        if (xw) c.x.assign(m.x.begin() + (size_t)r0 * xw, m.x.begin() + (size_t)(r0 + n) * xw);
        out.push_back(std::move(c));
    }
    return out;
}

} // namespace

struct MossEngine::Impl {
    vk::Context* ctx = nullptr;
    Kernels K;
    Arena ar;
    ms::Config cfg;
    ggml::ModelLoader mel_loader;   // only the two mel constant tensors

    // dims
    uint32_t ED = 0, EH = 0, EHD = 0, EFF = 0, EL = 0, DS = 0, EOUT = 0;
    uint32_t AH = 0;                                   // adapter hidden
    uint32_t HD = 0, NH = 0, NKV = 0, HDIM = 0, FF = 0, NL = 0, MAXPOS = 0;

    // encoder weights
    Arena::Id c1_w, c1_b, c2_b, c3_b, pe13, lnp_g, lnp_b, p1_b, p2_b;
    GMat c2, c3, cout, p1, p2, ad_gateup, ad_down;
    std::vector<EncLayer> enc;
    // LLM weights
    std::vector<LlmLayer> llm;
    Arena::Id final_norm, rope_tab;
    ChunkedMat embed;

    // runtime buffers
    vk::Buffer kcache, vcache, state, ids, xd, qkvd, attd, hd, part;
    vk::Buffer mel_in, cv1, cv2, cv3, ex, eh, eq, ek, ev, esc, ep, ectx, effh, eproj, eproj2, adh;
    vk::Buffer lx, lhn, lqkv, lqr, lsc, lp, lctx, lmlp;
    int cap_C = 0, cap_A = 0, cap_S = 0;
    uint32_t n_part = 0;

    struct Rec { std::unique_ptr<vk::Recording> rec; uint64_t used = 0; int A = 0, S = 0; };
    std::map<std::pair<int, int>, Rec> recs;           // keyed by (chunks, tail)
    std::unique_ptr<vk::Recording> dec_rec;
    uint32_t dec_steps = 0;
    uint64_t tick = 0;

    bool load(const ms::MossModel& m, std::string& err);
    bool alloc_static(std::string& err);
    bool ensure(int C, int A, int S, std::string& err);
    bool record_prefill(int C, int tail, Rec& r, std::string& err);
    // Prefill attention heads per score-matrix group: whole GQA groups within
    // a 64 MiB budget (at least one GQA group, however long the prompt).
    uint32_t prefill_hgroup(size_t S) const {
        const uint32_t gqa = NH / NKV;
        const size_t fit = std::min<size_t>(NH, (64ull << 20) / (S * S * 4));
        return std::max<uint32_t>(gqa, (uint32_t)(fit / gqa * gqa));
    }
    bool record_decode(uint32_t steps, std::string& err);
    bool lm_head(vk::Recording& rc, vk::Ref x, uint32_t x_off, std::string& err);
    bool dec_next(vk::Recording& rc, std::string& err);
    vk::Ref R(Arena::Id id) const { return ar.ref(id); }
};

MossEngine::MossEngine() : impl_(new Impl) {}
MossEngine::~MossEngine() {
    if (impl_ && impl_->ctx) impl_->ctx->fn().vkDeviceWaitIdle(impl_->ctx->device());
}

std::unique_ptr<MossEngine> MossEngine::create(const ms::MossModel& m, std::string& err) {
    vk::Context* ctx = vk::Context::get(err);
    if (!ctx) return nullptr;
    std::unique_ptr<MossEngine> e(new MossEngine());
    e->impl_->ctx = ctx;
    if (!e->impl_->K.init(*ctx, err) || !e->impl_->load(m, err) || !e->impl_->alloc_static(err))
        return nullptr;
    return e;
}

std::string MossEngine::describe() const {
    char buf[256];
    std::snprintf(buf, sizeof buf, "fast/vulkan '%s' weights=%.0f MiB",
                  impl_->ctx->info().name.c_str(), impl_->ar.total_bytes() / 1048576.0);
    return buf;
}

bool MossEngine::Impl::load(const ms::MossModel& m, std::string& err) {
    const auto t_load0 = std::chrono::steady_clock::now();
    const auto& ml = m.loader;
    cfg = m.config;
    const auto& ec = cfg.encoder;
    const auto& lc = cfg.llm;
    ED = ec.d_model; EH = ec.n_heads; EHD = ec.head_dim; EFF = ec.ff_dim; EL = ec.n_layers;
    DS = ec.downsample_hidden_size; EOUT = ec.output_dim; AH = cfg.adapter_hidden;
    HD = lc.hidden; NH = lc.n_heads; NKV = lc.n_kv_heads; HDIM = lc.head_dim; FF = lc.intermediate;
    NL = lc.n_layers; MAXPOS = lc.max_cache;
    if (EHD != 64 || HDIM != 128 || DS % 8 || cfg.frontend.n_mels != 128 || ec.n_window_infer % 100 ||
        cfg.adapter_input != EOUT || cfg.adapter_output != HD || !lc.tied_embeddings || MAXPOS > 4096 ||
        NKV == 0 || NH % NKV || cfg.max_new_tokens > MAXPOS ||
        cfg.prompt_prefix.size() + cfg.prompt_suffix.size() > kMaxPromptIds) {
        err = "fast moss: unsupported configuration";
        return false;
    }
    auto T = [&](const std::string& n) -> const ggml_tensor* {
        const ggml_tensor* t = ml.tensor(n.c_str());
        if (!t && err.empty()) err = "fast moss: missing tensor " + n;
        return t;
    };
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
    auto host_mat = [&](const std::string& n, HostMatrix& hm) {
        const ggml_tensor* t = T(n);
        return t && pack_gpu_matrix(t, hm, err);
    };
    PackJobs jobs;   // matrices repack in parallel before the upload
    // STARLING_FAST_PACKED=path overrides GGUF repacking for the tensors the
    // file contains (#319): source-quantized weights load straight into the
    // arena. Shapes are validated against the GGUF so a file built for a
    // different model fails loudly instead of misreading memory.
    std::unique_ptr<PackedWeights> packed;
    if (const char* pp = std::getenv("STARLING_FAST_PACKED")) {
        packed = PackedWeights::load(pp, err);
        if (!packed) return false;
        std::fprintf(stderr,
                     "[fast] packed override: %zu tensors, %.1f MB, rounding=%s source=%s\n",
                     packed->size(), packed->bytes() / 1e6, packed->rounding().c_str(),
                     packed->source_hash().c_str());
    }
    auto pack_one = [&](const std::string& n, const ggml_tensor* t, HostMatrix& hm,
                         std::string& e) {
        if (packed && packed->has(n)) {
            if (!packed->matrix(n, hm, e)) return false;
            const uint32_t want_n = (uint32_t)(ggml_nelements(t) / t->ne[0]);
            if (hm.K != (uint32_t)t->ne[0] || hm.N != want_n) {
                e = n + ": packed shape " + std::to_string(hm.N) + "x" + std::to_string(hm.K) +
                    " != GGUF " + std::to_string(want_n) + "x" + std::to_string(t->ne[0]);
                return false;
            }
            return true;
        }
        return pack_gpu_matrix(t, hm, e);
    };
    auto mat = [&](const std::string& n, GMat& g) {
        const ggml_tensor* t = T(n);
        if (!t) return false;
        jobs.add(&g, [&, t, n](HostMatrix& hm, std::string& e) {
            return pack_one(n, t, hm, e);
        });
        return true;
    };
    // Rows of `b` interleaved with rows of `a` (a0 b0 a1 b1 ...): the paired
    // GEMM/GEMV epilogues own both halves of each gated unit.
    auto interleave = [&](const std::string& a, const std::string& b, GMat& g) {
        const ggml_tensor* ta = T(a);
        const ggml_tensor* tb = T(b);
        if (!ta || !tb) return false;
        jobs.add(&g, [&, ta, tb, a, b](HostMatrix& ha, std::string& e) {
            HostMatrix hb;
            if (!pack_one(a, ta, ha, e) || !pack_one(b, tb, hb, e)) return false;
            const uint32_t n = ha.N;
            if (!concat_rows(ha, hb, e)) return false;
            std::vector<uint32_t> order(2 * n);
            for (uint32_t j = 0; j < n; ++j) { order[2 * j] = j; order[2 * j + 1] = n + j; }
            permute_rows(ha, order);
            return true;
        });
        return true;
    };

    // ---- mel constants (kept in a tiny loader for the shared frontend) ----
    {
        std::vector<float> w, f;
        const ggml_tensor* wt = T("audio.mel_window");
        const ggml_tensor* ft = T("audio.mel_filters");
        if (!wt || !ft || !tensor_to_f32(wt, w, err) || !tensor_to_f32(ft, f, err)) return false;
        mel_loader.add_owned_tensor("audio.mel_window", w, wt->ne[0], 1);
        mel_loader.add_owned_tensor("audio.mel_filters", f, ft->ne[0], ft->ne[1]);
    }

    // ---- audio encoder ----
    if (!vec_id("enc.conv1.weight", c1_w) || !vec_id("enc.conv1.bias", c1_b) ||
        !vec_id("enc.conv2.bias", c2_b) || !vec_id("enc.conv3.bias", c3_b))
        return false;
    // conv2/conv3 as implicit GEMMs: [oc][(kt*3 + kf)*cin + ic], f16.
    for (int ci = 2; ci <= 3; ++ci) {
        std::vector<float> w;
        if (!f32("enc.conv" + std::to_string(ci) + ".weight", w)) return false;
        if (w.size() != (size_t)DS * DS * 9) { err = "fast moss: conv shape"; return false; }
        std::vector<float> r((size_t)DS * 9 * DS);
        for (uint32_t oc = 0; oc < DS; ++oc)
            for (uint32_t ic = 0; ic < DS; ++ic)
                for (uint32_t kf = 0; kf < 3; ++kf)          // ggml kh (frequency)
                    for (uint32_t kt = 0; kt < 3; ++kt)      // ggml kw (time)
                        r[(size_t)oc * 9 * DS + (kt * 3 + kf) * DS + ic] =
                            w[(((size_t)oc * DS + ic) * 3 + kf) * 3 + kt];
        HostMatrix hm;
        if (!pack_gpu_matrix_raw(GGML_TYPE_F32, r.data(), DS, 9 * DS, hm, err)) return false;
        (ci == 2 ? c2 : c3) = arena_matrix(ar, std::move(hm));
    }
    // conv_out: columns reordered from ggml's (f + 16*c) to our (f*480 + c).
    {
        std::vector<float> w;
        if (!f32("enc.conv_out.weight", w)) return false;
        const uint32_t Kc = 16 * DS;
        if (w.size() != (size_t)ED * Kc) { err = "fast moss: conv_out shape"; return false; }
        std::vector<float> r(w.size());
        for (uint32_t n = 0; n < ED; ++n)
            for (uint32_t f = 0; f < 16; ++f)
                for (uint32_t c = 0; c < DS; ++c)
                    r[(size_t)n * Kc + f * DS + c] = w[(size_t)n * Kc + f + 16 * c];
        HostMatrix hm;
        if (!pack_gpu_matrix_raw(GGML_TYPE_F32, r.data(), ED, Kc, hm, err)) return false;
        cout = arena_matrix(ar, std::move(hm));
    }
    {
        std::vector<float> pe;
        if (!f32("enc.positional_embedding", pe)) return false;
        if (pe.size() < (size_t)13 * ED) { err = "fast moss: positional embedding shorter than 13 rows"; return false; }
        pe.resize((size_t)13 * ED);                  // rows [0, 13): one chunk of tokens
        pe13 = ar.add_f32(pe);
    }
    enc.resize(EL);
    for (uint32_t l = 0; l < EL; ++l) {
        EncLayer& Y = enc[l];
        const std::string p = "enc.blk." + std::to_string(l) + ".";
        if (!mat(p + "attn.q.weight", Y.q) || !mat(p + "attn.k.weight", Y.k) || !mat(p + "attn.v.weight", Y.v) ||
            !mat(p + "attn.o.weight", Y.o) || !mat(p + "ffn.fc1.weight", Y.fc1) || !mat(p + "ffn.fc2.weight", Y.fc2) ||
            !vec_id(p + "attn.q.bias", Y.q_b) || !vec_id(p + "attn.k.bias", Y.k_b) ||
            !vec_id(p + "attn.v.bias", Y.v_b) || !vec_id(p + "attn.o.bias", Y.o_b) ||
            !vec_id(p + "ffn.fc1.bias", Y.fc1_b) || !vec_id(p + "ffn.fc2.bias", Y.fc2_b) ||
            !vec_id(p + "attn_norm.weight", Y.ln1_g) || !vec_id(p + "attn_norm.bias", Y.ln1_b) ||
            !vec_id(p + "ffn_norm.weight", Y.ln2_g) || !vec_id(p + "ffn_norm.bias", Y.ln2_b))
            return false;
    }
    if (!vec_id("enc.ln_post.weight", lnp_g) || !vec_id("enc.ln_post.bias", lnp_b) ||
        !mat("enc.proj1.weight", p1) || !vec_id("enc.proj1.bias", p1_b) ||
        !mat("enc.proj2.weight", p2) || !vec_id("enc.proj2.bias", p2_b) ||
        !interleave("adapter.gate.weight", "adapter.up.weight", ad_gateup) ||
        !mat("adapter.down.weight", ad_down))
        return false;

    // ---- LLM ----
    llm.resize(NL);
    for (uint32_t l = 0; l < NL; ++l) {
        LlmLayer& Y = llm[l];
        const std::string p = "llm.blk." + std::to_string(l) + ".";
        const ggml_tensor* tq = T(p + "attn.q.weight");
        const ggml_tensor* tk = T(p + "attn.k.weight");
        const ggml_tensor* tv = T(p + "attn.v.weight");
        if (!tq || !tk || !tv) return false;
        jobs.add(&Y.qkv, [&, tq, tk, tv, p](HostMatrix& q, std::string& e) {
            HostMatrix k, v;
            return pack_one(p + "attn.q.weight", tq, q, e) &&
                   pack_one(p + "attn.k.weight", tk, k, e) &&
                   pack_one(p + "attn.v.weight", tv, v, e) &&
                   concat_rows(q, k, e) && concat_rows(q, v, e);
        });
        if (!mat(p + "attn.o.weight", Y.o) ||
            !interleave(p + "ffn.gate.weight", p + "ffn.up.weight", Y.gateup) ||
            !mat(p + "ffn.down.weight", Y.down) || !vec_id(p + "attn_norm.weight", Y.attn_norm) ||
            !vec_id(p + "ffn_norm.weight", Y.ffn_norm) || !vec_id(p + "attn.q_norm.weight", Y.q_norm) ||
            !vec_id(p + "attn.k_norm.weight", Y.k_norm))
            return false;
    }
    if (!vec_id("llm.final_norm.weight", final_norm)) return false;
    // Tied embedding / lm_head: repacked in row blocks (parallel), then
    // regrouped into chunks that each fit one storage-buffer binding.
    std::vector<HostMatrix> emb_blocks;
    const uint32_t emb_block = 8192;
    {
        const ggml_tensor* te = T("llm.embed.weight");
        if (!te) return false;
        const uint32_t N = (uint32_t)te->ne[1], Kc = (uint32_t)te->ne[0];
        if (N == 0 || Kc != HD) { err = "fast moss: embedding table shape mismatch"; return false; }
        if (packed && packed->has("llm.embed.weight")) {
            HostMatrix whole;
            if (!packed->matrix("llm.embed.weight", whole, err)) return false;
            if (whole.K != Kc || whole.N != N) {
                err = "llm.embed.weight: packed shape " + std::to_string(whole.N) + "x" +
                      std::to_string(whole.K) + " != GGUF " + std::to_string(N) + "x" + std::to_string(Kc);
                return false;
            }
            // Same 8192-row blocks the GGUF path repacks in.
            emb_blocks = split_rows(std::move(whole), emb_block);
        } else {
            const size_t rb = ggml_row_size(te->type, Kc);
            emb_blocks.resize((N + emb_block - 1) / emb_block);
            for (size_t b = 0; b < emb_blocks.size(); ++b) {
                const uint32_t r0 = (uint32_t)b * emb_block, n = std::min(emb_block, N - r0);
                jobs.add_host(&emb_blocks[b], [te, r0, n, Kc, rb](HostMatrix& hm, std::string& e) {
                    return pack_gpu_matrix_raw((int)te->type, (const uint8_t*)te->data + (size_t)r0 * rb, n, Kc, hm, e);
                });
            }
        }
    }
    if (!jobs.run(ar, err)) return false;
    {
        embed.K = emb_blocks[0].K; embed.fmt = emb_blocks[0].fmt;
        embed.N = 0;
        for (const HostMatrix& b : emb_blocks) embed.N += b.N;
        const size_t row_bytes = (emb_blocks[0].q.size() + emb_blocks[0].s.size()) * 4 / emb_blocks[0].N;
        const VkDeviceSize cap = std::min<VkDeviceSize>(256ull << 20, ctx->info().max_storage_range);
        embed.chunk_rows = (uint32_t)std::max<size_t>(emb_block, (cap / row_bytes) / emb_block * emb_block);
        if (((size_t)embed.N + embed.chunk_rows - 1) / embed.chunk_rows > 4) {
            err = "fast moss: embedding table needs more than 4 buffer chunks on this device";
            return false;
        }
        const size_t per_chunk = embed.chunk_rows / emb_block;
        for (size_t b0 = 0; b0 < emb_blocks.size(); b0 += per_chunk) {
            HostMatrix c = std::move(emb_blocks[b0]);
            for (size_t b = b0 + 1; b < std::min(emb_blocks.size(), b0 + per_chunk); ++b)
                if (!concat_rows(c, emb_blocks[b], err)) return false;
            embed.parts.push_back(arena_matrix(ar, std::move(c)));
        }
        std::vector<HostMatrix>().swap(emb_blocks);
    }
    // RoPE table as Transformers builds it: inv_freq and pos*inv_freq in f32.
    {
        const uint32_t half = HDIM / 2;
        std::vector<float> tab((size_t)2 * MAXPOS * half);
        for (uint32_t i = 0; i < half; ++i) {
            const float ex = (float)(2 * i) / (float)HDIM;
            const float inv = 1.0f / std::pow(lc.rope_theta, ex);
            for (uint32_t pos = 0; pos < MAXPOS; ++pos) {
                const float th = (float)pos * inv;
                tab[(size_t)pos * half + i] = (float)std::cos((double)th);
                tab[(size_t)MAXPOS * half + (size_t)pos * half + i] = (float)std::sin((double)th);
            }
        }
        rope_tab = ar.add_f32(tab);
    }
    const double t_pack = ms_since(t_load0);
    if (!ar.finalize(*ctx, err)) return false;
    if (env_on("STARLING_FAST_TIMING"))
        std::fprintf(stderr, "[fast-moss] load: repack %.1f ms, upload %.1f ms\n", t_pack, ms_since(t_load0) - t_pack);
    if (env_on("STARLING_FAST_VERBOSE"))
        std::fprintf(stderr, "[fast] moss: weights %.1f MiB on '%s' (embed %s, %zu chunk(s))\n",
                     ar.total_bytes() / 1048576.0, ctx->info().name.c_str(), fmt_name(embed.fmt),
                     embed.parts.size());
    return true;
}

bool MossEngine::Impl::alloc_static(std::string& err) {
    const size_t cache_elems = (size_t)NL * NKV * MAXPOS * HDIM;
    n_part = 0;
    for (const GMat& g : embed.parts) n_part += ceil_div(g.N, K.gemv_rows(g.N));
    auto mk = [&](vk::Buffer& b, size_t bytes, vk::Mem kind = vk::Mem::Device) {
        return ctx->create_buffer(b, bytes, kind, err);
    };
    return mk(kcache, cache_elems * 2) && mk(vcache, cache_elems * 2) &&
           mk(state, (16 + (size_t)cfg.max_new_tokens) * 4, vk::Mem::Readback) && mk(ids, kMaxPromptIds * 4, vk::Mem::Readback) &&
           mk(xd, HD * 4) && mk(qkvd, (size_t)(NH + 2 * NKV) * HDIM * 4) && mk(attd, (size_t)NH * HDIM * 4) &&
           mk(hd, (size_t)FF * 4) && mk(part, (size_t)n_part * 8);
}

bool MossEngine::Impl::ensure(int C, int A, int S, std::string& err) {
    if (C <= cap_C && A <= cap_A && S <= cap_S) return true;
    C = std::max(C, cap_C); A = std::max(A, cap_A); S = std::max(S, cap_S);
    recs.clear();
    const size_t P = 100, T1 = half_len((int)P), F1 = 64, T2 = half_len((int)T1), F2 = 32, T3 = half_len((int)T2), F3 = 16;
    const size_t W = 13 * (cfg.encoder.n_window_infer / 100);
    const size_t nwin = (A + W - 1) / W;
    const size_t ldPw = round_up((uint32_t)W, 8);
    const size_t ldPs = round_up((uint32_t)S, 8);
    // Prefill attention scores are processed in head groups (prefill_hgroup).
    // record_prefill sizes its groups from the actual buffer capacity, so a
    // shorter prompt recorded against these buffers can never overrun them.
    auto mk = [&](vk::Buffer& b, size_t bytes, vk::Mem kind = vk::Mem::Device) {
        return ctx->create_buffer(b, std::max<size_t>(bytes, 16), kind, err);
    };
    const size_t hgroup = prefill_hgroup((size_t)S);
    if (!mk(mel_in, (size_t)C * 128 * P * 4, ctx->info().uma ? vk::Mem::Device : vk::Mem::Upload) ||
        !mk(cv1, (size_t)C * T1 * F1 * DS * 2) || !mk(cv2, (size_t)C * T2 * F2 * DS * 2) ||
        !mk(cv3, (size_t)C * T3 * F3 * DS * 2) || !mk(ex, (size_t)A * ED * 4) || !mk(eh, (size_t)A * ED * 2) ||
        !mk(eq, (size_t)A * ED * 2) || !mk(ek, (size_t)A * ED * 2) || !mk(ev, (size_t)A * ED * 2) ||
        !mk(esc, nwin * EH * W * W * 4) || !mk(ep, nwin * EH * W * ldPw * 2) || !mk(ectx, (size_t)A * ED * 2) ||
        !mk(effh, (size_t)A * EFF * 2) || !mk(eproj, (size_t)A * ED * 2) || !mk(eproj2, (size_t)A * EOUT * 2) ||
        !mk(adh, (size_t)A * AH * 2) || !mk(lx, (size_t)S * HD * 4) || !mk(lhn, (size_t)S * HD * 2) ||
        !mk(lqkv, (size_t)S * (NH + 2 * NKV) * HDIM * 2) || !mk(lqr, (size_t)S * NH * HDIM * 2) ||
        !mk(lsc, hgroup * S * S * 4) || !mk(lp, hgroup * S * ldPs * 2) || !mk(lctx, (size_t)S * HD * 2) ||
        !mk(lmlp, (size_t)S * FF * 2))
        return false;
    cap_C = C; cap_A = A; cap_S = S;
    return true;
}

bool MossEngine::Impl::lm_head(vk::Recording& rc, vk::Ref x, uint32_t x_off, std::string& err) {
    uint32_t off = 0;
    for (size_t c = 0; c < embed.parts.size(); ++c) {
        const GMat& g = embed.parts[c];
        Kernels::GemvArgs a;
        a.x_off = x_off;
        a.y_off = off * 2;
        a.eps = cfg.llm.rms_norm_eps;
        a.row0 = (uint32_t)c * embed.chunk_rows;
        rc.label("lm_head");
        if (!K.gemv(rc, ar, g, x, vk::Ref(part), R(final_norm), vk::Ref(state), {}, 3, a, err)) return false;
        off += ceil_div(g.N, K.gemv_rows(g.N));
    }
    return true;
}

bool MossEngine::Impl::dec_next(vk::Recording& rc, std::string& err) {
    const char* name = embed.fmt == GpuFmt::W8 ? "dec_next" : embed.fmt == GpuFmt::W4 ? "dec_next_w4" : "dec_next_f16";
    const vk::Pipeline* p = ctx->pipeline(name, {256u, embed.chunk_rows}, err);
    if (!p) return false;
    std::vector<vk::Ref> b(11, vk::Ref(K.dummy()));
    b[0] = vk::Ref(part);
    for (size_t c = 0; c < embed.parts.size(); ++c) {
        b[1 + c] = R(embed.parts[c].q);
        if (embed.parts[c].has_s) b[5 + c] = R(embed.parts[c].s);
    }
    b[9] = vk::Ref(xd);
    b[10] = vk::Ref(state);
    struct { uint32_t n_part, K, eos, max_steps; } pc{n_part, HD, (uint32_t)cfg.eos_token_id, cfg.max_new_tokens};
    rc.label("dec_next");
    rc.dispatch(*p, b, &pc, sizeof(pc), 1);
    return true;
}

bool MossEngine::Impl::record_prefill(int C, int tail, Rec& r, std::string& err) {
    const auto& ec = cfg.encoder;
    const int P = C == 1 ? tail : 100;
    const int T1 = half_len(P), F1 = 64, T2 = half_len(T1), F2 = 32, T3 = half_len(T2), F3 = 16;
    const int64_t Tm = (int64_t)(C - 1) * 100 + tail;
    const uint32_t A = (uint32_t)ms::audio_token_length(Tm);
    const uint32_t Mtok = (uint32_t)T3;                     // tokens per chunk
    const uint32_t W = Mtok * (ec.n_window_infer / 100);
    const uint32_t S = (uint32_t)(cfg.prompt_prefix.size() + A + cfg.prompt_suffix.size());
    const uint32_t npre = (uint32_t)cfg.prompt_prefix.size();
    r.A = (int)A;
    r.S = (int)S;
    r.rec = std::make_unique<vk::Recording>(*ctx);
    vk::Recording& rc = *r.rec;
    rc.begin();

    // ---- conv stack ----
    rc.label("enc_conv1");
    if (!K.pk_conv(rc, 3, {DS, (uint32_t)P, 128, (uint32_t)T1, (uint32_t)F1, 0}, F1, (uint32_t)(C * T1),
                   vk::Ref(mel_in), R(c1_w), R(c1_b), {}, vk::Ref(cv1), err))
        return false;
    rc.barrier();
    auto conv = [&](const GMat& w, Arena::Id b, vk::Buffer& in, vk::Buffer& out, int ti, int fi, int to, int fo) {
        GemmCall c;
        c.b = BKind::F16; c.a_conv = true; c.epi = Epi::F16; c.act = Act::Gelu; c.bias_mode = 1;
        c.a.M = (uint32_t)(C * to * fo); c.a.N = DS; c.a.K = 9 * DS;
        c.a.lda = DS; c.a.ldb = 9 * DS; c.a.ldc = DS;
        c.a.cv_cin = DS; c.a.cv_ti = (uint32_t)ti; c.a.cv_fi = (uint32_t)fi;
        c.a.cv_to = (uint32_t)to; c.a.cv_fo = (uint32_t)fo;
        c.A = vk::Ref(in); c.Bq = R(w.q); c.C = vk::Ref(out); c.bias = R(b);
        return K.gemm(rc, c, err);
    };
    rc.label("enc_conv2");
    if (!conv(c2, c2_b, cv1, cv2, T1, F1, T2, F2)) return false;
    rc.barrier();
    rc.label("enc_conv3");
    if (!conv(c3, c3_b, cv2, cv3, T2, F2, T3, F3)) return false;
    rc.barrier();
    // conv_out over the first A (valid) tokens + per-chunk positional table.
    rc.label("enc_conv_out");
    {
        GemmCall c;
        c.b = BKind::F16; c.epi = Epi::F32; c.bias_mode = 3;
        c.a.M = A; c.a.N = ED; c.a.K = 16 * DS;
        c.a.lda = 16 * DS; c.a.ldb = 16 * DS; c.a.ldc = ED;
        c.a.bias_mod = Mtok;
        c.A = vk::Ref(cv3); c.Bq = R(cout.q); c.C = vk::Ref(ex); c.bias = R(pe13);
        if (!K.gemm(rc, c, err)) return false;
    }
    rc.barrier();

    // ---- encoder layers ----
    const uint32_t n_full = A / W, tail_w = A % W;
    const float escale = 1.0f / std::sqrt((float)EHD);
    auto attention = [&](uint32_t tok0, uint32_t nwin, uint32_t Wn) {
        if (nwin == 0 || Wn == 0) return true;
        const uint32_t ldP = round_up(Wn, 8);
        GemmCall c;
        c.b = BKind::F16; c.epi = Epi::F32;
        c.a.M = Wn; c.a.N = Wn; c.a.K = EHD;
        c.a.lda = ED; c.a.ldb = ED; c.a.ldc = Wn;
        c.a.a_off = tok0 * ED; c.a.b_off = tok0 * ED;
        c.a.nb_lo = EH;
        c.a.sa_hi = Wn * ED; c.a.sa_lo = EHD; c.a.sb_hi = Wn * ED; c.a.sb_lo = EHD;
        c.a.sc_hi = EH * Wn * Wn; c.a.sc_lo = Wn * Wn;
        c.batch = nwin * EH;
        c.A = vk::Ref(eq); c.Bq = vk::Ref(ek); c.C = vk::Ref(esc);
        rc.label("enc_attn_qk");
        if (!K.gemm(rc, c, err)) return false;
        rc.barrier();
        Kernels::SoftmaxArgs sa{Wn, Wn, 0, ldP, Wn * Wn, 0, Wn * ldP, escale, Wn, 0xffffffffu, Wn};
        rc.label("enc_softmax");
        if (!K.softmax(rc, false, Wn, nwin * EH, sa, vk::Ref(esc), {}, vk::Ref(ep), err)) return false;
        rc.barrier();
        GemmCall v;
        v.b = BKind::F16T; v.epi = Epi::F16;
        v.a.M = Wn; v.a.N = EHD; v.a.K = Wn;
        v.a.lda = ldP; v.a.ldb = ED; v.a.ldc = ED;
        v.a.b_off = tok0 * ED; v.a.c_off = tok0 * ED;
        v.a.nb_lo = EH;
        v.a.sa_hi = EH * Wn * ldP; v.a.sa_lo = Wn * ldP;
        v.a.sb_hi = Wn * ED; v.a.sb_lo = EHD; v.a.sc_hi = Wn * ED; v.a.sc_lo = EHD;
        v.batch = nwin * EH;
        v.A = vk::Ref(ep); v.Bq = vk::Ref(ev); v.C = vk::Ref(ectx);
        rc.label("enc_attn_pv");
        return K.gemm(rc, v, err);
    };
    const float eps = ec.layer_norm_eps;
    for (uint32_t l = 0; l < EL; ++l) {
        const EncLayer& Y = enc[l];
        rc.split();
        rc.label("norm");
        if (!K.norm(rc, 0, A, ED, vk::Ref(ex), ED, R(Y.ln1_g), R(Y.ln1_b), {}, {}, vk::Ref(eh), ED, eps, eps, err))
            return false;
        rc.barrier();
        rc.label("enc_qkv");
        if (!K.gemm_w(rc, ar, Y.q, vk::Ref(eh), A, ED, vk::Ref(eq), ED, Epi::F16, Act::None, R(Y.q_b), 1.0f, ~0u, err) ||
            !K.gemm_w(rc, ar, Y.k, vk::Ref(eh), A, ED, vk::Ref(ek), ED, Epi::F16, Act::None, R(Y.k_b), 1.0f, ~0u, err) ||
            !K.gemm_w(rc, ar, Y.v, vk::Ref(eh), A, ED, vk::Ref(ev), ED, Epi::F16, Act::None, R(Y.v_b), 1.0f, ~0u, err))
            return false;
        rc.barrier();
        if (!attention(0, n_full, W)) return false;
        if (!attention(n_full * W, tail_w ? 1u : 0u, tail_w)) return false;
        rc.barrier();
        rc.label("enc_attn_o");
        if (!K.gemm_w(rc, ar, Y.o, vk::Ref(ectx), A, ED, vk::Ref(ex), ED, Epi::Residual, Act::None, R(Y.o_b), 1.0f, ~0u, err))
            return false;
        rc.barrier();
        rc.label("norm");
        if (!K.norm(rc, 0, A, ED, vk::Ref(ex), ED, R(Y.ln2_g), R(Y.ln2_b), {}, {}, vk::Ref(eh), ED, eps, eps, err))
            return false;
        rc.barrier();
        rc.label("enc_ff_up");
        if (!K.gemm_w(rc, ar, Y.fc1, vk::Ref(eh), A, ED, vk::Ref(effh), EFF, Epi::F16, Act::Gelu, R(Y.fc1_b), 1.0f, ~0u, err))
            return false;
        rc.barrier();
        rc.label("enc_ff_down");
        if (!K.gemm_w(rc, ar, Y.fc2, vk::Ref(effh), A, EFF, vk::Ref(ex), ED, Epi::Residual, Act::None, R(Y.fc2_b), 1.0f, ~0u, err))
            return false;
        rc.barrier();
    }
    rc.split();
    rc.label("norm");
    if (!K.norm(rc, 0, A, ED, vk::Ref(ex), ED, R(lnp_g), R(lnp_b), {}, {}, vk::Ref(eh), ED, eps, eps, err))
        return false;
    rc.barrier();
    rc.label("enc_proj");
    if (!K.gemm_w(rc, ar, p1, vk::Ref(eh), A, ED, vk::Ref(eproj), ED, Epi::F16, Act::Gelu, R(p1_b), 1.0f, ~0u, err))
        return false;
    rc.barrier();
    if (!K.gemm_w(rc, ar, p2, vk::Ref(eproj), A, ED, vk::Ref(eproj2), EOUT, Epi::F16, Act::None, R(p2_b), 1.0f, ~0u, err))
        return false;
    rc.barrier();
    rc.label("adapter");
    if (!K.gemm_w(rc, ar, ad_gateup, vk::Ref(eproj2), A, EOUT, vk::Ref(adh), AH, Epi::SwiGlu, Act::None, {}, 1.0f, ~0u, err))
        return false;
    rc.barrier();
    // Adapter output lands directly in the LLM input rows [npre, npre + A).
    {
        GemmCall c;
        c.b = BKind::W4;
        if (ad_down.fmt == GpuFmt::W8) c.b = BKind::W8;
        if (ad_down.fmt == GpuFmt::F16) c.b = BKind::F16;
        c.epi = Epi::F32;
        c.a.M = A; c.a.N = ad_down.N; c.a.K = ad_down.K;
        c.a.lda = AH; c.a.ldb = ad_down.K; c.a.ldc = HD; c.a.c_off = npre * HD;
        c.A = vk::Ref(adh); c.Bq = R(ad_down.q);
        if (ad_down.has_s) c.Bs = R(ad_down.s);
        c.C = vk::Ref(lx);
        if (!K.gemm(rc, c, err)) return false;
    }
    // Prompt token embeddings (ids uploaded at load time: prefix then suffix).
    {
        const char* name = embed.fmt == GpuFmt::W8 ? "embed_rows_w8" : embed.fmt == GpuFmt::W4 ? "embed_rows" : "embed_rows_f16";
        const vk::Pipeline* pp = ctx->pipeline(name, {256u, embed.chunk_rows}, err);
        if (!pp) return false;
        std::vector<vk::Ref> b(10, vk::Ref(K.dummy()));
        b[0] = vk::Ref(ids);
        for (size_t c = 0; c < embed.parts.size(); ++c) {
            b[1 + c] = R(embed.parts[c].q);
            if (embed.parts[c].has_s) b[5 + c] = R(embed.parts[c].s);
        }
        b[9] = vk::Ref(lx);
        struct { uint32_t K, ids_off, out_row0; } a1{HD, 0, 0}, a2{HD, npre, npre + A};
        rc.label("embed");
        rc.dispatch(*pp, b, &a1, sizeof(a1), npre);
        rc.dispatch(*pp, b, &a2, sizeof(a2), (uint32_t)cfg.prompt_suffix.size());
    }
    rc.barrier();

    // ---- LLM prefill ----
    const uint32_t QKVW = (NH + 2 * NKV) * HDIM;
    const uint32_t ldS = S, ldP = round_up(S, 8);
    // Head groups are whole GQA groups (each group's first head starts a KV
    // head), as many as the score buffers allocated by ensure() hold.
    const uint32_t gqa = NH / NKV;
    uint32_t hgroup = (uint32_t)std::min<size_t>(
        {(size_t)NH, lsc.size / ((size_t)S * S * 4), lp.size / ((size_t)S * ldP * 2)});
    hgroup = std::max(gqa, hgroup / gqa * gqa);
    const float leps = cfg.llm.rms_norm_eps;
    const float lscale = 1.0f / std::sqrt((float)HDIM);
    for (uint32_t l = 0; l < NL; ++l) {
        const LlmLayer& Y = llm[l];
        const uint32_t coff = l * NKV * MAXPOS * HDIM;
        rc.split();
        rc.label("norm");
        if (!K.norm(rc, 2, S, HD, vk::Ref(lx), HD, R(Y.attn_norm), {}, {}, {}, vk::Ref(lhn), HD, leps, leps, err))
            return false;
        rc.barrier();
        rc.label("llm_qkv");
        if (!K.gemm_w(rc, ar, Y.qkv, vk::Ref(lhn), S, HD, vk::Ref(lqkv), QKVW, Epi::F16, Act::None, {}, 1.0f, ~0u, err))
            return false;
        rc.barrier();
        rc.label("rope_kv");
        {
            const vk::Pipeline* pp = ctx->pipeline("rope_kv", {HDIM, MAXPOS}, err);
            if (!pp) return false;
            struct { uint32_t H, KVH, cache_off; float eps; uint32_t pos0; } a{NH, NKV, coff, leps, 0};
            rc.dispatch(*pp, {vk::Ref(lqkv), vk::Ref(kcache), vk::Ref(vcache), vk::Ref(lqr), R(Y.q_norm),
                              R(Y.k_norm), R(rope_tab)},
                        &a, sizeof(a), S, NH + NKV);
        }
        rc.barrier();
        for (uint32_t h0 = 0; h0 < NH; h0 += hgroup) {
            const uint32_t hn = std::min(hgroup, NH - h0);
            GemmCall c;
            c.b = BKind::F16; c.epi = Epi::F32;
            c.a.M = S; c.a.N = S; c.a.K = HDIM;
            c.a.lda = NH * HDIM; c.a.ldb = HDIM; c.a.ldc = ldS;
            c.a.a_off = h0 * HDIM; c.a.b_off = coff + (h0 / (NH / NKV)) * MAXPOS * HDIM;
            c.a.nb_lo = hn; c.a.sa_lo = HDIM; c.a.sb_lo = MAXPOS * HDIM; c.a.gqa = NH / NKV;
            c.a.sc_lo = S * ldS;
            c.batch = hn;
            c.A = vk::Ref(lqr); c.Bq = vk::Ref(kcache); c.C = vk::Ref(lsc);
            rc.label("llm_attn_qk");
            if (!K.gemm(rc, c, err)) return false;
            rc.barrier();
            Kernels::SoftmaxArgs sa{S, ldS, 0, ldP, S * ldS, 0, S * ldP, lscale, S, 0u, S};
            rc.label("llm_softmax");
            if (!K.softmax(rc, false, S, hn, sa, vk::Ref(lsc), {}, vk::Ref(lp), err)) return false;
            rc.barrier();
            GemmCall v;
            v.b = BKind::F16T; v.epi = Epi::F16;
            v.a.M = S; v.a.N = HDIM; v.a.K = S;
            v.a.lda = ldP; v.a.ldb = HDIM; v.a.ldc = NH * HDIM;
            v.a.b_off = coff + (h0 / (NH / NKV)) * MAXPOS * HDIM; v.a.c_off = h0 * HDIM;
            v.a.nb_lo = hn; v.a.sa_lo = S * ldP; v.a.sb_lo = MAXPOS * HDIM; v.a.gqa = NH / NKV;
            v.a.sc_lo = HDIM;
            v.batch = hn;
            v.A = vk::Ref(lp); v.Bq = vk::Ref(vcache); v.C = vk::Ref(lctx);
            rc.label("llm_attn_pv");
            if (!K.gemm(rc, v, err)) return false;
            rc.barrier();
        }
        rc.label("llm_attn_o");
        if (!K.gemm_w(rc, ar, Y.o, vk::Ref(lctx), S, NH * HDIM, vk::Ref(lx), HD, Epi::Residual, Act::None, {}, 1.0f, ~0u, err))
            return false;
        rc.barrier();
        rc.label("norm");
        if (!K.norm(rc, 2, S, HD, vk::Ref(lx), HD, R(Y.ffn_norm), {}, {}, {}, vk::Ref(lhn), HD, leps, leps, err))
            return false;
        rc.barrier();
        rc.label("llm_ff_up");
        if (!K.gemm_w(rc, ar, Y.gateup, vk::Ref(lhn), S, HD, vk::Ref(lmlp), FF, Epi::SwiGlu, Act::None, {}, 1.0f, ~0u, err))
            return false;
        rc.barrier();
        rc.label("llm_ff_down");
        if (!K.gemm_w(rc, ar, Y.down, vk::Ref(lmlp), S, FF, vk::Ref(lx), HD, Epi::Residual, Act::None, {}, 1.0f, ~0u, err))
            return false;
        rc.barrier();
    }
    // First token: logits of the last prompt row only.
    rc.split();
    if (!lm_head(rc, vk::Ref(lx), (S - 1) * HD, err)) return false;
    rc.barrier();
    if (!dec_next(rc, err)) return false;
    rc.end();
    return true;
}

bool MossEngine::Impl::record_decode(uint32_t steps, std::string& err) {
    dec_rec = std::make_unique<vk::Recording>(*ctx);
    vk::Recording& rc = *dec_rec;
    rc.begin();
    const float leps = cfg.llm.rms_norm_eps;
    const float lscale = 1.0f / std::sqrt((float)HDIM);
    const vk::Pipeline* ap = ctx->pipeline("attn_decode", {HDIM, MAXPOS}, err);
    if (!ap) return false;
    for (uint32_t s = 0; s < steps; ++s) {
        if (s) rc.split();
        for (uint32_t l = 0; l < NL; ++l) {
            const LlmLayer& Y = llm[l];
            Kernels::GemvArgs a;
            a.eps = leps;
            rc.label("dec_qkv");
            if (!K.gemv(rc, ar, Y.qkv, vk::Ref(xd), vk::Ref(qkvd), R(Y.attn_norm), vk::Ref(state), {}, 0, a, err))
                return false;
            rc.barrier();
            rc.label("dec_attn");
            struct { uint32_t H, KVH, cache_off; float eps, scale; } pa{NH, NKV, l * NKV * MAXPOS * HDIM, leps, lscale};
            rc.dispatch(*ap, {vk::Ref(qkvd), vk::Ref(kcache), vk::Ref(vcache), vk::Ref(attd), R(Y.q_norm),
                              R(Y.k_norm), R(rope_tab), vk::Ref(state)},
                        &pa, sizeof(pa), NH);
            rc.barrier();
            rc.label("dec_o");
            if (!K.gemv(rc, ar, Y.o, vk::Ref(attd), vk::Ref(xd), {}, vk::Ref(state), {}, 1, a, err)) return false;
            rc.barrier();
            rc.label("dec_ff_up");
            if (!K.gemv(rc, ar, Y.gateup, vk::Ref(xd), vk::Ref(hd), R(Y.ffn_norm), vk::Ref(state), {}, 2, a, err))
                return false;
            rc.barrier();
            rc.label("dec_ff_down");
            if (!K.gemv(rc, ar, Y.down, vk::Ref(hd), vk::Ref(xd), {}, vk::Ref(state), {}, 1, a, err)) return false;
            rc.barrier();
        }
        if (!lm_head(rc, vk::Ref(xd), 0, err)) return false;
        rc.barrier();
        if (!dec_next(rc, err)) return false;
        rc.barrier();
    }
    rc.end();
    dec_steps = steps;
    return true;
}

bool MossEngine::generate(const float* pcm, size_t n, std::vector<int32_t>& out_ids, bool& eos,
                          std::string& err) {
    Impl& I = *impl_;
    const bool timing = env_on("STARLING_FAST_TIMING");
    const auto t0 = std::chrono::steady_clock::now();
    ms::MelFeatures mel;
    if (!ms::compute_log_mel(I.cfg, I.mel_loader, pcm, n, mel, err)) return false;
    const int64_t T = mel.n_frames;
    if (T <= 0) { err = "fast moss: empty audio"; return false; }
    const int C = (int)((T + 99) / 100);
    const int tail = (int)(T % 100 ? T % 100 : 100);
    const int P = C == 1 ? tail : 100;
    const int A = (int)ms::audio_token_length(T);
    const int S = (int)(I.cfg.prompt_prefix.size() + A + I.cfg.prompt_suffix.size());
    if (S + (int)I.cfg.max_new_tokens > (int)I.MAXPOS) { err = "MOSS generation exceeds cache"; return false; }
    if (!I.ensure(C, A, S, err)) return false;

    // Mel chunks [C][128][P] (bf16 values, zero-padded tail).
    std::vector<float> chunks((size_t)C * 128 * P, 0.0f);
    for (int ci = 0; ci < C; ++ci) {
        const int len = ci == C - 1 ? tail : 100;
        for (int f = 0; f < 128; ++f)
            for (int t = 0; t < len; ++t)
                chunks[((size_t)ci * 128 + f) * P + t] = ggml_bf16_to_fp32(mel.data[(size_t)f * T + ci * 100 + t]);
    }
    const double t_mel = ms_since(t0);

    auto key = std::make_pair(C, tail);
    auto it = I.recs.find(key);
    if (it == I.recs.end()) {
        if (I.recs.size() >= 8) {
            auto old = I.recs.begin();
            for (auto j = I.recs.begin(); j != I.recs.end(); ++j)
                if (j->second.used < old->second.used) old = j;
            I.recs.erase(old);
        }
        Impl::Rec r;
        if (!I.record_prefill(C, tail, r, err)) return false;
        it = I.recs.emplace(key, std::move(r)).first;
    }
    it->second.used = ++I.tick;
    const uint32_t steps = (uint32_t)std::max(1, std::atoi(std::getenv("STARLING_FAST_KSTEP") ? std::getenv("STARLING_FAST_KSTEP") : "16"));
    if (!I.dec_rec || I.dec_steps != steps) {
        if (!I.record_decode(steps, err)) return false;
    }

    // Inputs: mel chunks, prompt ids, fresh state.
    if (!I.ctx->upload(I.mel_in, 0, chunks.data(), chunks.size() * 4, err)) return false;
    {
        std::vector<uint32_t> idv;
        for (int32_t v : I.cfg.prompt_prefix) idv.push_back((uint32_t)v);
        for (int32_t v : I.cfg.prompt_suffix) idv.push_back((uint32_t)v);
        if (!I.ctx->upload(I.ids, 0, idv.data(), idv.size() * 4, err)) return false;
        uint32_t st[16] = {};
        st[0] = (uint32_t)S - 1;   // dec_next advances to S: the first generated token's position
        if (!I.ctx->upload(I.state, 0, st, sizeof st, err)) return false;
    }
    if (!it->second.rec->submit_and_wait(err)) return false;
    it->second.rec->report_profile("moss encoder+prefill");
    const double t_pre = ms_since(t0);

    std::vector<uint32_t> st(16 + (size_t)I.cfg.max_new_tokens);
    int rounds = 0;
    // The device stops at EOS or the budget; the host bound only guards
    // against a state buffer that never reports either.
    const int max_rounds = (int)(I.cfg.max_new_tokens / steps) + 2;
    for (;;) {
        if (!I.ctx->download(I.state, 0, st.data(), 16 * 4, err)) return false;
        if (st[2] != 0 || st[1] >= I.cfg.max_new_tokens) break;
        if (rounds >= max_rounds) { err = "fast moss: decode did not terminate"; return false; }
        if (!I.dec_rec->submit_and_wait(err)) return false;
        if (rounds == 0) I.dec_rec->report_profile("moss decode (K steps)");
        ++rounds;
    }
    const uint32_t ngen = std::min<uint32_t>(st[1], I.cfg.max_new_tokens);
    if (!I.ctx->download(I.state, 16 * 4, st.data() + 16, ngen * 4, err)) return false;
    out_ids.assign(st.begin() + 16, st.begin() + 16 + ngen);
    eos = !out_ids.empty() && out_ids.back() == I.cfg.eos_token_id;
    // Debug dump of the greedy token stream (#311 draft-acceptance studies):
    // STARLING_FAST_DUMP_TOKENS=path writes the generated ids (comma
    // separated); a path used repeatedly gets .1, .2 ... suffixes.
    if (const char* dp = std::getenv("STARLING_FAST_DUMP_TOKENS")) {
        static std::mutex mu;              // engine use is single-threaded
        static std::map<std::string, int> seq;   // today; guarded regardless
        std::string path;
        {
            std::lock_guard<std::mutex> lk(mu);
            path = dp;
            if (!seq.emplace(path, 0).second) path += "." + std::to_string(++seq[dp]);
        }
        if (FILE* f = std::fopen(path.c_str(), "w")) {
            for (size_t i = 0; i < out_ids.size(); ++i)
                std::fprintf(f, i ? ",%d" : "%d", (int)out_ids[i]);
            std::fclose(f);
        } else {
            std::fprintf(stderr, "[fast-moss] STARLING_FAST_DUMP_TOKENS: cannot write %s\n", path.c_str());
        }
    }
    if (timing)
        std::fprintf(stderr, "[fast-moss] audio=%.2fs mel=%.1fms enc+prefill=%.1fms decode=%.1fms (%u tokens, %d rounds) total=%.1fms (T=%lld A=%d S=%d)\n",
                     n / 16000.0, t_mel, t_pre - t_mel, ms_since(t0) - t_pre, ngen, rounds, ms_since(t0),
                     (long long)T, A, S);
    return true;
}

} // namespace starling::fast
