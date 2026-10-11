// audio_llm_engine.cpp — see audio_llm_engine.hpp.

#include "audio_llm_engine.hpp"

#include "kernels.hpp"
#include "packed_file.hpp"
#include "vk_runtime.hpp"
#include "weights.hpp"

#include "ark/loader.hpp"
#include "granite/loader.hpp"
#include "moss/loader.hpp"
#include "moss/mel.hpp"
#include "qwen3/loader.hpp"
#include "qwen3/mel.hpp"

#include "ggml.h"
#include "starling_ggml.h"   // starling_ggml_stop_requested (#325 coop stop)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace starling::fast {

constexpr size_t kMaxPromptIds = 256;   // prompt prefix + suffix ids (device buffer)

namespace ms = starling::ggml::moss;
namespace q3 = starling::ggml::qwen3;
namespace ak = starling::ggml::ark;
namespace gr = starling::ggml::granite;

namespace {

bool env_on(const char* name) {
    const char* v = std::getenv(name);
    return v && v[0] && v[0] != '0';
}
double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}
int half_len(int x) { return (x - 1) / 2 + 1; }

// Audio tokens for T valid mel frames: 13 per full 100-frame chunk plus the
// triple ceil-halved remainder (both families' post-CNN length).
int64_t audio_tokens(int64_t T) {
    if (T <= 0) return 0;
    const int64_t r = T % 100;
    return 13 * (T / 100) + (r ? half_len(half_len(half_len((int)r))) : 0);
}

// Mel features as the encoder consumes them: bf16, feature-major with row
// stride `stride`, `T` valid frames (frames beyond T are ignored).
struct MelView {
    std::vector<ggml_bf16_t> data;
    int64_t stride = 0, T = 0;
};

// What differs between the families the engine serves. Tensor names are
// the GGUF names; the per-layer encoder names are suffixes of "enc.blk.<l>.".
struct Spec {
    const char* tag = "";   // messages and timing lines ("moss", "qwen3", "ark", "granite")
    bool encoder = true;    // false: decoder-only, the host supplies the audio rows
    // ARK's Whisper encoder (with `encoder` false): the host supplies the
    // log-mel; conv stem, RoPE global-attention layers, ln_post and the
    // merge-by-`merge` GELU adapter run on the device.
    bool whisper = false;
    uint32_t merge = 0, rope_half = 0;
    std::string conv_out, pos_embed, proj1, proj2;
    std::string eq, ek, ev, eo, efc1, efc2;
    bool adapter = false;   // MOSS: SwiGLU adapter between projection and LLM
    // encoder
    uint32_t ED = 0, EH = 0, EHD = 0, EFF = 0, EL = 0, DS = 0, EOUT = 0, n_window_infer = 0, n_mels = 0;
    float enc_eps = 1e-5f;
    uint32_t AH = 0;        // adapter hidden (0 without an adapter)
    // LLM
    uint32_t HD = 0, NH = 0, NKV = 0, HDIM = 0, FF = 0, NL = 0, MAXPOS = 0;
    float rope_theta = 0, rms_eps = 0;
    bool tied = false;      // lm_head == llm.embed (and the config is supported)
    std::string lm_head;    // untied head tensor (Granite); "" = tied
    bool qkv_bias = false;  // Qwen2.5 (ARK): biased q/k/v projections
    bool qk_norm = true;    // Qwen3: per-head q/k RMSNorm
    float attn_scale = 0;   // 0 = 1/sqrt(head_dim) (Granite: attention_multiplier)
    float emb_mult = 1.0f, resid_mult = 1.0f;   // Granite (folded into weights)
    std::vector<int32_t> banned;   // ids never picked (ARK 0.6B bad_words_ids)
    int32_t eos = 0;
    uint32_t max_new_tokens = 0;
    std::vector<int32_t> prefix, suffix;
    // The family's CPU mel frontend over the mel constants in `ml`.
    std::function<bool(const ggml::ModelLoader& ml, const float* pcm, size_t n, MelView& mel,
                       std::string& err)> mel;
};

Spec moss_spec(const ms::Config& c) {
    Spec s;
    s.tag = "moss";
    s.conv_out = "enc.conv_out.weight"; s.pos_embed = "enc.positional_embedding";
    s.proj1 = "enc.proj1"; s.proj2 = "enc.proj2";
    s.eq = "attn.q"; s.ek = "attn.k"; s.ev = "attn.v"; s.eo = "attn.o"; s.efc1 = "ffn.fc1"; s.efc2 = "ffn.fc2";
    s.adapter = true;
    const auto& ec = c.encoder;
    const auto& lc = c.llm;
    s.ED = ec.d_model; s.EH = ec.n_heads; s.EHD = ec.head_dim; s.EFF = ec.ff_dim; s.EL = ec.n_layers;
    s.DS = ec.downsample_hidden_size; s.EOUT = ec.output_dim; s.n_window_infer = ec.n_window_infer;
    s.n_mels = c.frontend.n_mels; s.enc_eps = ec.layer_norm_eps;
    s.AH = c.adapter_hidden;
    s.HD = lc.hidden; s.NH = lc.n_heads; s.NKV = lc.n_kv_heads; s.HDIM = lc.head_dim; s.FF = lc.intermediate;
    s.NL = lc.n_layers; s.MAXPOS = lc.max_cache; s.rope_theta = lc.rope_theta; s.rms_eps = lc.rms_norm_eps;
    s.tied = lc.tied_embeddings && c.adapter_input == ec.output_dim && c.adapter_output == lc.hidden;
    s.eos = c.eos_token_id; s.max_new_tokens = c.max_new_tokens;
    s.prefix = c.prompt_prefix; s.suffix = c.prompt_suffix;
    s.mel = [c](const ggml::ModelLoader& ml, const float* pcm, size_t n, MelView& mel, std::string& err) {
        ms::MelFeatures f;
        if (!ms::compute_log_mel(c, ml, pcm, n, f, err)) return false;
        mel.stride = mel.T = f.n_frames;
        mel.data = std::move(f.data);
        return true;
    };
    return s;
}

// Qwen3-ASR decodes at most ~5 tokens per second of audio plus a fixed
// allowance, and the C API cuts audio into <= 30 s pieces, so a prompt and
// its budget stay far below the GGUF's 4096-slot max_cache. The KV cache is
// sized to this instead (UMA memory on phones: 4096 slots would be 470 MiB).
constexpr uint32_t kQwen3MaxPos = 1024;

Spec qwen3_spec(const q3::Config& c) {
    Spec s;
    s.tag = "qwen3";
    s.conv_out = "enc.out.weight"; s.pos_embed = "enc.pos_embed";
    s.proj1 = "proj.linear_1"; s.proj2 = "proj.linear_2";
    s.eq = "attn_q"; s.ek = "attn_k"; s.ev = "attn_v"; s.eo = "attn_o"; s.efc1 = "ff_up"; s.efc2 = "ff_down";
    s.adapter = false;
    const auto& ec = c.encoder;
    const auto& lc = c.llm;
    s.ED = ec.hidden; s.EH = ec.n_heads; s.EHD = ec.head_dim; s.EFF = ec.ffn_dim; s.EL = ec.n_layers;
    s.DS = ec.downsample_hidden; s.EOUT = c.projector.output_dim; s.n_window_infer = ec.n_window_infer;
    s.n_mels = c.frontend.n_mels; s.enc_eps = ec.layer_norm_eps;
    s.HD = lc.hidden; s.NH = lc.n_heads; s.NKV = lc.n_kv_heads; s.HDIM = lc.head_dim; s.FF = lc.intermediate;
    s.NL = lc.n_layers; s.MAXPOS = std::min(lc.max_cache, kQwen3MaxPos);
    s.rope_theta = lc.rope_theta; s.rms_eps = lc.rms_norm_eps;
    s.tied = lc.tied_embeddings && lc.has_qk_norm && c.projector.hidden == ec.hidden &&
             c.encoder.n_mel == c.frontend.n_mels;
    s.eos = c.eos_token_id; s.max_new_tokens = c.max_new_tokens;
    s.prefix = c.prompt_prefix; s.suffix = c.prompt_suffix;
    s.mel = [c](const ggml::ModelLoader& ml, const float* pcm, size_t n, MelView& mel, std::string& err) {
        q3::MelFeatures f;
        if (!q3::compute_log_mel(c, ml, pcm, n, f, err)) return false;
        mel.stride = f.n_frames;
        mel.T = f.valid_frames;
        mel.data = std::move(f.data);
        return true;
    };
    return s;
}

// ARK keeps the ggml engine's prompt layout (prefix + audio slots + suffix,
// the slot count from the uncapped mel length, so audio past the encoder's
// 30 s window becomes zero rows) and has no chunking of its own: 2048 slots
// cover ~2.5 min of audio plus the 200-token budget at half the GGUF's
// 4096-slot max_cache (UMA memory on phones).
constexpr uint32_t kArkMaxPos = 2048;

Spec ark_spec(const ak::Config& c) {
    Spec s;
    s.tag = "ark";
    s.encoder = false;
    const auto& lc = c.llm;
    s.HD = lc.hidden; s.NH = lc.n_heads; s.NKV = lc.n_kv_heads; s.HDIM = lc.head_dim; s.FF = lc.intermediate;
    s.NL = lc.n_layers; s.MAXPOS = std::min(lc.max_cache, kArkMaxPos);
    s.rope_theta = lc.rope_theta; s.rms_eps = lc.rms_norm_eps;
    s.tied = lc.tied_embeddings && !lc.has_qk_norm && lc.rope_scaling == "none";
    s.qkv_bias = true;
    s.qk_norm = false;
    s.banned = c.bad_words_ids;
    s.eos = c.eos_token_id; s.max_new_tokens = c.max_new_tokens;
    // The Whisper encoder + adapter on the device unless
    // STARLING_FAST_ARK_HOST_ENCODER=1 keeps them on ggml (decoder-only).
    const auto& ec = c.encoder;
    const auto& ac = c.adapter;
    s.whisper = !env_on("STARLING_FAST_ARK_HOST_ENCODER") && ec.use_rope && ac.act == "gelu" &&
                ac.input_size == ec.d_model * ac.merge_factor && ac.output_size == lc.hidden &&
                ec.conv_kernel == 3;
    s.ED = ec.d_model; s.EH = ec.n_heads; s.EHD = ec.head_dim; s.EFF = ec.ffn_dim; s.EL = ec.n_layers;
    s.n_mels = ec.num_mel_bins; s.enc_eps = ec.layer_norm_eps;
    s.merge = ac.merge_factor; s.rope_half = ec.rope_dim / 2;
    s.AH = ac.hidden_size; s.EOUT = ac.output_size;
    return s;
}

Spec granite_spec(const gr::Config& c) {
    Spec s;
    s.tag = "granite";
    s.encoder = false;
    const auto& lc = c.llm;
    s.HD = lc.hidden; s.NH = lc.n_heads; s.NKV = lc.n_kv_heads; s.HDIM = lc.head_dim; s.FF = lc.intermediate;
    s.NL = lc.n_layers; s.MAXPOS = lc.max_cache;   // 640: the C API chunks audio to fit it
    s.rope_theta = lc.rope_theta; s.rms_eps = lc.rms_norm_eps;
    if (!lc.tied_embeddings && !lc.has_qk_norm) s.lm_head = "llm.lm_head.weight";
    s.qk_norm = false;
    s.attn_scale = lc.attention_multiplier;
    s.emb_mult = lc.embedding_multiplier;
    s.resid_mult = lc.residual_multiplier;
    s.eos = c.eos_token_id; s.max_new_tokens = c.max_new_tokens;
    return s;
}

struct EncLayer {
    GMat q, k, v, o, fc1, fc2;
    Arena::Id q_b, k_b, v_b, o_b, fc1_b, fc2_b, ln1_g, ln1_b, ln2_g, ln2_b;
};
struct LlmLayer {
    GMat qkv, o, gateup, down;
    Arena::Id attn_norm, ffn_norm, q_norm, k_norm, qkv_b;
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

struct AudioLlmEngine::Impl {
    // Keeps the shared Vulkan context alive while this engine exists.
    // Declared first, so it is released last — after every buffer and
    // recording below; the last engine's release destroys the device.
    std::shared_ptr<vk::Context> ctx_hold;
    vk::Context* ctx = nullptr;
    Kernels K;
    Arena ar;
    Spec sp;
    ggml::ModelLoader mel_loader;   // only the two mel constant tensors

    // dims (copied from sp)
    uint32_t ED = 0, EH = 0, EHD = 0, EFF = 0, EL = 0, DS = 0, EOUT = 0;
    uint32_t AH = 0;                                   // adapter hidden
    uint32_t HD = 0, NH = 0, NKV = 0, HDIM = 0, FF = 0, NL = 0, MAXPOS = 0;

    // encoder weights
    Arena::Id c1_w, c1_b, c2_b, c3_b, pe13, lnp_g, lnp_b, p1_b, p2_b;
    GMat c2, c3, cout, p1, p2, ad_gateup, ad_down;
    std::vector<EncLayer> enc;
    // Whisper (ARK): conv stem as GEMMs over [k*cin + c] rows, RoPE tables;
    // the adapter is p1/p2 (+ p1_b/p2_b), ln_post lnp_g/lnp_b.
    GMat wc1, wc2;
    Arena::Id wc1_b, wc2_b, rope_cos, rope_sin;
    uint32_t rope_rows = 0;   // encoder positions the RoPE tables cover
    // LLM weights
    std::vector<LlmLayer> llm;
    Arena::Id final_norm, rope_tab;
    ChunkedMat embed;                   // input table (dec_next, embed_rows)
    ChunkedMat head;                    // lm_head: `embed` when tied
    std::vector<Arena::Id> head_bias;   // per head chunk: -3e38 on banned ids

    // runtime buffers
    vk::Buffer kcache, vcache, state, ids, xd, qkvd, attd, hd, part;
    vk::Buffer mel_in, cv1, cv2, cv3, ex, eh, eq, ek, ev, esc, ep, ectx, effh, eproj, eproj2, adh;
    vk::Buffer lx, lhn, lqkv, lqr, lsc, lp, lctx, lmlp;
    vk::Buffer apart;   // attention split partials [NH][S][2+HDIM]
    static constexpr uint32_t kAttnSplitMax = 8;
    int cap_C = 0, cap_A = 0, cap_S = 0;
    uint32_t n_part = 0;

    struct Rec { std::unique_ptr<vk::Recording> rec; uint64_t used = 0; int A = 0, S = 0, slot0 = 0; };
    std::map<std::pair<int, int>, Rec> recs;           // keyed by (chunks, tail)
    std::unique_ptr<vk::Recording> dec_rec;
    uint32_t dec_steps = 0;
    uint64_t tick = 0;

    bool load(const ggml::ModelLoader& ml, std::string& err);
    bool alloc_static(std::string& err);
    bool ensure(int C, int A, int S, std::string& err);
    bool record_prefill(int C, int tail, Rec& r, std::string& err);
    // Decoder-only prefill over S host-supplied rows (embedding lookup of
    // the text slots, then the LLM).
    bool record_prefill_rows(int S, Rec& r, std::string& err);
    // Whisper (ARK) prefill: L mel frames through conv stem, encoder layers
    // and adapter into the audio slots [slot0, slot0 + n_slots) of lx, then
    // the text-slot lookup and the LLM over S rows.
    bool record_prefill_whisper(int L, int S, int slot0, int n_slots, Rec& r, std::string& err);
    // Host-side prompt rows: audio slots from `audio` (or zero), text slots
    // left to the device lookup (ids, 0xffffffff on audio slots).
    bool prompt_rows(const std::vector<int32_t>& prompt_ids, const std::vector<uint8_t>& audio_mask,
                     const float* audio, int64_t audio_rows, std::vector<float>& rows,
                     std::vector<uint32_t>& idv, std::string& err);
    // The LLM prefill over the S rows of lx, ending with the first token.
    bool record_llm(vk::Recording& rc, uint32_t S, std::string& err);
    bool embed_lookup(vk::Recording& rc, uint32_t n, uint32_t ids_off, uint32_t out_row0, std::string& err);
    // The prefill recording for `key`, made by `record` on first use; the
    // least recently used of more than 8 is dropped.
    Rec* prefill_rec(std::pair<int, int> key, const std::function<bool(Rec&)>& record) {
        auto it = recs.find(key);
        if (it == recs.end()) {
            if (recs.size() >= 8) {
                auto old = recs.begin();
                for (auto j = recs.begin(); j != recs.end(); ++j)
                    if (j->second.used < old->second.used) old = j;
                recs.erase(old);
            }
            Rec r;
            if (!record(r)) return nullptr;
            it = recs.emplace(key, std::move(r)).first;
        }
        it->second.used = ++tick;
        return &it->second;
    }
    // Fresh decode state for an S-row prompt, then the prefill recording and
    // the greedy rounds; fills the generated ids.
    // `t_pre` receives the ms from t0 to the end of the prefill.
    bool run(vk::Recording& prefill, uint32_t S, uint32_t max_new, std::chrono::steady_clock::time_point t0,
             std::vector<int32_t>& out_ids, bool& eos, int& rounds, double& t_pre, std::string& err);
    float attn_scale() const { return sp.attn_scale > 0 ? sp.attn_scale : 1.0f / std::sqrt((float)HDIM); }
    // Prefill attention heads per score-matrix group: whole GQA groups within
    // a 64 MiB budget (at least one GQA group, however long the prompt).
    uint32_t prefill_hgroup(size_t S) const {
        const uint32_t gqa = NH / NKV;
        const size_t fit = std::min<size_t>(NH, (64ull << 20) / (S * S * 4));
        return std::max<uint32_t>(gqa, (uint32_t)(fit / gqa * gqa));
    }
    // Whisper attention heads per score group: a 64 MiB score budget (at
    // least one head, at most all).
    uint32_t whisper_hgroup(size_t T) const {
        return (uint32_t)std::max<size_t>(1, std::min<size_t>(EH, (64ull << 20) / (T * T * 4)));
    }
    bool record_decode(uint32_t steps, std::string& err);
    bool lm_head(vk::Recording& rc, vk::Ref x, uint32_t x_off, std::string& err);
    bool dec_next(vk::Recording& rc, std::string& err);
    vk::Ref R(Arena::Id id) const { return ar.ref(id); }
};

AudioLlmEngine::AudioLlmEngine() : impl_(new Impl) {}
AudioLlmEngine::~AudioLlmEngine() {
    // Skipped on a hung GPU: vkDeviceWaitIdle has no timeout (#325); the
    // runtime then leaks this engine's objects to the driver instead.
    if (impl_ && impl_->ctx && !impl_->ctx->gpu_hung())
        impl_->ctx->fn().vkDeviceWaitIdle(impl_->ctx->device());
}

namespace {
std::unique_ptr<AudioLlmEngine::Impl> make_impl(Spec sp, const ggml::ModelLoader& ml, std::string& err) {
    std::shared_ptr<vk::Context> hold = vk::Context::acquire(err);
    if (!hold) return nullptr;
    auto I = std::make_unique<AudioLlmEngine::Impl>();
    I->ctx = hold.get();
    I->ctx_hold = std::move(hold);
    I->sp = std::move(sp);
    if (!I->K.init(*I->ctx, err) || !I->load(ml, err) || !I->alloc_static(err)) return nullptr;
    return I;
}
} // namespace

std::unique_ptr<AudioLlmEngine> AudioLlmEngine::create(const ms::MossModel& m, std::string& err) {
    auto I = make_impl(moss_spec(m.config), m.loader, err);
    if (!I) return nullptr;
    std::unique_ptr<AudioLlmEngine> e(new AudioLlmEngine());
    e->impl_ = std::move(I);
    return e;
}

std::unique_ptr<AudioLlmEngine> AudioLlmEngine::create(const q3::Qwen3Model& m, std::string& err) {
    auto I = make_impl(qwen3_spec(m.config), m.loader, err);
    if (!I) return nullptr;
    std::unique_ptr<AudioLlmEngine> e(new AudioLlmEngine());
    e->impl_ = std::move(I);
    return e;
}

std::unique_ptr<AudioLlmEngine> AudioLlmEngine::create(const ak::ArkModel& m, std::string& err) {
    auto I = make_impl(ark_spec(m.config), m.loader, err);
    if (!I) return nullptr;
    std::unique_ptr<AudioLlmEngine> e(new AudioLlmEngine());
    e->impl_ = std::move(I);
    return e;
}

std::unique_ptr<AudioLlmEngine> AudioLlmEngine::create(const gr::GraniteModel& m, std::string& err) {
    auto I = make_impl(granite_spec(m.config), m.loader, err);
    if (!I) return nullptr;
    std::unique_ptr<AudioLlmEngine> e(new AudioLlmEngine());
    e->impl_ = std::move(I);
    return e;
}

size_t release_host_weights(const ggml::ModelLoader& loader, const std::vector<std::string>& prefixes) {
    size_t released = 0;
#if defined(__linux__)
    // Only ranges inside the loader's own host allocation: once a GPU ggml
    // backend has mirrored the weights, tensor() returns device tensors
    // whose data pointers are not host memory.
    if (!loader.ctx()) return 0;
    const uintptr_t lo = (uintptr_t)ggml_get_mem_buffer(loader.ctx());
    const uintptr_t hi = lo + ggml_get_mem_size(loader.ctx());
    const uintptr_t page = (uintptr_t)sysconf(_SC_PAGESIZE);
    for (const std::string& n : loader.tensor_names()) {
        if (std::none_of(prefixes.begin(), prefixes.end(),
                         [&](const std::string& pf) { return n.rfind(pf, 0) == 0; }))
            continue;
        const ggml_tensor* t = loader.tensor(n.c_str());
        if (!t || !t->data || (uintptr_t)t->data < lo || (uintptr_t)t->data + ggml_nbytes(t) > hi) continue;
        // Whole pages inside the tensor only: neighbours keep their bytes.
        const uintptr_t b = ((uintptr_t)t->data + page - 1) / page * page;
        const uintptr_t e = ((uintptr_t)t->data + ggml_nbytes(t)) / page * page;
        if (e > b && madvise((void*)b, e - b, MADV_DONTNEED) == 0) released += e - b;
    }
#else
    (void)loader;
    (void)prefixes;
#endif
    return released;
}

std::string AudioLlmEngine::describe() const {
    char buf[256];
    std::snprintf(buf, sizeof buf, "fast/vulkan '%s' weights=%.0f MiB",
                  impl_->ctx->info().name.c_str(), impl_->ar.total_bytes() / 1048576.0);
    return buf;
}

bool AudioLlmEngine::Impl::load(const ggml::ModelLoader& ml, std::string& err) {
    const auto t_load0 = std::chrono::steady_clock::now();
    const std::string tag = sp.tag;
    ED = sp.ED; EH = sp.EH; EHD = sp.EHD; EFF = sp.EFF; EL = sp.EL;
    DS = sp.DS; EOUT = sp.EOUT; AH = sp.AH;
    HD = sp.HD; NH = sp.NH; NKV = sp.NKV; HDIM = sp.HDIM; FF = sp.FF;
    NL = sp.NL; MAXPOS = sp.MAXPOS;
    // EOUT == HD without an adapter: the projection writes the LLM input rows.
    const bool enc_ok = !sp.encoder ||
        (EHD == 64 && DS % 8 == 0 && sp.n_mels == 128 && sp.n_window_infer % 100 == 0 &&
         sp.n_window_infer != 0 && (sp.adapter ? AH != 0 : EOUT == HD) &&
         sp.prefix.size() + sp.suffix.size() <= kMaxPromptIds);
    const bool whisper_ok = !sp.whisper ||
        (sp.n_mels % 8 == 0 && ED % 32 == 0 && EFF % 32 == 0 && EHD % 8 == 0 && EH * EHD == ED &&
         sp.rope_half * 2 <= EHD && sp.merge != 0 && AH % 32 == 0 && EOUT == HD);
    // The input-embedding multiplier is folded into the input table, so it
    // needs an untied head.
    if (!enc_ok || !whisper_ok || (HDIM != 128 && HDIM != 64) || (!sp.tied && sp.lm_head.empty()) || MAXPOS > 4096 ||
        NKV == 0 || NH % NKV || HD % 32 || FF % 32 || sp.max_new_tokens > MAXPOS ||
        (sp.emb_mult != 1.0f && sp.lm_head.empty())) {
        err = "fast " + tag + ": unsupported configuration";
        return false;
    }
    auto T = [&](const std::string& n) -> const ggml_tensor* {
        const ggml_tensor* t = ml.tensor(n.c_str());
        if (!t && err.empty()) err = "fast " + tag + ": missing tensor " + n;
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
    // Fused row groups need one format: when the parts come in different
    // types (Granite: Q5_K gate -> W8, Q4_K up -> W4), the others are
    // repacked to W8.
    using Part = std::pair<const ggml_tensor*, HostMatrix*>;
    auto same_fmt = [](std::initializer_list<Part> parts, std::string& e) {
        bool mixed = false;
        for (const Part& p : parts) mixed |= p.second->fmt != parts.begin()->second->fmt;
        if (!mixed) return true;
        for (const Part& p : parts)
            if (p.second->fmt != GpuFmt::W8 && !pack_gpu_matrix_w8(p.first, *p.second, e)) return false;
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
            if (!pack_one(a, ta, ha, e) || !pack_one(b, tb, hb, e) || !same_fmt({{ta, &ha}, {tb, &hb}}, e))
                return false;
            const uint32_t n = ha.N;
            if (!concat_rows(ha, hb, e)) return false;
            std::vector<uint32_t> order(2 * n);
            for (uint32_t j = 0; j < n; ++j) { order[2 * j] = j; order[2 * j + 1] = n + j; }
            permute_rows(ha, order);
            return true;
        });
        return true;
    };

    if (sp.encoder) {
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
            if (w.size() != (size_t)DS * DS * 9) { err = "fast " + tag + ": conv shape"; return false; }
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
            if (!f32(sp.conv_out, w)) return false;
            const uint32_t Kc = 16 * DS;
            if (w.size() != (size_t)ED * Kc) { err = "fast " + tag + ": conv_out shape"; return false; }
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
            if (!f32(sp.pos_embed, pe)) return false;
            if (pe.size() < (size_t)13 * ED) { err = "fast " + tag + ": positional embedding shorter than 13 rows"; return false; }
            pe.resize((size_t)13 * ED);                  // rows [0, 13): one chunk of tokens
            pe13 = ar.add_f32(pe);
        }
        enc.resize(EL);
        for (uint32_t l = 0; l < EL; ++l) {
            EncLayer& Y = enc[l];
            const std::string p = "enc.blk." + std::to_string(l) + ".";
            if (!mat(p + sp.eq + ".weight", Y.q) || !mat(p + sp.ek + ".weight", Y.k) || !mat(p + sp.ev + ".weight", Y.v) ||
                !mat(p + sp.eo + ".weight", Y.o) || !mat(p + sp.efc1 + ".weight", Y.fc1) ||
                !mat(p + sp.efc2 + ".weight", Y.fc2) ||
                !vec_id(p + sp.eq + ".bias", Y.q_b) || !vec_id(p + sp.ek + ".bias", Y.k_b) ||
                !vec_id(p + sp.ev + ".bias", Y.v_b) || !vec_id(p + sp.eo + ".bias", Y.o_b) ||
                !vec_id(p + sp.efc1 + ".bias", Y.fc1_b) || !vec_id(p + sp.efc2 + ".bias", Y.fc2_b) ||
                !vec_id(p + "attn_norm.weight", Y.ln1_g) || !vec_id(p + "attn_norm.bias", Y.ln1_b) ||
                !vec_id(p + "ffn_norm.weight", Y.ln2_g) || !vec_id(p + "ffn_norm.bias", Y.ln2_b))
                return false;
        }
        if (!vec_id("enc.ln_post.weight", lnp_g) || !vec_id("enc.ln_post.bias", lnp_b) ||
            !mat(sp.proj1 + ".weight", p1) || !vec_id(sp.proj1 + ".bias", p1_b) ||
            !mat(sp.proj2 + ".weight", p2) || !vec_id(sp.proj2 + ".bias", p2_b))
            return false;
        if (sp.adapter && (!interleave("adapter.gate.weight", "adapter.up.weight", ad_gateup) ||
                           !mat("adapter.down.weight", ad_down)))
            return false;
    }

    // ---- Whisper encoder + adapter (ARK) ----
    if (sp.whisper) {
        // Conv1d k3 weights [OC][IC][K] (GGUF ne = K, IC, OC) as GEMM rows
        // [OC][k * IC + ic]: the stem convolves time-major activations, whose
        // three input rows t*s-1 .. t*s+1 are one contiguous run of 3*IC.
        auto conv = [&](const char* n, uint32_t IC, GMat& g, Arena::Id& b) {
            std::vector<float> w;
            if (!f32(std::string(n) + ".weight", w) || !vec_id(std::string(n) + ".bias", b)) return false;
            if (w.size() != (size_t)ED * IC * 3) { err = "fast " + tag + ": " + n + " shape"; return false; }
            std::vector<float> r(w.size());
            for (uint32_t oc = 0; oc < ED; ++oc)
                for (uint32_t ic = 0; ic < IC; ++ic)
                    for (uint32_t k = 0; k < 3; ++k)
                        r[(size_t)oc * 3 * IC + k * IC + ic] = w[((size_t)oc * IC + ic) * 3 + k];
            HostMatrix hm;
            if (!pack_gpu_matrix_raw(GGML_TYPE_F32, r.data(), ED, 3 * IC, hm, err)) return false;
            g = arena_matrix(ar, std::move(hm));
            return true;
        };
        if (!conv("enc.conv1", sp.n_mels, wc1, wc1_b) || !conv("enc.conv2", ED, wc2, wc2_b)) return false;
        {
            const ggml_tensor* tc = T("enc.rope_cos");
            if (!tc || tc->ne[0] != (int64_t)sp.rope_half) {
                if (err.empty()) err = "fast " + tag + ": encoder RoPE table shape";
                return false;
            }
            if (!vec_id("enc.rope_cos", rope_cos) || !vec_id("enc.rope_sin", rope_sin)) return false;
            rope_rows = (uint32_t)tc->ne[1];
        }
        enc.resize(EL);
        for (uint32_t l = 0; l < EL; ++l) {
            EncLayer& Y = enc[l];
            const std::string p = "enc.blk." + std::to_string(l) + ".";
            // k has no bias (WhisperRoPESdpaAttention).
            if (!mat(p + "attn.q.weight", Y.q) || !mat(p + "attn.k.weight", Y.k) ||
                !mat(p + "attn.v.weight", Y.v) || !mat(p + "attn.o.weight", Y.o) ||
                !mat(p + "ffn.fc1.weight", Y.fc1) || !mat(p + "ffn.fc2.weight", Y.fc2) ||
                !vec_id(p + "attn.q.bias", Y.q_b) || !vec_id(p + "attn.v.bias", Y.v_b) ||
                !vec_id(p + "attn.o.bias", Y.o_b) || !vec_id(p + "ffn.fc1.bias", Y.fc1_b) ||
                !vec_id(p + "ffn.fc2.bias", Y.fc2_b) ||
                !vec_id(p + "attn_norm.weight", Y.ln1_g) || !vec_id(p + "attn_norm.bias", Y.ln1_b) ||
                !vec_id(p + "ffn_norm.weight", Y.ln2_g) || !vec_id(p + "ffn_norm.bias", Y.ln2_b))
                return false;
        }
        if (!vec_id("enc.ln_post.weight", lnp_g) || !vec_id("enc.ln_post.bias", lnp_b) ||
            !mat("adapter.fc0.weight", p1) || !vec_id("adapter.fc0.bias", p1_b) ||
            !mat("adapter.fc2.weight", p2) || !vec_id("adapter.fc2.bias", p2_b))
            return false;
    }

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
                   same_fmt({{tq, &q}, {tk, &k}, {tv, &v}}, e) &&
                   concat_rows(q, k, e) && concat_rows(q, v, e);
        });
        // Granite's residual multiplier scales both residual branches: folded
        // into the o and down projections (their outputs feed only the
        // residual adds).
        auto resid_mat = [&](const std::string& n, GMat& g) {
            const ggml_tensor* t = T(n);
            if (!t) return false;
            jobs.add(&g, [&, t, n](HostMatrix& hm, std::string& e) {
                return pack_one(n, t, hm, e) && (sp.resid_mult == 1.0f || scale_matrix(hm, sp.resid_mult, e));
            });
            return true;
        };
        if (!resid_mat(p + "attn.o.weight", Y.o) ||
            !interleave(p + "ffn.gate.weight", p + "ffn.up.weight", Y.gateup) ||
            !resid_mat(p + "ffn.down.weight", Y.down) || !vec_id(p + "attn_norm.weight", Y.attn_norm) ||
            !vec_id(p + "ffn_norm.weight", Y.ffn_norm))
            return false;
        if (sp.qk_norm &&
            (!vec_id(p + "attn.q_norm.weight", Y.q_norm) || !vec_id(p + "attn.k_norm.weight", Y.k_norm)))
            return false;
        if (sp.qkv_bias) {
            std::vector<float> b, bk, bv;
            if (!f32(p + "attn.q.bias", b) || !f32(p + "attn.k.bias", bk) || !f32(p + "attn.v.bias", bv))
                return false;
            if (b.size() != (size_t)NH * HDIM || bk.size() != (size_t)NKV * HDIM || bv.size() != bk.size()) {
                err = "fast " + tag + ": q/k/v bias shape";
                return false;
            }
            b.insert(b.end(), bk.begin(), bk.end());
            b.insert(b.end(), bv.begin(), bv.end());
            Y.qkv_b = ar.add_f32(b);
        }
    }
    if (!vec_id("llm.final_norm.weight", final_norm)) return false;
    // Embedding table (and an untied lm_head): repacked in row blocks
    // (parallel), then regrouped into chunks that each fit one storage-buffer
    // binding. Granite's embedding multiplier is folded into the input table.
    const uint32_t emb_block = 8192;
    struct Table { std::string name; ChunkedMat* dst; float scale; std::vector<HostMatrix> blocks; };
    std::vector<Table> tables;
    tables.reserve(2);   // the repack jobs hold pointers into `blocks`
    tables.push_back({"llm.embed.weight", &embed, sp.emb_mult, {}});
    if (!sp.lm_head.empty()) tables.push_back({sp.lm_head, &head, 1.0f, {}});
    for (Table& tb : tables) {
        const ggml_tensor* te = T(tb.name);
        if (!te) return false;
        const uint32_t N = (uint32_t)te->ne[1], Kc = (uint32_t)te->ne[0];
        if (N == 0 || Kc != HD) { err = "fast " + tag + ": " + tb.name + " shape mismatch"; return false; }
        if (packed && packed->has(tb.name)) {
            HostMatrix whole;
            if (!packed->matrix(tb.name, whole, err)) return false;
            if (whole.K != Kc || whole.N != N) {
                err = tb.name + ": packed shape " + std::to_string(whole.N) + "x" +
                      std::to_string(whole.K) + " != GGUF " + std::to_string(N) + "x" + std::to_string(Kc);
                return false;
            }
            // Same 8192-row blocks the GGUF path repacks in.
            tb.blocks = split_rows(std::move(whole), emb_block);
            if (tb.scale != 1.0f)
                for (HostMatrix& b : tb.blocks)
                    if (!scale_matrix(b, tb.scale, err)) return false;
        } else {
            const size_t rb = ggml_row_size(te->type, Kc);
            const float scale = tb.scale;
            tb.blocks.resize((N + emb_block - 1) / emb_block);
            for (size_t b = 0; b < tb.blocks.size(); ++b) {
                const uint32_t r0 = (uint32_t)b * emb_block, n = std::min(emb_block, N - r0);
                jobs.add_host(&tb.blocks[b], [te, r0, n, Kc, rb, scale](HostMatrix& hm, std::string& e) {
                    return pack_gpu_matrix_raw((int)te->type, (const uint8_t*)te->data + (size_t)r0 * rb, n, Kc, hm, e) &&
                           (scale == 1.0f || scale_matrix(hm, scale, e));
                });
            }
        }
    }
    if (!jobs.run(ar, err)) return false;
    for (Table& tb : tables) {
        ChunkedMat& cm = *tb.dst;
        std::vector<HostMatrix>& blocks = tb.blocks;
        cm.K = blocks[0].K; cm.fmt = blocks[0].fmt;
        cm.N = 0;
        for (const HostMatrix& b : blocks) cm.N += b.N;
        const size_t row_bytes = (blocks[0].q.size() + blocks[0].s.size()) * 4 / blocks[0].N;
        const VkDeviceSize cap = std::min<VkDeviceSize>(256ull << 20, ctx->info().max_storage_range);
        cm.chunk_rows = (uint32_t)std::max<size_t>(emb_block, (cap / row_bytes) / emb_block * emb_block);
        if (((size_t)cm.N + cm.chunk_rows - 1) / cm.chunk_rows > 4) {
            err = "fast " + tag + ": " + tb.name + " needs more than 4 buffer chunks on this device";
            return false;
        }
        const size_t per_chunk = cm.chunk_rows / emb_block;
        for (size_t b0 = 0; b0 < blocks.size(); b0 += per_chunk) {
            HostMatrix c = std::move(blocks[b0]);
            for (size_t b = b0 + 1; b < std::min(blocks.size(), b0 + per_chunk); ++b)
                if (!concat_rows(c, blocks[b], err)) return false;
            cm.parts.push_back(arena_matrix(ar, std::move(c)));
        }
        std::vector<HostMatrix>().swap(blocks);
    }
    if (sp.lm_head.empty()) head = embed;
    // Suppressed ids (ARK 0.6B: every special and codec id but EOS) get a
    // -3e38 logit bias in the lm_head argmax, prefill and decode alike.
    if (!sp.banned.empty()) {
        for (size_t c = 0; c < head.parts.size(); ++c) {
            const uint32_t r0 = (uint32_t)c * head.chunk_rows;
            std::vector<float> b(head.parts[c].N, 0.0f);
            for (int32_t id : sp.banned)
                if (id >= 0 && (uint32_t)id >= r0 && (uint32_t)id - r0 < b.size()) b[id - r0] = -3.0e38f;
            head_bias.push_back(ar.add_f32(b));
        }
    }
    // RoPE table as Transformers builds it: inv_freq and pos*inv_freq in f32.
    {
        const uint32_t half = HDIM / 2;
        std::vector<float> tab((size_t)2 * MAXPOS * half);
        for (uint32_t i = 0; i < half; ++i) {
            const float ex = (float)(2 * i) / (float)HDIM;
            const float inv = 1.0f / std::pow(sp.rope_theta, ex);
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
        std::fprintf(stderr, "[fast-%s] load: repack %.1f ms, upload %.1f ms\n", sp.tag, t_pack,
                     ms_since(t_load0) - t_pack);
    if (env_on("STARLING_FAST_VERBOSE"))
        std::fprintf(stderr, "[fast] %s: weights %.1f MiB on '%s' (embed %s, %zu chunk(s)%s%s)\n", sp.tag,
                     ar.total_bytes() / 1048576.0, ctx->info().name.c_str(), fmt_name(embed.fmt),
                     embed.parts.size(), sp.lm_head.empty() ? "" : ", lm_head ",
                     sp.lm_head.empty() ? "" : fmt_name(head.fmt));
    return true;
}

bool AudioLlmEngine::Impl::alloc_static(std::string& err) {
    const size_t cache_elems = (size_t)NL * NKV * MAXPOS * HDIM;
    n_part = 0;
    for (const GMat& g : head.parts) n_part += ceil_div(g.N, K.gemv_rows(g.N));
    auto mk = [&](vk::Buffer& b, size_t bytes, vk::Mem kind = vk::Mem::Device) {
        return ctx->create_buffer(b, bytes, kind, err);
    };
    // Prompt ids: the prefix/suffix (encoder models) or a whole prompt
    // (decoder-only models; one id per row, at most MAXPOS rows).
    const size_t n_ids = sp.encoder ? kMaxPromptIds : MAXPOS;
    return mk(kcache, cache_elems * 2) && mk(vcache, cache_elems * 2) &&
           mk(state, (16 + (size_t)sp.max_new_tokens) * 4, vk::Mem::Readback) && mk(ids, n_ids * 4, vk::Mem::Readback) &&
           mk(xd, HD * 4) && mk(qkvd, (size_t)(NH + 2 * NKV) * HDIM * 4) && mk(attd, (size_t)NH * HDIM * 4) &&
           mk(hd, (size_t)FF * 4) && mk(part, (size_t)n_part * 8) &&
           mk(apart, (size_t)NH * kAttnSplitMax * (2 + HDIM) * 4);
}

bool AudioLlmEngine::Impl::ensure(int C, int A, int S, std::string& err) {
    if (C <= cap_C && A <= cap_A && S <= cap_S) return true;
    C = std::max(C, cap_C); A = std::max(A, cap_A); S = std::max(S, cap_S);
    recs.clear();
    const size_t P = 100, T1 = half_len((int)P), F1 = 64, T2 = half_len((int)T1), F2 = 32, T3 = half_len((int)T2), F3 = 16;
    // Decoder-only engines (C = A = 0) get placeholder encoder buffers.
    const size_t W = std::max<size_t>(1, 13 * (sp.n_window_infer / 100));
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
    size_t mel_b = (size_t)C * 128 * P * 4, cv1_b = (size_t)C * T1 * F1 * DS * 2,
           cv2_b = (size_t)C * T2 * F2 * DS * 2, cv3_b = (size_t)C * T3 * F3 * DS * 2,
           esc_b = nwin * EH * W * W * 4, ep_b = nwin * EH * W * ldPw * 2, adh_b = (size_t)A * AH * 2;
    if (sp.whisper) {
        // C = mel frames (time-major f16 with a zero row either side), A =
        // encoder frames; global attention scores in head groups (64 MiB).
        const size_t hw = whisper_hgroup((size_t)A);
        mel_b = ((size_t)C + 2) * sp.n_mels * 2;
        cv1_b = ((size_t)C + 2) * ED * 2;
        cv2_b = cv3_b = 0;
        esc_b = hw * A * A * 4;
        ep_b = hw * A * round_up((uint32_t)A, 8) * 2;
        adh_b = ((size_t)A / sp.merge + 1) * AH * 2;
    }
    if (!mk(mel_in, mel_b, ctx->info().uma ? vk::Mem::Device : vk::Mem::Upload) ||
        !mk(cv1, cv1_b) || !mk(cv2, cv2_b) ||
        !mk(cv3, cv3_b) || !mk(ex, (size_t)A * ED * 4) || !mk(eh, (size_t)A * ED * 2) ||
        !mk(eq, (size_t)A * ED * 2) || !mk(ek, (size_t)A * ED * 2) || !mk(ev, (size_t)A * ED * 2) ||
        !mk(esc, esc_b) || !mk(ep, ep_b) || !mk(ectx, (size_t)A * ED * 2) ||
        !mk(effh, (size_t)A * EFF * 2) || !mk(eproj, (size_t)A * ED * 2) || !mk(eproj2, (size_t)A * EOUT * 2) ||
        !mk(adh, adh_b) || !mk(lx, (size_t)S * HD * 4) || !mk(lhn, (size_t)S * HD * 2) ||
        !mk(lqkv, (size_t)S * (NH + 2 * NKV) * HDIM * 2) || !mk(lqr, (size_t)S * NH * HDIM * 2) ||
        !mk(lsc, hgroup * S * S * 4) || !mk(lp, hgroup * S * ldPs * 2) || !mk(lctx, (size_t)S * NH * HDIM * 2) ||
        !mk(lmlp, (size_t)S * FF * 2))
        return false;
    cap_C = C; cap_A = A; cap_S = S;
    return true;
}

bool AudioLlmEngine::Impl::lm_head(vk::Recording& rc, vk::Ref x, uint32_t x_off, std::string& err) {
    uint32_t off = 0;
    for (size_t c = 0; c < head.parts.size(); ++c) {
        const GMat& g = head.parts[c];
        Kernels::GemvArgs a;
        a.x_off = x_off;
        a.y_off = off * 2;
        a.eps = sp.rms_eps;
        a.row0 = (uint32_t)c * head.chunk_rows;
        rc.label("lm_head");
        if (!K.gemv(rc, ar, g, x, vk::Ref(part), R(final_norm), vk::Ref(state),
                    head_bias.empty() ? vk::Ref() : R(head_bias[c]), 3, a, err))
            return false;
        off += ceil_div(g.N, K.gemv_rows(g.N));
    }
    return true;
}

bool AudioLlmEngine::Impl::dec_next(vk::Recording& rc, std::string& err) {
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
    struct { uint32_t n_part, K, eos, max_steps; } pc{n_part, HD, (uint32_t)sp.eos, sp.max_new_tokens};
    rc.label("dec_next");
    rc.dispatch(*p, b, &pc, sizeof(pc), 1);
    return true;
}

bool AudioLlmEngine::Impl::record_prefill(int C, int tail, Rec& r, std::string& err) {
    const int P = C == 1 ? tail : 100;
    const int T1 = half_len(P), F1 = 64, T2 = half_len(T1), F2 = 32, T3 = half_len(T2), F3 = 16;
    const int64_t Tm = (int64_t)(C - 1) * 100 + tail;
    const uint32_t A = (uint32_t)audio_tokens(Tm);
    const uint32_t Mtok = (uint32_t)T3;                     // tokens per chunk
    const uint32_t W = Mtok * (sp.n_window_infer / 100);
    const uint32_t S = (uint32_t)(sp.prefix.size() + A + sp.suffix.size());
    const uint32_t npre = (uint32_t)sp.prefix.size();
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
    const float eps = sp.enc_eps;
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
    // The last projection (MOSS: the adapter's down projection) lands
    // directly in the LLM input rows [npre, npre + A).
    auto to_llm_rows = [&](const GMat& w, vk::Buffer& in, uint32_t lda, vk::Ref bias) {
        GemmCall c;
        c.b = w.fmt == GpuFmt::W4 ? BKind::W4 : w.fmt == GpuFmt::W8 ? BKind::W8 : BKind::F16;
        c.epi = Epi::F32;
        c.bias_mode = bias.buf ? 1u : 0u;
        c.a.M = A; c.a.N = w.N; c.a.K = w.K;
        c.a.lda = lda; c.a.ldb = w.K; c.a.ldc = HD; c.a.c_off = npre * HD;
        c.A = vk::Ref(in); c.Bq = R(w.q);
        if (w.has_s) c.Bs = R(w.s);
        c.C = vk::Ref(lx);
        c.bias = bias;
        return K.gemm(rc, c, err);
    };
    if (sp.adapter) {
        if (!K.gemm_w(rc, ar, p2, vk::Ref(eproj), A, ED, vk::Ref(eproj2), EOUT, Epi::F16, Act::None, R(p2_b), 1.0f, ~0u, err))
            return false;
        rc.barrier();
        rc.label("adapter");
        if (!K.gemm_w(rc, ar, ad_gateup, vk::Ref(eproj2), A, EOUT, vk::Ref(adh), AH, Epi::SwiGlu, Act::None, {}, 1.0f, ~0u, err))
            return false;
        rc.barrier();
        if (!to_llm_rows(ad_down, adh, AH, {})) return false;
    } else if (!to_llm_rows(p2, eproj, ED, R(p2_b))) {
        return false;
    }
    // Prompt token embeddings (ids uploaded per call: prefix then suffix).
    if (!embed_lookup(rc, npre, 0, 0, err) ||
        !embed_lookup(rc, (uint32_t)sp.suffix.size(), npre, npre + A, err))
        return false;
    rc.barrier();
    if (!record_llm(rc, S, err)) return false;
    rc.end();
    return true;
}

bool AudioLlmEngine::Impl::embed_lookup(vk::Recording& rc, uint32_t n, uint32_t ids_off, uint32_t out_row0,
                                        std::string& err) {
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
    struct { uint32_t K, ids_off, out_row0; } a{HD, ids_off, out_row0};
    rc.label("embed");
    rc.dispatch(*pp, b, &a, sizeof(a), n);
    return true;
}

bool AudioLlmEngine::Impl::record_prefill_rows(int S, Rec& r, std::string& err) {
    r.A = 0;
    r.S = S;
    r.rec = std::make_unique<vk::Recording>(*ctx);
    vk::Recording& rc = *r.rec;
    rc.begin();
    // Text slots from the input table; the audio slots (id 0xffffffff) were
    // uploaded with the rows.
    if (!embed_lookup(rc, (uint32_t)S, 0, 0, err)) return false;
    rc.barrier();
    if (!record_llm(rc, (uint32_t)S, err)) return false;
    rc.end();
    return true;
}

bool AudioLlmEngine::Impl::record_prefill_whisper(int L, int S, int slot0, int n_slots, Rec& r, std::string& err) {
    const uint32_t T = (uint32_t)(L + 1) / 2;            // conv2 stride 2
    const uint32_t Na = std::min<uint32_t>(T / sp.merge, (uint32_t)n_slots);
    r.A = (int)T;
    r.S = S;
    r.slot0 = slot0;
    r.rec = std::make_unique<vk::Recording>(*ctx);
    vk::Recording& rc = *r.rec;
    rc.begin();
    const uint32_t IC = sp.n_mels;

    // ---- conv stem: two k3 Conv1d + GELU as GEMMs over time-major rows ----
    // mel_in is [L + 2][IC] f16 with zero rows at both ends (padding 1), so
    // output row t reads the contiguous run of input rows t*s .. t*s + 2.
    // conv1 writes rows 1..L of cv1, whose two pad rows are cleared here.
    rc.fill(cv1, 0, (VkDeviceSize)ED * 2, 0);
    rc.fill(cv1, ((VkDeviceSize)L + 1) * ED * 2, (VkDeviceSize)ED * 2, 0);
    rc.barrier();
    auto stem = [&](const GMat& w, Arena::Id b, vk::Buffer& in, uint32_t M, uint32_t cin, uint32_t stride,
                    vk::Buffer& out, uint32_t c_off, Epi epi) {
        GemmCall c;
        c.b = BKind::F16; c.epi = epi; c.act = Act::Gelu; c.bias_mode = 1;
        c.a.M = M; c.a.N = ED; c.a.K = 3 * cin;
        c.a.lda = stride * cin; c.a.ldb = 3 * cin; c.a.ldc = ED; c.a.c_off = c_off;
        c.A = vk::Ref(in); c.Bq = R(w.q); c.C = vk::Ref(out); c.bias = R(b);
        return K.gemm(rc, c, err);
    };
    rc.label("enc_conv1");
    if (!stem(wc1, wc1_b, mel_in, (uint32_t)L, IC, 1, cv1, ED, Epi::F16)) return false;
    rc.barrier();
    rc.label("enc_conv2");
    if (!stem(wc2, wc2_b, cv1, T, ED, 2, ex, 0, Epi::F32)) return false;
    rc.barrier();

    // ---- encoder layers: pre-LN, RoPE global attention, GELU MLP ----
    const vk::Pipeline* rope = ctx->pipeline("enc_rope", {}, err);
    if (!rope) return false;
    const uint32_t ldP = round_up(T, 8);
    const uint32_t hg = (uint32_t)std::max<size_t>(
        1, std::min<size_t>({(size_t)EH, esc.size / ((size_t)T * T * 4), ep.size / ((size_t)T * ldP * 2)}));
    const float escale = 1.0f / std::sqrt((float)EHD);
    const float eps = sp.enc_eps;
    for (uint32_t l = 0; l < EL; ++l) {
        const EncLayer& Y = enc[l];
        rc.split();
        rc.label("norm");
        if (!K.norm(rc, 0, T, ED, vk::Ref(ex), ED, R(Y.ln1_g), R(Y.ln1_b), {}, {}, vk::Ref(eh), ED, eps, eps, err))
            return false;
        rc.barrier();
        rc.label("enc_qkv");
        if (!K.gemm_w(rc, ar, Y.q, vk::Ref(eh), T, ED, vk::Ref(eq), ED, Epi::F16, Act::None, R(Y.q_b), 1.0f, ~0u, err) ||
            !K.gemm_w(rc, ar, Y.k, vk::Ref(eh), T, ED, vk::Ref(ek), ED, Epi::F16, Act::None, {}, 1.0f, ~0u, err) ||
            !K.gemm_w(rc, ar, Y.v, vk::Ref(eh), T, ED, vk::Ref(ev), ED, Epi::F16, Act::None, R(Y.v_b), 1.0f, ~0u, err))
            return false;
        rc.barrier();
        rc.label("enc_rope");
        {
            struct { uint32_t T, H, D, half_n; } a{T, EH, EHD, sp.rope_half};
            rc.dispatch(*rope, {vk::Ref(eq), vk::Ref(ek), R(rope_cos), R(rope_sin)}, &a, sizeof(a),
                        ceil_div(T * EH * sp.rope_half, 64));
        }
        rc.barrier();
        for (uint32_t h0 = 0; h0 < EH; h0 += hg) {
            const uint32_t hn = std::min(hg, EH - h0);
            GemmCall c;
            c.b = BKind::F16; c.epi = Epi::F32;
            c.a.M = T; c.a.N = T; c.a.K = EHD;
            c.a.lda = ED; c.a.ldb = ED; c.a.ldc = T;
            c.a.a_off = h0 * EHD; c.a.b_off = h0 * EHD;
            c.a.nb_lo = hn; c.a.sa_lo = EHD; c.a.sb_lo = EHD; c.a.sc_lo = T * T;
            c.batch = hn;
            c.A = vk::Ref(eq); c.Bq = vk::Ref(ek); c.C = vk::Ref(esc);
            rc.label("enc_attn_qk");
            if (!K.gemm(rc, c, err)) return false;
            rc.barrier();
            Kernels::SoftmaxArgs sa{T, T, 0, ldP, T * T, 0, T * ldP, escale, T, 0xffffffffu, T};
            rc.label("enc_softmax");
            if (!K.softmax(rc, false, T, hn, sa, vk::Ref(esc), {}, vk::Ref(ep), err)) return false;
            rc.barrier();
            GemmCall v;
            v.b = BKind::F16T; v.epi = Epi::F16;
            v.a.M = T; v.a.N = EHD; v.a.K = T;
            v.a.lda = ldP; v.a.ldb = ED; v.a.ldc = ED;
            v.a.b_off = h0 * EHD; v.a.c_off = h0 * EHD;
            v.a.nb_lo = hn; v.a.sa_lo = T * ldP; v.a.sb_lo = EHD; v.a.sc_lo = EHD;
            v.batch = hn;
            v.A = vk::Ref(ep); v.Bq = vk::Ref(ev); v.C = vk::Ref(ectx);
            rc.label("enc_attn_pv");
            if (!K.gemm(rc, v, err)) return false;
            rc.barrier();
        }
        rc.label("enc_attn_o");
        if (!K.gemm_w(rc, ar, Y.o, vk::Ref(ectx), T, ED, vk::Ref(ex), ED, Epi::Residual, Act::None, R(Y.o_b), 1.0f, ~0u, err))
            return false;
        rc.barrier();
        rc.label("norm");
        if (!K.norm(rc, 0, T, ED, vk::Ref(ex), ED, R(Y.ln2_g), R(Y.ln2_b), {}, {}, vk::Ref(eh), ED, eps, eps, err))
            return false;
        rc.barrier();
        rc.label("enc_ff_up");
        if (!K.gemm_w(rc, ar, Y.fc1, vk::Ref(eh), T, ED, vk::Ref(effh), EFF, Epi::F16, Act::Gelu, R(Y.fc1_b), 1.0f, ~0u, err))
            return false;
        rc.barrier();
        rc.label("enc_ff_down");
        if (!K.gemm_w(rc, ar, Y.fc2, vk::Ref(effh), T, EFF, vk::Ref(ex), ED, Epi::Residual, Act::None, R(Y.fc2_b), 1.0f, ~0u, err))
            return false;
        rc.barrier();
    }

    // ---- ln_post, then the adapter: `merge` consecutive frames are one
    // contiguous row of merge*ED, GELU MLP, written into the audio slots ----
    rc.split();
    rc.label("norm");
    if (!K.norm(rc, 0, T, ED, vk::Ref(ex), ED, R(lnp_g), R(lnp_b), {}, {}, vk::Ref(eh), ED, eps, eps, err))
        return false;
    rc.barrier();
    if (Na > 0) {
        rc.label("adapter");
        if (!K.gemm_w(rc, ar, p1, vk::Ref(eh), Na, sp.merge * ED, vk::Ref(adh), AH, Epi::F16, Act::Gelu, R(p1_b), 1.0f, ~0u, err))
            return false;
        rc.barrier();
        GemmCall c;
        c.b = p2.fmt == GpuFmt::W4 ? BKind::W4 : p2.fmt == GpuFmt::W8 ? BKind::W8 : BKind::F16;
        c.epi = Epi::F32;
        c.bias_mode = 1;
        c.a.M = Na; c.a.N = p2.N; c.a.K = p2.K;
        c.a.lda = AH; c.a.ldb = p2.K; c.a.ldc = HD; c.a.c_off = (uint32_t)slot0 * HD;
        c.A = vk::Ref(adh); c.Bq = R(p2.q);
        if (p2.has_s) c.Bs = R(p2.s);
        c.C = vk::Ref(lx);
        c.bias = R(p2_b);
        if (!K.gemm(rc, c, err)) return false;
    }
    // Text slots from the input table (audio slots carry the skip marker).
    if (!embed_lookup(rc, (uint32_t)S, 0, 0, err)) return false;
    rc.barrier();
    if (!record_llm(rc, (uint32_t)S, err)) return false;
    rc.end();
    return true;
}

bool AudioLlmEngine::Impl::record_llm(vk::Recording& rc, uint32_t S, std::string& err) {
    const uint32_t QKVW = (NH + 2 * NKV) * HDIM;
    const uint32_t ldS = S, ldP = round_up(S, 8);
    // Head groups are whole GQA groups (each group's first head starts a KV
    // head), as many as the score buffers allocated by ensure() hold.
    const uint32_t gqa = NH / NKV;
    uint32_t hgroup = (uint32_t)std::min<size_t>(
        {(size_t)NH, lsc.size / ((size_t)S * S * 4), lp.size / ((size_t)S * ldP * 2)});
    hgroup = std::max(gqa, hgroup / gqa * gqa);
    const float leps = sp.rms_eps;
    const float lscale = attn_scale();
    const uint32_t qk_norm = sp.qk_norm ? 1u : 0u;
    for (uint32_t l = 0; l < NL; ++l) {
        const LlmLayer& Y = llm[l];
        const uint32_t coff = l * NKV * MAXPOS * HDIM;
        rc.split();
        rc.label("norm");
        if (!K.norm(rc, 2, S, HD, vk::Ref(lx), HD, R(Y.attn_norm), {}, {}, {}, vk::Ref(lhn), HD, leps, leps, err))
            return false;
        rc.barrier();
        rc.label("llm_qkv");
        if (!K.gemm_w(rc, ar, Y.qkv, vk::Ref(lhn), S, HD, vk::Ref(lqkv), QKVW, Epi::F16, Act::None,
                      sp.qkv_bias ? R(Y.qkv_b) : vk::Ref(), 1.0f, ~0u, err))
            return false;
        rc.barrier();
        rc.label("rope_kv");
        {
            const vk::Pipeline* pp = ctx->pipeline("rope_kv", {HDIM, MAXPOS, qk_norm}, err);
            if (!pp) return false;
            struct { uint32_t H, KVH, cache_off; float eps; uint32_t pos0; } a{NH, NKV, coff, leps, 0};
            rc.dispatch(*pp, {vk::Ref(lqkv), vk::Ref(kcache), vk::Ref(vcache), vk::Ref(lqr),
                              qk_norm ? R(Y.q_norm) : vk::Ref(K.dummy()), qk_norm ? R(Y.k_norm) : vk::Ref(K.dummy()),
                              R(rope_tab)},
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
    return dec_next(rc, err);
}

bool AudioLlmEngine::Impl::record_decode(uint32_t steps, std::string& err) {
    dec_rec = std::make_unique<vk::Recording>(*ctx);
    vk::Recording& rc = *dec_rec;
    rc.begin();
    const float leps = sp.rms_eps;
    const float lscale = attn_scale();
    const uint32_t qk_norm = sp.qk_norm ? 1u : 0u;
    // Decode attention as a flash-decode KV split: S chunks per query head,
    // then a combine. The single-pass kernel runs one workgroup per head (16
    // on MOSS), too few to hide cache-load latency on an iGPU. The chunked
    // exp/sum/value accumulation differs from the single pass at ulp level,
    // so the split is on by default only where it was measured and gated
    // (AMD/RADV); other vendors, including the Pixel's PowerVR, keep the
    // single pass. STARLING_FAST_ATTN_SPLIT=S overrides: 0 = single pass,
    // 1..kAttnSplitMax = chunk count.
    static const int split_env = [] {
        const char* e = std::getenv("STARLING_FAST_ATTN_SPLIT");
        if (!e) return -1;
        char* end = nullptr;
        const unsigned long v = std::strtoul(e, &end, 10);
        if (*e >= '0' && *e <= '9' && *end == '\0' && v <= kAttnSplitMax) return (int)v;
        std::fprintf(stderr, "[fast] ignoring STARLING_FAST_ATTN_SPLIT='%s' (want 0..%u)\n", e, kAttnSplitMax);
        return -1;
    }();
    const uint32_t asplit = split_env >= 0 ? (uint32_t)split_env : ctx->info().vendor_id == 0x1002 ? 4 : 0;
    const vk::Pipeline* ap = asplit ? ctx->pipeline("attn_decode_split", {HDIM, MAXPOS, ceil_div(MAXPOS, asplit), qk_norm}, err)
                                    : ctx->pipeline("attn_decode", {HDIM, MAXPOS, qk_norm}, err);
    const vk::Pipeline* ac = asplit ? ctx->pipeline("attn_split_combine", {HDIM}, err) : nullptr;
    if (!ap || (asplit && !ac)) return false;
    for (uint32_t s = 0; s < steps; ++s) {
        // No split() per token: the decode recording is one command buffer
        // per round (one submit instead of `steps`). The per-step split
        // dated from the original MOSS WIP; the segments were submitted
        // back-to-back on one queue and ordered by barriers anyway, and
        // measured neutral on the Pixel 10 Pro (P2-2). Trade-off: a round
        // is now one command buffer of `steps` tokens (~1 s on the Pixel at
        // K=16), longer than the per-token jobs split() keeps short for
        // driver watchdogs. If a driver's watchdog trips on that, lower
        // STARLING_FAST_KSTEP.
        for (uint32_t l = 0; l < NL; ++l) {
            const LlmLayer& Y = llm[l];
            Kernels::GemvArgs a;
            a.eps = leps;
            rc.label("dec_qkv");
            if (!K.gemv(rc, ar, Y.qkv, vk::Ref(xd), vk::Ref(qkvd), R(Y.attn_norm), vk::Ref(state),
                        sp.qkv_bias ? R(Y.qkv_b) : vk::Ref(), 0, a, err))
                return false;
            rc.barrier();
            rc.label("dec_attn");
            // attn_decode ignores the trailing S (push constants are one fixed block).
            struct { uint32_t H, KVH, cache_off; float eps, scale; uint32_t S; } pa{
                NH, NKV, l * NKV * MAXPOS * HDIM, leps, lscale, asplit};
            rc.dispatch(*ap, {vk::Ref(qkvd), vk::Ref(kcache), vk::Ref(vcache), vk::Ref(asplit ? apart : attd),
                              qk_norm ? R(Y.q_norm) : vk::Ref(K.dummy()), qk_norm ? R(Y.k_norm) : vk::Ref(K.dummy()),
                              R(rope_tab), vk::Ref(state)},
                        &pa, sizeof(pa), NH, std::max(asplit, 1u));
            if (asplit) {
                rc.barrier();
                rc.label("dec_attn_combine");
                rc.dispatch(*ac, {vk::Ref(apart), vk::Ref(attd), vk::Ref(state)}, &asplit, sizeof(asplit), NH);
            }
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

bool AudioLlmEngine::generate(const float* pcm, size_t n, uint32_t budget, std::vector<int32_t>& out_ids,
                              bool& eos, std::string& err) {
    Impl& I = *impl_;
    const char* tag = I.sp.tag;
    const uint32_t max_new = budget ? std::min(budget, I.sp.max_new_tokens) : I.sp.max_new_tokens;
    const bool timing = env_on("STARLING_FAST_TIMING");
    const auto t0 = std::chrono::steady_clock::now();
    MelView mel;
    if (!I.sp.mel(I.mel_loader, pcm, n, mel, err)) return false;
    const int64_t T = mel.T;
    if (T <= 0) { err = std::string("fast ") + tag + ": empty audio"; return false; }
    const int C = (int)((T + 99) / 100);
    const int tail = (int)(T % 100 ? T % 100 : 100);
    const int P = C == 1 ? tail : 100;
    const int A = (int)audio_tokens(T);
    const int S = (int)(I.sp.prefix.size() + A + I.sp.suffix.size());
    if (S + (int)max_new > (int)I.MAXPOS) {
        err = std::string("fast ") + tag + ": generation exceeds the KV cache (audio too long)";
        return false;
    }
    if (!I.ensure(C, A, S, err)) return false;

    // Mel chunks [C][128][P] (bf16 values, zero-padded tail).
    std::vector<float> chunks((size_t)C * 128 * P, 0.0f);
    for (int ci = 0; ci < C; ++ci) {
        const int len = ci == C - 1 ? tail : 100;
        for (int f = 0; f < 128; ++f)
            for (int t = 0; t < len; ++t)
                chunks[((size_t)ci * 128 + f) * P + t] =
                    ggml_bf16_to_fp32(mel.data[(size_t)f * mel.stride + ci * 100 + t]);
    }
    const double t_mel = ms_since(t0);

    Impl::Rec* rec = I.prefill_rec({C, tail}, [&](Impl::Rec& r) { return I.record_prefill(C, tail, r, err); });
    if (!rec) return false;

    // Inputs: mel chunks, prompt ids (fresh state: run()).
    if (!I.ctx->upload(I.mel_in, 0, chunks.data(), chunks.size() * 4, err)) return false;
    {
        std::vector<uint32_t> idv;
        for (int32_t v : I.sp.prefix) idv.push_back((uint32_t)v);
        for (int32_t v : I.sp.suffix) idv.push_back((uint32_t)v);
        if (!I.ctx->upload(I.ids, 0, idv.data(), idv.size() * 4, err)) return false;
    }
    int rounds = 0;
    double t_pre = 0;
    if (!I.run(*rec->rec, (uint32_t)S, max_new, t0, out_ids, eos, rounds, t_pre, err)) return false;
    if (timing)
        std::fprintf(stderr, "[fast-%s] audio=%.2fs mel=%.1fms enc+prefill=%.1fms decode=%.1fms (%zu tokens, %d rounds) total=%.1fms (T=%lld A=%d S=%d)\n",
                     tag, n / 16000.0, t_mel, t_pre - t_mel, ms_since(t0) - t_pre, out_ids.size(), rounds,
                     ms_since(t0), (long long)T, A, S);
    return true;
}

bool AudioLlmEngine::Impl::run(vk::Recording& prefill, uint32_t S, uint32_t max_new,
                               std::chrono::steady_clock::time_point t0, std::vector<int32_t>& out_ids,
                               bool& eos, int& rounds, double& t_pre, std::string& err) {
    Impl& I = *this;
    const char* tag = sp.tag;
    const uint32_t steps = (uint32_t)std::max(1, std::atoi(std::getenv("STARLING_FAST_KSTEP") ? std::getenv("STARLING_FAST_KSTEP") : "16"));
    if (!dec_rec || dec_steps != steps) {
        if (!record_decode(steps, err)) return false;
    }
    {
        uint32_t st[16] = {};
        st[0] = S - 1;      // dec_next advances to S: the first generated token's position
        st[3] = max_new;    // this call's token budget (dec_next.comp)
        if (!ctx->upload(state, 0, st, sizeof st, err)) return false;
    }
    if (!prefill.submit_and_wait(err)) return false;
    prefill.report_profile(sp.encoder ? "encoder+prefill" : "prefill");
    t_pre = ms_since(t0);

    std::vector<uint32_t> st(16 + (size_t)I.sp.max_new_tokens);
    rounds = 0;
    // The device stops at EOS or the budget; the host bound only guards
    // against a state buffer that never reports either.
    const int max_rounds = (int)(max_new / steps) + 2;
    // Degradation watchdog (#325): the wedge is preceded by rounds that
    // blow past the session's own median (the bimodal 101-141 ms/token
    // band, P2-7). From the second round on, each fence wait is bounded at
    // ~6x the running round median (floor 20 s) so a stalling driver is
    // caught with the same wedge-marker semantics long before the 120 s
    // default — instead of feeding a dying driver until it hangs. The first
    // round has no history and keeps the 120 s default: a fixed floor there
    // would misfire on slow drivers (llvmpipe at a large KSTEP).
    // STARLING_FAST_STALL_MULT tunes the multiplier (0 disables);
    // STARLING_FAST_STALL_BUDGET_MS forces an absolute budget for every
    // round (validation hook).
    // Parsed once per process: the env does not change, and a bad value
    // should warn once, not on every transcription.
    static const double stall_mult = [] {
        double m = 6.0;
        if (const char* e = std::getenv("STARLING_FAST_STALL_MULT")) {
            char* end = nullptr;
            const double v = std::strtod(e, &end);
            if (end != e && *end == '\0' && std::isfinite(v) && v >= 0 && v <= 1000) m = v;
            else std::fprintf(stderr, "[fast] ignoring STARLING_FAST_STALL_MULT='%s' (want 0..1000)\n", e);
        }
        return m;
    }();
    static const uint64_t forced_budget_ms = [] {
        uint64_t b = 0;
        if (const char* e = std::getenv("STARLING_FAST_STALL_BUDGET_MS")) {
            char* end = nullptr;
            const unsigned long long v = std::strtoull(e, &end, 10);
            if (end != e && *end == '\0' && v > 0) b = v;
            else std::fprintf(stderr, "[fast] ignoring STARLING_FAST_STALL_BUDGET_MS='%s' (want ms > 0)\n", e);
        }
        return b;
    }();
    constexpr double kStallFloorMs = 20000.0;    // the watchdog never fires sooner
    constexpr double kStallCeilMs = 3600000.0;   // keeps the double->uint64 cast defined
    std::vector<double> round_ms;
    for (;;) {
        if (!I.ctx->download(I.state, 0, st.data(), 16 * 4, err)) return false;
        if (st[2] != 0 || st[1] >= max_new) break;
        // Cooperative stop (#325): a SIGTERM'd process checks this between
        // decode rounds and returns the valid prefix instead of dying with a
        // live VkDevice (the driver's async reaping of ~1.6 GB of GPU state
        // is the leading wedge correlate; RESEARCH_LOG P2-7). Checked after
        // the state refresh so the prefix includes the round just finished.
        if (starling_ggml_stop_requested()) {
            std::fprintf(stderr, "[fast-%s] stop requested: returning a %u-token prefix\n", tag,
                         std::min<uint32_t>(st[1], max_new));
            break;
        }
        if (rounds >= max_rounds) { err = std::string("fast ") + tag + ": decode did not terminate"; return false; }
        uint64_t budget_ms = 0;   // 0 = the runtime's 120 s default
        if (forced_budget_ms) budget_ms = forced_budget_ms;
        else if (stall_mult > 0 && !round_ms.empty()) {
            std::vector<double> srt(round_ms);
            std::nth_element(srt.begin(), srt.begin() + srt.size() / 2, srt.end());
            const double med_ms = srt[srt.size() / 2];
            budget_ms = (uint64_t)std::min(kStallCeilMs, std::max(kStallFloorMs, stall_mult * med_ms));
        }
        const auto t_round = std::chrono::steady_clock::now();
        if (!I.dec_rec->submit_and_wait(err, budget_ms)) return false;
        round_ms.push_back(ms_since(t_round));
        if (rounds == 0) I.dec_rec->report_profile("decode (K steps)");
        ++rounds;
    }
    const uint32_t ngen = std::min<uint32_t>(st[1], max_new);
    if (!I.ctx->download(I.state, 16 * 4, st.data() + 16, ngen * 4, err)) return false;
    out_ids.assign(st.begin() + 16, st.begin() + 16 + ngen);
    eos = !out_ids.empty() && out_ids.back() == I.sp.eos;
    // Debug dump of the greedy token stream (#311 draft-acceptance studies):
    // STARLING_FAST_DUMP_TOKENS=path writes the generated ids (comma
    // separated); a path used repeatedly gets .1, .2 ... suffixes. The
    // suffix counter is per process: separate processes (one bench per
    // clip) writing the same path overwrite each other — give each its own.
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
            std::fprintf(stderr, "[fast-%s] STARLING_FAST_DUMP_TOKENS: cannot write %s\n", tag, path.c_str());
        }
    }
    return true;
}

bool AudioLlmEngine::Impl::prompt_rows(const std::vector<int32_t>& prompt_ids, const std::vector<uint8_t>& audio_mask,
                                      const float* audio, int64_t audio_rows, std::vector<float>& rows,
                                      std::vector<uint32_t>& idv, std::string& err) {
    // Audio slots take the encoder rows (bf16-rounded like the ggml scatter;
    // zero once the rows run out) times the embedding multiplier the input
    // table already carries; text slots get their ids, audio slots the skip
    // marker for embed_rows.
    const size_t S = prompt_ids.size();
    rows.assign(S * HD, 0.0f);
    idv.resize(S);
    int64_t next = 0;
    for (size_t i = 0; i < S; ++i) {
        if (!audio_mask[i]) {
            if (prompt_ids[i] < 0 || (uint32_t)prompt_ids[i] >= embed.N) {
                err = std::string("fast ") + sp.tag + ": prompt id out of range";
                return false;
            }
            idv[i] = (uint32_t)prompt_ids[i];
            continue;
        }
        idv[i] = 0xffffffffu;
        if (next < audio_rows)
            for (size_t d = 0; d < HD; ++d)
                rows[i * HD + d] = ggml_bf16_to_fp32(ggml_fp32_to_bf16(audio[(size_t)next * HD + d])) * sp.emb_mult;
        ++next;
    }
    return true;
}

bool AudioLlmEngine::generate(const std::vector<int32_t>& prompt_ids, const std::vector<uint8_t>& audio_mask,
                              const float* audio, int64_t audio_rows, uint32_t budget,
                              std::vector<int32_t>& out_ids, bool& eos, std::string& err) {
    Impl& I = *impl_;
    const char* tag = I.sp.tag;
    if (I.sp.encoder || I.sp.whisper) {
        err = std::string("fast ") + tag + ": this engine encodes the audio itself";
        return false;
    }
    const uint32_t max_new = budget ? std::min(budget, I.sp.max_new_tokens) : I.sp.max_new_tokens;
    const auto t0 = std::chrono::steady_clock::now();
    const int S = (int)prompt_ids.size();
    if (S == 0 || audio_mask.size() != prompt_ids.size() || audio_rows < 0 || (audio_rows > 0 && !audio)) {
        err = std::string("fast ") + tag + ": invalid prompt";
        return false;
    }
    if (S + (int)max_new > (int)I.MAXPOS) {
        err = std::string("fast ") + tag + ": generation exceeds the KV cache (audio too long)";
        return false;
    }
    if (!I.ensure(0, 0, S, err)) return false;
    std::vector<float> rows;
    std::vector<uint32_t> idv;
    if (!I.prompt_rows(prompt_ids, audio_mask, audio, audio_rows, rows, idv, err)) return false;
    Impl::Rec* rec = I.prefill_rec({0, S}, [&](Impl::Rec& r) { return I.record_prefill_rows(S, r, err); });
    if (!rec) return false;
    if (!I.ctx->upload(I.lx, 0, rows.data(), rows.size() * 4, err) ||
        !I.ctx->upload(I.ids, 0, idv.data(), idv.size() * 4, err))
        return false;
    const double t_up = ms_since(t0);
    int rounds = 0;
    double t_pre = 0;
    if (!I.run(*rec->rec, (uint32_t)S, max_new, t0, out_ids, eos, rounds, t_pre, err)) return false;
    if (env_on("STARLING_FAST_TIMING"))
        std::fprintf(stderr, "[fast-%s] rows=%.1fms prefill=%.1fms decode=%.1fms (%zu tokens, %d rounds) total=%.1fms (S=%d)\n",
                     tag, t_up, t_pre - t_up, ms_since(t0) - t_pre, out_ids.size(), rounds, ms_since(t0), S);
    return true;
}

bool AudioLlmEngine::encodes_audio() const { return impl_->sp.encoder || impl_->sp.whisper; }

bool AudioLlmEngine::generate_from_mel(const std::vector<int32_t>& prompt_ids, const std::vector<uint8_t>& audio_mask,
                                       const uint16_t* mel_bf16, int64_t mel_T, uint32_t budget,
                                       std::vector<int32_t>& out_ids, bool& eos, std::string& err) {
    Impl& I = *impl_;
    const char* tag = I.sp.tag;
    if (!I.sp.whisper) { err = std::string("fast ") + tag + ": no on-device Whisper encoder"; return false; }
    const uint32_t max_new = budget ? std::min(budget, I.sp.max_new_tokens) : I.sp.max_new_tokens;
    const auto t0 = std::chrono::steady_clock::now();
    const int S = (int)prompt_ids.size();
    const int64_t T = (mel_T + 1) / 2;
    if (S == 0 || audio_mask.size() != prompt_ids.size() || !mel_bf16 || mel_T <= 0) {
        err = std::string("fast ") + tag + ": invalid prompt";
        return false;
    }
    // The adapter needs one whole merge group; the RoPE tables bound T.
    if (T < (int64_t)I.sp.merge || T > (int64_t)I.rope_rows) {
        err = std::string("fast ") + tag + ": audio length outside the encoder's range";
        return false;
    }
    if (S + (int)max_new > (int)I.MAXPOS) {
        err = std::string("fast ") + tag + ": generation exceeds the KV cache (audio too long)";
        return false;
    }
    // The audio slots must be one run (prefix, slots, suffix).
    int slot0 = -1, n_slots = 0;
    for (int i = 0; i < S; ++i) {
        if (!audio_mask[i]) continue;
        if (slot0 < 0) slot0 = i;
        else if (!audio_mask[i - 1]) { err = std::string("fast ") + tag + ": audio slots are not contiguous"; return false; }
        ++n_slots;
    }
    if (slot0 < 0) { err = std::string("fast ") + tag + ": prompt has no audio slots"; return false; }
    if (!I.ensure((int)mel_T, (int)T, S, err)) return false;
    // Text slots for the device lookup; audio slots start zeroed (the slots
    // past the adapter's rows stay zero, like the ggml scatter).
    std::vector<float> rows;
    std::vector<uint32_t> idv;
    if (!I.prompt_rows(prompt_ids, audio_mask, nullptr, 0, rows, idv, err)) return false;
    // Mel as time-major f16 rows [mel_T + 2][n_mels], zero rows at both ends.
    const size_t NM = I.sp.n_mels;
    std::vector<float> mt(((size_t)mel_T + 2) * NM, 0.0f);
    for (size_t f = 0; f < NM; ++f)
        for (int64_t t = 0; t < mel_T; ++t) {
            ggml_bf16_t b;
            b.bits = mel_bf16[f * (size_t)mel_T + (size_t)t];
            mt[((size_t)t + 1) * NM + f] = ggml_bf16_to_fp32(b);
        }
    const std::vector<uint32_t> mel16 = pack_f16(mt.data(), mt.size());

    const int key_s = S;
    Impl::Rec* rec = I.prefill_rec({(int)mel_T, key_s}, [&](Impl::Rec& r) {
        return I.record_prefill_whisper((int)mel_T, S, slot0, n_slots, r, err);
    });
    if (!rec) return false;
    if (rec->slot0 != slot0) {   // same lengths, different prompt layout
        I.recs.erase({(int)mel_T, key_s});
        rec = I.prefill_rec({(int)mel_T, key_s}, [&](Impl::Rec& r) {
            return I.record_prefill_whisper((int)mel_T, S, slot0, n_slots, r, err);
        });
        if (!rec) return false;
    }
    if (!I.ctx->upload(I.mel_in, 0, mel16.data(), mel16.size() * 4, err) ||
        !I.ctx->upload(I.lx, 0, rows.data(), rows.size() * 4, err) ||
        !I.ctx->upload(I.ids, 0, idv.data(), idv.size() * 4, err))
        return false;
    const double t_up = ms_since(t0);
    int rounds = 0;
    double t_pre = 0;
    if (!I.run(*rec->rec, (uint32_t)S, max_new, t0, out_ids, eos, rounds, t_pre, err)) return false;
    if (env_on("STARLING_FAST_TIMING"))
        std::fprintf(stderr, "[fast-%s] mel_T=%lld upload=%.1fms enc+prefill=%.1fms decode=%.1fms (%zu tokens, %d rounds) total=%.1fms (T=%lld S=%d)\n",
                     tag, (long long)mel_T, t_up, t_pre - t_up, ms_since(t0) - t_pre, out_ids.size(), rounds,
                     ms_since(t0), (long long)T, S);
    return true;
}

} // namespace starling::fast
