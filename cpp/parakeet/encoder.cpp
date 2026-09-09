// encoder.cpp — parakeet-tdt Conformer encoder graph build and execution.
#include "encoder.hpp"

#include "runtime/graph.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "runtime/backend.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace starling::ggml::parakeet {

Encoder::Encoder(const ParakeetModel& model)
    : model_(model), sub_(model.loader, model.config), config_(model.config) {}

Encoder::~Encoder() = default;

size_t Encoder::cache_size() const {
    const auto* cache = model_.loader.find_cache<ReplayCache>();
    return cache ? cache->by_T.size() : 0;
}

ggml_tensor* Encoder::build_graph(ggml_context* ctx,
                                  const std::vector<float>& mel,
                                  int n_mels, int T,
                                  GraphInputPool& pool,
                                  int& Tp, int& valid_len,
                                  const float* ph_scratch) const {
    const ModelLoader& ml = model_.loader;
    ggml_tensor* x = sub_.build_graph(ctx, mel, n_mels, T, pool, Tp, valid_len);

    const int pos_len = 2 * Tp - 1;
    const bool have_ph = ph_scratch != nullptr;
    // With cached per-T' projections the raw positional table is dead: no node
    // consumes it, so do not register it as a graph input (an unallocated
    // input would be skipped by the allocator and fail re-upload).
    ggml_tensor* pe = nullptr;
    std::vector<float> pe_vec;
    if (!have_ph) {
        rel_pos_encoding(Tp, (int)config_.d_model, pe_vec);
        float* pe_host = pool.alloc_f32(pe_vec.size());
        std::memcpy(pe_host, pe_vec.data(), pe_vec.size() * sizeof(float));
        int64_t pe_ne[2] = {(int)config_.d_model, pos_len};
        pe = graph_input_tensor(ctx, GGML_TYPE_F32, 2, pe_ne,
            pe_host, pe_vec.size() * sizeof(float));
    }

    const int dk = (int)config_.d_model / (int)config_.n_heads;
    for (int i = 0; i < (int)config_.n_layers; ++i) {
        ConformerLayer layer(ml, config_, i);
        ggml_tensor* ph_i = nullptr;
        if (have_ph) {
            // Persistent input: uploaded once right after this build, then
            // kept (contents are the per-layer cached projections).
            int64_t ph_ne[3] = {dk, pos_len, (int)config_.n_heads};
            const size_t ph_n = (size_t)dk * pos_len * config_.n_heads;
            ph_i = graph_input_tensor(ctx, GGML_TYPE_F32, 3, ph_ne,
                ph_scratch, ph_n * sizeof(float));
            mark_graph_input_persistent(ph_i);
        }
        x = layer.build_graph(ctx, x, Tp, pe, pos_len, valid_len, pool, ph_i);
    }

    ggml_tensor* jw = clone_weight(ctx, ml, "joint.enc.weight");
    ggml_tensor* jb = clone_weight(ctx, ml, "joint.enc.bias");
    return ggml_add(ctx, ggml_mul_mat(ctx, jw, x), jb);
}

// [starling pos-cache] Compute one layer's positional projection for this
// T' once: pe -> linear_pos -> head split -> [dk, P, H]. One small one-shot
// device compute per layer; the caller uploads the result into the encoder
// replay graph's persistent input and frees it immediately (the transient
// host footprint is ONE layer's ph, not all 24).
bool Encoder::compute_pos_layer(int Tp, int layer,
                                std::vector<float>& ph_out) const {
    const ModelLoader& ml = model_.loader;
    const int D = (int)config_.d_model;
    const int H = (int)config_.n_heads;
    const int dk = D / H;
    const int pos_len = 2 * Tp - 1;
    GraphInputPool pool;
    Backend& backend = global_backend();
    bool ok = backend.compute([&](ggml_context* ctx) -> ggml_tensor* {
        std::vector<float> pe_vec;
        rel_pos_encoding(Tp, D, pe_vec);
        float* pe_host = pool.alloc_f32(pe_vec.size());
        std::memcpy(pe_host, pe_vec.data(), pe_vec.size() * sizeof(float));
        int64_t pe_ne[2] = {D, pos_len};
        ggml_tensor* pe = graph_input_tensor(ctx, GGML_TYPE_F32, 2, pe_ne,
            pe_host, pe_vec.size() * sizeof(float));
        ggml_tensor* W = clone_weight(ctx, ml,
            ("encoder.layers." + std::to_string(layer) + ".self_attn.linear_pos.weight").c_str());
        ggml_tensor* p = ggml_mul_mat(ctx, W, pe);            // [D, P]
        p = ggml_reshape_3d(ctx, p, dk, H, pos_len);          // [dk, H, P]
        p = ggml_cont(ctx, ggml_permute(ctx, p, 0, 2, 1, 3)); // [dk, P, H]
        return p;
    }, ph_out);
    return ok;
}

bool Encoder::encode(const std::vector<float>& mel, int n_mels, int T,
                     std::vector<float>& out, int& out_Tp) const {
    Backend& backend = global_backend();
    if (!backend.is_gpu()) {
        GraphInputPool pool;
        int Tp = 0, valid_len = 0;
        bool ok = run_graph([&](ggml_context* ctx) {
            return build_graph(ctx, mel, n_mels, T, pool, Tp, valid_len);
        }, out);
        if (ok) out_Tp = Tp;
        return ok;
    }

    // get_or_init places the entry (stable address) first, then builds: the
    // ReplayGraph build lambda captures the stable pool. On a miss at capacity
    // the LRU mel length is evicted (its captured graph freed) before this one
    // is inserted. Bounded by STARLING_REPLAY_CACHE_SIZE (default 16) via the
    // shared LruCache (runtime/lru_cache.hpp) — without it each distinct T would
    // pin its own captured graph + private gallocr until the model unloads (the
    // Wave H OOM bug).
    auto& replay_cache_ = model_.loader.cache<ReplayCache>();
    if (!replay_cache_) replay_cache_ = std::make_unique<ReplayCache>(replay_cache_size());
    ReplayEntry& e = *replay_cache_->by_T.get_or_init(T,
        [this, &backend, &mel, n_mels, T](ReplayEntry& entry) {
            entry.T = T;
            // Positional projections are constant per T': cache them as
            // persistent inputs (skips ~25ms of per-pass GEMM on medium).
            // The scratch (stable address) backs every registered ph host
            // pointer; values are computed + uploaded layer by layer below,
            // syncing so a reused buffer is never read late.
            const int Tp0 = sub_.subsample_len(T);
            std::vector<float> probe;
            const bool ph_ok = compute_pos_layer(Tp0, 0, probe);
            if (ph_ok) entry.ph_scratch = std::move(probe);
            entry.graph = std::make_unique<ReplayGraph>(backend,
                [this, &entry, &mel, n_mels, T, ph_ok](ggml_context* ctx) {
                    return build_graph(ctx, mel, n_mels, T, entry.pool,
                                       entry.Tp, entry.valid_len,
                                       ph_ok ? entry.ph_scratch.data() : nullptr);
                });
            // Static persistent inputs (their own host data: e.g. the
            // transposed depthwise-conv kernels) upload immediately; the ph
            // inputs share one scratch and need the layer-by-layer dance.
            for (size_t i = 0; i < entry.graph->n_inputs(); ++i) {
                if (!entry.graph->input_persistent(i)) continue;
                if (entry.graph->input_host(i) == entry.ph_scratch.data()) continue;
                entry.graph->set_input(i, entry.graph->input_host(i),
                                       entry.graph->input_nbytes(i));
            }
            if (!ph_ok) return;
            std::vector<size_t> ph_idx;
            for (size_t i = 0; i < entry.graph->n_inputs(); ++i)
                if (entry.graph->input_persistent(i) &&
                    entry.graph->input_host(i) == entry.ph_scratch.data())
                    ph_idx.push_back(i);
            for (size_t k = 0; k < ph_idx.size(); ++k) {
                if (k > 0 && !compute_pos_layer(Tp0, (int)k, entry.ph_scratch)) {
                    // Poison the entry: later ph persistent inputs were
                    // registered but will never be uploaded — replaying this
                    // graph would read never-written garbage. Failing loudly
                    // (until the entry is evicted or the model unloads)
                    // beats silently wrong transcripts.
                    entry.graph.reset();
                    return;
                }
                entry.graph->set_input(ph_idx[k], entry.ph_scratch.data(),
                                       entry.ph_scratch.size() * sizeof(float));
                ggml_backend_synchronize(backend.handle());
            }
        });
    if (!e.graph) {
        // Poisoned by the initializer (pos-projection compute failed after
        // the replay graph was built): refuse to replay on never-uploaded
        // persistent inputs instead of emitting garbage.
        std::fprintf(stderr, "[encoder] T=%d replay entry poisoned: positional "
                             "projection compute failed; transcription aborted\n", T);
        return false;
    }

    // F1 instrumentation: split the encoder phase into host-mel-transpose,
    // H2D enqueue (set_input is async), and graph_compute+readback (the latter
    // holds the GPU sync, so its wall ~ per-replay GPU time). Gated by the same
    // flag as the ReplayGraph split so one env var lights up the whole chain.
    const bool t_on = std::getenv("STARLING_REPLAY_TIMING") != nullptr;
    const int64_t t_h0 = t_on ? ggml_time_us() : 0;
    // Subsampling registers the transposed mel first. Refresh it in the stable
    // pool buffer; all remaining inputs are shape constants/masks and are
    // re-uploaded as well because ReplayGraph does not promise persistence.
    float* mel_host = static_cast<float*>(const_cast<void*>(e.graph->input_host(0)));
    for (int t = 0; t < T; ++t)
        for (int f = 0; f < n_mels; ++f)
            mel_host[(size_t)t * n_mels + f] = mel[(size_t)f * T + t];
    const int64_t t_h1 = t_on ? ggml_time_us() : 0;
    for (size_t i = 0; i < e.graph->n_inputs(); ++i) {
        if (e.graph->input_persistent(i)) continue;  // ph buffers: upload once
        e.graph->set_input(i, e.graph->input_host(i), e.graph->input_nbytes(i));
    }
    const int64_t t_h2 = t_on ? ggml_time_us() : 0;
    bool ok = e.graph->compute(out);
    if (t_on) {
        const int64_t t_h3 = ggml_time_us();
        std::fprintf(stderr,
            "[enc-timing] T=%d Tp=%d transpose=%lldus h2d_enqueue=%lldus compute=%lldus n_inputs=%zu\n",
            T, e.Tp,
            (long long)(t_h1 - t_h0), (long long)(t_h2 - t_h1),
            (long long)(t_h3 - t_h2), e.graph->n_inputs());
    }
    if (ok) out_Tp = e.Tp;
    return ok;
}

} // namespace starling::ggml::parakeet
