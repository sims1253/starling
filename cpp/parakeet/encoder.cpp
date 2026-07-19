// encoder.cpp — parakeet-tdt Conformer encoder graph build and execution.
#include "encoder.hpp"

#include "runtime/graph.hpp"
#include "ggml.h"

#include <cstring>

namespace starling::ggml::parakeet {

Encoder::Encoder(const ParakeetModel& model)
    : model_(model), sub_(model.loader, model.config), config_(model.config),
      replay_cache_(std::make_shared<ReplayCache>()) {
    // The clearer holds no model pointer. If the model has already been freed,
    // the weak_ptr is expired; otherwise graphs are released before Backend.
    std::weak_ptr<ReplayCache> weak = replay_cache_;
    register_decode_cache_clearer([weak]() {
        if (auto cache = weak.lock()) cache->clear();
    });
}

Encoder::~Encoder() = default;

ggml_tensor* Encoder::build_graph(ggml_context* ctx,
                                  const std::vector<float>& mel,
                                  int n_mels, int T,
                                  GraphInputPool& pool,
                                  int& Tp, int& valid_len) const {
    const ModelLoader& ml = model_.loader;
    ggml_tensor* x = sub_.build_graph(ctx, mel, n_mels, T, pool, Tp, valid_len);

    const int pos_len = 2 * Tp - 1;
    std::vector<float> pe_vec;
    rel_pos_encoding(Tp, (int)config_.d_model, pe_vec);
    float* pe_host = pool.alloc_f32(pe_vec.size());
    std::memcpy(pe_host, pe_vec.data(), pe_vec.size() * sizeof(float));
    int64_t pe_ne[2] = {(int)config_.d_model, pos_len};
    ggml_tensor* pe = graph_input_tensor(ctx, GGML_TYPE_F32, 2, pe_ne,
        pe_host, pe_vec.size() * sizeof(float));

    for (int i = 0; i < (int)config_.n_layers; ++i) {
        ConformerLayer layer(ml, config_, i);
        x = layer.build_graph(ctx, x, Tp, pe, pos_len, valid_len, pool);
    }

    ggml_tensor* jw = clone_weight(ctx, ml, "joint.enc.weight");
    ggml_tensor* jb = clone_weight(ctx, ml, "joint.enc.bias");
    return ggml_add(ctx, ggml_mul_mat(ctx, jw, x), jb);
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

    auto it = replay_cache_->by_T.find(T);
    if (it == replay_cache_->by_T.end()) {
        auto entry = std::make_unique<ReplayEntry>();
        entry->T = T;
        ReplayEntry* e = entry.get();
        e->graph = std::make_unique<ReplayGraph>(backend,
            [this, e, &mel, n_mels, T](ggml_context* ctx) {
                return build_graph(ctx, mel, n_mels, T, e->pool,
                                   e->Tp, e->valid_len);
            });
        it = replay_cache_->by_T.emplace(T, std::move(entry)).first;
    }

    ReplayEntry& e = *it->second;
    // Subsampling registers the transposed mel first. Refresh it in the stable
    // pool buffer; all remaining inputs are shape constants/masks and are
    // re-uploaded as well because ReplayGraph does not promise persistence.
    float* mel_host = static_cast<float*>(const_cast<void*>(e.graph->input_host(0)));
    for (int t = 0; t < T; ++t)
        for (int f = 0; f < n_mels; ++f)
            mel_host[(size_t)t * n_mels + f] = mel[(size_t)f * T + t];
    for (size_t i = 0; i < e.graph->n_inputs(); ++i)
        e.graph->set_input(i, e.graph->input_host(i), e.graph->input_nbytes(i));

    bool ok = e.graph->compute(out);
    if (ok) out_Tp = e.Tp;
    return ok;
}

} // namespace starling::ggml::parakeet
