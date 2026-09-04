// imatrix.cpp — ImatrixCollector implementation. See imatrix.hpp for the
// design and the file format (imatrix_file.hpp).

#include "imatrix.hpp"

#include "ggml.h"

#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace starling::ggml {

ImatrixCollector& ImatrixCollector::instance() {
    static ImatrixCollector inst;
    return inst;
}

bool ImatrixCollector::enabled() {
    return std::getenv("STARLING_IMATRIX") != nullptr;
}

ImatrixCollector::ImatrixCollector() {
    const char* p = std::getenv("STARLING_IMATRIX");
    path_ = p ? p : "";
    if (!path_.empty()) {
        // Flush at normal process exit. The collector owns plain host memory
        // only (no ggml handles), so teardown order relative to the Backend
        // does not matter.
        std::atexit([] { ImatrixCollector::instance().flush(); });
        std::fprintf(stderr,
                     "[imatrix] collecting activation importance -> %s\n",
                     path_.c_str());
    }
}

bool ImatrixCollector::observe(ggml_tensor* node, bool ask) {
    if (node == nullptr || node->op != GGML_OP_MUL_MAT) return false;
    ggml_tensor* w = node->src[0];
    ggml_tensor* x = node->src[1];
    if (w == nullptr || x == nullptr) return false;
    // Only weight tensors cloned from the loader carry names; activations and
    // graph inputs are anonymous. This automatically excludes attention
    // score matmuls (both operands are activations).
    if (w->name[0] == '\0') return false;
    if (x->type != GGML_TYPE_F32) return false;

    if (ask) return true;  // observe this node — NOTE: the callback does NOT
                           // control placement (sched decides in split_graph);
                           // host-side reads are sound because collection
                           // pins STARLING_GGML_DEVICE=cpu

    // x is [K, N] row-major with K == w->ne[0] (the weight's input-channel
    // count). Accumulate x[k, j]^2 over all columns j.
    const int64_t K = x->ne[0];
    const int64_t N = ggml_nelements(x) / (K > 0 ? K : 1);
    if (K <= 0 || N <= 0 || x->data == nullptr) return true;

    const float* xp = (const float*)x->data;
    std::lock_guard<std::mutex> lock(mu_);
    Entry& e = entries_[w->name];
    if (e.sums.empty()) {
        e.sums.assign((size_t)K, 0.0f);
    } else if (e.sums.size() != (size_t)K) {
        // Same-named weight with a different row width should never happen;
        // keep the first shape rather than corrupt the entry.
        return true;
    }
    for (int64_t j = 0; j < N; ++j) {
        const float* col = xp + j * K;
        for (int64_t k = 0; k < K; ++k) e.sums[(size_t)k] += col[k] * col[k];
    }
    e.n_obs += 1;
    return true;
}

void ImatrixCollector::flush() {
    std::lock_guard<std::mutex> lock(mu_);
    if (flushed_ || path_.empty()) return;
    flushed_ = true;

    ImatrixMap map;
    uint64_t total_obs = 0;
    map.reserve(entries_.size());
    for (auto& kv : entries_) {
        ImatrixEntry e;
        e.values = std::move(kv.second.sums);
        e.n_obs = kv.second.n_obs;
        total_obs += e.n_obs;
        map.emplace(kv.first, std::move(e));
    }
    if (!imatrix_write(path_, map)) {
        std::fprintf(stderr, "[imatrix] ERROR: failed to write %s\n", path_.c_str());
        return;
    }
    std::fprintf(stderr,
                 "[imatrix] wrote %s: %zu tensors, %llu mul_mat observations\n",
                 path_.c_str(), map.size(), (unsigned long long)total_obs);
}

} // namespace starling::ggml
