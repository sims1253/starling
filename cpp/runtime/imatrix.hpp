// imatrix.hpp — activation importance collection for calibrated quantization.
//
// The Starling analogue of llama.cpp's imatrix: while the engine transcribes
// calibration audio, every MUL_MAT whose src[0] is a named weight tensor gets
// its src[1] (F32 activations) reduced to a per-input-channel sum of squares,
// keyed by the weight's GGUF name. The resulting map is exactly the
// `--imatrix` input starling-quantize weights quantization error by, so
// block scales land where activations actually live instead of assuming a
// uniform distribution.
//
// Enable by setting STARLING_IMATRIX=<output-path> in the environment before
// the engine loads: backend.cpp then routes every graph through
// ggml_backend_sched with an eval callback, which observes each node's
// activations. NOTE: in the vendored ggml the callback only gates
// observation, not backend placement — host-side activation reads are sound
// because the collection drivers pin STARLING_GGML_DEVICE=cpu. Collection
// mode trades speed for visibility and is not meant for serving.
//
// The map is flushed at process exit (std::atexit) and after an explicit
// starling_ggml_imatrix_flush() C-API call; writing touches no ggml state so
// it is safe in any teardown order.

#pragma once

#include "imatrix_file.hpp"

#include <mutex>

struct ggml_tensor;

namespace starling::ggml {

class ImatrixCollector {
public:
    static ImatrixCollector& instance();

    // True when STARLING_IMATRIX=<path> is set.
    static bool enabled();

    // The ggml_backend_sched_eval_callback body. ask=true: report interest in
    // MUL_MAT nodes with a named weight src[0] and F32 activations src[1].
    // ask=false: the node has run; accumulate src[1]^2 into the entry keyed
    // by the weight's name.
    bool observe(::ggml_tensor* node, bool ask);

    // Write the accumulated map to the STARLING_IMATRIX path. Idempotent.
    void flush();

private:
    ImatrixCollector();
    ImatrixCollector(const ImatrixCollector&) = delete;
    ImatrixCollector& operator=(const ImatrixCollector&) = delete;

    struct Entry {  // mirrored into ImatrixMap on flush
        std::vector<float> sums;
        uint64_t n_obs = 0;
    };

    std::mutex mu_;                 // guards entries_ (eval callback context)
    std::unordered_map<std::string, Entry> entries_;
    std::string path_;
    bool flushed_ = false;
};

} // namespace starling::ggml
