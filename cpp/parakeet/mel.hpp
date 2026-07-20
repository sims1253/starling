// mel.hpp — parakeet-tdt mel feature extraction.
//
// Two interchangeable front ends, both producing the SAME feat-major
// [n_mels, T] float32 post-CMVN tensor that the encoder consumes:
//
//   - MelFrontend (CPU): radix-2 FFT, the byte-exact reference. Used on the CPU
//     backend (and as the validation gate against golden/parakeet_tdt_*_mel.pt).
//   - GpuMel (GPU): STFT expressed as a DFT-matmul in a ggml graph, cached in a
//     per-T ReplayGraph so ggml-cuda captures + replays it. Host preemph/window
//     framing stay on host (double precision) to preserve byte-exactness; only
//     the per-frame FFT is replaced by the matmul.
//
// The mel filterbank + window are baked into the GGUF as
// preprocessor.featurizer.{fb,window}; we read them at construction.
//
// Output layout: feat-major `feats[m*T + t]` (n_mels rows × T frames), float32,
// after the full 8-step NeMo pipeline (preemph -> STFT -> power -> filterbank
// -> log -> CMVN). The encoder's subsampling layer transposes this to
// time-major [T, F] for the conv input.

#pragma once

#include "config.hpp"
#include "runtime/backend.hpp"
#include "runtime/graph_builder.hpp"
#include "runtime/model_loader.hpp"

#include <memory>
#include <vector>

namespace starling::ggml::parakeet {

// Shared mel constants read from the GGUF once (the MelKernel equivalent).
struct MelConstants {
    uint32_t n_mels = 0, n_fft = 0, n_bins = 0;  // n_bins = n_fft/2 + 1
    uint32_t win_length = 0, hop_length = 0;
    float    preemph = 0.0f, mag_power = 0.0f, log_zero_guard = 0.0f;
    std::string normalize;
    // window center-padded from win_length to n_fft (offset (n_fft-win_length)/2)
    std::vector<float> window;        // [n_fft]
    std::vector<float> filterbank;    // [n_mels * n_bins], fb[m*n_bins + b]

    void read_from(const ModelLoader& ml, const Config& cfg);
};

// CPU FFT mel (byte-exact reference). Computes feats[m*T + t], feat-major.
class MelFrontend {
public:
    MelFrontend(const MelConstants& c) : c_(c) {}
    // PCM is mono float32 [-1,1] at the model's sample rate (16k). Returns
    // feat-major [n_mels, T] in `feats`; T = 1 + S/hop.
    void compute(const float* pcm, size_t n_samples, std::vector<float>& feats,
                 int& out_T) const;
private:
    const MelConstants& c_;
};

// GPU DFT-as-matmul mel, cached per-T in a ReplayGraph. Holds the persistent
// graph across calls (caller keeps one instance alive, like Model does).
class GpuMel {
public:
    GpuMel(Backend& backend, const MelConstants& c);
    ~GpuMel();
    GpuMel(const GpuMel&) = delete;
    GpuMel& operator=(const GpuMel&) = delete;

    void compute(const float* pcm, size_t n_samples, std::vector<float>& feats,
                 int& out_T);
private:
    Backend& backend_;
    const MelConstants& c_;
    // DFT basis (built once): cos/sin per bin, [n_bins * n_fft], row-major.
    std::vector<float> dft_cos_, dft_sin_;
    // Per-T replay cache.
    struct MelReplay {
        int T = 0;
        std::unique_ptr<ReplayGraph> rg;
        GraphInputPool pool;
        float* xw_host = nullptr;       // [n_fft * T] windowed frames
        size_t xw_nbytes = 0;
    };
    std::unique_ptr<MelReplay> replay_;

    void build_basis();
    void build_or_reuse_replay(int T);
};

} // namespace starling::ggml::parakeet
