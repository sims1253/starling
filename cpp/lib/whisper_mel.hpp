// whisper_mel.hpp — Whisper-style log-mel frontend: reflect pad, Hann window
// from GGUF, pocketfft r2c, power=2, log10, global max-clamp(dynamic_range),
// normalize. The global max is taken over all computed M*fullT frames (flat
// split), clamp + normalize run in f32, and the output is f32 plus a bf16
// copy. The engines differ only in the points MelPolicy captures:
//   * valid-frame rule after fullT = S/H + 1 STFT frames:
//       T_FULLT_MINUS_1  (moss/ark/qwen3/voxtral)  T = fullT - 1 = S/H
//       T_FULLT          (granite)   T = fullT (torchaudio center=True keeps
//                                        every frame; the odd-frame drop and
//                                        the 80->160 pair-stack are
//                                        engine-side, in cpp/granite/mel.cpp)
//   * n_samples truncation: none (moss), or after the S<2 check with a >0
//     guard (ark).
//
#pragma once

#include "runtime/model_loader.hpp"
#include "ggml.h"
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace starling::ggml::lib {

struct MelPolicy {
    // Frontend constants (from each model's FrontendConfig). Stored as double;
    // the f32-discipline models round back to float at use, which is exact.
    uint32_t n_fft = 400, hop_length = 160, n_mels = 128;
    double mel_floor = 1e-10, dynamic_range = 8.0;
    double normalization_offset = 4.0, normalization_divisor = 4.0;

    enum TRule { T_FULLT_MINUS_1, T_FULLT };
    TRule t_rule = T_FULLT_MINUS_1;

    // n_samples truncation of the raw waveform (after the S<2 check).
    bool cap_n_samples = false;
    uint32_t n_samples = 0;

    // Fixed global log-mel max (streaming-safe normalization): when finite,
    // replaces the computed per-utterance max as the clamp/normalize reference.
    // NaN (default) keeps the legacy per-utterance max and is byte-identical
    // for every existing engine.
    double fixed_log_mel_max = std::numeric_limits<double>::quiet_NaN();
    const char* dump_env = nullptr;   // e.g. "STARLING_ARK_MEL_DUMP"
    const char* label = "GGUF";       // error-message prefix ("<label> mel ...")
};

struct MelOutput {
    std::vector<float> f32;         // [n_mels * n_frames], always filled
    std::vector<ggml_bf16_t> bf16;  // bf16 copy of f32
    size_t n_mels = 0, n_frames = 0;
};

bool compute_log_mel(const MelPolicy& p, const ModelLoader& ml, const float* pcm,
                     size_t S, MelOutput& out, std::string& err);

// ---- shared per-model adapter ----------------------------------------------
// The Whisper-style mel wrappers are structurally identical: they map
// their model's FrontendConfig (identical field names across engines) into a
// MelPolicy and differ only in the fixed policy bits below. EngineMelPolicy
// captures those bits; make_mel_policy is the one shared mapping. A new
// Whisper-style frontend = one EngineMelPolicy constant + a compute_log_mel
// call. (parakeet/mel.* is a different NeMo frontend and does not use this.)

// Fixed per-engine policy bits — everything not derivable from a
// FrontendConfig. See the engine rows in each cpp/<model>/mel.cpp.
struct EngineMelPolicy {
    MelPolicy::TRule t_rule;
    bool cap_n_samples;
    const char* dump_env;  // e.g. "STARLING_ARK_MEL_DUMP"
    const char* label;     // error-message prefix ("<label> mel ...")
};

// Build the full MelPolicy from any model's FrontendConfig (the field names
// n_fft/hop_length/n_mels/mel_floor/dynamic_range/normalization_offset/
// normalization_divisor/n_samples are shared) plus the engine's fixed bits.
// n_samples is only consulted when cap_n_samples is set.
template <typename FrontendConfig>
MelPolicy make_mel_policy(const FrontendConfig& c, const EngineMelPolicy& e) {
    MelPolicy p;
    p.n_fft = c.n_fft;
    p.hop_length = c.hop_length;
    p.n_mels = c.n_mels;
    p.mel_floor = c.mel_floor;
    p.dynamic_range = c.dynamic_range;
    p.normalization_offset = c.normalization_offset;
    p.normalization_divisor = c.normalization_divisor;
    p.n_samples = c.n_samples;
    p.t_rule = e.t_rule;
    p.cap_n_samples = e.cap_n_samples;
    p.dump_env = e.dump_env;
    p.label = e.label;
    return p;
}

} // namespace starling::ggml::lib
