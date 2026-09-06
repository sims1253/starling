// whisper_mel.hpp — Whisper-style log-mel frontend: reflect pad, Hann window
// from GGUF, pocketfft r2c, power=2, log10, global max-clamp(dynamic_range),
// normalize. The engines differ in a handful of REAL numerical/behavioral
// points, captured explicitly by MelPolicy:
//   * valid-frame rule after fullT = S/H + 1 STFT frames:
//       T_FULLT_MINUS_1  (moss/ark)  T = fullT - 1 = S/H
//       T_CEIL_S_OVER_H  (higgs)     T = ceil(S/H)
//       T_FLOOR_S_OVER_H (hojo)      T = S/H
//       T_FULLT          (granite)   T = fullT (torchaudio center=True keeps
//                                        every frame; the odd-frame drop and
//                                        the 80->160 pair-stack are
//                                        engine-side, in cpp/granite/mel.cpp)
//   * n_samples truncation: none (moss), after the S<2 check with a >0 guard
//     (ark/higgs), or unconditionally before the checks (hojo).
//   * global-max scope: all computed M*fullT frames with a flat split
//     (moss/ark) or kept frames only with a per-mel split (higgs/hojo).
//   * clamp + normalize arithmetic: f32 throughout (moss/ark/higgs) or with
//     double constants/intermediates (hojo).
//   * output: f32 always; bf16 alongside (moss/ark/higgs) or not (hojo).
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

    enum TRule { T_FULLT_MINUS_1, T_CEIL_S_OVER_H, T_FLOOR_S_OVER_H, T_FULLT };
    TRule t_rule = T_FULLT_MINUS_1;

    // n_samples truncation of the raw waveform before the STFT.
    bool cap_n_samples = false;       // cap at all
    bool cap_before_checks = false;   // hojo: unconditional, before the S<2 check
    uint32_t n_samples = 0;

    enum MaxScope { MAX_ALL_FRAMES, MAX_KEPT_FRAMES };
    MaxScope max_scope = MAX_ALL_FRAMES;

    bool norm_in_double = false;  // clamp+normalize with double intermediates
    bool emit_bf16 = true;        // bf16 copy alongside f32
    // Fixed global log-mel max (streaming-safe normalization): when finite,
    // replaces the computed per-utterance max as the clamp/normalize reference.
    // NaN (default) keeps the legacy per-utterance max and is byte-identical
    // for every existing engine.
    double fixed_log_mel_max = std::numeric_limits<double>::quiet_NaN();
    const char* dump_env = nullptr;   // e.g. "STARLING_HIGGS_MEL_DUMP"
    const char* label = "GGUF";       // error-message prefix ("<label> mel ...")
};

struct MelOutput {
    std::vector<float> f32;         // [n_mels * n_frames], always filled
    std::vector<ggml_bf16_t> bf16;  // filled only when policy.emit_bf16
    size_t n_mels = 0, n_frames = 0;
};

bool compute_log_mel(const MelPolicy& p, const ModelLoader& ml, const float* pcm,
                     size_t S, MelOutput& out, std::string& err);

// ---- shared per-model adapter ----------------------------------------------
// The moss/ark/higgs/hojo mel wrappers are structurally identical: they map
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
    bool cap_before_checks;
    MelPolicy::MaxScope max_scope;
    bool norm_in_double;
    bool emit_bf16;
    const char* dump_env;  // e.g. "STARLING_HIGGS_MEL_DUMP"
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
    p.cap_before_checks = e.cap_before_checks;
    p.max_scope = e.max_scope;
    p.norm_in_double = e.norm_in_double;
    p.emit_bf16 = e.emit_bf16;
    p.dump_env = e.dump_env;
    p.label = e.label;
    return p;
}

} // namespace starling::ggml::lib
