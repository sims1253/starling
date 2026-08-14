// whisper_mel.hpp — shared Whisper-style log-mel frontend.
//
// Unifies the four frontends in moss/ark/higgs/hojo mel.cpp (same algorithm:
// reflect pad, Hann window from GGUF, pocketfft r2c, power=2, log10, global
// max-clamp(dynamic_range), normalize). The copies differ in a handful of
// REAL numerical/behavioral points, captured explicitly by MelPolicy:
//   * valid-frame rule after fullT = S/H + 1 STFT frames:
//       T_FULLT_MINUS_1  (moss/ark)  T = fullT - 1 = S/H
//       T_CEIL_S_OVER_H  (higgs)     T = ceil(S/H)
//       T_FLOOR_S_OVER_H (hojo)      T = S/H
//   * n_samples truncation: none (moss), after the S<2 check with a >0 guard
//     (ark/higgs), or unconditionally before the checks (hojo).
//   * global-max scope: all computed M*fullT frames with a flat split
//     (moss/ark) or kept frames only with a per-mel split (higgs/hojo).
//   * clamp + normalize arithmetic: f32 throughout (moss/ark/higgs) or with
//     double constants/intermediates (hojo).
//   * output: f32 always; bf16 alongside (moss/ark/higgs) or not (hojo).
//
// Everything else (loops 1/2, accumulation orders, chunk-max reduction) is
// byte-identical across the four copies and lives once here.
#pragma once

#include "runtime/model_loader.hpp"
#include "ggml.h"
#include <cstdint>
#include <string>
#include <vector>

namespace starling::ggml::lib {

struct MelPolicy {
    // Frontend constants (from each model's FrontendConfig). Stored as double;
    // the f32-discipline models round back to float at use, which is exact.
    uint32_t n_fft = 400, hop_length = 160, n_mels = 128;
    double mel_floor = 1e-10, dynamic_range = 8.0;
    double normalization_offset = 4.0, normalization_divisor = 4.0;

    enum TRule { T_FULLT_MINUS_1, T_CEIL_S_OVER_H, T_FLOOR_S_OVER_H };
    TRule t_rule = T_FULLT_MINUS_1;

    // n_samples truncation of the raw waveform before the STFT.
    bool cap_n_samples = false;       // cap at all
    bool cap_before_checks = false;   // hojo: unconditional, before the S<2 check
    uint32_t n_samples = 0;

    enum MaxScope { MAX_ALL_FRAMES, MAX_KEPT_FRAMES };
    MaxScope max_scope = MAX_ALL_FRAMES;

    bool norm_in_double = false;  // clamp+normalize with double intermediates
    bool emit_bf16 = true;        // bf16 copy alongside f32
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

} // namespace starling::ggml::lib
