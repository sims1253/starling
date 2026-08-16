// mel.cpp — Whisper log-mel frontend for Hojo-ASR-V1.
//
// Thin adapter over the shared frontend (lib/whisper_mel.hpp); hojo's policy:
// S/H valid frames (padding=False, last-frame truncation load-bearing),
// unconditional n_samples (=40 s) truncation BEFORE the input checks, global
// max over the KEPT frames only, double-constant clamp/normalize arithmetic,
// f32-only output.
#include "mel.hpp"
#include "lib/whisper_mel.hpp"

namespace starling::ggml::hojo {
namespace {
const lib::EngineMelPolicy kMelPolicy = {
    lib::MelPolicy::T_FLOOR_S_OVER_H,
    /*cap_n_samples=*/true,
    /*cap_before_checks=*/true,
    lib::MelPolicy::MAX_KEPT_FRAMES,
    /*norm_in_double=*/true,
    /*emit_bf16=*/false,
    "STARLING_HOJO_MEL_DUMP",
    "Hojo",
};
} // namespace

bool compute_log_mel(const Config& cfg, const ModelLoader& ml, const float* pcm,
                     size_t S, MelFeatures& out, std::string& err) {
    lib::MelOutput mo;
    if (!lib::compute_log_mel(lib::make_mel_policy(cfg.frontend, kMelPolicy),
                              ml, pcm, S, mo, err))
        return false;
    out.n_mels = (int64_t) mo.n_mels;
    out.n_frames = (int64_t) mo.n_frames;
    out.data = std::move(mo.f32);
    return true;
}
} // namespace starling::ggml::hojo
