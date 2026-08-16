// mel.cpp — Whisper log-mel frontend for higgs-audio-v3-stt.
//
// Thin adapter over the shared frontend (lib/whisper_mel.hpp); higgs's policy:
// ceil(S/H) valid frames, n_samples truncation after the input checks, global
// max over the KEPT frames only, f32 arithmetic, bf16 + f32 output.
#include "mel.hpp"
#include "lib/whisper_mel.hpp"

namespace starling::ggml::higgs {
namespace {
const lib::EngineMelPolicy kMelPolicy = {
    lib::MelPolicy::T_CEIL_S_OVER_H,
    /*cap_n_samples=*/true,
    /*cap_before_checks=*/false,
    lib::MelPolicy::MAX_KEPT_FRAMES,
    /*norm_in_double=*/false,
    /*emit_bf16=*/true,
    "STARLING_HIGGS_MEL_DUMP",
    "Higgs",
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
    out.f32 = std::move(mo.f32);
    out.data = std::move(mo.bf16);
    return true;
}
} // namespace starling::ggml::higgs
