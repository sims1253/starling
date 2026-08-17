// mel.cpp — Whisper log-mel frontend for ARK-ASR-3B.
//
// Thin adapter over the shared frontend (lib/whisper_mel.hpp); ark's policy:
// fullT-1 valid frames, n_samples truncation after the input checks, global
// max over ALL computed frames (flat split), f32 arithmetic, bf16 + f32
// output.
#include "mel.hpp"
#include "lib/whisper_mel.hpp"

namespace starling::ggml::ark {
namespace {
const lib::EngineMelPolicy kMelPolicy = {
    lib::MelPolicy::T_FULLT_MINUS_1,
    /*cap_n_samples=*/true,
    /*cap_before_checks=*/false,
    lib::MelPolicy::MAX_ALL_FRAMES,
    /*norm_in_double=*/false,
    /*emit_bf16=*/true,
    "STARLING_ARK_MEL_DUMP",
    "ARK",
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
} // namespace starling::ggml::ark
