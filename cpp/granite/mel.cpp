// mel.cpp — torchaudio log-mel frontend for granite-speech-4.1-2b.
//
// Thin adapter over the shared frontend (lib/whisper_mel.hpp) plus the two
// granite-specific post steps the feature extractor applies after the
// normalize: drop the trailing frame when the count is odd, then stack
// consecutive pairs into 160-dim frames. Policy: torchaudio center=True keeps
// every fullT frame (T_FULLT), no n_samples cap, global max over ALL computed
// frames (the amax runs before the odd drop), f32 arithmetic, bf16 + f32 out.
#include "mel.hpp"
#include "lib/whisper_mel.hpp"

namespace starling::ggml::granite {
namespace {
const lib::EngineMelPolicy kMelPolicy = {
    lib::MelPolicy::T_FULLT,
    /*cap_n_samples=*/false,
    /*cap_before_checks=*/false,
    lib::MelPolicy::MAX_ALL_FRAMES,
    /*norm_in_double=*/false,
    /*emit_bf16=*/true,
    "STARLING_GRANITE_MEL_DUMP",
    "GRANITE",
};
} // namespace

bool compute_log_mel(const Config& cfg, const ModelLoader& ml, const float* pcm,
                     size_t S, MelFeatures& out, std::string& err) {
    lib::MelOutput mo;
    if (!lib::compute_log_mel(lib::make_mel_policy(cfg.frontend, kMelPolicy),
                              ml, pcm, S, mo, err))
        return false;
    // mo is feat-major [80, T_mel] (element (m, t) at m * T_mel + t). Drop the
    // trailing frame when odd, then pair-stack: frame t' of the output
    // concatenates mel[:, 2t'] and mel[:, 2t'+1] along the feature axis.
    const int64_t M = (int64_t) mo.n_mels;
    const int64_t T_full = (int64_t) mo.n_frames;
    int64_t T = T_full;
    if (T & 1) T -= 1;
    const int64_t T2 = T / 2;
    const int64_t M2 = M * 2;
    out.f32.assign((size_t) M2 * T2, 0.0f);
    out.data.assign((size_t) M2 * T2, ggml_bf16_t{0});
    for (int64_t t = 0; t < T2; ++t) {
        float* dst = out.f32.data() + (size_t) t * M2;
        ggml_bf16_t* dst16 = out.data.data() + (size_t) t * M2;
        for (int64_t m = 0; m < M; ++m) {
            dst[m] = mo.f32[(size_t) m * T_full + 2 * t];
            dst[M + m] = mo.f32[(size_t) m * T_full + 2 * t + 1];
            dst16[m] = mo.bf16[(size_t) m * T_full + 2 * t];
            dst16[M + m] = mo.bf16[(size_t) m * T_full + 2 * t + 1];
        }
    }
    out.n_mels = M2;
    out.n_frames = T2;
    return true;
}
} // namespace starling::ggml::granite
