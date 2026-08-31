// mel.cpp — whisper-style log-mel frontend for Nemotron-Labs-Audex-2B.
//
// Thin adapter over the shared frontend (lib/whisper_mel.hpp) plus the one
// audex-specific step around it: every clip is zero-padded to
// frontend.n_samples = 480000 samples BEFORE the mel (the WhisperFeatureExtractor's
// padding="max_length" — the encoder's expected_seq_length check makes 3000
// frames a hard shape requirement, so unlike qwen3 there is no short-clip
// minimum and no variable-length padding: T = S/H = 3000 always). Policy:
// torch.stft center=True gives fullT = S/H + 1 frames of which the LAST is
// dropped (T_FULLT_MINUS_1); the eager extractor drops that frame BEFORE the
// global max-clamp, so the max runs over the KEPT frames only
// (MAX_KEPT_FRAMES, the higgs/hojo rule); f32 arithmetic, bf16 + f32 out.
// The buffer keeps the shared feat-major layout ([T, 128] ggml view, time
// innermost) the encoder's conv reshape wants.
#include "mel.hpp"
#include "lib/whisper_mel.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

namespace starling::ggml::audex {
namespace {
const lib::EngineMelPolicy kMelPolicy = {
    lib::MelPolicy::T_FULLT_MINUS_1,
    /*cap_n_samples=*/false,
    /*cap_before_checks=*/false,
    lib::MelPolicy::MAX_KEPT_FRAMES,
    /*norm_in_double=*/false,
    /*emit_bf16=*/true,
    "STARLING_AUDEX_MEL_DUMP",
    "AUDEX",
};
} // namespace

bool compute_log_mel(const Config& cfg, const ModelLoader& ml, const float* pcm,
                     size_t S, MelFeatures& out, std::string& err) {
    // padding="max_length" (+ truncation=True): every clip becomes exactly
    // n_samples; the decode entry never feeds more than one chunk, so the
    // truncate arm is defensive only.
    std::vector<float> padded;
    const size_t want = cfg.frontend.n_samples;
    if (S != want) {
        padded.assign(want, 0.0f);
        if (S > 0) std::memcpy(padded.data(), pcm, std::min(S, want) * sizeof(float));
        pcm = padded.data();
        S = want;
    }
    lib::MelOutput mo;
    if (!lib::compute_log_mel(lib::make_mel_policy(cfg.frontend, kMelPolicy),
                              ml, pcm, S, mo, err))
        return false;
    // mo.bf16 is feat-major [128, T] (element (m, t) at m * T + t) — the
    // [T, 128] ggml layout with time innermost, exactly what the fixed
    // encoder graph reads. T is always 3000 here; keep the shape check so a
    // metadata/policy drift fails loudly instead of deshaping the encoder.
    if ((int64_t) mo.n_frames != 2 * (int64_t) cfg.encoder.max_pos_emb) {
        err = "AUDEX mel frame count does not match the fixed encoder input";
        return false;
    }
    out.data = std::move(mo.bf16);
    out.n_mels = (int64_t) mo.n_mels;
    out.n_frames = (int64_t) mo.n_frames;
    out.valid_frames = out.n_frames;
    return true;
}
} // namespace starling::ggml::audex
