// mel.cpp — Whisper log-mel frontend for higgs-audio-v3-stt.
//
// Thin adapter over the shared frontend (lib/whisper_mel.hpp); higgs's policy:
// ceil(S/H) valid frames, n_samples truncation after the input checks, global
// max over the KEPT frames only, f32 arithmetic, bf16 + f32 output. The
// shared body is byte-exact against the previous per-model copy (loop nests
// and reduction orders unchanged).
#include "mel.hpp"
#include "lib/whisper_mel.hpp"

namespace starling::ggml::higgs {

bool compute_log_mel(const Config& cfg, const ModelLoader& ml, const float* pcm,
                     size_t S, MelFeatures& out, std::string& err) {
    const auto& c = cfg.frontend;
    lib::MelPolicy p;
    p.n_fft = c.n_fft;
    p.hop_length = c.hop_length;
    p.n_mels = c.n_mels;
    p.mel_floor = c.mel_floor;
    p.dynamic_range = c.dynamic_range;
    p.normalization_offset = c.normalization_offset;
    p.normalization_divisor = c.normalization_divisor;
    p.t_rule = lib::MelPolicy::T_CEIL_S_OVER_H;
    p.cap_n_samples = true;
    p.cap_before_checks = false;
    p.n_samples = c.n_samples;
    p.max_scope = lib::MelPolicy::MAX_KEPT_FRAMES;
    p.norm_in_double = false;
    p.emit_bf16 = true;
    p.dump_env = "STARLING_HIGGS_MEL_DUMP";
    p.label = "Higgs";
    lib::MelOutput mo;
    if (!lib::compute_log_mel(p, ml, pcm, S, mo, err)) return false;
    out.n_mels = (int64_t) mo.n_mels;
    out.n_frames = (int64_t) mo.n_frames;
    out.f32 = std::move(mo.f32);
    out.data = std::move(mo.bf16);
    return true;
}
} // namespace starling::ggml::higgs
