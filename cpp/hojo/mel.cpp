// mel.cpp — Whisper log-mel frontend for Hojo-ASR-V1.
//
// Thin adapter over the shared frontend (lib/whisper_mel.hpp); hojo's policy:
// S/H valid frames (padding=False, last-frame truncation load-bearing),
// unconditional n_samples (=40 s) truncation BEFORE the input checks, global
// max over the KEPT frames only, double-constant clamp/normalize arithmetic,
// f32-only output. The shared body is byte-exact against the previous
// per-model copy.
#include "mel.hpp"
#include "lib/whisper_mel.hpp"

namespace starling::ggml::hojo {

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
    p.t_rule = lib::MelPolicy::T_FLOOR_S_OVER_H;
    p.cap_n_samples = true;
    p.cap_before_checks = true;
    p.n_samples = c.n_samples;
    p.max_scope = lib::MelPolicy::MAX_KEPT_FRAMES;
    p.norm_in_double = true;
    p.emit_bf16 = false;
    p.dump_env = "STARLING_HOJO_MEL_DUMP";
    p.label = "Hojo";
    lib::MelOutput mo;
    if (!lib::compute_log_mel(p, ml, pcm, S, mo, err)) return false;
    out.n_mels = (int64_t) mo.n_mels;
    out.n_frames = (int64_t) mo.n_frames;
    out.data = std::move(mo.f32);
    return true;
}
} // namespace starling::ggml::hojo
