// mel.cpp — Whisper log-mel frontend for moss-transcribe-preview-2B.
//
// Thin adapter over the shared frontend (lib/whisper_mel.hpp); moss's policy:
// fullT-1 valid frames, NO n_samples truncation (the windowing stage owns
// truncation), global max over ALL computed frames (flat split), f32
// arithmetic, bf16 + f32 output. n_fft=640 (vs 400 for ark/higgs/hojo) comes
// from config like every other constant. The shared body is byte-exact
// against the previous per-model copy.
#include "mel.hpp"
#include "lib/whisper_mel.hpp"

namespace starling::ggml::moss {

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
    p.t_rule = lib::MelPolicy::T_FULLT_MINUS_1;
    p.cap_n_samples = false;
    p.max_scope = lib::MelPolicy::MAX_ALL_FRAMES;
    p.norm_in_double = false;
    p.emit_bf16 = true;
    p.dump_env = "STARLING_MEL_DUMP";
    p.label = "MOSS";
    lib::MelOutput mo;
    if (!lib::compute_log_mel(p, ml, pcm, S, mo, err)) return false;
    out.n_mels = (int64_t) mo.n_mels;
    out.n_frames = (int64_t) mo.n_frames;
    out.f32 = std::move(mo.f32);
    out.data = std::move(mo.bf16);
    return true;
}
} // namespace starling::ggml::moss
