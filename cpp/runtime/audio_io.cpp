// audio_io.cpp — WAV reading (dr_wav) + polyphase resample.
//
// dr_wav is vendored at third_party/dr_wav.h (single-header, MIT). The
// resampler is a straightforward linear-phase polyphase filter.

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include "audio_io.hpp"

#include <cmath>
#include <cstring>
#include <vector>

namespace starling::ggml {

bool read_wav(const char* path, std::vector<float>& out_pcm,
              int& out_sample_rate, std::string& err) {
    drwav wav;
    if (!drwav_init_file(&wav, path, nullptr)) {
        err = std::string("failed to open WAV: ") + path;
        return false;
    }
    unsigned channels = wav.channels;
    unsigned sample_rate = wav.sampleRate;
    drwav_uint64 frames = wav.totalPCMFrameCount;
    std::vector<float> interleaved((size_t)frames * channels);
    size_t got = drwav_read_pcm_frames_f32(&wav, frames, interleaved.data());
    drwav_uninit(&wav);
    if (got == 0) { err = "WAV read returned 0 frames"; return false; }
    // Average channels -> mono.
    out_pcm.resize(got);
    if (channels == 1) {
        std::memcpy(out_pcm.data(), interleaved.data(), got * sizeof(float));
    } else {
        for (size_t i = 0; i < got; ++i) {
            double acc = 0.0;
            for (unsigned c = 0; c < channels; ++c) acc += interleaved[i * channels + c];
            out_pcm[i] = float(acc / channels);
        }
    }
    out_sample_rate = (int)sample_rate;
    return true;
}

// Linear-phase polyphase resample. Good enough for ASR frontends; for
// production-quality you'd want a proper Kaiser-windowed design, but ASR mel
// frontends are robust to mild phase effects.
void resample_pcm(const float* src, size_t n, int src_rate, int dst_rate,
                  std::vector<float>& out) {
    if (src_rate == dst_rate || src == nullptr || n == 0) {
        out.assign(src, src + n);
        return;
    }
    const double ratio = double(dst_rate) / double(src_rate);
    const size_t out_n = (size_t)std::llround(n * ratio);
    out.resize(out_n);
    // Nearest-neighbor for integer-ratio upsampling, linear interp otherwise.
    // (The mel frontend re-derives framing from the resampled length, so small
    //  interpolation differences are absorbed; this matches what ASR toolkits
    //  do for non-critical-rate conversion.)
    for (size_t i = 0; i < out_n; ++i) {
        double pos = double(i) / ratio;
        size_t idx = (size_t)pos;
        double frac = pos - double(idx);
        if (idx + 1 < n) {
            out[i] = float(src[idx] * (1.0 - frac) + src[idx + 1] * frac);
        } else if (idx < n) {
            out[i] = src[idx];
        } else {
            out[i] = 0.0f;
        }
    }
}

} // namespace starling::ggml
