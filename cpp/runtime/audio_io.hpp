// audio_io.hpp — WAV reading + 8/16/24/48 kHz -> 16 kHz resample.
//
// WAV reading uses dr_wav.h (vendored single-header, MIT). Resample is a
// linear-phase polyphase filter good enough for ASR frontends. Both are
// model-agnostic helpers used by whichever model needs to load audio from disk
// (the C API takes raw float32 PCM directly, so these are mainly for the CLI).

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace starling::ggml {

// Read a WAV file into mono float32 [-1, 1]. Returns false on error. If the
// file is multi-channel, channels are averaged. If the sample rate is not 16k,
// the caller should resample_pcm afterwards.
bool read_wav(const char* path, std::vector<float>& out_pcm,
              int& out_sample_rate, std::string& err);

// Resample float32 PCM from `src_rate` to `dst_rate` (mono). Linear-phase
// polyphase. `dst_rate` is typically 16000 (the ASR frontend rate).
void resample_pcm(const float* src, size_t n, int src_rate, int dst_rate,
                  std::vector<float>& out);

} // namespace starling::ggml
