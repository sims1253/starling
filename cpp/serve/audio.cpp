// audio.cpp — WAV/PCM audio decoding implementation.
//
// NOTE: dr_wav.h with DR_WAV_IMPLEMENTATION is already included in
// cpp/runtime/audio_io.cpp (part of starling_ggml_core, which starling-serve
// links). We only need the declarations here, not a second implementation.

#include "audio.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <sstream>

#include "dr_wav.h"

namespace starling::serve::audio {

// Decode in bounded batches so no single allocation scales with the header's
// (untrusted) frame count.
constexpr drwav_uint64 kWavDecodeBatchFrames = 16384;
// No real-world upload carries more channels than this; a fmt chunk claiming
// more is crafted (it would also scale the batch buffer by channels).
constexpr uint32_t kWavMaxChannels = 64;

bool wav_bytes_to_float32(const std::string& wav_bytes,
                          std::vector<float>& out, int& sr) {
    drwav wav;
    if (!drwav_init_memory(&wav, wav_bytes.data(), wav_bytes.size(), nullptr))
        return false;

    sr = static_cast<int>(wav.sampleRate);
    uint64_t total_frames = wav.totalPCMFrameCount;
    uint32_t channels = wav.channels;

    // The header's totalPCMFrameCount is derived from the data-chunk size
    // field, which is untrusted input: a crafted header claiming millions of
    // frames must not drive the allocation. Derive the maximum frame count
    // from the actual payload length and reject headers that overclaim (a
    // truncated or lying data-chunk size is a malformed file).
    uint32_t bytes_per_frame =
        ((wav.bitsPerSample + 7) / 8) * channels;
    if (channels == 0 || channels > kWavMaxChannels || total_frames == 0
        || bytes_per_frame == 0) {
        drwav_uninit(&wav);
        return false;
    }
    uint64_t max_frames = wav_bytes.size() / bytes_per_frame;
    if (total_frames > max_frames) {
        drwav_uninit(&wav);
        return false;
    }

    out.clear();
    out.reserve(static_cast<size_t>(total_frames));

    // Decode as interleaved int16 in fixed-size batches, then convert to mono
    // float32.
    std::vector<drwav_int16> interleaved;
    interleaved.resize(static_cast<size_t>(kWavDecodeBatchFrames) * channels);
    uint64_t decoded_frames = 0;
    while (decoded_frames < total_frames) {
        uint64_t want = std::min<uint64_t>(kWavDecodeBatchFrames,
                                           total_frames - decoded_frames);
        drwav_uint64 read = drwav_read_pcm_frames_s16(
            &wav, want, interleaved.data());
        if (read == 0) break;  // payload exhausted before the header's count
        size_t old = out.size();
        out.resize(old + static_cast<size_t>(read));
        if (channels == 1) {
            for (drwav_uint64 i = 0; i < read; ++i)
                out[old + i] = static_cast<float>(interleaved[i]) / 32768.0f;
        } else {
            // Mix down to mono by averaging channels.
            for (drwav_uint64 i = 0; i < read; ++i) {
                int sum = 0;
                for (uint32_t c = 0; c < channels; ++c)
                    sum += interleaved[i * channels + c];
                out[old + i] = static_cast<float>(sum) / (channels * 32768.0f);
            }
        }
        decoded_frames += read;
    }
    drwav_uninit(&wav);

    if (out.empty()) return false;
    return true;
}

std::vector<float> pcm16_to_float32(const std::string& bytes) {
    size_t nbytes = bytes.size();
    if (nbytes == 0) return {};
    if (nbytes % 2 == 1) nbytes--;  // drop odd trailing byte
    size_t nsamples = nbytes / 2;
    std::vector<float> out(nsamples);
    const auto* src = reinterpret_cast<const int16_t*>(bytes.data());
    for (size_t i = 0; i < nsamples; ++i)
        out[i] = static_cast<float>(src[i]) / 32768.0f;
    return out;
}

} // namespace starling::serve::audio
