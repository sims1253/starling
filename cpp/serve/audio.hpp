// audio.hpp — WAV/PCM audio decoding for the native server.
//
// Uses dr_wav (third_party/dr_wav.h, already vendored) for WAV decoding and
// provides PCM16→float32 conversion matching the Python helpers in
// src/starling/server.py.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace starling::serve::audio {

// Decode a WAV byte blob to mono float32 samples. Returns true on success;
// writes the sample rate to *sr. On failure returns false.
bool wav_bytes_to_float32(const std::string& wav_bytes,
                          std::vector<float>& out, int& sr);

// Convert raw PCM16 little-endian bytes to float32 samples.
std::vector<float> pcm16_to_float32(const std::string& bytes);

// Extract the audio payload from a multipart/form-data body, or return the body
// unchanged if it's not multipart (raw WAV path). Port of
// _extract_multipart_payload from server.py.
std::string extract_multipart_payload(const std::string& body,
                                      const std::string& content_type);

} // namespace starling::serve::audio
