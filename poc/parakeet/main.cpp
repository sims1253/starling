#include "engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

uint16_t u16(const std::vector<unsigned char> &data, size_t offset) {
  return static_cast<uint16_t>(data[offset]) |
         static_cast<uint16_t>(data[offset + 1] << 8);
}

uint32_t u32(const std::vector<unsigned char> &data, size_t offset) {
  return static_cast<uint32_t>(data[offset]) |
         (static_cast<uint32_t>(data[offset + 1]) << 8) |
         (static_cast<uint32_t>(data[offset + 2]) << 16) |
         (static_cast<uint32_t>(data[offset + 3]) << 24);
}

bool read_wav(const std::string &path, std::vector<float> &pcm,
              int &sample_rate, std::string &error) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    error = "cannot open WAV";
    return false;
  }
  std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(file)),
                                   std::istreambuf_iterator<char>());
  if (bytes.size() < 44 || std::memcmp(bytes.data(), "RIFF", 4) != 0 ||
      std::memcmp(bytes.data() + 8, "WAVE", 4) != 0) {
    error = "invalid WAV";
    return false;
  }
  uint16_t format = 0;
  uint16_t channels = 0;
  size_t data_offset = 0;
  size_t data_size = 0;
  size_t offset = 12;
  while (offset + 8 <= bytes.size()) {
    const size_t size = u32(bytes, offset + 4);
    if (std::memcmp(bytes.data() + offset, "fmt ", 4) == 0 &&
        offset + 8 + size <= bytes.size() && size >= 16) {
      format = u16(bytes, offset + 8);
      channels = u16(bytes, offset + 10);
      sample_rate = static_cast<int>(u32(bytes, offset + 12));
    } else if (std::memcmp(bytes.data() + offset, "data", 4) == 0 &&
               offset + 8 + size <= bytes.size()) {
      data_offset = offset + 8;
      data_size = size;
    }
    offset += 8 + size + (size & 1);
  }
  if (format == 0xfffe && data_offset >= 40)
    format = u16(bytes, data_offset - 24);
  if (sample_rate != 16000 || channels == 0 || data_offset == 0) {
    error = "WAV must be mono or convertible PCM at 16 kHz";
    return false;
  }
  const size_t frame_bytes = format == 3 ? 4 : 2;
  if (format != 1 && format != 3) {
    error = "WAV must use PCM16 or float32";
    return false;
  }
  const size_t frames = data_size / (frame_bytes * channels);
  pcm.resize(frames);
  for (size_t frame = 0; frame < frames; ++frame) {
    float sum = 0.0f;
    for (size_t channel = 0; channel < channels; ++channel) {
      const size_t at =
          data_offset + (frame * channels + channel) * frame_bytes;
      if (format == 3) {
        float value = 0.0f;
        std::memcpy(&value, bytes.data() + at, sizeof(value));
        sum += value;
      } else {
        int16_t value = 0;
        std::memcpy(&value, bytes.data() + at, sizeof(value));
        sum += static_cast<float>(value) / 32768.0f;
      }
    }
    pcm[frame] = sum / static_cast<float>(channels);
  }
  return true;
}

}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: parakeet_poc_cli MODEL [--load-only] [--wav PATH] "
                 "[--seconds N] [--repeat N] [--stream]\n";
    return 2;
  }
  bool load_only = false;
  bool stream_mode = false;
  double seconds = 1.0;
  int repeat = 1;
  std::string wav_path;
  for (int i = 2; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--load-only")
      load_only = true;
    else if (argument == "--stream")
      stream_mode = true;
    else if (argument == "--wav" && i + 1 < argc)
      wav_path = argv[++i];
    else if (argument == "--seconds" && i + 1 < argc)
      seconds = std::atof(argv[++i]);
    else if (argument == "--repeat" && i + 1 < argc)
      repeat = std::atoi(argv[++i]);
  }
  if (repeat < 1 || !std::isfinite(seconds) || seconds <= 0.0) {
    std::cerr << "seconds and repeat must be positive\n";
    return 1;
  }
  parakeet_poc::Engine engine;
  std::string error;
  const auto load_start = std::chrono::steady_clock::now();
  if (!engine.load(argv[1], error)) {
    std::cerr << error << '\n';
    return 1;
  }
  const auto load_end = std::chrono::steady_clock::now();
  std::cerr << "loaded in "
            << std::chrono::duration<double, std::milli>(load_end - load_start)
                   .count()
            << " ms\n";
  if (load_only)
    return 0;
  std::vector<float> pcm;
  int sample_rate = 0;
  if (!wav_path.empty()) {
    if (!read_wav(wav_path, pcm, sample_rate, error)) {
      std::cerr << error << '\n';
      return 1;
    }
  } else {
    pcm.resize(static_cast<size_t>(seconds * 16000.0));
  }
  if (repeat > 1) {
    if (pcm.size() >
        std::numeric_limits<size_t>::max() / static_cast<size_t>(repeat)) {
      std::cerr << "repeated audio is too large\n";
      return 1;
    }
    std::vector<float> repeated(pcm.size() * static_cast<size_t>(repeat));
    for (size_t iteration = 0; iteration < static_cast<size_t>(repeat);
         ++iteration)
      std::copy(pcm.begin(), pcm.end(),
                repeated.begin() + iteration * pcm.size());
    pcm.swap(repeated);
  }
  if (stream_mode) {
    parakeet_poc::Stream stream(engine);
    std::string stream_error;
    const auto stream_start = std::chrono::steady_clock::now();
    const size_t chunk = 16000;
    for (size_t offset = 0; offset < pcm.size(); offset += chunk) {
      const size_t length = std::min(chunk, pcm.size() - offset);
      stream.append(pcm.data() + offset, length, stream_error);
      if (!stream_error.empty()) {
        std::cerr << stream_error << '\n';
        return 1;
      }
    }
    auto final_text = stream.finish(stream_error);
    if (!final_text.has_value()) {
      std::cerr << stream_error << '\n';
      return 1;
    }
    const auto stream_end = std::chrono::steady_clock::now();
    std::cout << *final_text << '\n';
    const auto stream_stats = engine.stats();
    std::cerr << "stream "
              << std::chrono::duration<double, std::milli>(stream_end -
                                                           stream_start)
                     .count()
              << " ms, samples " << pcm.size() << ", audio_seconds "
              << static_cast<double>(pcm.size()) / 16000.0 << ", plans "
              << stream_stats.encoder_plans << ", calls "
              << stream_stats.transcribe_calls << ", cache_hits "
              << stream_stats.cache_hits << ", plan_hits "
              << stream_stats.plan_cache_hits << '\n';
    return 0;
  }
  std::string text;
  const auto start = std::chrono::steady_clock::now();
  if (!engine.transcribe(pcm.data(), pcm.size(), text, error)) {
    std::cerr << error << '\n';
    return 1;
  }
  const auto end = std::chrono::steady_clock::now();
  const auto stats = engine.stats();
  std::cout << text << '\n';
  std::cerr << "transcribe "
            << std::chrono::duration<double, std::milli>(end - start).count()
            << " ms, mel " << stats.last_mel_ms << " ms, encoder "
            << stats.last_encoder_ms << " ms, decode " << stats.last_decode_ms
            << " ms, plans " << stats.encoder_plans << ", plan_hits "
            << stats.plan_cache_hits << ", steps " << stats.decode_steps
            << ", samples " << pcm.size() << '\n';
  return 0;
}
