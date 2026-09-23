#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace parakeet_poc {

struct Stats {
  std::uint64_t transcribe_calls = 0;
  std::uint64_t cache_hits = 0;
  std::uint64_t encoder_plans = 0;
  std::uint64_t plan_cache_hits = 0;
  std::uint64_t decode_steps = 0;
  double last_mel_ms = 0.0;
  double last_encoder_ms = 0.0;
  double last_decode_ms = 0.0;
  double last_total_ms = 0.0;
};

struct StreamConfig {
  std::size_t sample_rate = 16000;
  std::size_t window_samples = 192000;
  std::size_t overlap_samples = 48000;
  std::size_t max_buffer_samples = 60 * 16000;
};

class Engine {
public:
  Engine();
  ~Engine();

  Engine(const Engine &) = delete;
  Engine &operator=(const Engine &) = delete;

  bool load(const char *path, std::string &error);
  bool transcribe(const float *pcm, std::size_t samples, std::string &text,
                  std::string &error);
  void reset();
  Stats stats() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class Stream {
public:
  Stream(Engine &engine, StreamConfig config = {});
  ~Stream();

  Stream(const Stream &) = delete;
  Stream &operator=(const Stream &) = delete;

  std::optional<std::string> append(const float *pcm, std::size_t samples,
                                    std::string &error);
  std::optional<std::string> finish(std::string &error);
  void reset();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}
