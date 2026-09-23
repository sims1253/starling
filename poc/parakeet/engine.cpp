#include "engine.hpp"

#include "cpu_kernels.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace parakeet_poc {
namespace {

constexpr int kSampleRate = 16000;
constexpr size_t kMaxSamples = 60 * kSampleRate;
constexpr size_t kGraphNodes = 32768;
constexpr float kDefaultPreemph = 0.97f;
constexpr float kDefaultLogGuard = 5.9604645e-08f;
constexpr float kNormEps = 1e-5f;

struct Config {
  int sample_rate = kSampleRate;
  int n_mels = 128;
  int n_fft = 512;
  int win_length = 400;
  int hop_length = 160;
  float preemph = kDefaultPreemph;
  float mag_power = 2.0f;
  float log_zero_guard = kDefaultLogGuard;
  std::string normalize = "per_feature";
  int d_model = 0;
  int n_layers = 0;
  int pred_out = 0;
  int n_heads = 0;
  int ff_dim = 0;
  int conv_kernel = 0;
  int subsampling_conv_channels = 0;
  std::string conv_norm_type = "batch_norm";
  bool xscaling = false;
  int pred_hidden = 0;
  int pred_rnn_layers = 0;
  int joint_hidden = 0;
  int max_symbols = 10;
  int vocab_size = 0;
  int blank_id = 0;
  std::vector<int> durations;
  std::vector<std::string> pieces;
};

struct HostInput {
  std::vector<float> data;
  ggml_tensor *tensor = nullptr;
  bool persistent = false;
};

struct CustomLinearData {
  cpu::LinearMatrix matrix;
};

struct Plan {
  int mel_frames = 0;
  int valid_frames = 0;
  int graph_frames = 0;
  int encoder_frames = 0;
  ggml_context *meta = nullptr;
  ggml_cgraph *graph = nullptr;
  ggml_gallocr_t allocator = nullptr;
  ggml_tensor *mel_input = nullptr;
  ggml_tensor *output = nullptr;
  std::vector<float> mel_host;
  std::vector<std::unique_ptr<HostInput>> inputs;
  std::vector<HostInput *> input_order;
  std::vector<std::unique_ptr<CustomLinearData>> custom_linear_data;

  ~Plan() {
    if (allocator)
      ggml_gallocr_free(allocator);
    if (meta)
      ggml_free(meta);
  }

  Plan() = default;
  Plan(const Plan &) = delete;
  Plan &operator=(const Plan &) = delete;
};

struct StepPlan {
  ggml_context *meta = nullptr;
  ggml_cgraph *graph = nullptr;
  ggml_gallocr_t allocator = nullptr;
  ggml_tensor *input = nullptr;
  ggml_tensor *token_output = nullptr;
  ggml_tensor *duration_output = nullptr;
  ggml_tensor *hidden_output = nullptr;
  ggml_tensor *cell_output = nullptr;
  std::vector<float> input_host;
  int joint_hidden = 0;
  int pred_hidden = 0;
  int layers = 0;
  int token_count = 0;
  int duration_count = 0;

  ~StepPlan() {
    if (allocator)
      ggml_gallocr_free(allocator);
    if (meta)
      ggml_free(meta);
  }

  StepPlan() = default;
  StepPlan(const StepPlan &) = delete;
  StepPlan &operator=(const StepPlan &) = delete;
};

struct DecoderWeights {
  std::vector<float> embedding;
  int embedding_rows = 0;
  int embedding_width = 0;
};

struct StepWorkspace {
  std::vector<float> layer_input;
  std::vector<float> gates;
  std::vector<float> recurrent;
  std::vector<float> updated_hidden;
  std::vector<float> updated_cell;
  std::vector<float> predicted;
  std::vector<float> fused;
  std::vector<float> logits;
};

struct Model {
  gguf_context *gguf = nullptr;
  ggml_context *weights = nullptr;
  ggml_backend_t backend = nullptr;
  ggml_backend_buffer_t weight_buffer = nullptr;
  std::map<std::string, ggml_tensor *> tensors;
  Config config;
  int threads = 1;
  StepWorkspace step_workspace;
  DecoderWeights decoder;
  std::unique_ptr<StepPlan> step_plan;
  std::vector<float> filterbank;
  std::vector<float> window;
  std::map<std::pair<int, int>, std::unique_ptr<Plan>> plans;
  std::vector<std::pair<int, int>> plan_order;
  std::vector<float> cached_pcm;
  std::string cached_text;
  bool cache_valid = false;
  Stats stats;

  ~Model() {
    step_plan.reset();
    plans.clear();
    if (weight_buffer)
      ggml_backend_buffer_free(weight_buffer);
    if (backend)
      ggml_backend_free(backend);
    if (gguf)
      gguf_free(gguf);
    if (weights)
      ggml_free(weights);
  }

  ggml_tensor *tensor(const std::string &name) const {
    auto it = tensors.find(name);
    return it == tensors.end() ? nullptr : it->second;
  }

  Plan *find_plan(const std::pair<int, int> &key) {
    auto it = plans.find(key);
    if (it == plans.end())
      return nullptr;
    for (auto order_it = plan_order.begin(); order_it != plan_order.end();
         ++order_it) {
      if (*order_it == key) {
        std::pair<int, int> saved = *order_it;
        plan_order.erase(order_it);
        plan_order.push_back(saved);
        break;
      }
    }
    ++stats.plan_cache_hits;
    return it->second.get();
  }

  Plan *install_plan(const std::pair<int, int> &key,
                     std::unique_ptr<Plan> plan) {
    plans.emplace(key, std::move(plan));
    plan_order.push_back(key);
    while (plan_order.size() > 4) {
      plans.erase(plan_order.front());
      plan_order.erase(plan_order.begin());
    }
    return plans.at(key).get();
  }
};

std::string env_value(const char *name) {
  const char *value = std::getenv(name);
  return value ? std::string(value) : std::string();
}

bool test_mode() {
  static const bool value = env_value("PARAKEET_POC_ALLOW_TEST_MODEL") == "1";
  return value;
}

bool reference_step_mode() {
  static const bool value = env_value("PARAKEET_POC_REFERENCE_STEP") == "1";
  return value;
}

int custom_linear_threads() {
  const std::string setting = env_value("PARAKEET_POC_THREADS");
  if (!setting.empty())
    return std::max(1, std::atoi(setting.c_str()));
  const int available = static_cast<int>(std::thread::hardware_concurrency());
  return std::max(1, available);
}

void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

int64_t key_id(gguf_context *gguf, const std::string &key) {
  return gguf_find_key(gguf, key.c_str());
}

bool kv_string(gguf_context *gguf, const std::string &key, std::string &value) {
  const int64_t id = key_id(gguf, key);
  if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_STRING)
    return false;
  const char *raw = gguf_get_val_str(gguf, id);
  if (!raw)
    return false;
  value = raw;
  return true;
}

bool kv_i64(gguf_context *gguf, const std::string &key, int64_t &value) {
  const int64_t id = key_id(gguf, key);
  if (id < 0)
    return false;
  switch (gguf_get_kv_type(gguf, id)) {
  case GGUF_TYPE_UINT8:
    value = gguf_get_val_u8(gguf, id);
    return true;
  case GGUF_TYPE_INT8:
    value = gguf_get_val_i8(gguf, id);
    return true;
  case GGUF_TYPE_UINT16:
    value = gguf_get_val_u16(gguf, id);
    return true;
  case GGUF_TYPE_INT16:
    value = gguf_get_val_i16(gguf, id);
    return true;
  case GGUF_TYPE_UINT32:
    value = gguf_get_val_u32(gguf, id);
    return true;
  case GGUF_TYPE_INT32:
    value = gguf_get_val_i32(gguf, id);
    return true;
  case GGUF_TYPE_UINT64:
    value = static_cast<int64_t>(gguf_get_val_u64(gguf, id));
    return true;
  case GGUF_TYPE_INT64:
    value = gguf_get_val_i64(gguf, id);
    return true;
  default:
    return false;
  }
}

int kv_int(gguf_context *gguf, const std::string &key, int fallback) {
  int64_t value = 0;
  return kv_i64(gguf, key, value) ? static_cast<int>(value) : fallback;
}

float kv_float(gguf_context *gguf, const std::string &key, float fallback) {
  const int64_t id = key_id(gguf, key);
  if (id < 0)
    return fallback;
  switch (gguf_get_kv_type(gguf, id)) {
  case GGUF_TYPE_FLOAT32:
    return gguf_get_val_f32(gguf, id);
  case GGUF_TYPE_FLOAT64:
    return static_cast<float>(gguf_get_val_f64(gguf, id));
  default: {
    int64_t value = 0;
    return kv_i64(gguf, key, value) ? static_cast<float>(value) : fallback;
  }
  }
}

std::vector<int> kv_int_array(gguf_context *gguf, const std::string &key) {
  std::vector<int> result;
  const int64_t id = key_id(gguf, key);
  if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_ARRAY)
    return result;
  const size_t count = gguf_get_arr_n(gguf, id);
  const auto type = gguf_get_arr_type(gguf, id);
  const void *raw = gguf_get_arr_data(gguf, id);
  if (!raw)
    return result;
  result.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    switch (type) {
    case GGUF_TYPE_UINT8:
      result.push_back(static_cast<int>(static_cast<const uint8_t *>(raw)[i]));
      break;
    case GGUF_TYPE_INT8:
      result.push_back(static_cast<int>(static_cast<const int8_t *>(raw)[i]));
      break;
    case GGUF_TYPE_UINT16:
      result.push_back(static_cast<int>(static_cast<const uint16_t *>(raw)[i]));
      break;
    case GGUF_TYPE_INT16:
      result.push_back(static_cast<int>(static_cast<const int16_t *>(raw)[i]));
      break;
    case GGUF_TYPE_UINT32:
      result.push_back(static_cast<int>(static_cast<const uint32_t *>(raw)[i]));
      break;
    case GGUF_TYPE_INT32:
      result.push_back(static_cast<int>(static_cast<const int32_t *>(raw)[i]));
      break;
    case GGUF_TYPE_UINT64:
      result.push_back(static_cast<int>(static_cast<const uint64_t *>(raw)[i]));
      break;
    case GGUF_TYPE_INT64:
      result.push_back(static_cast<int>(static_cast<const int64_t *>(raw)[i]));
      break;
    default:
      break;
    }
  }
  return result;
}

std::vector<std::string> kv_string_array(gguf_context *gguf,
                                         const std::string &key) {
  std::vector<std::string> result;
  const int64_t id = key_id(gguf, key);
  if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_ARRAY)
    return result;
  if (gguf_get_arr_type(gguf, id) != GGUF_TYPE_STRING)
    return result;
  result.reserve(gguf_get_arr_n(gguf, id));
  for (size_t i = 0; i < gguf_get_arr_n(gguf, id); ++i) {
    const char *value = gguf_get_arr_str(gguf, id, i);
    result.emplace_back(value ? value : "");
  }
  return result;
}

void read_tensor_f32(ggml_tensor *tensor, std::vector<float> &output) {
  require(tensor != nullptr, "missing tensor");
  require(tensor->data != nullptr, "tensor has no host data");
  const size_t count = static_cast<size_t>(ggml_nelements(tensor));
  output.resize(count);
  if (tensor->type == GGML_TYPE_F32) {
    std::memcpy(output.data(), tensor->data, count * sizeof(float));
    return;
  }
  if (tensor->type == GGML_TYPE_F16) {
    ggml_fp16_to_fp32_row(static_cast<const ggml_fp16_t *>(tensor->data),
                          output.data(), count);
    return;
  }
  if (tensor->type == GGML_TYPE_BF16) {
    ggml_bf16_to_fp32_row(static_cast<const ggml_bf16_t *>(tensor->data),
                          output.data(), count);
    return;
  }
  const ggml_type_traits *traits = ggml_get_type_traits(tensor->type);
  require(traits != nullptr && traits->to_float != nullptr,
          "unsupported tensor type");
  std::vector<uint8_t> raw(ggml_nbytes(tensor));
  std::memcpy(raw.data(), tensor->data, raw.size());
  traits->to_float(raw.data(), output.data(), static_cast<int64_t>(count));
}

size_t element_count(const std::vector<int64_t> &shape) {
  size_t result = 1;
  for (int64_t value : shape)
    result *= static_cast<size_t>(value);
  return result;
}

ggml_tensor *add_input(Plan &plan, ggml_type type,
                       const std::vector<int64_t> &shape,
                       std::vector<float> data, bool persistent) {
  auto input = std::make_unique<HostInput>();
  input->data = std::move(data);
  input->persistent = persistent;
  require(input->data.size() == element_count(shape), "input size mismatch");
  ggml_tensor *tensor = nullptr;
  if (shape.size() == 1)
    tensor = ggml_new_tensor_1d(plan.meta, type, shape[0]);
  else if (shape.size() == 2)
    tensor = ggml_new_tensor_2d(plan.meta, type, shape[0], shape[1]);
  else if (shape.size() == 3)
    tensor = ggml_new_tensor_3d(plan.meta, type, shape[0], shape[1], shape[2]);
  else if (shape.size() == 4)
    tensor = ggml_new_tensor_4d(plan.meta, type, shape[0], shape[1], shape[2],
                                shape[3]);
  require(tensor != nullptr, "input allocation failed");
  ggml_set_input(tensor);
  input->tensor = tensor;
  plan.inputs.push_back(std::move(input));
  plan.input_order.push_back(plan.inputs.back().get());
  return tensor;
}

ggml_tensor *weight(Model &model, const std::string &name) {
  ggml_tensor *tensor = model.tensor(name);
  require(tensor != nullptr, "missing tensor " + name);
  return tensor;
}

cpu::QuantType graph_quant_type(ggml_type type) {
  switch (type) {
  case GGML_TYPE_F32:
    return cpu::QuantType::F32;
  case GGML_TYPE_Q4_K:
    return cpu::QuantType::Q4K;
  case GGML_TYPE_Q6_K:
    return cpu::QuantType::Q6K;
  case GGML_TYPE_Q8_0:
    return cpu::QuantType::Q8_0;
  default:
    throw std::runtime_error("unsupported custom linear weight type");
  }
}

void custom_linear_callback(ggml_tensor *destination, int thread_index, int,
                            void *user_data) {
  if (destination->src[0] == nullptr || destination->src[1] == nullptr ||
      destination->src[0]->type != GGML_TYPE_F32 ||
      destination->src[0]->nb[0] != sizeof(float) ||
      destination->src[0]->nb[1] !=
          static_cast<std::size_t>(destination->src[0]->ne[0]) * sizeof(float))
    return;
  const int samples = static_cast<int>(destination->ne[1] * destination->ne[2]);
  if (user_data == nullptr)
    return;
  const CustomLinearData *data =
      static_cast<const CustomLinearData *>(user_data);
  const cpu::LinearMatrix &matrix = data->matrix;
  const float *input = static_cast<const float *>(destination->src[0]->data);
  float *output = static_cast<float *>(destination->data);
  if (thread_index != 0)
    return;
  const int threads = custom_linear_threads();
  if (matrix.type == cpu::QuantType::F32)
    cpu::linear_f32(matrix, input, output, samples, threads);
  else
    cpu::linear_q8_batch(matrix, input, output, samples, threads);
}

ggml_tensor *linear(Model &model, Plan &plan, ggml_context *context,
                    ggml_tensor *input, const std::string &weight_name,
                    const std::string &bias_name) {
  ggml_tensor *weights = weight(model, weight_name);
  ggml_tensor *bias = bias_name.empty() ? nullptr : model.tensor(bias_name);
  const std::string custom_prefix =
      env_value("PARAKEET_POC_CUSTOM_LINEAR_PREFIX");
  const bool use_custom =
      env_value("PARAKEET_POC_REFERENCE_LINEAR") == "1" ? false
      : custom_prefix.empty()                           ? true
                              : weight_name.rfind(custom_prefix, 0) == 0;
  const bool contiguous_input =
      input->type == GGML_TYPE_F32 && input->nb[0] == sizeof(float) &&
      input->nb[1] == static_cast<std::size_t>(input->ne[0]) * sizeof(float) &&
      (input->ne[2] <= 1 ||
       input->nb[2] == input->nb[1] * static_cast<std::size_t>(input->ne[1]));
  if (!use_custom || !contiguous_input) {
    ggml_tensor *output = ggml_mul_mat(context, weights, input);
    if (bias)
      output = ggml_add(context, output, bias);
    return output;
  }
  auto data = std::make_unique<CustomLinearData>();
  data->matrix.data = weights->data;
  data->matrix.type = graph_quant_type(weights->type);
  data->matrix.input = static_cast<int>(weights->ne[0]);
  data->matrix.output = static_cast<int>(weights->ne[1]);
  data->matrix.row_bytes = weights->nb[1];
  if (bias)
    data->matrix.bias = static_cast<const float *>(bias->data);
  if (data->matrix.output % 2 != 0)
    data->matrix.packed = cpu::pack_matrix(data->matrix);
  CustomLinearData *callback_data = data.get();
  plan.custom_linear_data.push_back(std::move(data));
  std::vector<ggml_tensor *> args{input, weights};
  if (bias)
    args.push_back(bias);
  return ggml_custom_4d(context, GGML_TYPE_F32, weights->ne[1], input->ne[1],
                        input->ne[2], 1, args.data(),
                        static_cast<int>(args.size()), custom_linear_callback,
                        1, callback_data);
}

ggml_tensor *layer_norm(Model &model, ggml_context *context, ggml_tensor *input,
                        const std::string &name) {
  ggml_tensor *output = ggml_norm(context, input, 1e-5f);
  output = ggml_mul(context, output, weight(model, name + ".weight"));
  return ggml_add(context, output, weight(model, name + ".bias"));
}

ggml_tensor *feed_forward(Model &model, Plan &plan, ggml_context *context,
                          ggml_tensor *input, const std::string &name) {
  const std::string first_bias = name + ".linear1.bias";
  const std::string second_bias = name + ".linear2.bias";
  ggml_tensor *output =
      linear(model, plan, context, input, name + ".linear1.weight", first_bias);
  output = ggml_silu(context, output);
  return linear(model, plan, context, output, name + ".linear2.weight",
                second_bias);
}

void rfft(std::vector<std::complex<double>> &values, int n) {
  std::vector<double> real(n);
  std::vector<double> imag(n);
  for (int i = 0; i < n; ++i) {
    real[i] = values[i].real();
    imag[i] = values[i].imag();
  }
  for (int i = 1, j = 0; i < n; ++i) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1)
      j ^= bit;
    j ^= bit;
    if (i < j) {
      std::swap(real[i], real[j]);
      std::swap(imag[i], imag[j]);
    }
  }
  constexpr double pi = 3.14159265358979323846;
  for (int length = 2; length <= n; length <<= 1) {
    const double angle = -2.0 * pi / length;
    const double wr = std::cos(angle);
    const double wi = std::sin(angle);
    for (int i = 0; i < n; i += length) {
      double current_real = 1.0;
      double current_imag = 0.0;
      for (int k = 0; k < length / 2; ++k) {
        const int u = i + k;
        const int v = u + length / 2;
        const double tr = current_real * real[v] - current_imag * imag[v];
        const double ti = current_real * imag[v] + current_imag * real[v];
        real[v] = real[u] - tr;
        imag[v] = imag[u] - ti;
        real[u] += tr;
        imag[u] += ti;
        const double next_real = current_real * wr - current_imag * wi;
        current_imag = current_real * wi + current_imag * wr;
        current_real = next_real;
      }
    }
  }
  for (int i = 0; i < n; ++i)
    values[i] = std::complex<double>(real[i], imag[i]);
}

void compute_mel(const Model &model, const float *pcm, size_t samples,
                 std::vector<float> &features, int &frames, int &valid_frames) {
  const Config &config = model.config;
  require(samples > 0 && samples <= kMaxSamples, "PCM length out of range");
  require(samples >= static_cast<size_t>(config.hop_length),
          "PCM is shorter than one mel hop");
  const int n_fft = config.n_fft;
  const int hop = config.hop_length;
  const int bins = n_fft / 2 + 1;
  std::vector<double> pre(samples);
  pre[0] = static_cast<double>(pcm[0]);
  for (size_t i = 1; i < samples; ++i) {
    pre[i] = static_cast<double>(pcm[i]) -
             static_cast<double>(config.preemph) * pcm[i - 1];
  }
  const int padding = n_fft / 2;
  std::vector<double> padded(samples + n_fft, 0.0);
  for (size_t i = 0; i < samples; ++i)
    padded[padding + i] = pre[i];
  frames = 1 + static_cast<int>(samples / hop);
  valid_frames = std::min(static_cast<int>(samples / hop), frames);
  features.assign(static_cast<size_t>(config.n_mels) * frames, 0.0f);
  std::vector<std::complex<double>> buffer(n_fft);
  std::vector<double> power(bins);
  for (int frame = 0; frame < frames; ++frame) {
    const size_t start = static_cast<size_t>(frame) * hop;
    for (int i = 0; i < n_fft; ++i) {
      buffer[i] = std::complex<double>(
          static_cast<double>(static_cast<float>(
              padded[start + i] * static_cast<double>(model.window[i]))),
          0.0);
    }
    rfft(buffer, n_fft);
    for (int bin = 0; bin < bins; ++bin) {
      const float re = static_cast<float>(buffer[bin].real());
      const float im = static_cast<float>(buffer[bin].imag());
      const double magnitude = std::sqrt(static_cast<double>(re) * re +
                                         static_cast<double>(im) * im);
      power[bin] = std::pow(magnitude, static_cast<double>(config.mag_power));
    }
    for (int mel = 0; mel < config.n_mels; ++mel) {
      double value = 0.0;
      const float *filter =
          model.filterbank.data() + static_cast<size_t>(mel) * bins;
      for (int bin = 0; bin < bins; ++bin)
        value += static_cast<double>(filter[bin]) * power[bin];
      features[static_cast<size_t>(mel) * frames + frame] =
          static_cast<float>(std::log(value + config.log_zero_guard));
    }
  }
  if (config.normalize == "per_feature" && valid_frames > 0) {
    const double divisor =
        valid_frames > 1 ? static_cast<double>(valid_frames - 1) : 1.0;
    for (int mel = 0; mel < config.n_mels; ++mel) {
      float *row = features.data() + static_cast<size_t>(mel) * frames;
      double mean = 0.0;
      for (int frame = 0; frame < valid_frames; ++frame)
        mean += row[frame];
      mean /= valid_frames;
      double variance = 0.0;
      for (int frame = 0; frame < valid_frames; ++frame) {
        const double delta = row[frame] - mean;
        variance += delta * delta;
      }
      const double scale = std::sqrt(variance / divisor) + kNormEps;
      for (int frame = 0; frame < frames; ++frame) {
        row[frame] = frame < valid_frames
                         ? static_cast<float>((row[frame] - mean) / scale)
                         : 0.0f;
      }
    }
  }
}

int subsample_length(int length) {
  for (int i = 0; i < 3; ++i)
    length = (length - 1) / 2 + 1;
  return length;
}

int valid_subsample_length(int length, int valid) {
  for (int i = 0; i < 3; ++i)
    valid = (valid - 1) / 2 + 1;
  return std::min(length, valid);
}

void read_batch_norm(Model &model, const std::string &prefix, int channels,
                     std::vector<float> &scale, std::vector<float> &shift) {
  std::vector<float> gamma, beta, mean, variance;
  read_tensor_f32(weight(model, prefix + "weight"), gamma);
  read_tensor_f32(weight(model, prefix + "bias"), beta);
  read_tensor_f32(weight(model, prefix + "running_mean"), mean);
  read_tensor_f32(weight(model, prefix + "running_var"), variance);
  require((int)gamma.size() == channels && (int)beta.size() == channels &&
              (int)mean.size() == channels && (int)variance.size() == channels,
          "batch norm shape mismatch");
  scale.resize(channels);
  shift.resize(channels);
  for (int i = 0; i < channels; ++i) {
    scale[i] = gamma[i] / std::sqrt(variance[i] + kNormEps);
    shift[i] = beta[i] - mean[i] * scale[i];
  }
}

ggml_tensor *build_subsampling(Model &model, Plan &plan, int frames,
                               int valid_frames) {
  const Config &config = model.config;
  const int channels = config.subsampling_conv_channels;
  ggml_tensor *x = plan.mel_input;
  ggml_tensor *first_weight = weight(model, "encoder.pre_encode.conv.0.weight");
  ggml_tensor *first_bias = weight(model, "encoder.pre_encode.conv.0.bias");
  x = ggml_conv_2d(plan.meta, first_weight, x, 2, 2, 1, 1, 1, 1);
  x = ggml_add(plan.meta, x,
               ggml_reshape_4d(plan.meta, first_bias, 1, 1, channels, 1));
  x = ggml_relu(plan.meta, x);
  const std::string names[2][4] = {
      {"encoder.pre_encode.conv.2.weight", "encoder.pre_encode.conv.2.bias",
       "encoder.pre_encode.conv.3.weight", "encoder.pre_encode.conv.3.bias"},
      {"encoder.pre_encode.conv.5.weight", "encoder.pre_encode.conv.5.bias",
       "encoder.pre_encode.conv.6.weight", "encoder.pre_encode.conv.6.bias"},
  };
  for (int stage = 0; stage < 2; ++stage) {
    ggml_tensor *depthwise = weight(model, names[stage][0]);
    ggml_tensor *depthwise_bias = weight(model, names[stage][1]);
    ggml_tensor *pointwise = weight(model, names[stage][2]);
    ggml_tensor *pointwise_bias = weight(model, names[stage][3]);
    x = ggml_conv_2d_dw_direct(plan.meta, depthwise, x, 2, 2, 1, 1, 1, 1);
    x = ggml_cont(plan.meta, x);
    x = ggml_add(plan.meta, x,
                 ggml_reshape_4d(plan.meta, depthwise_bias, 1, 1, channels, 1));
    x = ggml_conv_2d(plan.meta, pointwise, x, 1, 1, 0, 0, 1, 1);
    x = ggml_add(plan.meta, x,
                 ggml_reshape_4d(plan.meta, pointwise_bias, 1, 1, channels, 1));
    x = ggml_relu(plan.meta, x);
  }
  const int output_width = static_cast<int>(x->ne[0]);
  const int output_frames = static_cast<int>(x->ne[1]);
  plan.graph_frames = output_frames;
  ggml_tensor *flattened = ggml_reshape_2d(
      plan.meta, ggml_cont(plan.meta, ggml_permute(plan.meta, x, 0, 2, 1, 3)),
      static_cast<int64_t>(channels) * output_width, output_frames);
  const int valid_output = valid_subsample_length(output_frames, valid_frames);
  if (valid_output < output_frames) {
    std::vector<float> mask(output_frames, 0.0f);
    for (int i = 0; i < valid_output; ++i)
      mask[i] = 1.0f;
    ggml_tensor *mask_tensor = add_input(
        plan, GGML_TYPE_F32, {1, output_frames}, std::move(mask), true);
    flattened = ggml_mul(plan.meta, flattened, mask_tensor);
  }
  ggml_tensor *output_weight = weight(model, "encoder.pre_encode.out.weight");
  ggml_tensor *output_bias = weight(model, "encoder.pre_encode.out.bias");
  ggml_tensor *output = ggml_mul_mat(plan.meta, output_weight, flattened);
  output = ggml_add(plan.meta, output, output_bias);
  plan.encoder_frames = valid_output;
  require(output_frames == subsample_length(frames),
          "subsampling shape mismatch");
  return output;
}

void build_step_plan(Model &model) {
  const Config &config = model.config;
  const int hidden = config.pred_hidden;
  const int layers = config.pred_rnn_layers;
  const int joint = config.joint_hidden;
  const int token_count = config.vocab_size + 1;
  const int duration_count = static_cast<int>(config.durations.size());
  auto plan = std::make_unique<StepPlan>();
  plan->joint_hidden = joint;
  plan->pred_hidden = hidden;
  plan->layers = layers;
  plan->token_count = token_count;
  plan->duration_count = duration_count;
  const int input_size = joint + hidden + 2 * layers * hidden;
  const size_t metadata_size = ggml_tensor_overhead() * 4096 +
                               ggml_graph_overhead_custom(4096, false) +
                               8 * 1024 * 1024;
  ggml_init_params params{metadata_size, nullptr, true};
  plan->meta = ggml_init(params);
  require(plan->meta != nullptr, "decoder metadata allocation failed");
  int64_t input_shape[1] = {input_size};
  plan->input = ggml_new_tensor_1d(plan->meta, GGML_TYPE_F32, input_shape[0]);
  ggml_set_input(plan->input);
  plan->input_host.resize(input_size, 0.0f);
  auto view = [&](int offset, int length) {
    return ggml_view_1d(plan->meta, plan->input, length,
                        static_cast<size_t>(offset) * sizeof(float));
  };
  ggml_tensor *encoder_row = view(0, joint);
  ggml_tensor *embedding = view(joint, hidden);
  ggml_tensor *layer_input = embedding;
  std::vector<ggml_tensor *> hidden_nodes;
  std::vector<ggml_tensor *> cell_nodes;
  for (int layer = 0; layer < layers; ++layer) {
    const int state_offset = joint + hidden + 2 * layer * hidden;
    ggml_tensor *hidden_input = view(state_offset, hidden);
    ggml_tensor *cell_input = view(state_offset + hidden, hidden);
    const std::string suffix = "_l" + std::to_string(layer);
    ggml_tensor *input_weight =
        weight(model, "decoder.prediction.dec_rnn.lstm.weight_ih" + suffix);
    ggml_tensor *recurrent_weight =
        weight(model, "decoder.prediction.dec_rnn.lstm.weight_hh" + suffix);
    ggml_tensor *input_bias =
        weight(model, "decoder.prediction.dec_rnn.lstm.bias_ih" + suffix);
    ggml_tensor *recurrent_bias =
        weight(model, "decoder.prediction.dec_rnn.lstm.bias_hh" + suffix);
    ggml_tensor *gates = ggml_add(
        plan->meta,
        ggml_add(plan->meta,
                 ggml_mul_mat(plan->meta, input_weight, layer_input),
                 input_bias),
        ggml_add(plan->meta,
                 ggml_mul_mat(plan->meta, recurrent_weight, hidden_input),
                 recurrent_bias));
    ggml_tensor *input_gate =
        ggml_sigmoid(plan->meta, ggml_view_1d(plan->meta, gates, hidden, 0));
    ggml_tensor *forget_gate = ggml_sigmoid(
        plan->meta, ggml_view_1d(plan->meta, gates, hidden,
                                 static_cast<size_t>(hidden) * sizeof(float)));
    ggml_tensor *candidate =
        ggml_tanh(plan->meta, ggml_view_1d(plan->meta, gates, hidden,
                                           static_cast<size_t>(2 * hidden) *
                                               sizeof(float)));
    ggml_tensor *output_gate =
        ggml_sigmoid(plan->meta, ggml_view_1d(plan->meta, gates, hidden,
                                              static_cast<size_t>(3 * hidden) *
                                                  sizeof(float)));
    ggml_tensor *cell =
        ggml_add(plan->meta, ggml_mul(plan->meta, forget_gate, cell_input),
                 ggml_mul(plan->meta, input_gate, candidate));
    ggml_tensor *hidden_state =
        ggml_mul(plan->meta, output_gate, ggml_tanh(plan->meta, cell));
    hidden_nodes.push_back(hidden_state);
    cell_nodes.push_back(cell);
    layer_input = hidden_state;
  }
  auto concatenate = [&](const std::vector<ggml_tensor *> &values) {
    ggml_tensor *result = nullptr;
    for (ggml_tensor *value : values)
      result = result ? ggml_concat(plan->meta, result, value, 0) : value;
    return result;
  };
  plan->hidden_output = concatenate(hidden_nodes);
  plan->cell_output = concatenate(cell_nodes);
  ggml_tensor *predicted = ggml_add(
      plan->meta,
      ggml_mul_mat(plan->meta, weight(model, "joint.pred.weight"), layer_input),
      weight(model, "joint.pred.bias"));
  ggml_tensor *fused =
      ggml_relu(plan->meta, ggml_add(plan->meta, encoder_row, predicted));
  ggml_tensor *logits =
      ggml_add(plan->meta,
               ggml_mul_mat(plan->meta,
                            weight(model, "joint.joint_net.2.weight"), fused),
               weight(model, "joint.joint_net.2.bias"));
  plan->token_output =
      ggml_argmax(plan->meta, ggml_view_1d(plan->meta, logits, token_count, 0));
  plan->duration_output =
      ggml_argmax(plan->meta, ggml_view_1d(plan->meta, logits, duration_count,
                                           static_cast<size_t>(token_count) *
                                               sizeof(float)));
  ggml_set_output(plan->token_output);
  ggml_set_output(plan->duration_output);
  ggml_set_output(plan->hidden_output);
  ggml_set_output(plan->cell_output);
  plan->graph = ggml_new_graph_custom(plan->meta, 4096, false);
  ggml_build_forward_expand(plan->graph, plan->token_output);
  ggml_build_forward_expand(plan->graph, plan->duration_output);
  ggml_build_forward_expand(plan->graph, plan->hidden_output);
  ggml_build_forward_expand(plan->graph, plan->cell_output);
  plan->allocator =
      ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
  require(plan->allocator != nullptr, "decoder allocator allocation failed");
  require(ggml_gallocr_alloc_graph(plan->allocator, plan->graph),
          "decoder graph allocation failed");
  model.step_plan = std::move(plan);
}

void run_step(Model &model, StepPlan &plan, const float *encoder_row, int token,
              const std::vector<float> &hidden_state,
              const std::vector<float> &cell_state, int &next_token,
              int &next_duration, std::vector<float> &next_hidden,
              std::vector<float> &next_cell) {
  std::copy(encoder_row, encoder_row + plan.joint_hidden,
            plan.input_host.begin());
  if (token < 0) {
    std::fill(plan.input_host.begin() + plan.joint_hidden,
              plan.input_host.begin() + plan.joint_hidden + plan.pred_hidden,
              0.0f);
  } else {
    require(token < model.decoder.embedding_rows, "decoder token out of range");
    std::copy_n(model.decoder.embedding.data() +
                    static_cast<size_t>(token) * plan.pred_hidden,
                plan.pred_hidden, plan.input_host.begin() + plan.joint_hidden);
  }
  const int state_offset = plan.joint_hidden + plan.pred_hidden;
  for (int layer = 0; layer < plan.layers; ++layer) {
    const int offset = state_offset + 2 * layer * plan.pred_hidden;
    std::copy_n(hidden_state.data() +
                    static_cast<size_t>(layer) * plan.pred_hidden,
                plan.pred_hidden, plan.input_host.begin() + offset);
    std::copy_n(
        cell_state.data() + static_cast<size_t>(layer) * plan.pred_hidden,
        plan.pred_hidden, plan.input_host.begin() + offset + plan.pred_hidden);
  }
  require(plan.input->data != nullptr, "decoder input is not allocated");
  ggml_backend_tensor_set(plan.input, plan.input_host.data(), 0,
                          plan.input_host.size() * sizeof(float));
  require(ggml_backend_graph_compute(model.backend, plan.graph) ==
              GGML_STATUS_SUCCESS,
          "decoder graph execution failed");
  ggml_backend_tensor_get(plan.token_output, &next_token, 0, sizeof(int32_t));
  ggml_backend_tensor_get(plan.duration_output, &next_duration, 0,
                          sizeof(int32_t));
  next_hidden.resize(static_cast<size_t>(plan.layers) * plan.pred_hidden);
  next_cell.resize(next_hidden.size());
  ggml_backend_tensor_get(plan.hidden_output, next_hidden.data(), 0,
                          next_hidden.size() * sizeof(float));
  ggml_backend_tensor_get(plan.cell_output, next_cell.data(), 0,
                          next_cell.size() * sizeof(float));
}

cpu::QuantType custom_quant_type(ggml_type type) {
  switch (type) {
  case GGML_TYPE_F32:
    return cpu::QuantType::F32;
  case GGML_TYPE_Q4_K:
    return cpu::QuantType::Q4K;
  case GGML_TYPE_Q6_K:
    return cpu::QuantType::Q6K;
  case GGML_TYPE_Q8_0:
    return cpu::QuantType::Q8_0;
  default:
    throw std::runtime_error("custom step received unsupported weight type");
  }
}

cpu::LinearMatrix custom_matrix(ggml_tensor *weights, ggml_tensor *bias) {
  require(weights != nullptr && weights->data != nullptr,
          "custom step weight is missing");
  require(bias == nullptr || bias->data != nullptr,
          "custom step bias is missing");
  cpu::LinearMatrix matrix;
  matrix.data = weights->data;
  matrix.type = custom_quant_type(weights->type);
  matrix.input = static_cast<int>(weights->ne[0]);
  matrix.output = static_cast<int>(weights->ne[1]);
  matrix.row_bytes = weights->nb[1];
  if (bias)
    matrix.bias = static_cast<const float *>(bias->data);
  return matrix;
}

void custom_linear(const cpu::LinearMatrix &matrix, const float *input,
                   float *output, int threads) {
  if (matrix.type == cpu::QuantType::F32)
    cpu::linear_f32(matrix, input, output, 1, threads);
  else
    cpu::linear_q8(matrix, input, output, threads);
}

float sigmoid(float value) { return 1.0f / (1.0f + std::exp(-value)); }

void run_step_custom(Model &model, const float *encoder_row, int token,
                     const std::vector<float> &hidden_state,
                     const std::vector<float> &cell_state, int &next_token,
                     int &next_duration, std::vector<float> &next_hidden,
                     std::vector<float> &next_cell) {
  const Config &config = model.config;
  require(model.decoder.embedding_width == config.pred_hidden,
          "custom step embedding width mismatch");
  StepWorkspace &workspace = model.step_workspace;
  workspace.layer_input.resize(static_cast<std::size_t>(config.pred_hidden));
  std::fill(workspace.layer_input.begin(), workspace.layer_input.end(), 0.0f);
  if (token >= 0) {
    require(token < model.decoder.embedding_rows,
            "custom step token out of range");
    std::copy_n(model.decoder.embedding.data() +
                    static_cast<std::size_t>(token) * config.pred_hidden,
                config.pred_hidden, workspace.layer_input.data());
  }
  workspace.gates.resize(static_cast<std::size_t>(4 * config.pred_hidden));
  workspace.recurrent.resize(static_cast<std::size_t>(4 * config.pred_hidden));
  workspace.updated_hidden.resize(static_cast<std::size_t>(config.pred_hidden));
  workspace.updated_cell.resize(static_cast<std::size_t>(config.pred_hidden));
  for (int layer = 0; layer < config.pred_rnn_layers; ++layer) {
    const std::string suffix = "_l" + std::to_string(layer);
    const std::string prefix = "decoder.prediction.dec_rnn.lstm.";
    const cpu::LinearMatrix input_matrix =
        custom_matrix(weight(model, prefix + "weight_ih" + suffix),
                      weight(model, prefix + "bias_ih" + suffix));
    const cpu::LinearMatrix recurrent_matrix =
        custom_matrix(weight(model, prefix + "weight_hh" + suffix),
                      weight(model, prefix + "bias_hh" + suffix));
    const float *hidden = hidden_state.data() +
                          static_cast<std::size_t>(layer) * config.pred_hidden;
    const float *cell = cell_state.data() +
                        static_cast<std::size_t>(layer) * config.pred_hidden;
    custom_linear(input_matrix, workspace.layer_input.data(),
                  workspace.gates.data(), 1);
    custom_linear(recurrent_matrix, hidden, workspace.recurrent.data(), 1);
    for (int index = 0; index < 4 * config.pred_hidden; ++index)
      workspace.gates[index] += workspace.recurrent[index];
    std::vector<float> &updated_hidden = workspace.updated_hidden;
    std::vector<float> &updated_cell = workspace.updated_cell;
    for (int index = 0; index < config.pred_hidden; ++index) {
      const float input_gate = sigmoid(workspace.gates[index]);
      const float forget_gate =
          sigmoid(workspace.gates[index + config.pred_hidden]);
      const float candidate =
          std::tanh(workspace.gates[index + 2 * config.pred_hidden]);
      const float output_gate =
          sigmoid(workspace.gates[index + 3 * config.pred_hidden]);
      updated_cell[index] = forget_gate * cell[index] + input_gate * candidate;
      updated_hidden[index] = output_gate * std::tanh(updated_cell[index]);
    }
    std::copy(updated_hidden.begin(), updated_hidden.end(),
              next_hidden.begin() +
                  static_cast<std::size_t>(layer) * config.pred_hidden);
    std::copy(updated_cell.begin(), updated_cell.end(),
              next_cell.begin() +
                  static_cast<std::size_t>(layer) * config.pred_hidden);
    workspace.layer_input.swap(updated_hidden);
  }
  const cpu::LinearMatrix predicted_matrix = custom_matrix(
      weight(model, "joint.pred.weight"), weight(model, "joint.pred.bias"));
  workspace.predicted.resize(static_cast<std::size_t>(config.joint_hidden));
  custom_linear(predicted_matrix, workspace.layer_input.data(),
                workspace.predicted.data(), 1);
  workspace.fused.resize(static_cast<std::size_t>(config.joint_hidden));
  for (int index = 0; index < config.joint_hidden; ++index)
    workspace.fused[index] =
        std::max(0.0f, encoder_row[index] + workspace.predicted[index]);
  const cpu::LinearMatrix logits_matrix =
      custom_matrix(weight(model, "joint.joint_net.2.weight"),
                    weight(model, "joint.joint_net.2.bias"));
  workspace.logits.resize(static_cast<std::size_t>(config.vocab_size + 1 +
                                                   config.durations.size()));
  custom_linear(logits_matrix, workspace.fused.data(), workspace.logits.data(),
                1);
  const int token_count = config.vocab_size + 1;
  const auto token_end = workspace.logits.begin() + token_count;
  next_token = static_cast<int>(
      std::distance(workspace.logits.begin(),
                    std::max_element(workspace.logits.begin(), token_end)));
  const auto duration_begin = token_end;
  const auto duration_end = workspace.logits.end();
  next_duration = static_cast<int>(std::distance(
      duration_begin, std::max_element(duration_begin, duration_end)));
}

std::string decode_encoded(Model &model, const std::vector<float> &encoded,
                           int frames, std::uint64_t &steps) {
  const Config &config = model.config;
  require(model.step_plan != nullptr, "decoder plan is not built");
  StepPlan &plan = *model.step_plan;
  std::vector<float> hidden_state(
      static_cast<size_t>(plan.layers) * plan.pred_hidden, 0.0f);
  std::vector<float> cell_state(hidden_state.size(), 0.0f);
  std::vector<float> next_hidden(hidden_state.size(), 0.0f);
  std::vector<float> next_cell(hidden_state.size(), 0.0f);
  std::vector<int32_t> ids;
  int token = -1;
  bool emitted = false;
  int time = 0;
  while (time < frames) {
    int symbols = 0;
    int skip = 1;
    while (symbols < config.max_symbols && time < frames) {
      const float *encoder_row =
          encoded.data() + static_cast<size_t>(time) * plan.joint_hidden;
      int next_token = 0;
      int next_duration = 0;
      if (env_value("PARAKEET_POC_DEBUG_STEP") == "1" && steps == 0) {
        std::vector<float> reference_hidden(hidden_state.size());
        std::vector<float> reference_cell(cell_state.size());
        int reference_token = 0;
        int reference_duration = 0;
        run_step(model, plan, encoder_row, emitted ? token : -1, hidden_state,
                 cell_state, reference_token, reference_duration,
                 reference_hidden, reference_cell);
        run_step_custom(model, encoder_row, emitted ? token : -1, hidden_state,
                        cell_state, next_token, next_duration, next_hidden,
                        next_cell);
        float maximum_hidden = 0.0f;
        float maximum_cell = 0.0f;
        for (size_t index = 0; index < next_hidden.size(); ++index)
          maximum_hidden =
              std::max(maximum_hidden,
                       std::fabs(next_hidden[index] - reference_hidden[index]));
        for (size_t index = 0; index < next_cell.size(); ++index)
          maximum_cell =
              std::max(maximum_cell,
                       std::fabs(next_cell[index] - reference_cell[index]));
        std::cerr << "step reference " << reference_token << '/'
                  << reference_duration << " custom " << next_token << '/'
                  << next_duration << " hidden " << maximum_hidden << " cell "
                  << maximum_cell << '\n';
      }
      if (reference_step_mode()) {
        run_step(model, plan, encoder_row, emitted ? token : -1, hidden_state,
                 cell_state, next_token, next_duration, next_hidden, next_cell);
      } else {
        run_step_custom(model, encoder_row, emitted ? token : -1, hidden_state,
                        cell_state, next_token, next_duration, next_hidden,
                        next_cell);
      }
      ids.push_back(next_token);
      if (next_token != config.blank_id) {
        token = next_token;
        emitted = true;
        hidden_state.swap(next_hidden);
        cell_state.swap(next_cell);
      }
      skip = config.durations[next_duration];
      time += skip;
      ++symbols;
      if (skip != 0)
        break;
    }
    if (skip == 0)
      ++time;
    if (symbols == config.max_symbols)
      ++time;
  }
  std::string result;
  static const unsigned char meta_space[] = {0xe2, 0x96, 0x81};
  for (int32_t id : ids) {
    if (id < 0 || id >= static_cast<int32_t>(config.pieces.size()))
      continue;
    result += config.pieces[id];
  }
  std::string decoded;
  decoded.reserve(result.size());
  for (size_t i = 0; i < result.size();) {
    if (i + 3 <= result.size() &&
        static_cast<unsigned char>(result[i]) == meta_space[0] &&
        static_cast<unsigned char>(result[i + 1]) == meta_space[1] &&
        static_cast<unsigned char>(result[i + 2]) == meta_space[2]) {
      decoded.push_back(' ');
      i += 3;
    } else {
      decoded.push_back(result[i++]);
    }
  }
  if (!decoded.empty() && decoded.front() == ' ')
    decoded.erase(decoded.begin());
  steps = ids.size();
  return decoded;
}

void require_matrix_shape(Model &model, const std::string &name, int input,
                          int output) {
  ggml_tensor *tensor = weight(model, name);
  require(tensor->ne[0] == input && tensor->ne[1] == output &&
              tensor->ne[2] == 1 && tensor->ne[3] == 1,
          "matrix shape mismatch: " + name);
}

void load_decoder(Model &model) {
  const Config &config = model.config;
  DecoderWeights decoder;
  for (int layer = 0; layer < config.pred_rnn_layers; ++layer) {
    const std::string suffix = "_l" + std::to_string(layer);
    require_matrix_shape(model,
                         "decoder.prediction.dec_rnn.lstm.weight_ih" + suffix,
                         config.pred_hidden, 4 * config.pred_hidden);
    require_matrix_shape(model,
                         "decoder.prediction.dec_rnn.lstm.weight_hh" + suffix,
                         config.pred_hidden, 4 * config.pred_hidden);
    require_matrix_shape(model,
                         "decoder.prediction.dec_rnn.lstm.bias_ih" + suffix,
                         4 * config.pred_hidden, 1);
    require_matrix_shape(model,
                         "decoder.prediction.dec_rnn.lstm.bias_hh" + suffix,
                         4 * config.pred_hidden, 1);
  }
  require_matrix_shape(model, "joint.pred.weight", config.pred_hidden,
                       config.joint_hidden);
  require_matrix_shape(model, "joint.pred.bias", config.joint_hidden, 1);
  require_matrix_shape(model, "joint.joint_net.2.weight", config.joint_hidden,
                       config.vocab_size + 1 +
                           static_cast<int>(config.durations.size()));
  require_matrix_shape(
      model, "joint.joint_net.2.bias",
      config.vocab_size + 1 + static_cast<int>(config.durations.size()), 1);
  ggml_tensor *embedding = weight(model, "decoder.prediction.embed.weight");
  require(embedding->ne[0] == config.pred_hidden, "embedding width mismatch");
  decoder.embedding_width = static_cast<int>(embedding->ne[0]);
  decoder.embedding_rows = static_cast<int>(embedding->ne[1]);
  read_tensor_f32(embedding, decoder.embedding);
  const int needed_rows = config.vocab_size + 1;
  decoder.embedding.resize(
      static_cast<size_t>(needed_rows) * decoder.embedding_width, 0.0f);
  decoder.embedding_rows = needed_rows;
  model.decoder = std::move(decoder);
}

void load_model(Model &model, const char *path) {
  require(path != nullptr && *path != '\0', "empty model path");
  gguf_init_params params{false, &model.weights};
  model.gguf = gguf_init_from_file(path, params);
  require(model.gguf != nullptr, "failed to open GGUF");
  const int64_t tensor_count = gguf_get_n_tensors(model.gguf);
  require(tensor_count > 0, "GGUF has no tensors");
  for (int64_t i = 0; i < tensor_count; ++i) {
    const char *name = gguf_get_tensor_name(model.gguf, i);
    ggml_tensor *tensor = name ? ggml_get_tensor(model.weights, name) : nullptr;
    require(name != nullptr && tensor != nullptr, "failed to map GGUF tensor");
    model.tensors.emplace(name, tensor);
  }
  std::string architecture;
  require(kv_string(model.gguf, "general.architecture", architecture),
          "missing GGUF architecture");
  require(architecture == "parakeet_tdt", "GGUF is not parakeet_tdt");
  std::string profile;
  if (!test_mode()) {
    require(kv_string(model.gguf, "starling.numeric_profile", profile),
            "missing numeric profile");
    require(profile == "quantized" || profile == "f32_exact",
            "POC requires a quantized GGUF");
    bool has_quantized_linear = false;
    for (const auto &item : model.tensors) {
      const bool linear =
          item.first.find("linear") != std::string::npos ||
          item.first.find("pre_encode.out") != std::string::npos ||
          item.first.find("joint.enc") != std::string::npos;
      has_quantized_linear = has_quantized_linear ||
                             (linear && ggml_is_quantized(item.second->type));
    }
    require(has_quantized_linear, "POC requires quantized linear weights");
  }
  model.config.sample_rate =
      kv_int(model.gguf, "parakeet.preprocessor.sample_rate", kSampleRate);
  model.config.n_mels = kv_int(model.gguf, "parakeet.preprocessor.n_mels", 128);
  model.config.n_fft = kv_int(model.gguf, "parakeet.preprocessor.n_fft", 512);
  model.config.win_length =
      kv_int(model.gguf, "parakeet.preprocessor.win_length", 400);
  model.config.hop_length =
      kv_int(model.gguf, "parakeet.preprocessor.hop_length", 160);
  model.config.preemph =
      kv_float(model.gguf, "parakeet.preprocessor.preemph", kDefaultPreemph);
  model.config.mag_power =
      kv_float(model.gguf, "parakeet.preprocessor.mag_power", 2.0f);
  model.config.log_zero_guard = kv_float(
      model.gguf, "parakeet.preprocessor.log_zero_guard", kDefaultLogGuard);
  if (!kv_string(model.gguf, "parakeet.preprocessor.normalize",
                 model.config.normalize))
    model.config.normalize = "per_feature";
  model.config.d_model = kv_int(model.gguf, "parakeet.encoder.d_model", 0);
  model.config.n_layers = kv_int(model.gguf, "parakeet.encoder.n_layers", 0);
  model.config.pred_out = kv_int(model.gguf, "parakeet.encoder.pred_out", 0);
  model.config.n_heads = kv_int(model.gguf, "parakeet.encoder.n_heads", 0);
  model.config.ff_dim =
      kv_int(model.gguf, "parakeet.encoder.feedforward_dim", 0);
  if (model.config.ff_dim == 0)
    model.config.ff_dim = kv_int(model.gguf, "parakeet.encoder.ff_dim", 0);
  model.config.conv_kernel =
      kv_int(model.gguf, "parakeet.encoder.conv_kernel", 9);
  model.config.subsampling_conv_channels =
      kv_int(model.gguf, "parakeet.encoder.subsampling_conv_channels", 256);
  if (!kv_string(model.gguf, "parakeet.encoder.conv_norm_type",
                 model.config.conv_norm_type))
    model.config.conv_norm_type = "batch_norm";
  model.config.xscaling =
      kv_int(model.gguf, "parakeet.encoder.xscaling", 0) != 0;
  model.config.pred_hidden =
      kv_int(model.gguf, "parakeet.decoder.pred_hidden", 0);
  model.config.pred_rnn_layers =
      kv_int(model.gguf, "parakeet.decoder.pred_rnn_layers", 0);
  model.config.joint_hidden =
      kv_int(model.gguf, "parakeet.joint.joint_hidden", 0);
  model.config.max_symbols =
      kv_int(model.gguf, "parakeet.decoding.max_symbols", 10);
  model.config.vocab_size = kv_int(model.gguf, "parakeet.vocab_size", 0);
  model.config.blank_id = kv_int(model.gguf, "parakeet.blank_id", 0);
  model.config.durations = kv_int_array(model.gguf, "parakeet.tdt.durations");
  model.config.pieces =
      kv_string_array(model.gguf, "parakeet.tokenizer.pieces");
  require(model.config.sample_rate == kSampleRate, "POC requires 16 kHz audio");
  require(model.config.n_fft > 0 &&
              (model.config.n_fft & (model.config.n_fft - 1)) == 0,
          "n_fft must be a power of two");
  require(model.config.win_length > 0 &&
              model.config.win_length <= model.config.n_fft,
          "invalid window length");
  require(model.config.hop_length > 0 && model.config.n_mels > 0,
          "invalid frontend shape");
  require(model.config.d_model > 0 && model.config.n_layers > 0 &&
              model.config.n_heads > 0,
          "invalid encoder shape");
  require(model.config.d_model % model.config.n_heads == 0,
          "d_model must divide by head count");
  require(model.config.ff_dim > 0 && model.config.conv_kernel > 0 &&
              model.config.subsampling_conv_channels > 0,
          "invalid encoder shape");
  require(model.config.pred_hidden > 0 && model.config.pred_rnn_layers > 0 &&
              model.config.joint_hidden > 0,
          "invalid decoder shape");
  require(model.config.vocab_size > 0 &&
              model.config.blank_id == model.config.vocab_size,
          "invalid vocabulary contract");
  require(model.config.durations.size() > 0 &&
              model.config.pieces.size() >=
                  static_cast<size_t>(model.config.blank_id),
          "invalid tokenizer or durations");
  if (!test_mode()) {
    require(model.config.n_mels == 128 && model.config.n_fft == 512 &&
                model.config.win_length == 400 &&
                model.config.hop_length == 160,
            "POC is specialized for Parakeet v3 frontend");
    require(model.config.d_model == 1024 && model.config.n_layers == 24 &&
                model.config.n_heads == 8 && model.config.ff_dim == 4096,
            "POC is specialized for Parakeet v3 encoder");
    require(model.config.conv_kernel == 9 &&
                model.config.subsampling_conv_channels == 256,
            "POC is specialized for Parakeet v3 convolution stack");
    require(model.config.pred_hidden == 640 &&
                model.config.pred_rnn_layers == 2 &&
                model.config.joint_hidden == 640,
            "POC is specialized for Parakeet v3 decoder");
    require(model.config.vocab_size == 8192 && model.config.blank_id == 8192,
            "POC is specialized for Parakeet v3 vocabulary");
    require(model.config.durations == std::vector<int>({0, 1, 2, 3, 4}),
            "POC is specialized for Parakeet v3 durations");
  }
  std::vector<float> filterbank;
  read_tensor_f32(weight(model, "preprocessor.featurizer.fb"), filterbank);
  const int bins = model.config.n_fft / 2 + 1;
  require((int)filterbank.size() == model.config.n_mels * bins,
          "mel filterbank shape mismatch");
  std::vector<float> source_window;
  read_tensor_f32(weight(model, "preprocessor.featurizer.window"),
                  source_window);
  require((int)source_window.size() == model.config.win_length,
          "mel window shape mismatch");
  model.filterbank = std::move(filterbank);
  model.window.assign(model.config.n_fft, 0.0f);
  const int window_offset = (model.config.n_fft - model.config.win_length) / 2;
  std::copy(source_window.begin(), source_window.end(),
            model.window.begin() + window_offset);
  model.backend = ggml_backend_cpu_init();
  require(model.backend != nullptr, "failed to create CPU backend");
  const std::string thread_setting = env_value("PARAKEET_POC_THREADS");
  int threads = static_cast<int>(std::thread::hardware_concurrency());
  if (threads < 1)
    threads = 1;
  if (!thread_setting.empty())
    threads = std::max(1, std::atoi(thread_setting.c_str()));
  model.threads = threads;
  ggml_backend_cpu_set_n_threads(model.backend, threads);
  model.weight_buffer = ggml_backend_cpu_buffer_from_ptr(
      ggml_get_mem_buffer(model.weights), ggml_get_mem_size(model.weights));
  require(model.weight_buffer != nullptr, "failed to bind CPU weight buffer");
  for (const auto &item : model.tensors)
    item.second->buffer = model.weight_buffer;
  load_decoder(model);
  build_step_plan(model);
}

std::vector<float> relative_position(int frames, int model_dim) {
  const int positions = 2 * frames - 1;
  std::vector<float> result(static_cast<size_t>(positions) * model_dim, 0.0f);
  std::vector<double> divisor(model_dim / 2);
  const double factor = -std::log(10000.0) / model_dim;
  for (int i = 0; i < model_dim / 2; ++i)
    divisor[i] = std::exp(2.0 * i * factor);
  for (int position = 0; position < positions; ++position) {
    const double value = static_cast<double>(frames - 1 - position);
    float *row = result.data() + static_cast<size_t>(position) * model_dim;
    for (int i = 0; i < model_dim / 2; ++i) {
      row[2 * i] = static_cast<float>(std::sin(value * divisor[i]));
      row[2 * i + 1] = static_cast<float>(std::cos(value * divisor[i]));
    }
  }
  return result;
}

ggml_tensor *build_attention(Model &model, Plan &plan, int layer,
                             ggml_tensor *input, ggml_tensor *positional,
                             int frames, int valid_frames) {
  const Config &config = model.config;
  const int model_dim = config.d_model;
  const int heads = config.n_heads;
  const int head_dim = model_dim / heads;
  const std::string prefix =
      "encoder.layers." + std::to_string(layer) + ".self_attn.";
  auto to_heads = [&](ggml_tensor *value, int count) {
    value = ggml_reshape_3d(plan.meta, value, head_dim, heads, count);
    return ggml_cont(plan.meta, ggml_permute(plan.meta, value, 0, 2, 1, 3));
  };
  ggml_tensor *query = to_heads(
      linear(model, plan, plan.meta, input, prefix + "linear_q.weight", ""),
      frames);
  ggml_tensor *key = to_heads(
      linear(model, plan, plan.meta, input, prefix + "linear_k.weight", ""),
      frames);
  ggml_tensor *value = to_heads(
      linear(model, plan, plan.meta, input, prefix + "linear_v.weight", ""),
      frames);
  ggml_tensor *projected_position = linear(model, plan, plan.meta, positional,
                                           prefix + "linear_pos.weight", "");
  ggml_tensor *position_heads = to_heads(projected_position, 2 * frames - 1);
  ggml_tensor *bias_u = ggml_reshape_3d(
      plan.meta, weight(model, prefix + "pos_bias_u"), head_dim, 1, heads);
  ggml_tensor *bias_v = ggml_reshape_3d(
      plan.meta, weight(model, prefix + "pos_bias_v"), head_dim, 1, heads);
  ggml_tensor *query_u = ggml_add(plan.meta, query, bias_u);
  ggml_tensor *query_v = ggml_add(plan.meta, query, bias_v);
  ggml_tensor *bias = ggml_mul_mat(plan.meta, position_heads, query_v);
  bias = ggml_pad_ext(plan.meta, bias, 1, 0, 0, 0, 0, 0, 0, 0);
  bias = ggml_reshape_3d(plan.meta, bias, frames, 2 * frames, heads);
  bias = ggml_view_3d(plan.meta, bias, frames, 2 * frames - 1, heads,
                      bias->nb[1], bias->nb[2], bias->nb[1]);
  bias = ggml_cont(plan.meta, bias);
  bias = ggml_reshape_3d(plan.meta, bias, 2 * frames - 1, frames, heads);
  bias = ggml_view_3d(plan.meta, bias, frames, frames, heads, bias->nb[1],
                      bias->nb[2], 0);
  bias = ggml_cont(plan.meta, bias);
  ggml_tensor *scores = ggml_mul_mat(plan.meta, key, query_u);
  scores = ggml_add(plan.meta, scores, bias);
  ggml_tensor *mask = nullptr;
  if (valid_frames < frames) {
    std::vector<float> values(static_cast<size_t>(frames) * frames,
                              -std::numeric_limits<float>::infinity());
    for (int row = 0; row < frames; ++row) {
      for (int column = 0; column < valid_frames; ++column)
        values[static_cast<size_t>(row) * frames + column] = 0.0f;
    }
    mask = add_input(plan, GGML_TYPE_F32, {frames, frames}, std::move(values),
                     true);
  }
  ggml_tensor *attention =
      ggml_soft_max_ext(plan.meta, scores, mask,
                        1.0f / std::sqrt(static_cast<float>(head_dim)), 0.0f);
  ggml_tensor *value_transposed =
      ggml_cont(plan.meta, ggml_permute(plan.meta, value, 1, 0, 2, 3));
  ggml_tensor *context = ggml_mul_mat(plan.meta, value_transposed, attention);
  context = ggml_cont(plan.meta, ggml_permute(plan.meta, context, 0, 2, 1, 3));
  context = ggml_reshape_2d(plan.meta, context, model_dim, frames);
  if (valid_frames < frames) {
    std::vector<float> values(frames, 0.0f);
    for (int i = 0; i < valid_frames; ++i)
      values[i] = 1.0f;
    ggml_tensor *query_mask =
        add_input(plan, GGML_TYPE_F32, {1, frames}, std::move(values), true);
    context = ggml_mul(plan.meta, context, query_mask);
  }
  return linear(model, plan, plan.meta, context, prefix + "linear_out.weight",
                prefix + "linear_out.bias");
}

ggml_tensor *build_convolution(Model &model, Plan &plan, int layer,
                               ggml_tensor *input, int frames,
                               int valid_frames) {
  const Config &config = model.config;
  const int channels = config.d_model;
  const int kernel_size = config.conv_kernel;
  const std::string prefix =
      "encoder.layers." + std::to_string(layer) + ".conv.";
  ggml_tensor *first = weight(model, prefix + "pointwise_conv1.weight");
  if (first->type != GGML_TYPE_F16 && !ggml_is_quantized(first->type))
    first = ggml_cast(plan.meta, first, GGML_TYPE_F16);
  first = ggml_reshape_2d(plan.meta, first, channels, 2 * channels);
  ggml_tensor *output = ggml_mul_mat(plan.meta, first, input);
  ggml_tensor *first_bias = model.tensor(prefix + "pointwise_conv1.bias");
  if (first_bias)
    output = ggml_add(plan.meta, output, first_bias);
  ggml_tensor *left =
      ggml_view_2d(plan.meta, output, channels, frames, output->nb[1], 0);
  ggml_tensor *right =
      ggml_cont(plan.meta,
                ggml_view_2d(plan.meta, output, channels, frames, output->nb[1],
                             static_cast<size_t>(channels) * output->nb[0]));
  output = ggml_mul(plan.meta, left, ggml_sigmoid(plan.meta, right));
  if (valid_frames < frames) {
    std::vector<float> mask(frames, 0.0f);
    for (int i = 0; i < valid_frames; ++i)
      mask[i] = 1.0f;
    output = ggml_mul(
        plan.meta, output,
        add_input(plan, GGML_TYPE_F32, {1, frames}, std::move(mask), true));
  }
  ggml_tensor *depthwise = weight(model, prefix + "depthwise_conv.weight");
  if (ggml_is_quantized(depthwise->type)) {
    std::vector<float> values;
    read_tensor_f32(depthwise, values);
    depthwise = add_input(plan, GGML_TYPE_F32, {kernel_size, 1, 1, channels},
                          std::move(values), true);
  }
  depthwise =
      ggml_reshape_4d(plan.meta, depthwise, kernel_size, 1, 1, channels);
  ggml_tensor *transposed =
      ggml_cont(plan.meta, ggml_transpose(plan.meta, output));
  ggml_tensor *convolution_input =
      ggml_reshape_4d(plan.meta, transposed, frames, 1, channels, 1);
  output = ggml_conv_2d_dw_direct(plan.meta, depthwise, convolution_input, 1, 1,
                                  (kernel_size - 1) / 2, 0, 1, 1);
  output = ggml_reshape_2d(plan.meta, output, frames, channels);
  output = ggml_cont(plan.meta, ggml_transpose(plan.meta, output));
  ggml_tensor *depthwise_bias = model.tensor(prefix + "depthwise_conv.bias");
  if (depthwise_bias)
    output = ggml_add(plan.meta, output, depthwise_bias);
  std::vector<float> scale, shift;
  read_batch_norm(model, prefix + "batch_norm.", channels, scale, shift);
  output = ggml_add(
      plan.meta,
      ggml_mul(
          plan.meta, output,
          add_input(plan, GGML_TYPE_F32, {channels}, std::move(scale), true)),
      add_input(plan, GGML_TYPE_F32, {channels}, std::move(shift), true));
  output = ggml_silu(plan.meta, output);
  ggml_tensor *second = weight(model, prefix + "pointwise_conv2.weight");
  if (second->type != GGML_TYPE_F16 && !ggml_is_quantized(second->type))
    second = ggml_cast(plan.meta, second, GGML_TYPE_F16);
  second = ggml_reshape_2d(plan.meta, second, channels, channels);
  output = ggml_mul_mat(plan.meta, second, output);
  ggml_tensor *second_bias = model.tensor(prefix + "pointwise_conv2.bias");
  if (second_bias)
    output = ggml_add(plan.meta, output, second_bias);
  return output;
}

ggml_tensor *build_encoder(Model &model, Plan &plan) {
  const Config &config = model.config;
  const size_t metadata_size = ggml_tensor_overhead() * 20000 +
                               ggml_graph_overhead_custom(kGraphNodes, false) +
                               32 * 1024 * 1024;
  ggml_init_params params{metadata_size, nullptr, true};
  plan.meta = ggml_init(params);
  require(plan.meta != nullptr, "metadata context allocation failed");
  plan.mel_host.assign(static_cast<size_t>(config.n_mels) * plan.mel_frames,
                       0.0f);
  plan.mel_input =
      add_input(plan, GGML_TYPE_F32, {config.n_mels, plan.mel_frames, 1, 1},
                std::vector<float>(plan.mel_host.size(), 0.0f), false);
  ggml_tensor *x =
      build_subsampling(model, plan, plan.mel_frames, plan.valid_frames);
  const int graph_frames = plan.graph_frames;
  std::vector<float> positions =
      relative_position(graph_frames, config.d_model);
  ggml_tensor *positional =
      add_input(plan, GGML_TYPE_F32, {config.d_model, 2 * graph_frames - 1},
                std::move(positions), true);
  if (config.xscaling)
    x = ggml_scale(plan.meta, x, std::sqrt(static_cast<float>(config.d_model)));
  for (int layer = 0; layer < config.n_layers; ++layer) {
    const std::string prefix = "encoder.layers." + std::to_string(layer) + ".";
    ggml_tensor *first =
        layer_norm(model, plan.meta, x, prefix + "norm_feed_forward1");
    first =
        feed_forward(model, plan, plan.meta, first, prefix + "feed_forward1");
    x = ggml_add(plan.meta, x, ggml_scale(plan.meta, first, 0.5f));
    ggml_tensor *attention_input =
        layer_norm(model, plan.meta, x, prefix + "norm_self_att");
    x = ggml_add(plan.meta, x,
                 build_attention(model, plan, layer, attention_input,
                                 positional, graph_frames,
                                 plan.encoder_frames));
    ggml_tensor *convolution_input =
        layer_norm(model, plan.meta, x, prefix + "norm_conv");
    x = ggml_add(plan.meta, x,
                 build_convolution(model, plan, layer, convolution_input,
                                   graph_frames, plan.encoder_frames));
    ggml_tensor *second =
        layer_norm(model, plan.meta, x, prefix + "norm_feed_forward2");
    second =
        feed_forward(model, plan, plan.meta, second, prefix + "feed_forward2");
    x = ggml_add(plan.meta, x, ggml_scale(plan.meta, second, 0.5f));
    x = layer_norm(model, plan.meta, x, prefix + "norm_out");
  }
  x = ggml_add(plan.meta,
               ggml_mul_mat(plan.meta, weight(model, "joint.enc.weight"), x),
               weight(model, "joint.enc.bias"));
  plan.output = x;
  ggml_set_output(plan.output);
  plan.graph = ggml_new_graph_custom(plan.meta, kGraphNodes, false);
  ggml_build_forward_expand(plan.graph, plan.output);
  plan.allocator =
      ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
  require(plan.allocator != nullptr, "graph allocator allocation failed");
  require(ggml_gallocr_alloc_graph(plan.allocator, plan.graph),
          "graph allocation failed");
  for (HostInput *input : plan.input_order) {
    if (input->tensor == plan.mel_input)
      continue;
    if (input->tensor && input->tensor->data)
      ggml_backend_tensor_set(input->tensor, input->data.data(), 0,
                              input->data.size() * sizeof(float));
  }
  model.stats.encoder_plans++;
  return x;
}

std::unique_ptr<Plan> make_plan(Model &model, int frames, int valid_frames) {
  auto plan = std::make_unique<Plan>();
  plan->mel_frames = frames;
  plan->valid_frames = valid_frames;
  build_encoder(model, *plan);
  return plan;
}

void run_plan(Model &model, Plan &plan, const std::vector<float> &features,
              std::vector<float> &output) {
  require(plan.mel_input && plan.output, "plan is not built");
  require(plan.mel_host.size() == features.size(), "mel shape mismatch");
  const int frames = plan.mel_frames;
  const int features_per_frame = model.config.n_mels;
  for (int frame = 0; frame < frames; ++frame) {
    for (int feature = 0; feature < features_per_frame; ++feature) {
      plan.mel_host[static_cast<size_t>(frame) * features_per_frame + feature] =
          features[static_cast<size_t>(feature) * frames + frame];
    }
  }
  if (plan.mel_input->data)
    ggml_backend_tensor_set(plan.mel_input, plan.mel_host.data(), 0,
                            plan.mel_host.size() * sizeof(float));
  require(ggml_backend_graph_compute(model.backend, plan.graph) ==
              GGML_STATUS_SUCCESS,
          "CPU graph execution failed");
  output.resize(static_cast<size_t>(plan.encoder_frames) *
                model.config.joint_hidden);
  ggml_backend_tensor_get(plan.output, output.data(), 0,
                          output.size() * sizeof(float));
  if (model.stats.encoder_plans == 1) {
    for (float value : output) {
      if (!std::isfinite(value))
        throw std::runtime_error(
            "CPU graph produced non-finite encoder output");
    }
  }
}

std::vector<std::string> split_words(const std::string &text) {
  std::vector<std::string> words;
  size_t i = 0;
  while (i < text.size()) {
    while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i])))
      ++i;
    const size_t start = i;
    while (i < text.size() &&
           !std::isspace(static_cast<unsigned char>(text[i])))
      ++i;
    if (i > start)
      words.push_back(text.substr(start, i - start));
  }
  return words;
}

std::string normalize_word(std::string word) {
  std::string result;
  for (unsigned char value : word) {
    if (value >= 'A' && value <= 'Z')
      result.push_back(static_cast<char>(value - 'A' + 'a'));
    else if (std::isalnum(value) || value == '\'')
      result.push_back(static_cast<char>(value));
  }
  return result;
}

std::vector<std::string> stitch(const std::vector<std::string> &committed,
                                const std::vector<std::string> &incoming) {
  if (committed.empty())
    return incoming;
  if (incoming.empty())
    return committed;
  const size_t limit =
      std::min<size_t>(24, std::min(committed.size(), incoming.size()));
  size_t best_old = committed.size();
  size_t best_new = 0;
  for (size_t length = limit; length >= 2; --length) {
    bool match = true;
    for (size_t i = 0; i < length; ++i) {
      if (normalize_word(committed[committed.size() - length + i]) !=
          normalize_word(incoming[i])) {
        match = false;
        break;
      }
    }
    if (match) {
      best_old = committed.size() - length;
      best_new = length;
      break;
    }
  }
  std::vector<std::string> result = committed;
  result.resize(best_old);
  result.insert(result.end(), incoming.begin() + best_new, incoming.end());
  return result;
}

std::string join_words(const std::vector<std::string> &words) {
  std::string result;
  for (const std::string &word : words) {
    if (!result.empty())
      result.push_back(' ');
    result += word;
  }
  return result;
}

}

struct Engine::Impl {
  std::unique_ptr<Model> model;
};

Engine::Engine() : impl_(std::make_unique<Impl>()) {}
Engine::~Engine() = default;

bool Engine::load(const char *path, std::string &error) {
  error.clear();
  try {
    auto model = std::make_unique<Model>();
    load_model(*model, path);
    impl_->model = std::move(model);
    return true;
  } catch (const std::exception &exception) {
    error = exception.what();
    impl_->model.reset();
    return false;
  }
}

bool Engine::transcribe(const float *pcm, size_t samples, std::string &text,
                        std::string &error) {
  error.clear();
  text.clear();
  if (!impl_->model) {
    error = "engine is not loaded";
    return false;
  }
  if (!pcm || samples == 0) {
    error = "PCM input is empty";
    return false;
  }
  for (size_t i = 0; i < samples; ++i) {
    if (!std::isfinite(pcm[i])) {
      error = "PCM input contains a non-finite sample";
      return false;
    }
  }
  try {
    Model &model = *impl_->model;
    ++model.stats.transcribe_calls;
    if (model.cache_valid && model.cached_pcm.size() == samples &&
        std::memcmp(model.cached_pcm.data(), pcm, samples * sizeof(float)) ==
            0) {
      ++model.stats.cache_hits;
      text = model.cached_text;
      model.stats.last_mel_ms = 0.0;
      model.stats.last_encoder_ms = 0.0;
      model.stats.last_decode_ms = 0.0;
      model.stats.last_total_ms = 0.0;
      return true;
    }
    const auto total_start = std::chrono::steady_clock::now();
    std::vector<float> features;
    int frames = 0;
    int valid_frames = 0;
    const auto mel_start = std::chrono::steady_clock::now();
    compute_mel(model, pcm, samples, features, frames, valid_frames);
    const auto encoder_start = std::chrono::steady_clock::now();
    const auto key = std::make_pair(frames, valid_frames);
    Plan *plan = model.find_plan(key);
    if (plan == nullptr) {
      auto built = make_plan(model, frames, valid_frames);
      plan = model.install_plan(key, std::move(built));
    }
    std::vector<float> encoded;
    run_plan(model, *plan, features, encoded);
    const auto decode_start = std::chrono::steady_clock::now();
    std::uint64_t steps = 0;
    text = decode_encoded(model, encoded, plan->encoder_frames, steps);
    const auto total_end = std::chrono::steady_clock::now();
    model.stats.last_mel_ms =
        std::chrono::duration<double, std::milli>(encoder_start - mel_start)
            .count();
    model.stats.last_encoder_ms =
        std::chrono::duration<double, std::milli>(decode_start - encoder_start)
            .count();
    model.stats.last_decode_ms =
        std::chrono::duration<double, std::milli>(total_end - decode_start)
            .count();
    model.stats.last_total_ms =
        std::chrono::duration<double, std::milli>(total_end - total_start)
            .count();
    model.stats.decode_steps += steps;
    model.cached_pcm.assign(pcm, pcm + samples);
    model.cached_text = text;
    model.cache_valid = true;
    return true;
  } catch (const std::exception &exception) {
    error = exception.what();
    return false;
  }
}

void Engine::reset() {
  if (!impl_->model)
    return;
  impl_->model->cached_pcm.clear();
  impl_->model->cached_text.clear();
  impl_->model->cache_valid = false;
}

Stats Engine::stats() const {
  return impl_->model ? impl_->model->stats : Stats{};
}

struct Stream::Impl {
  Engine *engine = nullptr;
  StreamConfig config;
  std::vector<float> pending;
  std::vector<std::string> committed;
  std::string error;
};

Stream::Stream(Engine &engine, StreamConfig config)
    : impl_(std::make_unique<Impl>()) {
  impl_->engine = &engine;
  impl_->config = config;
  if (config.sample_rate != kSampleRate || config.window_samples == 0 ||
      config.overlap_samples >= config.window_samples ||
      config.max_buffer_samples < config.window_samples) {
    impl_->error = "invalid Parakeet stream window configuration";
  }
}

Stream::~Stream() = default;

std::optional<std::string> Stream::append(const float *pcm, size_t samples,
                                          std::string &error) {
  error.clear();
  if (!impl_->error.empty()) {
    error = impl_->error;
    return std::nullopt;
  }
  if (samples && !pcm) {
    error = "PCM input is null";
    return std::nullopt;
  }
  if (impl_->pending.size() + samples > impl_->config.max_buffer_samples) {
    error = "Parakeet stream buffer limit exceeded";
    return std::nullopt;
  }
  if (samples)
    impl_->pending.insert(impl_->pending.end(), pcm, pcm + samples);
  bool changed = false;
  while (impl_->pending.size() >= impl_->config.window_samples) {
    std::string text;
    std::string transcribe_error;
    if (!impl_->engine->transcribe(impl_->pending.data(),
                                   impl_->config.window_samples, text,
                                   transcribe_error)) {
      error = transcribe_error;
      return std::nullopt;
    }
    impl_->committed = stitch(impl_->committed, split_words(text));
    const size_t advance =
        impl_->config.window_samples - impl_->config.overlap_samples;
    impl_->pending.erase(impl_->pending.begin(),
                         impl_->pending.begin() + advance);
    changed = true;
  }
  if (!changed)
    return std::nullopt;
  return join_words(impl_->committed);
}

std::optional<std::string> Stream::finish(std::string &error) {
  error.clear();
  if (!impl_->error.empty()) {
    error = impl_->error;
    return std::nullopt;
  }
  if (!impl_->pending.empty()) {
    std::string text;
    if (!impl_->engine->transcribe(impl_->pending.data(), impl_->pending.size(),
                                   text, error))
      return std::nullopt;
    impl_->committed = stitch(impl_->committed, split_words(text));
    impl_->pending.clear();
  }
  return join_words(impl_->committed);
}

void Stream::reset() {
  impl_->pending.clear();
  impl_->committed.clear();
  impl_->error.clear();
  if (impl_->config.sample_rate != kSampleRate ||
      impl_->config.window_samples == 0 ||
      impl_->config.overlap_samples >= impl_->config.window_samples ||
      impl_->config.max_buffer_samples < impl_->config.window_samples) {
    impl_->error = "invalid Parakeet stream window configuration";
  }
}

}
