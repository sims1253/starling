#include "cpu_kernels.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(PARAKEET_HAVE_CALLGRIND)
#include <valgrind/callgrind.h>
#else
#define CALLGRIND_ZERO_STATS                                                   \
  do {                                                                         \
  } while (0)
#define CALLGRIND_START_INSTRUMENTATION                                        \
  do {                                                                         \
  } while (0)
#define CALLGRIND_STOP_INSTRUMENTATION                                         \
  do {                                                                         \
  } while (0)
#endif

namespace {

using namespace parakeet_poc::cpu;

void check(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

struct Model {
  gguf_context *gguf = nullptr;
  ggml_context *weights = nullptr;
  ggml_backend_t backend = nullptr;
  ggml_backend_buffer_t buffer = nullptr;

  ~Model() {
    if (buffer)
      ggml_backend_buffer_free(buffer);
    if (backend)
      ggml_backend_free(backend);
    if (gguf)
      gguf_free(gguf);
    if (weights)
      ggml_free(weights);
  }
};

QuantType type_of(ggml_tensor *tensor) {
  switch (tensor->type) {
  case GGML_TYPE_Q4_K:
    return QuantType::Q4K;
  case GGML_TYPE_Q6_K:
    return QuantType::Q6K;
  case GGML_TYPE_Q8_0:
    return QuantType::Q8_0;
  case GGML_TYPE_F32:
    return QuantType::F32;
  default:
    throw std::runtime_error("unsupported benchmark tensor type");
  }
}

void load(Model &model, const char *path, int threads) {
  gguf_init_params params{false, &model.weights};
  model.gguf = gguf_init_from_file(path, params);
  check(model.gguf != nullptr, "failed to open GGUF");
  model.backend = ggml_backend_cpu_init();
  check(model.backend != nullptr, "failed to create CPU backend");
  ggml_backend_cpu_set_n_threads(model.backend, threads);
  model.buffer = ggml_backend_cpu_buffer_from_ptr(
      ggml_get_mem_buffer(model.weights), ggml_get_mem_size(model.weights));
  check(model.buffer != nullptr, "failed to bind GGUF buffer");
}

struct Graph {
  ggml_context *context = nullptr;
  ggml_cgraph *graph = nullptr;
  ggml_gallocr_t allocator = nullptr;
  ggml_tensor *input = nullptr;
  ggml_tensor *output = nullptr;

  Graph() = default;
  Graph(const Graph &) = delete;
  Graph &operator=(const Graph &) = delete;
  Graph(Graph &&other) noexcept
      : context(other.context), graph(other.graph), allocator(other.allocator),
        input(other.input), output(other.output) {
    other.context = nullptr;
    other.graph = nullptr;
    other.allocator = nullptr;
    other.input = nullptr;
    other.output = nullptr;
  }
  Graph &operator=(Graph &&other) noexcept {
    if (this != &other) {
      if (allocator)
        ggml_gallocr_free(allocator);
      if (context)
        ggml_free(context);
      context = other.context;
      graph = other.graph;
      allocator = other.allocator;
      input = other.input;
      output = other.output;
      other.context = nullptr;
      other.graph = nullptr;
      other.allocator = nullptr;
      other.input = nullptr;
      other.output = nullptr;
    }
    return *this;
  }
  ~Graph() {
    if (allocator)
      ggml_gallocr_free(allocator);
    if (context)
      ggml_free(context);
  }
};

Graph make_graph(Model &model, ggml_tensor *weight, ggml_tensor *bias,
                 int input_width, int samples) {
  Graph result;
  ggml_init_params params{8 * 1024 * 1024, nullptr, true};
  result.context = ggml_init(params);
  check(result.context != nullptr, "failed to create benchmark context");
  result.input =
      ggml_new_tensor_2d(result.context, GGML_TYPE_F32, input_width, samples);
  ggml_set_input(result.input);
  result.output = ggml_mul_mat(result.context, weight, result.input);
  if (bias)
    result.output = ggml_add(result.context, result.output, bias);
  result.graph = ggml_new_graph_custom(result.context, 64, false);
  ggml_build_forward_expand(result.graph, result.output);
  result.allocator =
      ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
  check(result.allocator != nullptr, "failed to create graph allocator");
  check(ggml_gallocr_alloc_graph(result.allocator, result.graph),
        "failed to allocate benchmark graph");
  return result;
}

void run_ggml(Model &model, Graph &graph, int iterations) {
  for (int index = 0; index < iterations; ++index)
    check(ggml_backend_graph_compute(model.backend, graph.graph) ==
              GGML_STATUS_SUCCESS,
          "ggml benchmark compute failed");
}

void run_custom(const LinearMatrix &matrix, const float *input, float *output,
                int samples, int threads, int iterations) {
  const char *mode = std::getenv("PARAKEET_KERNEL_MODE");
  const bool single = mode != nullptr && std::string(mode) == "single";
  const bool raw = mode != nullptr && std::string(mode) == "raw";
  const bool exact = mode != nullptr && std::string(mode) == "exact";
  for (int index = 0; index < iterations; ++index) {
    if (matrix.type == QuantType::F32) {
      linear_f32(matrix, input, output, samples, threads);
    } else if (exact) {
      linear_q8_exact(matrix, input, output, samples, threads);
    } else if (single) {
      for (int sample = 0; sample < samples; ++sample)
        linear_q8_range(
            matrix, input + static_cast<std::size_t>(sample) * matrix.input,
            output + static_cast<std::size_t>(sample) * matrix.output, 1, 0,
            matrix.output);
    } else {
      if (raw) {
        LinearMatrix unpacked = matrix;
        unpacked.packed = nullptr;
        linear_q8_batch(unpacked, input, output, samples, threads);
      } else {
        linear_q8_batch(matrix, input, output, samples, threads);
      }
    }
  }
}

double time_ggml(Model &model, Graph &graph, int iterations) {
  const auto start = std::chrono::steady_clock::now();
  run_ggml(model, graph, iterations);
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count() /
         iterations;
}

double time_custom(const LinearMatrix &matrix, const float *input,
                   float *output, int samples, int threads, int iterations) {
  const auto start = std::chrono::steady_clock::now();
  run_custom(matrix, input, output, samples, threads, iterations);
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count() /
         iterations;
}

std::string value_arg(int argc, char **argv, int &index, const char *name) {
  check(index + 1 < argc, std::string("missing value for ") + name);
  return argv[++index];
}

}

int main(int argc, char **argv) {
  try {
    if (argc < 3) {
      std::cerr << "usage: parakeet_kernel_bench MODEL TENSOR "
                   "[--iterations N] [--threads N] [--samples N] "
                   "[--warmup N] [--kernel both|custom|ggml] [--callgrind]\n";
      return 2;
    }
    int iterations = 20;
    int threads = 2;
    int samples = 744;
    int warmup = 2;
    std::string kernel = "both";
    bool callgrind = false;
    for (int index = 3; index < argc; ++index) {
      const std::string argument = argv[index];
      if (argument == "--iterations")
        iterations = std::max(
            1, std::atoi(value_arg(argc, argv, index, "iterations").c_str()));
      else if (argument == "--threads")
        threads = std::max(
            1, std::atoi(value_arg(argc, argv, index, "threads").c_str()));
      else if (argument == "--samples")
        samples = std::max(
            1, std::atoi(value_arg(argc, argv, index, "samples").c_str()));
      else if (argument == "--warmup")
        warmup = std::max(
            0, std::atoi(value_arg(argc, argv, index, "warmup").c_str()));
      else if (argument == "--kernel")
        kernel = value_arg(argc, argv, index, "kernel");
      else if (argument == "--callgrind")
        callgrind = true;
      else
        throw std::runtime_error("unknown benchmark option: " + argument);
    }
    check(kernel == "both" || kernel == "custom" || kernel == "ggml",
          "kernel must be both, custom, or ggml");
    if (callgrind && kernel == "both")
      kernel = "custom";

    Model model;
    load(model, argv[1], threads);
    ggml_tensor *weight = ggml_get_tensor(model.weights, argv[2]);
    check(weight != nullptr, "benchmark tensor not found");
    const std::string name = argv[2];
    const std::string weight_suffix = ".weight";
    ggml_tensor *bias = nullptr;
    if (name.size() > weight_suffix.size() &&
        name.compare(name.size() - weight_suffix.size(), weight_suffix.size(),
                     weight_suffix) == 0) {
      const std::string bias_name =
          name.substr(0, name.size() - weight_suffix.size()) + ".bias";
      bias = ggml_get_tensor(model.weights, bias_name.c_str());
    }
    const int input_width = static_cast<int>(weight->ne[0]);
    const int output_width = static_cast<int>(weight->ne[1]);
    std::vector<float> input(static_cast<std::size_t>(input_width) * samples);
    for (int sample = 0; sample < samples; ++sample)
      for (int index = 0; index < input_width; ++index)
        input[static_cast<std::size_t>(sample) * input_width + index] =
            std::sin(static_cast<float>(sample * input_width + index) *
                     0.013f) *
            0.25f;

    Graph graph = make_graph(model, weight, bias, input_width, samples);
    check(graph.input->data != nullptr && graph.output->data != nullptr,
          "benchmark graph has no buffers");
    ggml_backend_tensor_set(graph.input, input.data(), 0,
                            input.size() * sizeof(float));
    LinearMatrix matrix{weight->data,
                        bias ? static_cast<const float *>(bias->data) : nullptr,
                        type_of(weight),
                        input_width,
                        output_width,
                        weight->nb[1],
                        nullptr};
    matrix.packed = pack_matrix(matrix);
    std::vector<float> reference_output(static_cast<std::size_t>(output_width) *
                                        samples);
    std::vector<float> custom_output(reference_output.size());
    run_ggml(model, graph, 1);
    ggml_backend_tensor_get(graph.output, reference_output.data(), 0,
                            reference_output.size() * sizeof(float));
    run_custom(matrix, input.data(), custom_output.data(), samples, threads, 1);
    float maximum_error = 0.0f;
    for (size_t index = 0; index < reference_output.size(); ++index)
      maximum_error =
          std::max(maximum_error,
                   std::fabs(custom_output[index] - reference_output[index]));

    if (kernel == "ggml" || kernel == "both") {
      run_ggml(model, graph, warmup);
      run_custom(matrix, input.data(), custom_output.data(), samples, threads,
                 warmup);
    }
    if (kernel == "custom" || kernel == "both") {
      run_custom(matrix, input.data(), custom_output.data(), samples, threads,
                 warmup);
      run_ggml(model, graph, warmup);
    }

    double ggml_ms = -1.0;
    double custom_ms = -1.0;
    if (callgrind) {
      CALLGRIND_ZERO_STATS;
      CALLGRIND_START_INSTRUMENTATION;
      if (kernel == "ggml")
        run_ggml(model, graph, iterations);
      else
        run_custom(matrix, input.data(), custom_output.data(), samples, threads,
                   iterations);
      CALLGRIND_STOP_INSTRUMENTATION;
    } else {
      if (kernel == "ggml" || kernel == "both")
        ggml_ms = time_ggml(model, graph, iterations);
      if (kernel == "custom" || kernel == "both")
        custom_ms = time_custom(matrix, input.data(), custom_output.data(),
                                samples, threads, iterations);
    }

    std::cout << "tensor=" << argv[2]
              << " type=" << static_cast<int>(matrix.type)
              << " in=" << input_width << " out=" << output_width
              << " samples=" << samples << " threads=" << threads
              << " iterations=" << iterations << " warmup=" << warmup
              << " max_abs=" << maximum_error
              << " packed=" << (matrix.packed ? 1 : 0) << " ggml_ms=" << ggml_ms
              << " custom_ms=" << custom_ms
              << " speedup=" << (custom_ms > 0.0 ? ggml_ms / custom_ms : 0.0)
              << " callgrind=" << (callgrind ? 1 : 0)
              << " backend=" << backend_name() << '\n';
    return 0;
  } catch (const std::exception &exception) {
    std::cerr << exception.what() << '\n';
    return 1;
  }
}
