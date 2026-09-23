#include "engine.hpp"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

struct TensorWriter {
  ggml_context *context = nullptr;
  gguf_context *output = nullptr;

  TensorWriter() {
    ggml_init_params params{8 * 1024 * 1024, nullptr, false};
    context = ggml_init(params);
    output = gguf_init_empty();
    check(context != nullptr && output != nullptr, "fixture context");
  }

  ~TensorWriter() {
    if (output)
      gguf_free(output);
    if (context)
      ggml_free(context);
  }

  void tensor(const std::string &name, const std::vector<int64_t> &shape,
              const std::vector<float> &values) {
    size_t count = 1;
    for (int64_t value : shape)
      count *= static_cast<size_t>(value);
    check(values.size() == count, "fixture tensor size");
    ggml_tensor *tensor = nullptr;
    if (shape.size() == 1)
      tensor = ggml_new_tensor_1d(context, GGML_TYPE_F32, shape[0]);
    else if (shape.size() == 2)
      tensor = ggml_new_tensor_2d(context, GGML_TYPE_F32, shape[0], shape[1]);
    else if (shape.size() == 3)
      tensor = ggml_new_tensor_3d(context, GGML_TYPE_F32, shape[0], shape[1],
                                  shape[2]);
    else
      tensor = ggml_new_tensor_4d(context, GGML_TYPE_F32, shape[0], shape[1],
                                  shape[2], shape[3]);
    check(tensor != nullptr, "fixture tensor");
    ggml_set_name(tensor, name.c_str());
    std::copy(values.begin(), values.end(), static_cast<float *>(tensor->data));
    gguf_add_tensor(output, tensor);
  }

  void matrix(const std::string &name, int input, int output_dim,
              float value = 0.0f) {
    tensor(name, {input, output_dim},
           std::vector<float>(static_cast<size_t>(input) * output_dim, value));
  }

  void vector1(const std::string &name, int count, float value = 0.0f) {
    tensor(name, {count}, std::vector<float>(count, value));
  }
};

std::filesystem::path make_fixture() {
  const auto path =
      std::filesystem::temp_directory_path() /
      ("starling-parakeet-poc-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()) +
       ".gguf");
  TensorWriter writer;
  gguf_set_val_str(writer.output, "general.architecture", "parakeet_tdt");
  gguf_set_val_u32(writer.output, "starling.format_version", 1);
  gguf_set_val_u32(writer.output, "parakeet.preprocessor.sample_rate", 16000);
  gguf_set_val_u32(writer.output, "parakeet.preprocessor.n_mels", 4);
  gguf_set_val_u32(writer.output, "parakeet.preprocessor.n_fft", 8);
  gguf_set_val_u32(writer.output, "parakeet.preprocessor.win_length", 8);
  gguf_set_val_u32(writer.output, "parakeet.preprocessor.hop_length", 2);
  gguf_set_val_f32(writer.output, "parakeet.preprocessor.preemph", 0.97f);
  gguf_set_val_f32(writer.output, "parakeet.preprocessor.mag_power", 2.0f);
  gguf_set_val_f32(writer.output, "parakeet.preprocessor.log_zero_guard",
                   5.9604645e-08f);
  gguf_set_val_str(writer.output, "parakeet.preprocessor.normalize",
                   "per_feature");
  gguf_set_val_u32(writer.output, "parakeet.encoder.d_model", 8);
  gguf_set_val_u32(writer.output, "parakeet.encoder.n_layers", 1);
  gguf_set_val_u32(writer.output, "parakeet.encoder.pred_out", 8);
  gguf_set_val_u32(writer.output, "parakeet.encoder.n_heads", 2);
  gguf_set_val_u32(writer.output, "parakeet.encoder.feedforward_dim", 16);
  gguf_set_val_u32(writer.output, "parakeet.encoder.conv_kernel", 3);
  gguf_set_val_u32(writer.output, "parakeet.encoder.subsampling_conv_channels",
                   4);
  gguf_set_val_str(writer.output, "parakeet.encoder.conv_norm_type",
                   "batch_norm");
  gguf_set_val_u32(writer.output, "parakeet.encoder.xscaling", 0);
  gguf_set_val_u32(writer.output, "parakeet.decoder.pred_hidden", 8);
  gguf_set_val_u32(writer.output, "parakeet.decoder.pred_rnn_layers", 1);
  gguf_set_val_u32(writer.output, "parakeet.joint.joint_hidden", 8);
  gguf_set_val_u32(writer.output, "parakeet.decoding.max_symbols", 2);
  gguf_set_val_u32(writer.output, "parakeet.vocab_size", 3);
  gguf_set_val_u32(writer.output, "parakeet.blank_id", 3);
  const int32_t durations[] = {0, 1};
  gguf_set_arr_data(writer.output, "parakeet.tdt.durations", GGUF_TYPE_INT32,
                    durations, 2);
  const char *pieces[] = {"a", "b", "c"};
  gguf_set_arr_str(writer.output, "parakeet.tokenizer.pieces", pieces, 3);

  writer.tensor("preprocessor.featurizer.fb", {5, 4},
                std::vector<float>(20, 1.0f));
  writer.vector1("preprocessor.featurizer.window", 8, 1.0f);
  writer.tensor("encoder.pre_encode.conv.0.weight", {3, 3, 1, 4},
                std::vector<float>(36, 0.01f));
  writer.vector1("encoder.pre_encode.conv.0.bias", 4, 0.01f);
  writer.tensor("encoder.pre_encode.conv.2.weight", {3, 3, 1, 4},
                std::vector<float>(36, 0.01f));
  writer.vector1("encoder.pre_encode.conv.2.bias", 4, 0.01f);
  writer.tensor("encoder.pre_encode.conv.3.weight", {1, 1, 4, 4},
                std::vector<float>(16, 0.01f));
  writer.vector1("encoder.pre_encode.conv.3.bias", 4, 0.01f);
  writer.tensor("encoder.pre_encode.conv.5.weight", {3, 3, 1, 4},
                std::vector<float>(36, 0.01f));
  writer.vector1("encoder.pre_encode.conv.5.bias", 4, 0.01f);
  writer.tensor("encoder.pre_encode.conv.6.weight", {1, 1, 4, 4},
                std::vector<float>(16, 0.01f));
  writer.vector1("encoder.pre_encode.conv.6.bias", 4, 0.01f);
  writer.matrix("encoder.pre_encode.out.weight", 4, 8, 0.01f);
  writer.vector1("encoder.pre_encode.out.bias", 8, 0.01f);

  writer.matrix("encoder.layers.0.self_attn.linear_q.weight", 8, 8, 0.01f);
  writer.matrix("encoder.layers.0.self_attn.linear_k.weight", 8, 8, 0.01f);
  writer.matrix("encoder.layers.0.self_attn.linear_v.weight", 8, 8, 0.01f);
  writer.matrix("encoder.layers.0.self_attn.linear_pos.weight", 8, 8, 0.01f);
  writer.tensor("encoder.layers.0.self_attn.pos_bias_u", {4, 2},
                std::vector<float>(8, 0.01f));
  writer.tensor("encoder.layers.0.self_attn.pos_bias_v", {4, 2},
                std::vector<float>(8, 0.01f));
  writer.matrix("encoder.layers.0.self_attn.linear_out.weight", 8, 8, 0.01f);
  writer.vector1("encoder.layers.0.norm_feed_forward1.weight", 8, 1.0f);
  writer.vector1("encoder.layers.0.norm_feed_forward1.bias", 8, 0.0f);
  writer.matrix("encoder.layers.0.feed_forward1.linear1.weight", 8, 16, 0.01f);
  writer.vector1("encoder.layers.0.feed_forward1.linear1.bias", 16, 0.01f);
  writer.matrix("encoder.layers.0.feed_forward1.linear2.weight", 16, 8, 0.01f);
  writer.vector1("encoder.layers.0.feed_forward1.linear2.bias", 8, 0.01f);
  writer.vector1("encoder.layers.0.norm_self_att.weight", 8, 1.0f);
  writer.vector1("encoder.layers.0.norm_self_att.bias", 8, 0.0f);
  writer.vector1("encoder.layers.0.norm_conv.weight", 8, 1.0f);
  writer.vector1("encoder.layers.0.norm_conv.bias", 8, 0.0f);
  writer.tensor("encoder.layers.0.conv.pointwise_conv1.weight", {1, 8, 16},
                std::vector<float>(128, 0.01f));
  writer.vector1("encoder.layers.0.conv.pointwise_conv1.bias", 16, 0.01f);
  writer.tensor("encoder.layers.0.conv.depthwise_conv.weight", {3, 1, 1, 8},
                std::vector<float>(24, 0.01f));
  writer.vector1("encoder.layers.0.conv.depthwise_conv.bias", 8, 0.01f);
  writer.vector1("encoder.layers.0.conv.batch_norm.weight", 8, 1.0f);
  writer.vector1("encoder.layers.0.conv.batch_norm.bias", 8, 0.0f);
  writer.vector1("encoder.layers.0.conv.batch_norm.running_mean", 8, 0.0f);
  writer.vector1("encoder.layers.0.conv.batch_norm.running_var", 8, 1.0f);
  writer.tensor("encoder.layers.0.conv.pointwise_conv2.weight", {1, 8, 8},
                std::vector<float>(64, 0.01f));
  writer.vector1("encoder.layers.0.conv.pointwise_conv2.bias", 8, 0.01f);
  writer.vector1("encoder.layers.0.norm_feed_forward2.weight", 8, 1.0f);
  writer.vector1("encoder.layers.0.norm_feed_forward2.bias", 8, 0.0f);
  writer.matrix("encoder.layers.0.feed_forward2.linear1.weight", 8, 16, 0.01f);
  writer.vector1("encoder.layers.0.feed_forward2.linear1.bias", 16, 0.01f);
  writer.matrix("encoder.layers.0.feed_forward2.linear2.weight", 16, 8, 0.01f);
  writer.vector1("encoder.layers.0.feed_forward2.linear2.bias", 8, 0.01f);
  writer.vector1("encoder.layers.0.norm_out.weight", 8, 1.0f);
  writer.vector1("encoder.layers.0.norm_out.bias", 8, 0.0f);
  writer.matrix("joint.enc.weight", 8, 8, 0.01f);
  writer.vector1("joint.enc.bias", 8, 0.0f);
  writer.matrix("decoder.prediction.embed.weight", 8, 4, 0.01f);
  writer.matrix("decoder.prediction.dec_rnn.lstm.weight_ih_l0", 8, 32, 0.01f);
  writer.matrix("decoder.prediction.dec_rnn.lstm.weight_hh_l0", 8, 32, 0.01f);
  writer.matrix("decoder.prediction.dec_rnn.lstm.bias_ih_l0", 32, 1, 0.0f);
  writer.matrix("decoder.prediction.dec_rnn.lstm.bias_hh_l0", 32, 1, 0.0f);
  writer.matrix("joint.pred.weight", 8, 8, 0.01f);
  writer.vector1("joint.pred.bias", 8, 0.0f);
  writer.matrix("joint.joint_net.2.weight", 8, 6, 0.01f);
  writer.vector1("joint.joint_net.2.bias", 6, 0.0f);
  check(gguf_write_to_file(writer.output, path.string().c_str(), false),
        "fixture write");
  return path;
}

void remove_fixture(const std::filesystem::path &path) {
  std::error_code error;
  std::filesystem::remove(path, error);
}

void enable_test_models() {
#if defined(_WIN32)
  _putenv_s("PARAKEET_POC_ALLOW_TEST_MODEL", "1");
#else
  setenv("PARAKEET_POC_ALLOW_TEST_MODEL", "1", 1);
#endif
}

}

int main() {
  try {
    enable_test_models();
    const auto fixture = make_fixture();
    parakeet_poc::Engine engine;
    std::string error;
    check(engine.load(fixture.string().c_str(), error), error);
    std::vector<float> pcm(32, 0.1f);
    std::string first;
    check(engine.transcribe(pcm.data(), pcm.size(), first, error), error);
    auto stats = engine.stats();
    check(stats.transcribe_calls == 1 && stats.encoder_plans == 1,
          "first transcribe stats");
    std::string second;
    check(engine.transcribe(pcm.data(), pcm.size(), second, error), error);
    check(first == second, "exact cache changed result");
    stats = engine.stats();
    check(stats.transcribe_calls == 2 && stats.cache_hits == 1, "cache stats");
    pcm[0] = 0.2f;
    check(engine.transcribe(pcm.data(), pcm.size(), second, error), error);
    stats = engine.stats();
    check(stats.transcribe_calls == 3 && stats.cache_hits == 1 &&
              stats.plan_cache_hits == 1,
          "altered input cache stats");

    parakeet_poc::Stream stream(engine, {16000, 8, 2});
    std::string stream_error;
    check(stream.append(pcm.data(), 12, stream_error).has_value(),
          stream_error);
    check(stream.finish(stream_error).has_value(), stream_error);
    parakeet_poc::Stream invalid(engine, {8000, 8, 2});
    check(!invalid.append(pcm.data(), 1, stream_error).has_value(),
          "invalid stream accepted");
    parakeet_poc::Stream capped(engine, {16000, 8, 2, 8});
    check(!capped.append(pcm.data(), 9, stream_error).has_value(),
          "oversized stream accepted");
    remove_fixture(fixture);
    std::puts("PARAKEET POC TESTS OK");
    return 0;
  } catch (const std::exception &exception) {
    std::cerr << exception.what() << '\n';
    return 1;
  }
}
