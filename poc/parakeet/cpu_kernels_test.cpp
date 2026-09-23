#include "cpu_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using parakeet_poc::cpu::LinearMatrix;
using parakeet_poc::cpu::QuantType;

void check(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

float half_to_float(std::uint16_t value) {
  const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000u) << 16;
  std::uint32_t exponent = (value >> 10) & 0x1fu;
  std::uint32_t mantissa = value & 0x03ffu;
  std::uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      exponent = 113;
      while ((mantissa & 0x0400u) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x03ffu;
      bits = sign | (exponent << 23) | (mantissa << 13);
    }
  } else if (exponent == 0x1fu) {
    bits = sign | 0x7f800000u | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
  }
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

std::uint16_t float_to_half(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t sign = (bits >> 16) & 0x8000u;
  const std::uint32_t exponent = (bits >> 23) & 0xffu;
  const std::uint32_t mantissa = bits & 0x007fffffu;
  if (exponent == 0xffu)
    return static_cast<std::uint16_t>(sign | 0x7c00u |
                                      (mantissa ? 0x0200u : 0));
  int half_exponent = static_cast<int>(exponent) - 112;
  if (half_exponent >= 31)
    return static_cast<std::uint16_t>(sign | 0x7c00u);
  if (half_exponent <= 0) {
    if (half_exponent < -10)
      return static_cast<std::uint16_t>(sign);
    std::uint32_t half_mantissa = mantissa >> (14 - half_exponent);
    const std::uint32_t remainder =
        mantissa & ((1u << (13 - half_exponent)) - 1u);
    const std::uint32_t midpoint = 1u << (12 - half_exponent);
    if (remainder > midpoint || (remainder == midpoint && (half_mantissa & 1u)))
      ++half_mantissa;
    return static_cast<std::uint16_t>(sign | half_mantissa);
  }
  std::uint32_t half_mantissa = mantissa >> 13;
  const std::uint32_t remainder = mantissa & 0x1fffu;
  if (remainder > 0x1000u || (remainder == 0x1000u && (half_mantissa & 1u)))
    ++half_mantissa;
  if (half_mantissa == 0x400u) {
    half_mantissa = 0;
    ++half_exponent;
    if (half_exponent >= 31)
      return static_cast<std::uint16_t>(sign | 0x7c00u);
  }
  return static_cast<std::uint16_t>(
      sign | (static_cast<std::uint32_t>(half_exponent) << 10) | half_mantissa);
}

struct Q4KBlock {
  std::uint16_t d;
  std::uint16_t dmin;
  std::uint8_t scales[12];
  std::uint8_t qs[128];
};

struct Q6KBlock {
  std::uint8_t ql[128];
  std::uint8_t qh[64];
  std::int8_t scales[16];
  std::uint16_t d;
};

struct Q8Block {
  std::uint16_t d;
  std::int8_t qs[32];
};

void unpack_q4(const Q4KBlock &block, std::uint8_t scales[8],
               std::uint8_t mins[8]) {
  for (int index = 0; index < 4; ++index) {
    scales[index] = block.scales[index] & 63u;
    mins[index] = block.scales[index + 4] & 63u;
  }
  for (int index = 4; index < 8; ++index) {
    scales[index] =
        static_cast<std::uint8_t>((block.scales[index + 4] & 0x0fu) |
                                  ((block.scales[index - 4] >> 6) << 4));
    mins[index] = static_cast<std::uint8_t>((block.scales[index + 4] >> 4) |
                                            ((block.scales[index] >> 6) << 4));
  }
}

void dequant_q4(const std::uint8_t *row, int width, std::vector<float> &out) {
  const Q4KBlock *blocks = reinterpret_cast<const Q4KBlock *>(row);
  out.resize(width);
  for (int block_index = 0; block_index < width / 256; ++block_index) {
    const Q4KBlock &block = blocks[block_index];
    std::uint8_t scales[8];
    std::uint8_t mins[8];
    unpack_q4(block, scales, mins);
    for (int sub = 0; sub < 8; ++sub) {
      const float d = half_to_float(block.d) * scales[sub];
      const float m = half_to_float(block.dmin) * mins[sub];
      for (int lane = 0; lane < 32; ++lane) {
        const int q = (sub & 1) == 0 ? (block.qs[(sub / 2) * 32 + lane] & 0x0f)
                                     : (block.qs[(sub / 2) * 32 + lane] >> 4);
        out[block_index * 256 + sub * 32 + lane] =
            d * static_cast<float>(q) - m;
      }
    }
  }
}

void dequant_q6(const std::uint8_t *row, int width, std::vector<float> &out) {
  const Q6KBlock *blocks = reinterpret_cast<const Q6KBlock *>(row);
  out.resize(width);
  for (int block_index = 0; block_index < width / 256; ++block_index) {
    const Q6KBlock &block = blocks[block_index];
    const float d = half_to_float(block.d);
    for (int group = 0; group < 2; ++group) {
      const std::uint8_t *ql = block.ql + group * 64;
      const std::uint8_t *qh = block.qh + group * 32;
      const std::int8_t *scales = block.scales + group * 8;
      for (int lane = 0; lane < 32; ++lane) {
        const int q[4] = {
            ((ql[lane] & 0x0f) | ((qh[lane] & 3) << 4)) - 32,
            ((ql[lane + 32] & 0x0f) | (((qh[lane] >> 2) & 3) << 4)) - 32,
            ((ql[lane] >> 4) | (((qh[lane] >> 4) & 3) << 4)) - 32,
            ((ql[lane + 32] >> 4) | (((qh[lane] >> 6) & 3) << 4)) - 32,
        };
        for (int part = 0; part < 4; ++part) {
          out[block_index * 256 + group * 128 + part * 32 + lane] =
              d * scales[lane / 16 + part * 2] * static_cast<float>(q[part]);
        }
      }
    }
  }
}

void dequant_q8(const std::uint8_t *row, int width, std::vector<float> &out) {
  const Q8Block *blocks = reinterpret_cast<const Q8Block *>(row);
  out.resize(width);
  for (int block_index = 0; block_index < width / 32; ++block_index) {
    const Q8Block &block = blocks[block_index];
    const float d = half_to_float(block.d);
    for (int lane = 0; lane < 32; ++lane)
      out[block_index * 32 + lane] = d * static_cast<float>(block.qs[lane]);
  }
}

std::vector<float> make_input(int width, int samples, unsigned seed) {
  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
  std::vector<float> result(static_cast<std::size_t>(width) * samples);
  for (int sample = 0; sample < samples; ++sample)
    for (int index = 0; index < width; ++index)
      result[static_cast<std::size_t>(sample) * width + index] =
          distribution(generator);
  return result;
}

std::vector<float> reference(const std::uint8_t *weights, QuantType type,
                             int width, int output, const float *bias,
                             const std::vector<float> &input, int samples) {
  std::vector<float> result(static_cast<std::size_t>(output) * samples);
  std::vector<float> row;
  const std::size_t row_bytes =
      type == QuantType::Q4K    ? sizeof(Q4KBlock) * (width / 256)
      : type == QuantType::Q6K  ? sizeof(Q6KBlock) * (width / 256)
      : type == QuantType::Q8_0 ? sizeof(Q8Block) * (width / 32)
                                : sizeof(float) * width;
  for (int out = 0; out < output; ++out) {
    const std::uint8_t *source = weights + out * row_bytes;
    if (type == QuantType::Q4K)
      dequant_q4(source, width, row);
    else if (type == QuantType::Q6K)
      dequant_q6(source, width, row);
    else if (type == QuantType::Q8_0)
      dequant_q8(source, width, row);
    else
      row.assign(reinterpret_cast<const float *>(source),
                 reinterpret_cast<const float *>(source) + width);
    for (int sample = 0; sample < samples; ++sample) {
      float value = bias ? bias[out] : 0.0f;
      for (int index = 0; index < width; ++index)
        value += row[index] *
                 input[static_cast<std::size_t>(sample) * width + index];
      result[static_cast<std::size_t>(sample) * output + out] = value;
    }
  }
  return result;
}

void compare(const std::vector<float> &actual,
             const std::vector<float> &expected, float tolerance,
             const std::string &name) {
  check(actual.size() == expected.size(), name + " size");
  for (size_t index = 0; index < actual.size(); ++index) {
    const float error = std::fabs(actual[index] - expected[index]);
    check(error <= tolerance * (1.0f + std::fabs(expected[index])),
          name + " mismatch at " + std::to_string(index) +
              " actual=" + std::to_string(actual[index]) +
              " expected=" + std::to_string(expected[index]));
  }
}

void make_q4(std::vector<std::uint8_t> &data, int width, unsigned seed) {
  std::mt19937 generator(seed);
  std::uniform_int_distribution<int> byte_distribution(0, 255);
  for (int block = 0; block < width / 256; ++block) {
    Q4KBlock &value = *reinterpret_cast<Q4KBlock *>(data.data() + block * 144);
    value.d = float_to_half(0.01f + (block % 3) * 0.002f);
    value.dmin = float_to_half(0.003f + (block % 2) * 0.001f);
    const int scales[8] = {7, 9, 11, 13, 17, 19, 21, 23};
    const int mins[8] = {2, 3, 4, 5, 6, 7, 8, 9};
    for (int index = 0; index < 12; ++index)
      value.scales[index] = 0;
    for (int index = 0; index < 4; ++index) {
      value.scales[index] = static_cast<std::uint8_t>(scales[index]);
      value.scales[index + 4] = static_cast<std::uint8_t>(mins[index]);
    }
    for (int index = 4; index < 8; ++index) {
      value.scales[index + 4] = static_cast<std::uint8_t>(
          (scales[index] & 15) | ((mins[index] & 15) << 4));
      value.scales[index - 4] |=
          static_cast<std::uint8_t>((scales[index] >> 4) << 6);
      value.scales[index] |= static_cast<std::uint8_t>((mins[index] >> 4) << 6);
    }
    for (std::uint8_t &q : value.qs)
      q = static_cast<std::uint8_t>(byte_distribution(generator));
  }
}

void make_q6(std::vector<std::uint8_t> &data, int width, unsigned seed) {
  std::mt19937 generator(seed);
  std::uniform_int_distribution<int> byte_distribution(0, 255);
  for (int block = 0; block < width / 256; ++block) {
    Q6KBlock &value = *reinterpret_cast<Q6KBlock *>(data.data() + block * 210);
    value.d = float_to_half(0.01f);
    for (std::uint8_t &q : value.ql)
      q = static_cast<std::uint8_t>(byte_distribution(generator));
    for (std::uint8_t &q : value.qh)
      q = static_cast<std::uint8_t>(byte_distribution(generator));
    for (std::int8_t &scale : value.scales)
      scale = static_cast<std::int8_t>(byte_distribution(generator) % 31 - 15);
  }
}

void make_q8(std::vector<std::uint8_t> &data, int width, unsigned seed) {
  std::mt19937 generator(seed);
  std::uniform_int_distribution<int> byte_distribution(-127, 127);
  for (int block = 0; block < width / 32; ++block) {
    Q8Block &value = *reinterpret_cast<Q8Block *>(data.data() + block * 34);
    value.d = float_to_half(0.02f);
    for (std::int8_t &q : value.qs)
      q = static_cast<std::int8_t>(byte_distribution(generator));
  }
}

void test_type(QuantType type, int width, int output, int samples,
               const std::vector<std::uint8_t> &weights,
               std::size_t row_bytes) {
  const auto input = make_input(width, samples, 77);
  const std::vector<float> bias(output, 0.125f);
  LinearMatrix matrix{weights.data(), bias.data(), type,   width,
                      output,         row_bytes,   nullptr};
  std::vector<float> actual(static_cast<std::size_t>(output) * samples);
  parakeet_poc::cpu::linear_f32(matrix, input.data(), actual.data(), samples,
                                2);
  const auto expected = reference(weights.data(), type, width, output,
                                  bias.data(), input, samples);
  compare(actual, expected, 2e-5f, "f32");
  if (type != QuantType::F32) {
    std::fill(actual.begin(), actual.end(), 0.0f);
    parakeet_poc::cpu::linear_q8_range(matrix, input.data(), actual.data(),
                                       samples, 0, output);
    for (int sample = 0; sample < samples; ++sample) {
      std::vector<float> single(static_cast<std::size_t>(output));
      parakeet_poc::cpu::linear_q8_range(
          matrix, input.data() + static_cast<std::size_t>(sample) * width,
          single.data(), 1, 0, output);
      for (int row = 0; row < output; ++row) {
        const float batch_value =
            actual[static_cast<std::size_t>(sample) * output + row];
        const float difference = std::fabs(batch_value - single[row]);
        check(difference <= 2e-5f * (1.0f + std::fabs(single[row])),
              std::string("q8 batch mismatch type=") +
                  std::to_string(static_cast<int>(type)) + " sample=" +
                  std::to_string(sample) + " row=" + std::to_string(row) +
                  " batch=" + std::to_string(batch_value) +
                  " single=" + std::to_string(single[row]));
      }
    }
    LinearMatrix packed_matrix = matrix;
    packed_matrix.packed = parakeet_poc::cpu::pack_matrix(matrix);
    std::vector<float> packed_output(actual.size());
    parakeet_poc::cpu::linear_q8_batch(packed_matrix, input.data(),
                                       packed_output.data(), samples, 2);
    compare(packed_output, actual, 2e-5f, "packed q8");
  }
}

}

int main() {
  try {
    const int q4_width = 512;
    std::vector<std::uint8_t> q4(static_cast<std::size_t>(q4_width / 256) *
                                 144 * 5);
    make_q4(q4, q4_width, 1);
    test_type(QuantType::Q4K, q4_width, 5, 16, q4,
              static_cast<std::size_t>(q4_width / 256) * 144);

    const int q6_width = 512;
    std::vector<std::uint8_t> q6(static_cast<std::size_t>(q6_width / 256) *
                                 210 * 5);
    make_q6(q6, q6_width, 2);
    test_type(QuantType::Q6K, q6_width, 5, 16, q6,
              static_cast<std::size_t>(q6_width / 256) * 210);

    const int q8_width = 64;
    std::vector<std::uint8_t> q8(static_cast<std::size_t>(q8_width / 32) * 34 *
                                 5);
    make_q8(q8, q8_width, 3);
    test_type(QuantType::Q8_0, q8_width, 5, 16, q8,
              static_cast<std::size_t>(q8_width / 32) * 34);

    const int f32_width = 17;
    std::vector<float> f32(static_cast<std::size_t>(f32_width) * 5);
    for (size_t index = 0; index < f32.size(); ++index)
      f32[index] = static_cast<float>(static_cast<int>(index % 13) - 6) * 0.03f;
    std::vector<std::uint8_t> raw(f32.size() * sizeof(float));
    std::memcpy(raw.data(), f32.data(), raw.size());
    test_type(QuantType::F32, f32_width, 5, 3, raw, f32_width * sizeof(float));

    std::cout << "CPU KERNEL TESTS OK " << parakeet_poc::cpu::backend_name()
              << '\n';
    return 0;
  } catch (const std::exception &exception) {
    std::cerr << exception.what() << '\n';
    return 1;
  }
}
