#include "cpu_kernels.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) ||             \
    defined(_M_IX86)
#define PARAKEET_POC_X86 1
#include <immintrin.h>
#else
#define PARAKEET_POC_X86 0
#endif

#if PARAKEET_POC_X86 && (defined(__GNUC__) || defined(__clang__))
#define PARAKEET_POC_TARGET_AVX2 __attribute__((target("avx2,fma")))
#define PARAKEET_POC_ALWAYS_INLINE __attribute__((always_inline))
#elif PARAKEET_POC_X86 && defined(_MSC_VER)
#define PARAKEET_POC_TARGET_AVX2
#define PARAKEET_POC_ALWAYS_INLINE __forceinline
#else
#define PARAKEET_POC_TARGET_AVX2
#define PARAKEET_POC_ALWAYS_INLINE inline
#endif

namespace parakeet_poc {
namespace cpu {
namespace {

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

struct Q8KBlock {
  float d;
  std::int8_t qs[256];
  std::int16_t bsums[16];
};

static_assert(sizeof(Q4KBlock) == 144, "Q4K block layout");
static_assert(sizeof(Q6KBlock) == 210, "Q6K block layout");
static_assert(sizeof(Q8Block) == 34, "Q8 block layout");
static_assert(sizeof(Q8KBlock) == 292, "Q8K block layout");

float half_to_float(std::uint16_t value) {
  const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000u) << 16;
  std::uint32_t exponent = (value >> 10) & 0x1fu;
  std::uint32_t mantissa = value & 0x03ffu;
  std::uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      exponent = 127 - 15 + 1;
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
    bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
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
  int half_exponent = static_cast<int>(exponent) - 127 + 15;
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

int nearest_int(float value) {
#if PARAKEET_POC_X86 && (defined(__GNUC__) || defined(__clang__))
  return _mm_cvtss_si32(_mm_set_ss(value));
#else
  return static_cast<int>(std::nearbyint(value));
#endif
}

void require(bool condition, const char *message) {
  if (!condition)
    throw std::invalid_argument(message);
}

std::size_t type_size(QuantType type) {
  switch (type) {
  case QuantType::F32:
    return sizeof(float);
  case QuantType::Q4K:
    return sizeof(Q4KBlock);
  case QuantType::Q6K:
    return sizeof(Q6KBlock);
  case QuantType::Q8_0:
    return sizeof(Q8Block);
  }
  return 0;
}

void validate(const LinearMatrix &matrix, int samples) {
  require(matrix.data != nullptr, "linear matrix data is null");
  require(matrix.input > 0 && matrix.output > 0,
          "linear matrix dimensions are invalid");
  require(samples > 0, "linear sample count is invalid");
  require(matrix.row_bytes >= static_cast<std::size_t>(matrix.input) *
                                  type_size(matrix.type) /
                                  (matrix.type == QuantType::F32 ? 1
                                   : matrix.type == QuantType::Q4K ||
                                           matrix.type == QuantType::Q6K
                                       ? 256
                                       : 32),
          "linear matrix row is too small");
  if (matrix.type == QuantType::Q4K || matrix.type == QuantType::Q6K)
    require(matrix.input % 256 == 0, "K-quant input width is invalid");
  if (matrix.type == QuantType::Q8_0)
    require(matrix.input % 32 == 0, "Q8 input width is invalid");
}

bool avx2_available() {
#if PARAKEET_POC_X86 && (defined(__GNUC__) || defined(__clang__))
  static const bool value =
      __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
  return value;
#else
  return false;
#endif
}

template <typename Function>
void parallel_rows(int output, int threads, Function function) {
  threads = std::max(1, std::min(threads, output));
  if (threads == 1) {
    for (int row = 0; row < output; ++row)
      function(row);
    return;
  }
  std::atomic<int> next{0};
  std::vector<std::thread> workers;
  workers.reserve(static_cast<std::size_t>(threads));
  for (int worker = 0; worker < threads; ++worker) {
    workers.emplace_back([&]() {
      for (;;) {
        const int row = next.fetch_add(1, std::memory_order_relaxed);
        if (row >= output)
          return;
        function(row);
      }
    });
  }
  for (std::thread &worker : workers)
    worker.join();
}

void unpack_q4k(const Q4KBlock &block, std::uint8_t scales[8],
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

void initialize_output(float *output, int samples, int row, int output_width,
                       const float *bias) {
  for (int sample = 0; sample < samples; ++sample)
    output[static_cast<std::size_t>(sample) * output_width + row] =
        bias ? bias[row] : 0.0f;
}

void add_product(float *output, const float *input, float weight, int samples,
                 int input_width, int output_width, int output_row, int element,
                 bool vector) {
  (void)vector;
  for (int sample = 0; sample < samples; ++sample)
    output[static_cast<std::size_t>(sample) * output_width + output_row] +=
        weight *
        input[static_cast<std::size_t>(sample) * input_width + element];
}

void q4k_row_f32_impl(const std::uint8_t *row, int input_width,
                      const float *input, float *output, int samples,
                      int output_row, int output_width, const float *bias,
                      bool vector) {
  initialize_output(output, samples, output_row, output_width, bias);
  const Q4KBlock *blocks = reinterpret_cast<const Q4KBlock *>(row);
  const int block_count = input_width / 256;
  for (int block_index = 0; block_index < block_count; ++block_index) {
    const Q4KBlock &block = blocks[block_index];
    std::uint8_t scales[8];
    std::uint8_t mins[8];
    unpack_q4k(block, scales, mins);
    const float d = half_to_float(block.d);
    const float dmin = half_to_float(block.dmin);
    for (int sub = 0; sub < 8; ++sub) {
      const int source = (sub / 2) * 32;
      const bool high = (sub & 1) != 0;
      const float weight_scale = d * static_cast<float>(scales[sub]);
      const float weight_min = dmin * static_cast<float>(mins[sub]);
      for (int lane = 0; lane < 32; ++lane) {
        const int quantized = high ? (block.qs[source + lane] >> 4)
                                   : (block.qs[source + lane] & 0x0f);
        const float weight =
            weight_scale * static_cast<float>(quantized) - weight_min;
        const int element = block_index * 256 + sub * 32 + lane;
        add_product(output, input, weight, samples, input_width, output_width,
                    output_row, element, vector);
      }
    }
  }
}

void q6k_row_f32_impl(const std::uint8_t *row, int input_width,
                      const float *input, float *output, int samples,
                      int output_row, int output_width, const float *bias,
                      bool vector) {
  initialize_output(output, samples, output_row, output_width, bias);
  const Q6KBlock *blocks = reinterpret_cast<const Q6KBlock *>(row);
  const int block_count = input_width / 256;
  for (int block_index = 0; block_index < block_count; ++block_index) {
    const Q6KBlock &block = blocks[block_index];
    const float d = half_to_float(block.d);
    for (int group = 0; group < 2; ++group) {
      const std::uint8_t *ql = block.ql + group * 64;
      const std::uint8_t *qh = block.qh + group * 32;
      const std::int8_t *scales = block.scales + group * 8;
      for (int lane = 0; lane < 32; ++lane) {
        const int scale_group = lane / 16;
        const int quantized[4] = {
            static_cast<int>((ql[lane] & 0x0f) | ((qh[lane] & 0x03u) << 4)) -
                32,
            static_cast<int>((ql[lane + 32] & 0x0f) |
                             (((qh[lane] >> 2) & 0x03u) << 4)) -
                32,
            static_cast<int>((ql[lane] >> 4) |
                             (((qh[lane] >> 4) & 0x03u) << 4)) -
                32,
            static_cast<int>((ql[lane + 32] >> 4) |
                             (((qh[lane] >> 6) & 0x03u) << 4)) -
                32,
        };
        const int scale_index[4] = {scale_group, scale_group + 2,
                                    scale_group + 4, scale_group + 6};
        const int offsets[4] = {lane, lane + 32, lane + 64, lane + 96};
        for (int part = 0; part < 4; ++part) {
          const float weight = d *
                               static_cast<float>(scales[scale_index[part]]) *
                               static_cast<float>(quantized[part]);
          const int element = block_index * 256 + group * 128 + offsets[part];
          add_product(output, input, weight, samples, input_width, output_width,
                      output_row, element, vector);
        }
      }
    }
  }
}

void q8_row_f32_impl(const std::uint8_t *row, int input_width,
                     const float *input, float *output, int samples,
                     int output_row, int output_width, const float *bias,
                     bool vector) {
  initialize_output(output, samples, output_row, output_width, bias);
  const Q8Block *blocks = reinterpret_cast<const Q8Block *>(row);
  const int block_count = input_width / 32;
  for (int block_index = 0; block_index < block_count; ++block_index) {
    const Q8Block &block = blocks[block_index];
    const float d = half_to_float(block.d);
    for (int lane = 0; lane < 32; ++lane) {
      const float weight = d * static_cast<float>(block.qs[lane]);
      const int element = block_index * 32 + lane;
      add_product(output, input, weight, samples, input_width, output_width,
                  output_row, element, vector);
    }
  }
}

void f32_row(const std::uint8_t *row, int input_width, const float *input,
             float *output, int samples, int output_row, int output_width,
             const float *bias, bool vector) {
  initialize_output(output, samples, output_row, output_width, bias);
  const float *weights = reinterpret_cast<const float *>(row);
  (void)vector;
  for (int index = 0; index < input_width; ++index) {
    const float weight = weights[index];
    for (int sample = 0; sample < samples; ++sample)
      output[static_cast<std::size_t>(sample) * output_width + output_row] +=
          weight *
          input[static_cast<std::size_t>(sample) * input_width + index];
  }
}

#if PARAKEET_POC_X86
PARAKEET_POC_TARGET_AVX2 int horizontal_sum(__m256i value) {
  __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(value),
                              _mm256_extracti128_si256(value, 1));
  sum = _mm_hadd_epi32(sum, sum);
  sum = _mm_hadd_epi32(sum, sum);
  return _mm_cvtsi128_si32(sum);
}

PARAKEET_POC_TARGET_AVX2 int dot_q4_sub_avx2(const std::uint8_t *weight,
                                             const std::int8_t *input,
                                             bool high) {
  __m256i packed =
      _mm256_loadu_si256(reinterpret_cast<const __m256i *>(weight));
  const __m256i mask = _mm256_set1_epi8(0x0f);
  packed = high ? _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask)
                : _mm256_and_si256(packed, mask);
  const __m256i values =
      _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input));
  const __m256i products = _mm256_maddubs_epi16(packed, values);
  return horizontal_sum(_mm256_madd_epi16(products, _mm256_set1_epi16(1)));
}

PARAKEET_POC_TARGET_AVX2 int dot_signed_16(const std::int8_t *weight,
                                           const std::int8_t *input) {
  const __m128i weights =
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(weight));
  const __m128i inputs =
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(input));
  const __m256i weight16 = _mm256_cvtepi8_epi16(weights);
  const __m256i input16 = _mm256_cvtepi8_epi16(inputs);
  const __m256i product = _mm256_mullo_epi16(weight16, input16);
  const __m256i ones = _mm256_set1_epi16(1);
  return horizontal_sum(_mm256_madd_epi16(product, ones));
}
#endif

#if PARAKEET_POC_X86
PARAKEET_POC_TARGET_AVX2 __m256i q4_scale_shuffle(int index) {
  const __m256i pattern =
      _mm256_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                       1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);
  return _mm256_add_epi8(pattern,
                         _mm256_set1_epi8(static_cast<char>(2 * index)));
}

PARAKEET_POC_TARGET_AVX2 __m128i q6_scale_shuffle(int index) {
  return _mm_unpacklo_epi64(_mm_set1_epi8(static_cast<char>(2 * index)),
                            _mm_set1_epi8(static_cast<char>(2 * index + 1)));
}

PARAKEET_POC_TARGET_AVX2 float horizontal_sum_float(__m256 value) {
  __m128 reduced = _mm_add_ps(_mm256_castps256_ps128(value),
                              _mm256_extractf128_ps(value, 1));
  reduced = _mm_hadd_ps(reduced, reduced);
  reduced = _mm_hadd_ps(reduced, reduced);
  return _mm_cvtss_f32(reduced);
}

PARAKEET_POC_TARGET_AVX2 float dot_q4k_q8k_avx2_fast(const std::uint8_t *row,
                                                     const Q8KBlock *input,
                                                     int width) {
  const Q4KBlock *blocks = reinterpret_cast<const Q4KBlock *>(row);
  __m256 accumulated = _mm256_setzero_ps();
  __m128 accumulated_min = _mm_setzero_ps();
  const __m256i mask_low = _mm256_set1_epi8(0x0f);
  for (int block = 0; block < width / 256; ++block) {
    const Q4KBlock &weight = blocks[block];
    const Q8KBlock &activation = input[block];
    const float d = activation.d * half_to_float(weight.d);
    const float dmin = -activation.d * half_to_float(weight.dmin);
    std::uint32_t scales[4];
    std::memcpy(scales, weight.scales, sizeof(scales[0]) * 3);
    scales[3] = ((scales[2] >> 4) & 0x0f0f0f0fu) |
                (((scales[1] >> 6) & 0x03030303u) << 4);
    const std::uint32_t auxiliary = scales[1] & 0x3f3f3f3fu;
    scales[1] =
        (scales[2] & 0x0f0f0f0fu) | (((scales[0] >> 6) & 0x03030303u) << 4);
    scales[2] = auxiliary;
    scales[0] &= 0x3f3f3f3fu;
    const __m256i scales_and_min = _mm256_cvtepu8_epi16(_mm_set_epi32(
        static_cast<int>(scales[3]), static_cast<int>(scales[2]),
        static_cast<int>(scales[1]), static_cast<int>(scales[0])));
    const __m256i activation_sums =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(activation.bsums));
    const __m128i sums =
        _mm_hadd_epi16(_mm256_extracti128_si256(activation_sums, 0),
                       _mm256_extracti128_si256(activation_sums, 1));
    const __m128i min_products =
        _mm_madd_epi16(_mm256_extracti128_si256(scales_and_min, 1), sums);
    accumulated_min = _mm_fmadd_ps(
        _mm_set1_ps(dmin), _mm_cvtepi32_ps(min_products), accumulated_min);
    const __m128i scale_low = _mm256_extracti128_si256(scales_and_min, 0);
    const __m256i scale_values = _mm256_insertf128_si256(
        _mm256_castsi128_si256(scale_low), scale_low, 1);
    __m256i sum = _mm256_setzero_si256();
    const std::uint8_t *q4 = weight.qs;
    const std::int8_t *q8 = activation.qs;
    for (int part = 0; part < 4; ++part) {
      const __m256i scale_low_part =
          _mm256_shuffle_epi8(scale_values, q4_scale_shuffle(2 * part));
      const __m256i scale_high_part =
          _mm256_shuffle_epi8(scale_values, q4_scale_shuffle(2 * part + 1));
      const __m256i q4_bits =
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q4));
      q4 += 32;
      const __m256i q4_low = _mm256_and_si256(q4_bits, mask_low);
      const __m256i q4_high =
          _mm256_and_si256(_mm256_srli_epi16(q4_bits, 4), mask_low);
      const __m256i q8_low =
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8));
      q8 += 32;
      __m256i product_low = _mm256_maddubs_epi16(q4_low, q8_low);
      product_low = _mm256_madd_epi16(scale_low_part, product_low);
      const __m256i q8_high =
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8));
      q8 += 32;
      __m256i product_high = _mm256_maddubs_epi16(q4_high, q8_high);
      product_high = _mm256_madd_epi16(scale_high_part, product_high);
      sum = _mm256_add_epi32(sum, _mm256_add_epi32(product_low, product_high));
    }
    accumulated = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sum),
                                  accumulated);
  }
  accumulated_min = _mm_add_ps(accumulated_min,
                               _mm_movehl_ps(accumulated_min, accumulated_min));
  accumulated_min =
      _mm_add_ss(accumulated_min, _mm_movehdup_ps(accumulated_min));
  return horizontal_sum_float(accumulated) + _mm_cvtss_f32(accumulated_min);
}

PARAKEET_POC_TARGET_AVX2 float dot_q6k_q8k_avx2_fast(const std::uint8_t *row,
                                                     const Q8KBlock *input,
                                                     int width) {
  const Q6KBlock *blocks = reinterpret_cast<const Q6KBlock *>(row);
  __m256 accumulated = _mm256_setzero_ps();
  const __m256i mask_three = _mm256_set1_epi8(3);
  const __m256i mask_fifteen = _mm256_set1_epi8(15);
  for (int block = 0; block < width / 256; ++block) {
    const Q6KBlock &weight = blocks[block];
    const Q8KBlock &activation = input[block];
    const float d = activation.d * half_to_float(weight.d);
    const __m256i activation_sums =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(activation.bsums));
    const __m128i scales =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(weight.scales));
    const __m256i scales_16 = _mm256_cvtepi8_epi16(scales);
    const __m256i scale_subtract =
        _mm256_slli_epi32(_mm256_madd_epi16(activation_sums, scales_16), 5);
    __m256i sum = _mm256_setzero_si256();
    const std::uint8_t *ql = weight.ql;
    const std::uint8_t *qh = weight.qh;
    const std::int8_t *q8 = activation.qs;
    int scale_index = 0;
    for (int part = 0; part < 2; ++part) {
      const __m256i ql_low =
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ql));
      ql += 32;
      const __m256i ql_high =
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ql));
      ql += 32;
      const __m256i qh_values =
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(qh));
      qh += 32;
      const __m256i high_0 =
          _mm256_slli_epi16(_mm256_and_si256(qh_values, mask_three), 4);
      const __m256i high_1 = _mm256_slli_epi16(
          _mm256_and_si256(qh_values, _mm256_set1_epi8(12)), 2);
      const __m256i high_2 = _mm256_and_si256(qh_values, _mm256_set1_epi8(48));
      const __m256i high_3 = _mm256_srli_epi16(
          _mm256_and_si256(qh_values, _mm256_set1_epi8(-64)), 2);
      const __m256i q_low =
          _mm256_or_si256(_mm256_and_si256(ql_low, mask_fifteen), high_0);
      const __m256i q_high =
          _mm256_or_si256(_mm256_and_si256(ql_high, mask_fifteen), high_1);
      const __m256i q_low_high = _mm256_or_si256(
          _mm256_and_si256(_mm256_srli_epi16(ql_low, 4), mask_fifteen), high_2);
      const __m256i q_high_high = _mm256_or_si256(
          _mm256_and_si256(_mm256_srli_epi16(ql_high, 4), mask_fifteen),
          high_3);
      const __m256i q8_0 =
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8));
      q8 += 32;
      const __m256i q8_1 =
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8));
      q8 += 32;
      const __m256i q8_2 =
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8));
      q8 += 32;
      const __m256i q8_3 =
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8));
      q8 += 32;
      __m256i product_0 = _mm256_maddubs_epi16(q_low, q8_0);
      __m256i product_1 = _mm256_maddubs_epi16(q_high, q8_1);
      __m256i product_2 = _mm256_maddubs_epi16(q_low_high, q8_2);
      __m256i product_3 = _mm256_maddubs_epi16(q_high_high, q8_3);
      product_0 =
          _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm_shuffle_epi8(
                                scales, q6_scale_shuffle(scale_index + 0))),
                            product_0);
      product_1 =
          _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm_shuffle_epi8(
                                scales, q6_scale_shuffle(scale_index + 1))),
                            product_1);
      product_2 =
          _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm_shuffle_epi8(
                                scales, q6_scale_shuffle(scale_index + 2))),
                            product_2);
      product_3 =
          _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm_shuffle_epi8(
                                scales, q6_scale_shuffle(scale_index + 3))),
                            product_3);
      scale_index += 4;
      sum = _mm256_add_epi32(sum, _mm256_add_epi32(product_0, product_1));
      sum = _mm256_add_epi32(sum, _mm256_add_epi32(product_2, product_3));
    }
    sum = _mm256_sub_epi32(sum, scale_subtract);
    accumulated = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sum),
                                  accumulated);
  }
  return horizontal_sum_float(accumulated);
}
#endif

#if !PARAKEET_POC_X86
int dot_signed_16_scalar(const std::int8_t *weight, const std::int8_t *input) {
  int result = 0;
  for (int index = 0; index < 16; ++index)
    result += static_cast<int>(weight[index]) * static_cast<int>(input[index]);
  return result;
}
#endif

void quantize_q8k_into(const float *input, Q8KBlock *result, int width) {
  require(width % 256 == 0, "Q8K input width is invalid");
  for (int block_index = 0; block_index < width / 256; ++block_index) {
    const float *source = input + block_index * 256;
    Q8KBlock &destination = result[block_index];
    float maximum = 0.0f;
    float maximum_absolute = 0.0f;
    for (int index = 0; index < 256; ++index) {
      const float value = source[index];
      const float absolute = std::fabs(value);
      if (absolute > maximum_absolute) {
        maximum_absolute = absolute;
        maximum = value;
      }
    }
    if (maximum_absolute == 0.0f) {
      destination.d = 0.0f;
      std::memset(destination.qs, 0, sizeof(destination.qs));
      std::memset(destination.bsums, 0, sizeof(destination.bsums));
      continue;
    }
    const float inverse_scale = -127.0f / maximum;
    for (int index = 0; index < 256; ++index) {
      int quantized = nearest_int(inverse_scale * source[index]);
      quantized = std::min(quantized, 127);
      destination.qs[index] = static_cast<std::int8_t>(quantized);
    }
    for (int group = 0; group < 16; ++group) {
      int sum = 0;
      for (int index = 0; index < 16; ++index)
        sum += destination.qs[group * 16 + index];
      destination.bsums[group] = static_cast<std::int16_t>(sum);
    }
    destination.d = 1.0f / inverse_scale;
  }
}

std::vector<Q8KBlock> quantize_q8k(const float *input, int width) {
  std::vector<Q8KBlock> result(static_cast<std::size_t>(width / 256));
  quantize_q8k_into(input, result.data(), width);
  return result;
}

struct PackedQ4K8 {
  std::uint8_t q[8][256];
  float scale[8][8];
  float min[8][8];
};

struct PackedQ6K8 {
  std::int8_t q[8][256];
  std::int8_t raw_scale[16][8];
  float scale[16][8];
};

static_assert(sizeof(PackedQ4K8) == 2560, "packed Q4K8 layout");
static_assert(sizeof(PackedQ6K8) == 2688, "packed Q6K8 layout");

std::shared_ptr<PackedMatrix> pack_matrix_impl(const LinearMatrix &matrix) {
  if (matrix.type != QuantType::Q4K && matrix.type != QuantType::Q6K)
    return nullptr;
  auto packed = std::make_shared<PackedMatrix>();
  packed->type = matrix.type;
  packed->input = matrix.input;
  packed->output = matrix.output;
  packed->groups = (matrix.output + 7) / 8;
  const std::size_t block_size =
      matrix.type == QuantType::Q4K ? sizeof(PackedQ4K8) : sizeof(PackedQ6K8);
  const int input_blocks = matrix.input / 256;
  packed->row_bytes = block_size * static_cast<std::size_t>(input_blocks);
  const std::size_t bytes =
      static_cast<std::size_t>(packed->groups) * packed->row_bytes;
  packed->words.resize((bytes + sizeof(std::uint64_t) - 1) /
                       sizeof(std::uint64_t));
  auto *storage = reinterpret_cast<std::uint8_t *>(packed->words.data());
  for (int group = 0; group < packed->groups; ++group) {
    auto *group_storage = storage + group * packed->row_bytes;
    for (int block = 0; block < input_blocks; ++block) {
      if (matrix.type == QuantType::Q4K) {
        auto *destination = reinterpret_cast<PackedQ4K8 *>(
            group_storage + static_cast<std::size_t>(block) * block_size);
        std::memset(destination, 0, sizeof(*destination));
        for (int row = 0; row < 8; ++row) {
          const int output_row = group * 8 + row;
          if (output_row >= matrix.output)
            continue;
          const std::uint8_t *source =
              static_cast<const std::uint8_t *>(matrix.data) +
              static_cast<std::size_t>(output_row) * matrix.row_bytes;
          const Q4KBlock *source_block =
              reinterpret_cast<const Q4KBlock *>(source) + block;
          std::uint8_t scales[8];
          std::uint8_t mins[8];
          unpack_q4k(*source_block, scales, mins);
          const float d = half_to_float(source_block->d);
          const float dmin = half_to_float(source_block->dmin);
          for (int sub = 0; sub < 8; ++sub) {
            destination->scale[sub][row] = d * scales[sub];
            destination->min[sub][row] = dmin * mins[sub];
            const int source_index = (sub / 2) * 32;
            for (int lane = 0; lane < 32; ++lane) {
              const int quantized =
                  (sub & 1) == 0
                      ? (source_block->qs[source_index + lane] & 0x0f)
                      : (source_block->qs[source_index + lane] >> 4);
              const int element = sub * 32 + lane;
              destination->q[row][element] =
                  static_cast<std::uint8_t>(quantized);
            }
          }
        }
      } else {
        auto *destination = reinterpret_cast<PackedQ6K8 *>(
            group_storage + static_cast<std::size_t>(block) * block_size);
        std::memset(destination, 0, sizeof(*destination));
        for (int row = 0; row < 8; ++row) {
          const int output_row = group * 8 + row;
          if (output_row >= matrix.output)
            continue;
          const std::uint8_t *source =
              static_cast<const std::uint8_t *>(matrix.data) +
              static_cast<std::size_t>(output_row) * matrix.row_bytes;
          const Q6KBlock *source_block =
              reinterpret_cast<const Q6KBlock *>(source) + block;
          const float d = half_to_float(source_block->d);
          for (int group_index = 0; group_index < 2; ++group_index) {
            const std::uint8_t *ql = source_block->ql + group_index * 64;
            const std::uint8_t *qh = source_block->qh + group_index * 32;
            const std::int8_t *scales = source_block->scales + group_index * 8;
            for (int part = 0; part < 4; ++part) {
              for (int half = 0; half < 2; ++half) {
                const int chunk = group_index * 8 + part * 2 + half;
                destination->raw_scale[chunk][row] = scales[half + part * 2];
                destination->scale[chunk][row] = d * scales[half + part * 2];
                for (int lane = 0; lane < 16; ++lane) {
                  const int source_lane = half * 16 + lane;
                  const int ql_index =
                      part == 1 || part == 3 ? source_lane + 32 : source_lane;
                  const int low = part == 0 || part == 1 ? (ql[ql_index] & 0x0f)
                                                         : (ql[ql_index] >> 4);
                  const int high = part == 0   ? ((qh[source_lane] >> 0) & 3)
                                   : part == 1 ? ((qh[source_lane] >> 2) & 3)
                                   : part == 2 ? ((qh[source_lane] >> 4) & 3)
                                               : ((qh[source_lane] >> 6) & 3);
                  const int element =
                      group_index * 128 + part * 32 + half * 16 + lane;
                  destination->q[row][element] =
                      static_cast<std::int8_t>((low | (high << 4)) - 32);
                }
              }
            }
          }
        }
      }
    }
  }
  return packed;
}

struct BatchQ8 {
  int input = 0;
  int samples = 0;
  int blocks = 0;
  std::vector<float> scales;
  std::vector<std::int8_t> values;
  std::vector<std::int8_t> sample_values;
  std::vector<std::int16_t> sums;
};

BatchQ8 make_batch_q8(const float *input, int width, int samples) {
  require(width % 256 == 0 && samples > 0, "Q8K batch shape is invalid");
  BatchQ8 result;
  result.input = width;
  result.samples = samples;
  result.blocks = width / 256;
  result.scales.resize(static_cast<std::size_t>(result.blocks) * samples);
  result.values.resize(static_cast<std::size_t>(width) * samples);
  result.sample_values.resize(static_cast<std::size_t>(width) * samples);
  result.sums.resize(static_cast<std::size_t>(result.blocks) * 16 * samples);
  for (int sample = 0; sample < samples; ++sample) {
    const float *source = input + static_cast<std::size_t>(sample) * width;
    for (int block = 0; block < result.blocks; ++block) {
      const float *block_source = source + block * 256;
      float maximum = 0.0f;
      float maximum_absolute = 0.0f;
      for (int index = 0; index < 256; ++index) {
        const float value = block_source[index];
        const float absolute = std::fabs(value);
        if (absolute > maximum_absolute) {
          maximum_absolute = absolute;
          maximum = value;
        }
      }
      const float inverse_scale =
          maximum_absolute == 0.0f ? 0.0f : -127.0f / maximum;
      const float scale = inverse_scale == 0.0f ? 0.0f : 1.0f / inverse_scale;
      result.scales[static_cast<std::size_t>(block) * samples + sample] = scale;
      for (int index = 0; index < 256; ++index) {
        int quantized = nearest_int(block_source[index] * inverse_scale);
        quantized = std::min(quantized, 127);
        const std::int8_t quantized_value = static_cast<std::int8_t>(quantized);
        result.values[static_cast<std::size_t>(block * 256 + index) * samples +
                      sample] = quantized_value;
        result.sample_values[static_cast<std::size_t>(sample) * width +
                             block * 256 + index] = quantized_value;
      }
      for (int group = 0; group < 16; ++group) {
        int sum = 0;
        for (int index = 0; index < 16; ++index)
          sum += result.values[static_cast<std::size_t>(block * 256 +
                                                        group * 16 + index) *
                                   samples +
                               sample];
        result.sums[static_cast<std::size_t>(block * 16 + group) * samples +
                    sample] = static_cast<std::int16_t>(sum);
      }
    }
  }
  return result;
}

std::vector<Q8Block> quantize_q8(const float *input, int width) {
  require(width % 32 == 0, "Q8 input width is invalid");
  std::vector<Q8Block> result(static_cast<std::size_t>(width / 32));
  for (int block_index = 0; block_index < width / 32; ++block_index) {
    const float *source = input + block_index * 32;
    Q8Block &destination = result[block_index];
    float maximum_absolute = 0.0f;
    for (int index = 0; index < 32; ++index)
      maximum_absolute = std::max(maximum_absolute, std::fabs(source[index]));
    const float scale = maximum_absolute / 127.0f;
    const float inverse_scale = scale == 0.0f ? 0.0f : 1.0f / scale;
    destination.d = float_to_half(scale);
    for (int index = 0; index < 32; ++index)
      destination.qs[index] =
          static_cast<std::int8_t>(std::round(source[index] * inverse_scale));
  }
  return result;
}

void q4k_batch_row_scalar(const std::uint8_t *row, const BatchQ8 &batch,
                          float *output, int output_row, int output_width,
                          const float *bias) {
  for (int sample = 0; sample < batch.samples; ++sample)
    output[static_cast<std::size_t>(sample) * output_width + output_row] =
        bias ? bias[output_row] : 0.0f;
  const Q4KBlock *blocks = reinterpret_cast<const Q4KBlock *>(row);
  for (int block = 0; block < batch.blocks; ++block) {
    std::uint8_t scales[8];
    std::uint8_t mins[8];
    unpack_q4k(blocks[block], scales, mins);
    const float d = half_to_float(blocks[block].d);
    const float dmin = half_to_float(blocks[block].dmin);
    for (int sub = 0; sub < 8; ++sub) {
      int quantized[32];
      const int source = (sub / 2) * 32;
      for (int lane = 0; lane < 32; ++lane)
        quantized[lane] = (sub & 1) == 0
                              ? (blocks[block].qs[source + lane] & 0x0f)
                              : (blocks[block].qs[source + lane] >> 4);
      for (int sample = 0; sample < batch.samples; ++sample) {
        int dot = 0;
        for (int lane = 0; lane < 32; ++lane)
          dot += quantized[lane] *
                 batch.values[static_cast<std::size_t>(block * 256 + sub * 32 +
                                                       lane) *
                                  batch.samples +
                              sample];
        const int sum =
            batch.sums[static_cast<std::size_t>(block * 16 + sub * 2) *
                           batch.samples +
                       sample] +
            batch.sums[static_cast<std::size_t>(block * 16 + sub * 2 + 1) *
                           batch.samples +
                       sample];
        output[static_cast<std::size_t>(sample) * output_width + output_row] +=
            batch.scales[static_cast<std::size_t>(block) * batch.samples +
                         sample] *
            (d * scales[sub] * static_cast<float>(dot) -
             dmin * mins[sub] * static_cast<float>(sum));
      }
    }
  }
}

#if PARAKEET_POC_X86
[[maybe_unused]] PARAKEET_POC_TARGET_AVX2 void
q4k_batch_row_avx2(const std::uint8_t *row, const BatchQ8 &batch, float *output,
                   int output_row, int output_width, const float *bias) {
  constexpr int max_blocks = 64;
  if (batch.blocks > max_blocks) {
    q4k_batch_row_scalar(row, batch, output, output_row, output_width, bias);
    return;
  }
  std::uint8_t quantized[max_blocks][8][32];
  float scales[max_blocks][8];
  float mins[max_blocks][8];
  const Q4KBlock *blocks = reinterpret_cast<const Q4KBlock *>(row);
  for (int block = 0; block < batch.blocks; ++block) {
    std::uint8_t block_scales[8];
    std::uint8_t block_mins[8];
    unpack_q4k(blocks[block], block_scales, block_mins);
    const float d = half_to_float(blocks[block].d);
    const float dmin = half_to_float(blocks[block].dmin);
    for (int sub = 0; sub < 8; ++sub) {
      scales[block][sub] = d * static_cast<float>(block_scales[sub]);
      mins[block][sub] = dmin * static_cast<float>(block_mins[sub]);
      const int source = (sub / 2) * 32;
      for (int lane = 0; lane < 32; ++lane)
        quantized[block][sub][lane] =
            (sub & 1) == 0 ? (blocks[block].qs[source + lane] & 0x0f)
                           : (blocks[block].qs[source + lane] >> 4);
    }
  }
  for (int sample = 0; sample < batch.samples; ++sample)
    output[static_cast<std::size_t>(sample) * output_width + output_row] =
        bias ? bias[output_row] : 0.0f;
  int sample = 0;
  for (; sample + 8 <= batch.samples; sample += 8) {
    __m256 accumulated = _mm256_setzero_ps();
    for (int block = 0; block < batch.blocks; ++block) {
      for (int sub = 0; sub < 8; ++sub) {
        __m256i dots = _mm256_setzero_si256();
        for (int lane = 0; lane < 32; ++lane) {
          const __m128i values =
              _mm_loadl_epi64(reinterpret_cast<const __m128i *>(
                  batch.values.data() +
                  (static_cast<std::size_t>(block * 256 + sub * 32 + lane) *
                       batch.samples +
                   sample)));
          dots = _mm256_add_epi32(
              dots, _mm256_mullo_epi32(
                        _mm256_cvtepi8_epi32(values),
                        _mm256_set1_epi32(quantized[block][sub][lane])));
        }
        const __m128i sums0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(
            batch.sums.data() +
            (static_cast<std::size_t>(block * 16 + sub * 2) * batch.samples +
             sample)));
        const __m128i sums1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(
            batch.sums.data() +
            (static_cast<std::size_t>(block * 16 + sub * 2 + 1) *
                 batch.samples +
             sample)));
        const __m256i sums = _mm256_add_epi32(_mm256_cvtepi16_epi32(sums0),
                                              _mm256_cvtepi16_epi32(sums1));
        const __m256 dot_values = _mm256_cvtepi32_ps(dots);
        const __m256 sum_values = _mm256_cvtepi32_ps(sums);
        const __m256 contribution = _mm256_sub_ps(
            _mm256_mul_ps(_mm256_set1_ps(scales[block][sub]), dot_values),
            _mm256_mul_ps(_mm256_set1_ps(mins[block][sub]), sum_values));
        accumulated = _mm256_add_ps(
            accumulated,
            _mm256_mul_ps(_mm256_loadu_ps(batch.scales.data() +
                                          static_cast<std::size_t>(block) *
                                              batch.samples +
                                          sample),
                          contribution));
      }
    }
    float values[8];
    _mm256_storeu_ps(values, accumulated);
    for (int lane = 0; lane < 8; ++lane)
      output[static_cast<std::size_t>(sample + lane) * output_width +
             output_row] += values[lane];
  }
  for (; sample < batch.samples; ++sample) {
    for (int block = 0; block < batch.blocks; ++block) {
      for (int sub = 0; sub < 8; ++sub) {
        int dot = 0;
        for (int lane = 0; lane < 32; ++lane)
          dot += quantized[block][sub][lane] *
                 batch.values[static_cast<std::size_t>(block * 256 + sub * 32 +
                                                       lane) *
                                  batch.samples +
                              sample];
        const int sum =
            batch.sums[static_cast<std::size_t>(block * 16 + sub * 2) *
                           batch.samples +
                       sample] +
            batch.sums[static_cast<std::size_t>(block * 16 + sub * 2 + 1) *
                           batch.samples +
                       sample];
        output[static_cast<std::size_t>(sample) * output_width + output_row] +=
            batch.scales[static_cast<std::size_t>(block) * batch.samples +
                         sample] *
            (scales[block][sub] * static_cast<float>(dot) -
             mins[block][sub] * static_cast<float>(sum));
      }
    }
  }
}
#endif

#if PARAKEET_POC_X86
[[maybe_unused]] PARAKEET_POC_TARGET_AVX2 void
packed_q4k_rows2(const std::uint8_t *weight, std::size_t block_stride,
                 const BatchQ8 &batch, float *output, int output_width,
                 int group, int row_offset, const float *bias) {
  const int output_row0 = group * 8 + row_offset;
  const int output_row1 = output_row0 + 1;
  int sample = 0;
  for (; sample + 8 <= batch.samples; sample += 8) {
    float values0[8];
    float values1[8];
    for (int lane = 0; lane < 8; ++lane) {
      values0[lane] =
          bias && output_row0 < output_width ? bias[output_row0] : 0.0f;
      values1[lane] =
          bias && output_row1 < output_width ? bias[output_row1] : 0.0f;
    }
    for (int block = 0; block < batch.blocks; ++block) {
      const auto *block_weight = reinterpret_cast<const PackedQ4K8 *>(
          weight + static_cast<std::size_t>(block) * block_stride);
      float weighted0[8] = {};
      float weighted1[8] = {};
      int correction0[8] = {};
      int correction1[8] = {};
      for (int part = 0; part < 4; ++part) {
        const std::uint8_t *q0 = block_weight->q[row_offset] + part * 64;
        const std::uint8_t *q1 = block_weight->q[row_offset + 1] + part * 64;
        __m256i low0[8];
        __m256i high0[8];
        __m256i low1[8];
        __m256i high1[8];
        for (int lane = 0; lane < 8; ++lane) {
          low0[lane] = _mm256_setzero_si256();
          high0[lane] = _mm256_setzero_si256();
          low1[lane] = _mm256_setzero_si256();
          high1[lane] = _mm256_setzero_si256();
        }
        for (int lane = 0; lane < 8; ++lane) {
          const std::int8_t *q8 =
              batch.sample_values.data() +
              static_cast<std::size_t>(sample + lane) * batch.input +
              block * 256 + part * 64;
          const __m256i q8_low =
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8));
          const __m256i q8_high =
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8 + 32));
          low0[lane] = _mm256_maddubs_epi16(
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q0)),
              q8_low);
          low0[lane] = _mm256_madd_epi16(low0[lane], _mm256_set1_epi16(1));
          high0[lane] = _mm256_maddubs_epi16(
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q0 + 32)),
              q8_high);
          high0[lane] = _mm256_madd_epi16(high0[lane], _mm256_set1_epi16(1));
          low1[lane] = _mm256_maddubs_epi16(
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q1)),
              q8_low);
          low1[lane] = _mm256_madd_epi16(low1[lane], _mm256_set1_epi16(1));
          high1[lane] = _mm256_maddubs_epi16(
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q1 + 32)),
              q8_high);
          high1[lane] = _mm256_madd_epi16(high1[lane], _mm256_set1_epi16(1));
        }
        for (int lane = 0; lane < 8; ++lane) {
          weighted0[lane] += block_weight->scale[2 * part][row_offset] *
                                 horizontal_sum(low0[lane]) +
                             block_weight->scale[2 * part + 1][row_offset] *
                                 horizontal_sum(high0[lane]);
          weighted1[lane] += block_weight->scale[2 * part][row_offset + 1] *
                                 horizontal_sum(low1[lane]) +
                             block_weight->scale[2 * part + 1][row_offset + 1] *
                                 horizontal_sum(high1[lane]);
        }
      }
      for (int sub = 0; sub < 8; ++sub) {
        for (int lane = 0; lane < 8; ++lane) {
          const int sum =
              batch.sums[static_cast<std::size_t>(block * 16 + sub * 2) *
                             batch.samples +
                         sample + lane] +
              batch.sums[static_cast<std::size_t>(block * 16 + sub * 2 + 1) *
                             batch.samples +
                         sample + lane];
          correction0[lane] += block_weight->min[sub][row_offset] * sum;
          correction1[lane] += block_weight->min[sub][row_offset + 1] * sum;
        }
      }
      for (int lane = 0; lane < 8; ++lane) {
        const float activation_scale =
            batch.scales[static_cast<std::size_t>(block) * batch.samples +
                         sample + lane];
        values0[lane] +=
            activation_scale * (weighted0[lane] - correction0[lane]);
        values1[lane] +=
            activation_scale * (weighted1[lane] - correction1[lane]);
      }
    }
    for (int lane = 0; lane < 8; ++lane) {
      if (output_row0 < output_width)
        output[static_cast<std::size_t>(sample + lane) * output_width +
               output_row0] = values0[lane];
      if (output_row1 < output_width)
        output[static_cast<std::size_t>(sample + lane) * output_width +
               output_row1] = values1[lane];
    }
  }
  for (; sample < batch.samples; ++sample) {
    for (int row = 0; row < 2; ++row) {
      const int output_row = output_row0 + row;
      if (output_row >= output_width)
        continue;
      float value = bias ? bias[output_row] : 0.0f;
      for (int block = 0; block < batch.blocks; ++block) {
        const auto *block_weight = reinterpret_cast<const PackedQ4K8 *>(
            weight + static_cast<std::size_t>(block) * block_stride);
        for (int sub = 0; sub < 8; ++sub) {
          int dot = 0;
          for (int lane = 0; lane < 32; ++lane)
            dot += block_weight->q[row_offset + row][sub * 32 + lane] *
                   batch.values[static_cast<std::size_t>(block * 256 +
                                                         sub * 32 + lane) *
                                    batch.samples +
                                sample];
          const int sum =
              batch.sums[static_cast<std::size_t>(block * 16 + sub * 2) *
                             batch.samples +
                         sample] +
              batch.sums[static_cast<std::size_t>(block * 16 + sub * 2 + 1) *
                             batch.samples +
                         sample];
          value +=
              batch.scales[static_cast<std::size_t>(block) * batch.samples +
                           sample] *
              (block_weight->scale[sub][row_offset + row] * dot -
               block_weight->min[sub][row_offset + row] * sum);
        }
      }
      output[static_cast<std::size_t>(sample) * output_width + output_row] =
          value;
    }
  }
}
#endif

#if PARAKEET_POC_X86
PARAKEET_POC_TARGET_AVX2 void packed_q4k8_group(const std::uint8_t *weight,
                                                std::size_t block_stride,
                                                const BatchQ8 &batch,
                                                float *output, int output_width,
                                                int group, const float *bias) {
  for (int sample = 0; sample < batch.samples; ++sample)
    for (int row = 0; row < 8; ++row) {
      const int output_row = group * 8 + row;
      if (output_row < output_width)
        output[static_cast<std::size_t>(sample) * output_width + output_row] =
            bias ? bias[output_row] : 0.0f;
    }

  int sample = 0;
  for (; sample + 8 <= batch.samples; sample += 8) {
    __m256 accumulated[8];
    for (int row = 0; row < 8; ++row)
      accumulated[row] = _mm256_set1_ps(bias && group * 8 + row < output_width
                                            ? bias[group * 8 + row]
                                            : 0.0f);
    for (int block = 0; block < batch.blocks; ++block) {
      const auto *block_weight = reinterpret_cast<const PackedQ4K8 *>(
          weight + static_cast<std::size_t>(block) * block_stride);
      const __m256 activation_scale = _mm256_loadu_ps(
          batch.scales.data() +
          static_cast<std::size_t>(block) * batch.samples + sample);
      for (int sub = 0; sub < 8; ++sub) {
        __m256i dots[8];
        for (int row = 0; row < 8; ++row)
          dots[row] = _mm256_setzero_si256();
        for (int lane = 0; lane < 32; ++lane) {
          const int local_element = sub * 32 + lane;
          const int element = block * 256 + local_element;
          const __m256i values = _mm256_cvtepi8_epi32(
              _mm_loadl_epi64(reinterpret_cast<const __m128i *>(
                  batch.values.data() +
                  static_cast<std::size_t>(element) * batch.samples + sample)));
          dots[0] = _mm256_add_epi32(
              dots[0], _mm256_mullo_epi32(
                           _mm256_set1_epi32(block_weight->q[0][local_element]),
                           values));
          dots[1] = _mm256_add_epi32(
              dots[1], _mm256_mullo_epi32(
                           _mm256_set1_epi32(block_weight->q[1][local_element]),
                           values));
          dots[2] = _mm256_add_epi32(
              dots[2], _mm256_mullo_epi32(
                           _mm256_set1_epi32(block_weight->q[2][local_element]),
                           values));
          dots[3] = _mm256_add_epi32(
              dots[3], _mm256_mullo_epi32(
                           _mm256_set1_epi32(block_weight->q[3][local_element]),
                           values));
          dots[4] = _mm256_add_epi32(
              dots[4], _mm256_mullo_epi32(
                           _mm256_set1_epi32(block_weight->q[4][local_element]),
                           values));
          dots[5] = _mm256_add_epi32(
              dots[5], _mm256_mullo_epi32(
                           _mm256_set1_epi32(block_weight->q[5][local_element]),
                           values));
          dots[6] = _mm256_add_epi32(
              dots[6], _mm256_mullo_epi32(
                           _mm256_set1_epi32(block_weight->q[6][local_element]),
                           values));
          dots[7] = _mm256_add_epi32(
              dots[7], _mm256_mullo_epi32(
                           _mm256_set1_epi32(block_weight->q[7][local_element]),
                           values));
        }
        const __m128i sums0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(
            batch.sums.data() +
            static_cast<std::size_t>(block * 16 + sub * 2) * batch.samples +
            sample));
        const __m128i sums1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(
            batch.sums.data() +
            static_cast<std::size_t>(block * 16 + sub * 2 + 1) * batch.samples +
            sample));
        const __m256i sums = _mm256_cvtepi16_epi32(_mm_add_epi16(sums0, sums1));
        for (int row = 0; row < 8; ++row) {
          const __m256 contribution = _mm256_sub_ps(
              _mm256_mul_ps(_mm256_set1_ps(block_weight->scale[sub][row]),
                            _mm256_cvtepi32_ps(dots[row])),
              _mm256_mul_ps(_mm256_set1_ps(block_weight->min[sub][row]),
                            _mm256_cvtepi32_ps(sums)));
          accumulated[row] =
              _mm256_fmadd_ps(activation_scale, contribution, accumulated[row]);
        }
      }
    }
    for (int row = 0; row < 8; ++row) {
      float values[8];
      _mm256_storeu_ps(values, accumulated[row]);
      const int output_row = group * 8 + row;
      if (output_row < output_width)
        for (int lane = 0; lane < 8; ++lane)
          output[static_cast<std::size_t>(sample + lane) * output_width +
                 output_row] = values[lane];
    }
  }

  for (; sample < batch.samples; ++sample) {
    for (int block = 0; block < batch.blocks; ++block) {
      const auto *block_weight = reinterpret_cast<const PackedQ4K8 *>(
          weight + static_cast<std::size_t>(block) * block_stride);
      const float activation_scale =
          batch
              .scales[static_cast<std::size_t>(block) * batch.samples + sample];
      for (int sub = 0; sub < 8; ++sub) {
        int dots[8] = {};
        for (int lane = 0; lane < 32; ++lane) {
          const int local_element = sub * 32 + lane;
          const int element = block * 256 + local_element;
          for (int row = 0; row < 8; ++row)
            dots[row] +=
                block_weight->q[row][local_element] *
                batch.values[static_cast<std::size_t>(element) * batch.samples +
                             sample];
        }
        const int sum =
            batch.sums[static_cast<std::size_t>(block * 16 + sub * 2) *
                           batch.samples +
                       sample] +
            batch.sums[static_cast<std::size_t>(block * 16 + sub * 2 + 1) *
                           batch.samples +
                       sample];
        for (int row = 0; row < 8; ++row) {
          const int output_row = group * 8 + row;
          if (output_row < output_width)
            output[static_cast<std::size_t>(sample) * output_width +
                   output_row] +=
                activation_scale *
                (block_weight->scale[sub][row] * static_cast<float>(dots[row]) -
                 block_weight->min[sub][row] * static_cast<float>(sum));
        }
      }
    }
  }
}

PARAKEET_POC_TARGET_AVX2 void packed_q6k8_group(const std::uint8_t *weight,
                                                std::size_t block_stride,
                                                const BatchQ8 &batch,
                                                float *output, int output_width,
                                                int group, const float *bias) {
  for (int sample = 0; sample < batch.samples; ++sample)
    for (int row = 0; row < 8; ++row) {
      const int output_row = group * 8 + row;
      if (output_row < output_width)
        output[static_cast<std::size_t>(sample) * output_width + output_row] =
            bias ? bias[output_row] : 0.0f;
    }

  int sample = 0;
  for (; sample + 8 <= batch.samples; sample += 8) {
    __m256 accumulated[8];
    for (int row = 0; row < 8; ++row)
      accumulated[row] = _mm256_set1_ps(bias && group * 8 + row < output_width
                                            ? bias[group * 8 + row]
                                            : 0.0f);
    for (int block = 0; block < batch.blocks; ++block) {
      const auto *block_weight = reinterpret_cast<const PackedQ6K8 *>(
          weight + static_cast<std::size_t>(block) * block_stride);
      const __m256 activation_scale = _mm256_loadu_ps(
          batch.scales.data() +
          static_cast<std::size_t>(block) * batch.samples + sample);
      for (int chunk = 0; chunk < 16; ++chunk) {
        const int base =
            chunk / 8 * 128 + (chunk % 8) / 2 * 32 + (chunk % 2) * 16;
        __m256i dots[8];
        for (int row = 0; row < 8; ++row)
          dots[row] = _mm256_setzero_si256();
        for (int lane = 0; lane < 16; ++lane) {
          const int local_element = base + lane;
          const int element = block * 256 + local_element;
          const __m256i values = _mm256_cvtepi8_epi32(
              _mm_loadl_epi64(reinterpret_cast<const __m128i *>(
                  batch.values.data() +
                  static_cast<std::size_t>(element) * batch.samples + sample)));
          const std::int8_t *q0 = block_weight->q[0];
          const std::int8_t *q1 = block_weight->q[1];
          const std::int8_t *q2 = block_weight->q[2];
          const std::int8_t *q3 = block_weight->q[3];
          const std::int8_t *q4 = block_weight->q[4];
          const std::int8_t *q5 = block_weight->q[5];
          const std::int8_t *q6 = block_weight->q[6];
          const std::int8_t *q7 = block_weight->q[7];
          dots[0] = _mm256_add_epi32(
              dots[0],
              _mm256_mullo_epi32(_mm256_set1_epi32(q0[local_element]), values));
          dots[1] = _mm256_add_epi32(
              dots[1],
              _mm256_mullo_epi32(_mm256_set1_epi32(q1[local_element]), values));
          dots[2] = _mm256_add_epi32(
              dots[2],
              _mm256_mullo_epi32(_mm256_set1_epi32(q2[local_element]), values));
          dots[3] = _mm256_add_epi32(
              dots[3],
              _mm256_mullo_epi32(_mm256_set1_epi32(q3[local_element]), values));
          dots[4] = _mm256_add_epi32(
              dots[4],
              _mm256_mullo_epi32(_mm256_set1_epi32(q4[local_element]), values));
          dots[5] = _mm256_add_epi32(
              dots[5],
              _mm256_mullo_epi32(_mm256_set1_epi32(q5[local_element]), values));
          dots[6] = _mm256_add_epi32(
              dots[6],
              _mm256_mullo_epi32(_mm256_set1_epi32(q6[local_element]), values));
          dots[7] = _mm256_add_epi32(
              dots[7],
              _mm256_mullo_epi32(_mm256_set1_epi32(q7[local_element]), values));
        }
        for (int row = 0; row < 8; ++row) {
          const __m256 contribution =
              _mm256_mul_ps(_mm256_set1_ps(block_weight->scale[chunk][row]),
                            _mm256_cvtepi32_ps(dots[row]));
          accumulated[row] =
              _mm256_fmadd_ps(activation_scale, contribution, accumulated[row]);
        }
      }
    }
    for (int row = 0; row < 8; ++row) {
      float values[8];
      _mm256_storeu_ps(values, accumulated[row]);
      const int output_row = group * 8 + row;
      if (output_row < output_width)
        for (int lane = 0; lane < 8; ++lane)
          output[static_cast<std::size_t>(sample + lane) * output_width +
                 output_row] = values[lane];
    }
  }

  for (; sample < batch.samples; ++sample) {
    for (int block = 0; block < batch.blocks; ++block) {
      const auto *block_weight = reinterpret_cast<const PackedQ6K8 *>(
          weight + static_cast<std::size_t>(block) * block_stride);
      const float activation_scale =
          batch
              .scales[static_cast<std::size_t>(block) * batch.samples + sample];
      for (int chunk = 0; chunk < 16; ++chunk) {
        const int base =
            chunk / 8 * 128 + (chunk % 8) / 2 * 32 + (chunk % 2) * 16;
        int dot[8] = {};
        for (int lane = 0; lane < 16; ++lane) {
          const int local_element = base + lane;
          const int element = block * 256 + local_element;
          for (int row = 0; row < 8; ++row)
            dot[row] +=
                block_weight->q[row][local_element] *
                batch.values[static_cast<std::size_t>(element) * batch.samples +
                             sample];
        }
        for (int row = 0; row < 8; ++row) {
          const int output_row = group * 8 + row;
          if (output_row < output_width)
            output[static_cast<std::size_t>(sample) * output_width +
                   output_row] += activation_scale *
                                  block_weight->scale[chunk][row] *
                                  static_cast<float>(dot[row]);
        }
      }
    }
  }
}
#endif

float dot_q4k_q8k(const std::uint8_t *row, const std::vector<Q8KBlock> &input,
                  int width, bool vector) {
#if PARAKEET_POC_X86
  if (vector)
    return dot_q4k_q8k_avx2_fast(row, input.data(), width);
#endif
  const Q4KBlock *blocks = reinterpret_cast<const Q4KBlock *>(row);
  float result = 0.0f;
  for (int block_index = 0; block_index < width / 256; ++block_index) {
    const Q4KBlock &block = blocks[block_index];
    const Q8KBlock &activation = input[block_index];
    std::uint8_t scales[8];
    std::uint8_t mins[8];
    unpack_q4k(block, scales, mins);
    const float d = half_to_float(block.d);
    const float dmin = half_to_float(block.dmin);
    float block_sum = 0.0f;
    for (int sub = 0; sub < 8; ++sub) {
      const int source = (sub / 2) * 32;
      const std::uint8_t *weights = block.qs + source;
      const std::int8_t *activations = activation.qs + sub * 32;
      int dot = 0;
      if (vector) {
#if PARAKEET_POC_X86
        dot = dot_q4_sub_avx2(weights, activations, (sub & 1) != 0);
#else
        dot = dot_unsigned_signed_16_scalar(weights, activations,
                                            (sub & 1) != 0) +
              dot_unsigned_signed_16_scalar(weights + 16, activations + 16,
                                            (sub & 1) != 0);
#endif
      } else {
        for (int lane = 0; lane < 32; ++lane) {
          const int quantized =
              (sub & 1) == 0 ? (weights[lane] & 0x0f) : (weights[lane] >> 4);
          dot += quantized * static_cast<int>(activations[lane]);
        }
      }
      const int activation_sum =
          activation.bsums[sub * 2] + activation.bsums[sub * 2 + 1];
      block_sum +=
          d * static_cast<float>(scales[sub]) * static_cast<float>(dot) -
          dmin * static_cast<float>(mins[sub]) *
              static_cast<float>(activation_sum);
    }
    result += activation.d * block_sum;
  }
  return result;
}

#if PARAKEET_POC_X86
[[maybe_unused]] PARAKEET_POC_TARGET_AVX2 void
q4k_batch_row_avx2_fast(const std::uint8_t *row, const BatchQ8 &batch,
                        float *output, int output_row, int output_width,
                        const float *bias) {
  for (int sample = 0; sample < batch.samples; ++sample)
    output[static_cast<std::size_t>(sample) * output_width + output_row] =
        bias ? bias[output_row] : 0.0f;
  int sample = 0;
  for (; sample + 8 <= batch.samples; sample += 8) {
    float values[8];
    for (int lane = 0; lane < 8; ++lane)
      values[lane] = bias ? bias[output_row] : 0.0f;
    const Q4KBlock *blocks = reinterpret_cast<const Q4KBlock *>(row);
    for (int block = 0; block < batch.blocks; ++block) {
      const Q4KBlock &weight = blocks[block];
      std::uint8_t scales[8];
      std::uint8_t mins[8];
      unpack_q4k(weight, scales, mins);
      const float d = half_to_float(weight.d);
      const float dmin = half_to_float(weight.dmin);
      __m256i sums[8];
      for (int lane = 0; lane < 8; ++lane)
        sums[lane] = _mm256_setzero_si256();
      const std::uint8_t *q4 = weight.qs;
      for (int part = 0; part < 4; ++part) {
        const __m256i q4_bits =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q4));
        q4 += 32;
        const __m256i q4_low =
            _mm256_and_si256(q4_bits, _mm256_set1_epi8(0x0f));
        const __m256i q4_high = _mm256_and_si256(_mm256_srli_epi16(q4_bits, 4),
                                                 _mm256_set1_epi8(0x0f));
        const __m256i scale_values = _mm256_setr_epi16(
            static_cast<short>(scales[0]), static_cast<short>(scales[1]),
            static_cast<short>(scales[2]), static_cast<short>(scales[3]),
            static_cast<short>(scales[4]), static_cast<short>(scales[5]),
            static_cast<short>(scales[6]), static_cast<short>(scales[7]),
            static_cast<short>(scales[0]), static_cast<short>(scales[1]),
            static_cast<short>(scales[2]), static_cast<short>(scales[3]),
            static_cast<short>(scales[4]), static_cast<short>(scales[5]),
            static_cast<short>(scales[6]), static_cast<short>(scales[7]));
        const __m256i scale_low =
            _mm256_shuffle_epi8(scale_values, q4_scale_shuffle(2 * part));
        const __m256i scale_high =
            _mm256_shuffle_epi8(scale_values, q4_scale_shuffle(2 * part + 1));
        for (int lane = 0; lane < 8; ++lane) {
          const std::int8_t *q8 =
              batch.sample_values.data() +
              static_cast<std::size_t>(sample + lane) * batch.input +
              block * 256 + part * 64;
          __m256i product_low = _mm256_maddubs_epi16(
              q4_low,
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8)));
          product_low = _mm256_madd_epi16(scale_low, product_low);
          __m256i product_high = _mm256_maddubs_epi16(
              q4_high,
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8 + 32)));
          product_high = _mm256_madd_epi16(scale_high, product_high);
          sums[lane] = _mm256_add_epi32(
              sums[lane], _mm256_add_epi32(product_low, product_high));
        }
      }
      for (int lane = 0; lane < 8; ++lane) {
        const int dot = horizontal_sum(sums[lane]);
        int correction = 0;
        for (int sub = 0; sub < 8; ++sub)
          correction +=
              static_cast<int>(mins[sub]) *
              (batch.sums[static_cast<std::size_t>(block * 16 + sub * 2) *
                              batch.samples +
                          sample + lane] +
               batch.sums[static_cast<std::size_t>(block * 16 + sub * 2 + 1) *
                              batch.samples +
                          sample + lane]);
        values[lane] +=
            batch.scales[static_cast<std::size_t>(block) * batch.samples +
                         sample + lane] *
            (d * static_cast<float>(dot) -
             dmin * static_cast<float>(correction));
      }
    }
    for (int lane = 0; lane < 8; ++lane)
      output[static_cast<std::size_t>(sample + lane) * output_width +
             output_row] = values[lane];
  }
  for (; sample < batch.samples; ++sample) {
    const Q4KBlock *blocks = reinterpret_cast<const Q4KBlock *>(row);
    for (int block = 0; block < batch.blocks; ++block) {
      const Q4KBlock &weight = blocks[block];
      std::uint8_t scales[8];
      std::uint8_t mins[8];
      unpack_q4k(weight, scales, mins);
      const float d = half_to_float(weight.d);
      const float dmin = half_to_float(weight.dmin);
      for (int sub = 0; sub < 8; ++sub) {
        const int source = (sub / 2) * 32;
        int dot = 0;
        for (int lane = 0; lane < 32; ++lane) {
          const int quantized = (sub & 1) == 0
                                    ? (weight.qs[source + lane] & 0x0f)
                                    : (weight.qs[source + lane] >> 4);
          dot += quantized * batch.values[static_cast<std::size_t>(
                                              block * 256 + sub * 32 + lane) *
                                              batch.samples +
                                          sample];
        }
        const int sum =
            batch.sums[static_cast<std::size_t>(block * 16 + sub * 2) *
                           batch.samples +
                       sample] +
            batch.sums[static_cast<std::size_t>(block * 16 + sub * 2 + 1) *
                           batch.samples +
                       sample];
        output[static_cast<std::size_t>(sample) * output_width + output_row] +=
            batch.scales[static_cast<std::size_t>(block) * batch.samples +
                         sample] *
            (d * scales[sub] * static_cast<float>(dot) -
             dmin * mins[sub] * static_cast<float>(sum));
      }
    }
  }
}
#endif

#if PARAKEET_POC_X86
PARAKEET_POC_TARGET_AVX2 void
q4k_batch_row_avx2_fast2(const std::uint8_t *row, const BatchQ8 &batch,
                         float *output, int output_row, int output_width,
                         const float *bias) {
  for (int sample = 0; sample < batch.samples; ++sample)
    output[static_cast<std::size_t>(sample) * output_width + output_row] =
        bias ? bias[output_row] : 0.0f;
  int sample = 0;
  for (; sample + 8 <= batch.samples; sample += 8) {
    float values[8];
    for (int lane = 0; lane < 8; ++lane)
      values[lane] = bias ? bias[output_row] : 0.0f;
    const Q4KBlock *blocks = reinterpret_cast<const Q4KBlock *>(row);
    for (int block = 0; block < batch.blocks; ++block) {
      const Q4KBlock &weight = blocks[block];
      std::uint8_t scales[8];
      std::uint8_t mins[8];
      unpack_q4k(weight, scales, mins);
      const float d = half_to_float(weight.d);
      const float dmin = half_to_float(weight.dmin);
      int weighted[8] = {};
      int correction[8] = {};
      const std::uint8_t *q4 = weight.qs;
      for (int part = 0; part < 4; ++part) {
        const __m256i q4_bits =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q4));
        q4 += 32;
        const __m256i q4_low =
            _mm256_and_si256(q4_bits, _mm256_set1_epi8(0x0f));
        const __m256i q4_high = _mm256_and_si256(_mm256_srli_epi16(q4_bits, 4),
                                                 _mm256_set1_epi8(0x0f));
        __m256i low_dots[8];
        __m256i high_dots[8];
        for (int lane = 0; lane < 8; ++lane) {
          low_dots[lane] = _mm256_setzero_si256();
          high_dots[lane] = _mm256_setzero_si256();
        }
        for (int lane = 0; lane < 8; ++lane) {
          const std::int8_t *q8 =
              batch.sample_values.data() +
              static_cast<std::size_t>(sample + lane) * batch.input +
              block * 256 + part * 64;
          low_dots[lane] = _mm256_maddubs_epi16(
              q4_low,
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8)));
          low_dots[lane] =
              _mm256_madd_epi16(low_dots[lane], _mm256_set1_epi16(1));
          high_dots[lane] = _mm256_maddubs_epi16(
              q4_high,
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8 + 32)));
          high_dots[lane] =
              _mm256_madd_epi16(high_dots[lane], _mm256_set1_epi16(1));
        }
        for (int lane = 0; lane < 8; ++lane)
          weighted[lane] +=
              scales[2 * part] * horizontal_sum(low_dots[lane]) +
              scales[2 * part + 1] * horizontal_sum(high_dots[lane]);
      }
      for (int lane = 0; lane < 8; ++lane) {
        for (int sub = 0; sub < 8; ++sub)
          correction[lane] +=
              static_cast<int>(mins[sub]) *
              (batch.sums[static_cast<std::size_t>(block * 16 + sub * 2) *
                              batch.samples +
                          sample + lane] +
               batch.sums[static_cast<std::size_t>(block * 16 + sub * 2 + 1) *
                              batch.samples +
                          sample + lane]);
        values[lane] +=
            batch.scales[static_cast<std::size_t>(block) * batch.samples +
                         sample + lane] *
            (d * static_cast<float>(weighted[lane]) -
             dmin * static_cast<float>(correction[lane]));
      }
    }
    for (int lane = 0; lane < 8; ++lane)
      output[static_cast<std::size_t>(sample + lane) * output_width +
             output_row] = values[lane];
  }
  for (; sample < batch.samples; ++sample) {
    const Q4KBlock *blocks = reinterpret_cast<const Q4KBlock *>(row);
    for (int block = 0; block < batch.blocks; ++block) {
      const Q4KBlock &weight = blocks[block];
      std::uint8_t scales[8];
      std::uint8_t mins[8];
      unpack_q4k(weight, scales, mins);
      const float d = half_to_float(weight.d);
      const float dmin = half_to_float(weight.dmin);
      for (int sub = 0; sub < 8; ++sub) {
        const int source = (sub / 2) * 32;
        int dot = 0;
        for (int lane = 0; lane < 32; ++lane) {
          const int quantized = (sub & 1) == 0
                                    ? (weight.qs[source + lane] & 0x0f)
                                    : (weight.qs[source + lane] >> 4);
          dot += quantized * batch.values[static_cast<std::size_t>(
                                              block * 256 + sub * 32 + lane) *
                                              batch.samples +
                                          sample];
        }
        const int sum =
            batch.sums[static_cast<std::size_t>(block * 16 + sub * 2) *
                           batch.samples +
                       sample] +
            batch.sums[static_cast<std::size_t>(block * 16 + sub * 2 + 1) *
                           batch.samples +
                       sample];
        output[static_cast<std::size_t>(sample) * output_width + output_row] +=
            batch.scales[static_cast<std::size_t>(block) * batch.samples +
                         sample] *
            (d * scales[sub] * static_cast<float>(dot) -
             dmin * mins[sub] * static_cast<float>(sum));
      }
    }
  }
}
#endif

#if PARAKEET_POC_X86
[[maybe_unused]] PARAKEET_POC_TARGET_AVX2 void
packed_q6k_rows2(const std::uint8_t *weight, std::size_t block_stride,
                 const BatchQ8 &batch, float *output, int output_width,
                 int group, int row_offset, const float *bias) {
  const int output_row0 = group * 8 + row_offset;
  const int output_row1 = output_row0 + 1;
  int sample = 0;
  for (; sample + 8 <= batch.samples; sample += 8) {
    float values0[8];
    float values1[8];
    for (int lane = 0; lane < 8; ++lane) {
      values0[lane] =
          bias && output_row0 < output_width ? bias[output_row0] : 0.0f;
      values1[lane] =
          bias && output_row1 < output_width ? bias[output_row1] : 0.0f;
    }
    for (int block = 0; block < batch.blocks; ++block) {
      const auto *block_weight = reinterpret_cast<const PackedQ6K8 *>(
          weight + static_cast<std::size_t>(block) * block_stride);
      float weighted0[8] = {};
      float weighted1[8] = {};
      const std::int8_t *q0 = block_weight->q[row_offset];
      const std::int8_t *q1 = block_weight->q[row_offset + 1];
      for (int pair = 0; pair < 8; ++pair) {
        const std::int8_t *q0_chunk = q0 + pair * 32;
        const std::int8_t *q1_chunk = q1 + pair * 32;
        const __m256i offset = _mm256_set1_epi8(static_cast<char>(-32));
        const __m256i q0_unsigned = _mm256_sub_epi8(
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q0_chunk)),
            offset);
        const __m256i q1_unsigned = _mm256_sub_epi8(
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q1_chunk)),
            offset);
        for (int lane = 0; lane < 8; ++lane) {
          const std::int8_t *q8 =
              batch.sample_values.data() +
              static_cast<std::size_t>(sample + lane) * batch.input +
              block * 256 + pair * 32;
          const __m256i q8_values =
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8));
          const __m256i products0 =
              _mm256_maddubs_epi16(q0_unsigned, q8_values);
          const __m256i products1 =
              _mm256_maddubs_epi16(q1_unsigned, q8_values);
          const __m256i dots0_low =
              _mm256_cvtepi16_epi32(_mm256_castsi256_si128(products0));
          const __m256i dots0_high =
              _mm256_cvtepi16_epi32(_mm256_extracti128_si256(products0, 1));
          const __m256i dots1_low =
              _mm256_cvtepi16_epi32(_mm256_castsi256_si128(products1));
          const __m256i dots1_high =
              _mm256_cvtepi16_epi32(_mm256_extracti128_si256(products1, 1));
          const int sum0 =
              batch.sums[static_cast<std::size_t>(block * 16 + pair * 2) *
                             batch.samples +
                         sample + lane];
          const int sum1 =
              batch.sums[static_cast<std::size_t>(block * 16 + pair * 2 + 1) *
                             batch.samples +
                         sample + lane];
          weighted0[lane] += block_weight->scale[pair * 2][row_offset] *
                                 (horizontal_sum(dots0_low) - 32 * sum0) +
                             block_weight->scale[pair * 2 + 1][row_offset] *
                                 (horizontal_sum(dots0_high) - 32 * sum1);
          weighted1[lane] += block_weight->scale[pair * 2][row_offset + 1] *
                                 (horizontal_sum(dots1_low) - 32 * sum0) +
                             block_weight->scale[pair * 2 + 1][row_offset + 1] *
                                 (horizontal_sum(dots1_high) - 32 * sum1);
        }
      }
      for (int lane = 0; lane < 8; ++lane) {
        const float activation_scale =
            batch.scales[static_cast<std::size_t>(block) * batch.samples +
                         sample + lane];
        values0[lane] += activation_scale * weighted0[lane];
        values1[lane] += activation_scale * weighted1[lane];
      }
    }
    for (int lane = 0; lane < 8; ++lane) {
      if (output_row0 < output_width)
        output[static_cast<std::size_t>(sample + lane) * output_width +
               output_row0] = values0[lane];
      if (output_row1 < output_width)
        output[static_cast<std::size_t>(sample + lane) * output_width +
               output_row1] = values1[lane];
    }
  }
  for (; sample < batch.samples; ++sample) {
    for (int row = 0; row < 2; ++row) {
      const int output_row = output_row0 + row;
      if (output_row >= output_width)
        continue;
      float value = bias ? bias[output_row] : 0.0f;
      for (int block = 0; block < batch.blocks; ++block) {
        const auto *block_weight = reinterpret_cast<const PackedQ6K8 *>(
            weight + static_cast<std::size_t>(block) * block_stride);
        for (int chunk = 0; chunk < 16; ++chunk) {
          const int base =
              chunk / 8 * 128 + (chunk % 8) / 2 * 32 + (chunk % 2) * 16;
          int dot = 0;
          for (int lane = 0; lane < 16; ++lane)
            dot += block_weight->q[row_offset + row][base + lane] *
                   batch.values[static_cast<std::size_t>(block * 256 + base +
                                                         lane) *
                                    batch.samples +
                                sample];
          value +=
              batch.scales[static_cast<std::size_t>(block) * batch.samples +
                           sample] *
              block_weight->scale[chunk][row_offset + row] * dot;
        }
      }
      output[static_cast<std::size_t>(sample) * output_width + output_row] =
          value;
    }
  }
}
#endif

#if PARAKEET_POC_X86
PARAKEET_POC_TARGET_AVX2 void
q4k_rows2_avx2(const std::uint8_t *row0, const std::uint8_t *row1,
               const BatchQ8 &batch, float *output, int output_row0,
               int output_row1, int output_width, const float *bias) {
  const Q4KBlock *blocks0 = reinterpret_cast<const Q4KBlock *>(row0);
  const Q4KBlock *blocks1 = reinterpret_cast<const Q4KBlock *>(row1);
  int sample = 0;
  for (; sample + 8 <= batch.samples; sample += 8) {
    float values0[8];
    float values1[8];
    for (int lane = 0; lane < 8; ++lane) {
      values0[lane] = bias ? bias[output_row0] : 0.0f;
      values1[lane] = bias ? bias[output_row1] : 0.0f;
    }
    for (int block = 0; block < batch.blocks; ++block) {
      const Q4KBlock &weight0 = blocks0[block];
      const Q4KBlock &weight1 = blocks1[block];
      std::uint8_t scales0[8];
      std::uint8_t mins0[8];
      std::uint8_t scales1[8];
      std::uint8_t mins1[8];
      unpack_q4k(weight0, scales0, mins0);
      unpack_q4k(weight1, scales1, mins1);
      const float d0 = half_to_float(weight0.d);
      const float dmin0 = half_to_float(weight0.dmin);
      const float d1 = half_to_float(weight1.d);
      const float dmin1 = half_to_float(weight1.dmin);
      __m256i dots0[8];
      __m256i dots1[8];
      for (int lane = 0; lane < 8; ++lane) {
        dots0[lane] = _mm256_setzero_si256();
        dots1[lane] = _mm256_setzero_si256();
      }
      const std::uint8_t *q40 = weight0.qs;
      const std::uint8_t *q41 = weight1.qs;
      for (int part = 0; part < 4; ++part) {
        const __m256i q4_bits0 =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q40));
        const __m256i q4_bits1 =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q41));
        q40 += 32;
        q41 += 32;
        const __m256i q4_low0 =
            _mm256_and_si256(q4_bits0, _mm256_set1_epi8(0x0f));
        const __m256i q4_high0 = _mm256_and_si256(
            _mm256_srli_epi16(q4_bits0, 4), _mm256_set1_epi8(0x0f));
        const __m256i q4_low1 =
            _mm256_and_si256(q4_bits1, _mm256_set1_epi8(0x0f));
        const __m256i q4_high1 = _mm256_and_si256(
            _mm256_srli_epi16(q4_bits1, 4), _mm256_set1_epi8(0x0f));
        const __m256i scale_values0 = _mm256_setr_epi16(
            static_cast<short>(scales0[0]), static_cast<short>(scales0[1]),
            static_cast<short>(scales0[2]), static_cast<short>(scales0[3]),
            static_cast<short>(scales0[4]), static_cast<short>(scales0[5]),
            static_cast<short>(scales0[6]), static_cast<short>(scales0[7]),
            static_cast<short>(scales0[0]), static_cast<short>(scales0[1]),
            static_cast<short>(scales0[2]), static_cast<short>(scales0[3]),
            static_cast<short>(scales0[4]), static_cast<short>(scales0[5]),
            static_cast<short>(scales0[6]), static_cast<short>(scales0[7]));
        const __m256i scale_values1 = _mm256_setr_epi16(
            static_cast<short>(scales1[0]), static_cast<short>(scales1[1]),
            static_cast<short>(scales1[2]), static_cast<short>(scales1[3]),
            static_cast<short>(scales1[4]), static_cast<short>(scales1[5]),
            static_cast<short>(scales1[6]), static_cast<short>(scales1[7]),
            static_cast<short>(scales1[0]), static_cast<short>(scales1[1]),
            static_cast<short>(scales1[2]), static_cast<short>(scales1[3]),
            static_cast<short>(scales1[4]), static_cast<short>(scales1[5]),
            static_cast<short>(scales1[6]), static_cast<short>(scales1[7]));
        const __m256i scale_low0 =
            _mm256_shuffle_epi8(scale_values0, q4_scale_shuffle(2 * part));
        const __m256i scale_high0 =
            _mm256_shuffle_epi8(scale_values0, q4_scale_shuffle(2 * part + 1));
        const __m256i scale_low1 =
            _mm256_shuffle_epi8(scale_values1, q4_scale_shuffle(2 * part));
        const __m256i scale_high1 =
            _mm256_shuffle_epi8(scale_values1, q4_scale_shuffle(2 * part + 1));
        for (int lane = 0; lane < 8; ++lane) {
          const std::int8_t *q8 =
              batch.sample_values.data() +
              static_cast<std::size_t>(sample + lane) * batch.input +
              block * 256 + part * 64;
          const __m256i q8_low =
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8));
          const __m256i q8_high =
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8 + 32));
          __m256i product0 = _mm256_maddubs_epi16(q4_low0, q8_low);
          product0 = _mm256_madd_epi16(scale_low0, product0);
          __m256i product0_high = _mm256_maddubs_epi16(q4_high0, q8_high);
          product0_high = _mm256_madd_epi16(scale_high0, product0_high);
          product0 = _mm256_add_epi32(product0, product0_high);
          __m256i product1 = _mm256_maddubs_epi16(q4_low1, q8_low);
          product1 = _mm256_madd_epi16(scale_low1, product1);
          __m256i product1_high = _mm256_maddubs_epi16(q4_high1, q8_high);
          product1_high = _mm256_madd_epi16(scale_high1, product1_high);
          product1 = _mm256_add_epi32(product1, product1_high);
          dots0[lane] = _mm256_add_epi32(dots0[lane], product0);
          dots1[lane] = _mm256_add_epi32(dots1[lane], product1);
        }
      }
      for (int lane = 0; lane < 8; ++lane) {
        int correction0 = 0;
        int correction1 = 0;
        for (int sub = 0; sub < 8; ++sub) {
          const int sum =
              batch.sums[static_cast<std::size_t>(block * 16 + sub * 2) *
                             batch.samples +
                         sample + lane] +
              batch.sums[static_cast<std::size_t>(block * 16 + sub * 2 + 1) *
                             batch.samples +
                         sample + lane];
          correction0 += static_cast<int>(mins0[sub]) * sum;
          correction1 += static_cast<int>(mins1[sub]) * sum;
        }
        const float activation_scale =
            batch.scales[static_cast<std::size_t>(block) * batch.samples +
                         sample + lane];
        values0[lane] += activation_scale *
                         (d0 * static_cast<float>(horizontal_sum(dots0[lane])) -
                          dmin0 * static_cast<float>(correction0));
        values1[lane] += activation_scale *
                         (d1 * static_cast<float>(horizontal_sum(dots1[lane])) -
                          dmin1 * static_cast<float>(correction1));
      }
    }
    for (int lane = 0; lane < 8; ++lane) {
      output[static_cast<std::size_t>(sample + lane) * output_width +
             output_row0] = values0[lane];
      output[static_cast<std::size_t>(sample + lane) * output_width +
             output_row1] = values1[lane];
    }
  }
  for (; sample < batch.samples; ++sample) {
    for (int row = 0; row < 2; ++row) {
      const int output_row = row == 0 ? output_row0 : output_row1;
      const Q4KBlock *blocks = row == 0 ? blocks0 : blocks1;
      output[static_cast<std::size_t>(sample) * output_width + output_row] =
          bias ? bias[output_row] : 0.0f;
      for (int block = 0; block < batch.blocks; ++block) {
        const Q4KBlock &weight = blocks[block];
        std::uint8_t scales[8];
        std::uint8_t mins[8];
        unpack_q4k(weight, scales, mins);
        const float d = half_to_float(weight.d);
        const float dmin = half_to_float(weight.dmin);
        for (int sub = 0; sub < 8; ++sub) {
          const int source = (sub / 2) * 32;
          int dot = 0;
          for (int lane = 0; lane < 32; ++lane) {
            const int quantized = (sub & 1) == 0
                                      ? (weight.qs[source + lane] & 0x0f)
                                      : (weight.qs[source + lane] >> 4);
            dot += quantized * batch.values[static_cast<std::size_t>(
                                                block * 256 + sub * 32 + lane) *
                                                batch.samples +
                                            sample];
          }
          const int sum =
              batch.sums[static_cast<std::size_t>(block * 16 + sub * 2) *
                             batch.samples +
                         sample] +
              batch.sums[static_cast<std::size_t>(block * 16 + sub * 2 + 1) *
                             batch.samples +
                         sample];
          output[static_cast<std::size_t>(sample) * output_width +
                 output_row] +=
              batch.scales[static_cast<std::size_t>(block) * batch.samples +
                           sample] *
              (d * scales[sub] * static_cast<float>(dot) -
               dmin * mins[sub] * static_cast<float>(sum));
        }
      }
    }
  }
}
#endif

#if PARAKEET_POC_X86
PARAKEET_POC_TARGET_AVX2 void
q6k_rows2_avx2(const std::uint8_t *row0, const std::uint8_t *row1,
               const BatchQ8 &batch, float *output, int output_row0,
               int output_row1, int output_width, const float *bias) {
  const Q6KBlock *blocks0 = reinterpret_cast<const Q6KBlock *>(row0);
  const Q6KBlock *blocks1 = reinterpret_cast<const Q6KBlock *>(row1);
  const __m256i mask_three = _mm256_set1_epi8(3);
  const __m256i mask_fifteen = _mm256_set1_epi8(15);
  int sample = 0;
  for (; sample + 8 <= batch.samples; sample += 8) {
    float values0[8];
    float values1[8];
    for (int lane = 0; lane < 8; ++lane) {
      values0[lane] = bias ? bias[output_row0] : 0.0f;
      values1[lane] = bias ? bias[output_row1] : 0.0f;
    }
    for (int block = 0; block < batch.blocks; ++block) {
      const Q6KBlock &weight0 = blocks0[block];
      const Q6KBlock &weight1 = blocks1[block];
      const float d0 = half_to_float(weight0.d);
      const float d1 = half_to_float(weight1.d);
      __m256i dots0[8];
      __m256i dots1[8];
      for (int lane = 0; lane < 8; ++lane) {
        dots0[lane] = _mm256_setzero_si256();
        dots1[lane] = _mm256_setzero_si256();
      }
      const std::uint8_t *ql0 = weight0.ql;
      const std::uint8_t *ql1 = weight1.ql;
      const std::uint8_t *qh0 = weight0.qh;
      const std::uint8_t *qh1 = weight1.qh;
      int scale_index = 0;
      for (int part = 0; part < 2; ++part) {
        const __m256i ql_low0 =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ql0));
        const __m256i ql_high0 =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ql0 + 32));
        const __m256i ql_low1 =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ql1));
        const __m256i ql_high1 =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ql1 + 32));
        const __m256i qh_values0 =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(qh0));
        const __m256i qh_values1 =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(qh1));
        ql0 += 64;
        ql1 += 64;
        qh0 += 32;
        qh1 += 32;
        const __m256i q4_low0 = _mm256_or_si256(
            _mm256_and_si256(ql_low0, mask_fifteen),
            _mm256_slli_epi16(_mm256_and_si256(qh_values0, mask_three), 4));
        const __m256i q4_high0 = _mm256_or_si256(
            _mm256_and_si256(ql_high0, mask_fifteen),
            _mm256_slli_epi16(
                _mm256_and_si256(qh_values0, _mm256_set1_epi8(12)), 2));
        const __m256i q4_low_high0 = _mm256_or_si256(
            _mm256_and_si256(_mm256_srli_epi16(ql_low0, 4), mask_fifteen),
            _mm256_and_si256(qh_values0, _mm256_set1_epi8(48)));
        const __m256i q4_high_high0 = _mm256_or_si256(
            _mm256_and_si256(_mm256_srli_epi16(ql_high0, 4), mask_fifteen),
            _mm256_srli_epi16(
                _mm256_and_si256(qh_values0, _mm256_set1_epi8(-64)), 2));
        const __m256i q4_low1 = _mm256_or_si256(
            _mm256_and_si256(ql_low1, mask_fifteen),
            _mm256_slli_epi16(_mm256_and_si256(qh_values1, mask_three), 4));
        const __m256i q4_high1 = _mm256_or_si256(
            _mm256_and_si256(ql_high1, mask_fifteen),
            _mm256_slli_epi16(
                _mm256_and_si256(qh_values1, _mm256_set1_epi8(12)), 2));
        const __m256i q4_low_high1 = _mm256_or_si256(
            _mm256_and_si256(_mm256_srli_epi16(ql_low1, 4), mask_fifteen),
            _mm256_and_si256(qh_values1, _mm256_set1_epi8(48)));
        const __m256i q4_high_high1 = _mm256_or_si256(
            _mm256_and_si256(_mm256_srli_epi16(ql_high1, 4), mask_fifteen),
            _mm256_srli_epi16(
                _mm256_and_si256(qh_values1, _mm256_set1_epi8(-64)), 2));
        const __m128i scales0 =
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(weight0.scales));
        const __m128i scales1 =
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(weight1.scales));
        const __m256i scale0 = _mm256_cvtepi8_epi16(
            _mm_shuffle_epi8(scales0, q6_scale_shuffle(scale_index + 0)));
        const __m256i scale1 = _mm256_cvtepi8_epi16(
            _mm_shuffle_epi8(scales0, q6_scale_shuffle(scale_index + 1)));
        const __m256i scale2 = _mm256_cvtepi8_epi16(
            _mm_shuffle_epi8(scales0, q6_scale_shuffle(scale_index + 2)));
        const __m256i scale3 = _mm256_cvtepi8_epi16(
            _mm_shuffle_epi8(scales0, q6_scale_shuffle(scale_index + 3)));
        const __m256i scale4 = _mm256_cvtepi8_epi16(
            _mm_shuffle_epi8(scales1, q6_scale_shuffle(scale_index + 0)));
        const __m256i scale5 = _mm256_cvtepi8_epi16(
            _mm_shuffle_epi8(scales1, q6_scale_shuffle(scale_index + 1)));
        const __m256i scale6 = _mm256_cvtepi8_epi16(
            _mm_shuffle_epi8(scales1, q6_scale_shuffle(scale_index + 2)));
        const __m256i scale7 = _mm256_cvtepi8_epi16(
            _mm_shuffle_epi8(scales1, q6_scale_shuffle(scale_index + 3)));
        for (int lane = 0; lane < 8; ++lane) {
          const std::int8_t *q8 =
              batch.sample_values.data() +
              static_cast<std::size_t>(sample + lane) * batch.input +
              block * 256 + part * 128;
          const __m256i q8_0 =
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8));
          const __m256i q8_1 =
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8 + 32));
          const __m256i q8_2 =
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8 + 64));
          const __m256i q8_3 =
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8 + 96));
          __m256i product0 = _mm256_maddubs_epi16(q4_low0, q8_0);
          product0 = _mm256_madd_epi16(scale0, product0);
          product0 = _mm256_add_epi32(
              product0,
              _mm256_madd_epi16(scale1, _mm256_maddubs_epi16(q4_high0, q8_1)));
          product0 = _mm256_add_epi32(
              product0, _mm256_madd_epi16(
                            scale2, _mm256_maddubs_epi16(q4_low_high0, q8_2)));
          product0 = _mm256_add_epi32(
              product0, _mm256_madd_epi16(
                            scale3, _mm256_maddubs_epi16(q4_high_high0, q8_3)));
          __m256i product1 = _mm256_maddubs_epi16(q4_low1, q8_0);
          product1 = _mm256_madd_epi16(scale4, product1);
          product1 = _mm256_add_epi32(
              product1,
              _mm256_madd_epi16(scale5, _mm256_maddubs_epi16(q4_high1, q8_1)));
          product1 = _mm256_add_epi32(
              product1, _mm256_madd_epi16(
                            scale6, _mm256_maddubs_epi16(q4_low_high1, q8_2)));
          product1 = _mm256_add_epi32(
              product1, _mm256_madd_epi16(
                            scale7, _mm256_maddubs_epi16(q4_high_high1, q8_3)));
          dots0[lane] = _mm256_add_epi32(dots0[lane], product0);
          dots1[lane] = _mm256_add_epi32(dots1[lane], product1);
        }
        scale_index += 4;
      }
      for (int lane = 0; lane < 8; ++lane) {
        int correction0 = 0;
        int correction1 = 0;
        for (int chunk = 0; chunk < 16; ++chunk) {
          const int sum =
              batch.sums[static_cast<std::size_t>(block * 16 + chunk) *
                             batch.samples +
                         sample + lane];
          correction0 += static_cast<int>(weight0.scales[chunk]) * sum;
          correction1 += static_cast<int>(weight1.scales[chunk]) * sum;
        }
        const float activation_scale =
            batch.scales[static_cast<std::size_t>(block) * batch.samples +
                         sample + lane];
        values0[lane] +=
            activation_scale * d0 *
            static_cast<float>(horizontal_sum(dots0[lane]) - 32 * correction0);
        values1[lane] +=
            activation_scale * d1 *
            static_cast<float>(horizontal_sum(dots1[lane]) - 32 * correction1);
      }
    }
    for (int lane = 0; lane < 8; ++lane) {
      output[static_cast<std::size_t>(sample + lane) * output_width +
             output_row0] = values0[lane];
      output[static_cast<std::size_t>(sample + lane) * output_width +
             output_row1] = values1[lane];
    }
  }
  for (; sample < batch.samples; ++sample) {
    for (int row = 0; row < 2; ++row) {
      const int output_row = row == 0 ? output_row0 : output_row1;
      const Q6KBlock *blocks = row == 0 ? blocks0 : blocks1;
      output[static_cast<std::size_t>(sample) * output_width + output_row] =
          bias ? bias[output_row] : 0.0f;
      for (int block = 0; block < batch.blocks; ++block) {
        const Q6KBlock &weight = blocks[block];
        const float d = half_to_float(weight.d);
        for (int group = 0; group < 2; ++group) {
          const std::uint8_t *ql = weight.ql + group * 64;
          const std::uint8_t *qh = weight.qh + group * 32;
          for (int part = 0; part < 4; ++part) {
            for (int half = 0; half < 2; ++half) {
              const int base = group * 128 + part * 32 + half * 16;
              int dot = 0;
              for (int lane = 0; lane < 16; ++lane) {
                const int source_lane = half * 16 + lane;
                const int ql_index =
                    part == 1 || part == 3 ? source_lane + 32 : source_lane;
                const int low = part == 0 || part == 1 ? (ql[ql_index] & 0x0f)
                                                       : (ql[ql_index] >> 4);
                const int high = part == 0   ? ((qh[source_lane] >> 0) & 3)
                                 : part == 1 ? ((qh[source_lane] >> 2) & 3)
                                 : part == 2 ? ((qh[source_lane] >> 4) & 3)
                                             : ((qh[source_lane] >> 6) & 3);
                dot += ((low | (high << 4)) - 32) *
                       batch.values[static_cast<std::size_t>(block * 256 +
                                                             base + lane) *
                                        batch.samples +
                                    sample];
              }
              const int chunk = group * 8 + part * 2 + half;
              output[static_cast<std::size_t>(sample) * output_width +
                     output_row] +=
                  batch.scales[static_cast<std::size_t>(block) * batch.samples +
                               sample] *
                  (d * weight.scales[chunk] * static_cast<float>(dot));
            }
          }
        }
      }
    }
  }
}
#endif

void q6k_batch_row_scalar(const std::uint8_t *row, const BatchQ8 &batch,
                          float *output, int output_row, int output_width,
                          const float *bias) {
  for (int sample = 0; sample < batch.samples; ++sample)
    output[static_cast<std::size_t>(sample) * output_width + output_row] =
        bias ? bias[output_row] : 0.0f;
  const Q6KBlock *blocks = reinterpret_cast<const Q6KBlock *>(row);
  for (int block = 0; block < batch.blocks; ++block) {
    const float d = half_to_float(blocks[block].d);
    for (int group = 0; group < 2; ++group) {
      const std::uint8_t *ql = blocks[block].ql + group * 64;
      const std::uint8_t *qh = blocks[block].qh + group * 32;
      const std::int8_t *scales = blocks[block].scales + group * 8;
      for (int part = 0; part < 4; ++part) {
        for (int half = 0; half < 2; ++half) {
          int quantized[16];
          for (int lane = 0; lane < 16; ++lane) {
            const int source_lane = half * 16 + lane;
            const int ql_index =
                part == 1 || part == 3 ? source_lane + 32 : source_lane;
            const int low = part == 0 || part == 1 ? (ql[ql_index] & 0x0f)
                                                   : (ql[ql_index] >> 4);
            const int high = part == 0   ? ((qh[source_lane] >> 0) & 3)
                             : part == 1 ? ((qh[source_lane] >> 2) & 3)
                             : part == 2 ? ((qh[source_lane] >> 4) & 3)
                                         : ((qh[source_lane] >> 6) & 3);
            quantized[lane] = (low | (high << 4)) - 32;
          }
          for (int sample = 0; sample < batch.samples; ++sample) {
            int dot = 0;
            const int base = group * 128 + part * 32 + half * 16;
            for (int lane = 0; lane < 16; ++lane)
              dot += quantized[lane] *
                     batch.values[static_cast<std::size_t>(block * 256 + base +
                                                           lane) *
                                      batch.samples +
                                  sample];
            output[static_cast<std::size_t>(sample) * output_width +
                   output_row] +=
                batch.scales[static_cast<std::size_t>(block) * batch.samples +
                             sample] *
                (d * scales[half + part * 2] * static_cast<float>(dot));
          }
        }
      }
    }
  }
}

#if PARAKEET_POC_X86
PARAKEET_POC_TARGET_AVX2 void
q6k_batch_row_avx2(const std::uint8_t *row, const BatchQ8 &batch, float *output,
                   int output_row, int output_width, const float *bias) {
  constexpr int max_blocks = 64;
  if (batch.blocks > max_blocks) {
    q6k_batch_row_scalar(row, batch, output, output_row, output_width, bias);
    return;
  }
  std::int8_t quantized[max_blocks][16][16];
  float scales[max_blocks][16];
  const Q6KBlock *blocks = reinterpret_cast<const Q6KBlock *>(row);
  for (int block = 0; block < batch.blocks; ++block) {
    const float d = half_to_float(blocks[block].d);
    for (int group = 0; group < 2; ++group) {
      const std::uint8_t *ql = blocks[block].ql + group * 64;
      const std::uint8_t *qh = blocks[block].qh + group * 32;
      const std::int8_t *block_scales = blocks[block].scales + group * 8;
      for (int part = 0; part < 4; ++part) {
        for (int half = 0; half < 2; ++half) {
          const int chunk = group * 8 + part * 2 + half;
          scales[block][chunk] =
              d * static_cast<float>(block_scales[half + part * 2]);
          for (int lane = 0; lane < 16; ++lane) {
            const int source_lane = half * 16 + lane;
            const int ql_index =
                part == 1 || part == 3 ? source_lane + 32 : source_lane;
            const int low = part == 0 || part == 1 ? (ql[ql_index] & 0x0f)
                                                   : (ql[ql_index] >> 4);
            const int high = part == 0   ? ((qh[source_lane] >> 0) & 3)
                             : part == 1 ? ((qh[source_lane] >> 2) & 3)
                             : part == 2 ? ((qh[source_lane] >> 4) & 3)
                                         : ((qh[source_lane] >> 6) & 3);
            quantized[block][chunk][lane] =
                static_cast<std::int8_t>((low | (high << 4)) - 32);
          }
        }
      }
    }
  }
  for (int sample = 0; sample < batch.samples; ++sample)
    output[static_cast<std::size_t>(sample) * output_width + output_row] =
        bias ? bias[output_row] : 0.0f;
  int sample = 0;
  for (; sample + 8 <= batch.samples; sample += 8) {
    __m256 accumulated = _mm256_setzero_ps();
    for (int block = 0; block < batch.blocks; ++block) {
      for (int chunk = 0; chunk < 16; ++chunk) {
        __m256i dots = _mm256_setzero_si256();
        const int group = chunk / 8;
        const int part = (chunk % 8) / 2;
        const int half = chunk % 2;
        const int base = group * 128 + part * 32 + half * 16;
        for (int lane = 0; lane < 16; ++lane) {
          const __m128i values =
              _mm_loadl_epi64(reinterpret_cast<const __m128i *>(
                  batch.values.data() +
                  (static_cast<std::size_t>(block * 256 + base + lane) *
                       batch.samples +
                   sample)));
          dots = _mm256_add_epi32(
              dots, _mm256_mullo_epi32(
                        _mm256_cvtepi8_epi32(values),
                        _mm256_set1_epi32(quantized[block][chunk][lane])));
        }
        const __m256 contribution = _mm256_mul_ps(
            _mm256_set1_ps(scales[block][chunk]), _mm256_cvtepi32_ps(dots));
        accumulated = _mm256_add_ps(
            accumulated,
            _mm256_mul_ps(_mm256_loadu_ps(batch.scales.data() +
                                          static_cast<std::size_t>(block) *
                                              batch.samples +
                                          sample),
                          contribution));
      }
    }
    float values[8];
    _mm256_storeu_ps(values, accumulated);
    for (int lane = 0; lane < 8; ++lane)
      output[static_cast<std::size_t>(sample + lane) * output_width +
             output_row] += values[lane];
  }
  for (; sample < batch.samples; ++sample) {
    for (int block = 0; block < batch.blocks; ++block) {
      for (int chunk = 0; chunk < 16; ++chunk) {
        const int group = chunk / 8;
        const int part = (chunk % 8) / 2;
        const int half = chunk % 2;
        const int base = group * 128 + part * 32 + half * 16;
        int dot = 0;
        for (int lane = 0; lane < 16; ++lane)
          dot +=
              quantized[block][chunk][lane] *
              batch.values[static_cast<std::size_t>(block * 256 + base + lane) *
                               batch.samples +
                           sample];
        output[static_cast<std::size_t>(sample) * output_width + output_row] +=
            batch.scales[static_cast<std::size_t>(block) * batch.samples +
                         sample] *
            scales[block][chunk] * static_cast<float>(dot);
      }
    }
  }
}
#endif

float dot_q6k_q8k(const std::uint8_t *row, const std::vector<Q8KBlock> &input,
                  int width, bool vector) {
#if PARAKEET_POC_X86
  if (vector)
    return dot_q6k_q8k_avx2_fast(row, input.data(), width);
#endif
  const Q6KBlock *blocks = reinterpret_cast<const Q6KBlock *>(row);
  float result = 0.0f;
  for (int block_index = 0; block_index < width / 256; ++block_index) {
    const Q6KBlock &block = blocks[block_index];
    const Q8KBlock &activation = input[block_index];
    const float d = half_to_float(block.d);
    int32_t total = 0;
    for (int group = 0; group < 2; ++group) {
      const std::uint8_t *ql = block.ql + group * 64;
      const std::uint8_t *qh = block.qh + group * 32;
      const std::int8_t *scales = block.scales + group * 8;
      for (int part = 0; part < 4; ++part) {
        for (int half = 0; half < 2; ++half) {
          std::int8_t quantized[16];
          for (int lane = 0; lane < 16; ++lane) {
            const int source_lane = half * 16 + lane;
            const int ql_index =
                part == 1 || part == 3 ? source_lane + 32 : source_lane;
            const int low = part == 0 || part == 1 ? (ql[ql_index] & 0x0f)
                                                   : (ql[ql_index] >> 4);
            const int high = part == 0   ? ((qh[source_lane] >> 0) & 3)
                             : part == 1 ? ((qh[source_lane] >> 2) & 3)
                             : part == 2 ? ((qh[source_lane] >> 4) & 3)
                                         : ((qh[source_lane] >> 6) & 3);
            quantized[lane] =
                static_cast<std::int8_t>((low | (high << 4)) - 32);
          }
          int dot = 0;
          if (vector) {
#if PARAKEET_POC_X86
            dot = dot_signed_16(quantized, activation.qs + group * 128 +
                                               part * 32 + half * 16);
#else
            dot = dot_signed_16_scalar(quantized, activation.qs + group * 128 +
                                                      part * 32 + half * 16);
#endif
          } else {
            for (int lane = 0; lane < 16; ++lane)
              dot += static_cast<int>(quantized[lane]) *
                     static_cast<int>(activation.qs[group * 128 + part * 32 +
                                                    half * 16 + lane]);
          }
          total += static_cast<int>(scales[half + part * 2]) * dot;
        }
      }
    }
    result += d * static_cast<float>(total) * activation.d;
  }
  return result;
}

PARAKEET_POC_TARGET_AVX2 float dot_q8_q8(const std::uint8_t *row,
                                         const std::vector<Q8Block> &input,
                                         int width, bool vector) {
  const Q8Block *blocks = reinterpret_cast<const Q8Block *>(row);
  float result = 0.0f;
  for (int block_index = 0; block_index < width / 32; ++block_index) {
    const Q8Block &block = blocks[block_index];
    const Q8Block &activation = input[block_index];
    int dot = 0;
    if (vector) {
#if PARAKEET_POC_X86
      __m256i accumulated = _mm256_setzero_si256();
      for (int half = 0; half < 2; ++half) {
        const __m128i weights = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(block.qs + half * 16));
        const __m128i inputs = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(activation.qs + half * 16));
        const __m256i weight16 = _mm256_cvtepi8_epi16(weights);
        const __m256i input16 = _mm256_cvtepi8_epi16(inputs);
        const __m256i product = _mm256_mullo_epi16(weight16, input16);
        accumulated = _mm256_add_epi32(
            accumulated, _mm256_madd_epi16(product, _mm256_set1_epi16(1)));
      }
      dot = horizontal_sum(accumulated);
#else
      dot = dot_signed_16_scalar(
                reinterpret_cast<const std::int8_t *>(block.qs),
                reinterpret_cast<const std::int8_t *>(activation.qs)) +
            dot_signed_16_scalar(
                reinterpret_cast<const std::int8_t *>(block.qs + 16),
                reinterpret_cast<const std::int8_t *>(activation.qs + 16));
#endif
    } else {
      for (int lane = 0; lane < 32; ++lane)
        dot += static_cast<int>(block.qs[lane]) *
               static_cast<int>(activation.qs[lane]);
    }
    result += half_to_float(block.d) * half_to_float(activation.d) *
              static_cast<float>(dot);
  }
  return result;
}

}

namespace {

void linear_f32_row(const LinearMatrix &matrix, const float *input,
                    float *output, int samples, int row, bool vector) {
  const std::uint8_t *source = static_cast<const std::uint8_t *>(matrix.data) +
                               static_cast<std::size_t>(row) * matrix.row_bytes;
  switch (matrix.type) {
  case QuantType::F32:
    f32_row(source, matrix.input, input, output, samples, row, matrix.output,
            matrix.bias, vector);
    break;
  case QuantType::Q4K:
    q4k_row_f32_impl(source, matrix.input, input, output, samples, row,
                     matrix.output, matrix.bias, vector);
    break;
  case QuantType::Q6K:
    q6k_row_f32_impl(source, matrix.input, input, output, samples, row,
                     matrix.output, matrix.bias, vector);
    break;
  case QuantType::Q8_0:
    q8_row_f32_impl(source, matrix.input, input, output, samples, row,
                    matrix.output, matrix.bias, vector);
    break;
  }
}

}

std::shared_ptr<PackedMatrix> pack_matrix(const LinearMatrix &matrix) {
  return pack_matrix_impl(matrix);
}

void linear_f32(const LinearMatrix &matrix, const float *input, float *output,
                int samples, int threads) {
  validate(matrix, samples);
  require(input != nullptr && output != nullptr, "linear buffers are null");
  const bool vector = avx2_available();
  parallel_rows(matrix.output, threads, [&](int row) {
    linear_f32_row(matrix, input, output, samples, row, vector);
  });
}

void linear_f32_range(const LinearMatrix &matrix, const float *input,
                      float *output, int samples, int row_begin, int row_end) {
  validate(matrix, samples);
  require(input != nullptr && output != nullptr, "linear buffers are null");
  require(row_begin >= 0 && row_end >= row_begin && row_end <= matrix.output,
          "linear row range is invalid");
  const bool vector = avx2_available();
  for (int row = row_begin; row < row_end; ++row)
    linear_f32_row(matrix, input, output, samples, row, vector);
}

void linear_q8(const LinearMatrix &matrix, const float *input, float *output,
               int threads) {
  validate(matrix, 1);
  require(input != nullptr && output != nullptr, "linear buffers are null");
  require(matrix.type != QuantType::F32,
          "Q8 linear requires quantized weights");
  const bool vector = avx2_available();
  if (matrix.type == QuantType::Q4K || matrix.type == QuantType::Q6K) {
    const std::vector<Q8KBlock> activation = quantize_q8k(input, matrix.input);
    parallel_rows(matrix.output, threads, [&](int row) {
      const std::uint8_t *source =
          static_cast<const std::uint8_t *>(matrix.data) +
          static_cast<std::size_t>(row) * matrix.row_bytes;
      const float value =
          matrix.type == QuantType::Q4K
              ? dot_q4k_q8k(source, activation, matrix.input, vector)
              : dot_q6k_q8k(source, activation, matrix.input, vector);
      output[row] = value + (matrix.bias ? matrix.bias[row] : 0.0f);
    });
    return;
  }
  const std::vector<Q8Block> activation = quantize_q8(input, matrix.input);
  parallel_rows(matrix.output, threads, [&](int row) {
    const std::uint8_t *source =
        static_cast<const std::uint8_t *>(matrix.data) +
        static_cast<std::size_t>(row) * matrix.row_bytes;
    output[row] = dot_q8_q8(source, activation, matrix.input, vector) +
                  (matrix.bias ? matrix.bias[row] : 0.0f);
  });
}

PARAKEET_POC_TARGET_AVX2 float dot_q4k_q8k_blocks(const std::uint8_t *row,
                                                  const Q8KBlock *input,
                                                  int width, bool vector) {
#if PARAKEET_POC_X86
  if (vector)
    return dot_q4k_q8k_avx2_fast(row, input, width);
#endif
  const std::vector<Q8KBlock> activation(input, input + width / 256);
  return dot_q4k_q8k(row, activation, width, vector);
}

PARAKEET_POC_TARGET_AVX2 float dot_q6k_q8k_blocks(const std::uint8_t *row,
                                                  const Q8KBlock *input,
                                                  int width, bool vector) {
#if PARAKEET_POC_X86
  if (vector)
    return dot_q6k_q8k_avx2_fast(row, input, width);
#endif
  const std::vector<Q8KBlock> activation(input, input + width / 256);
  return dot_q6k_q8k(row, activation, width, vector);
}

void linear_q8_exact(const LinearMatrix &matrix, const float *input,
                     float *output, int samples, int threads) {
  validate(matrix, samples);
  require(input != nullptr && output != nullptr, "linear buffers are null");
  if (matrix.type != QuantType::Q4K && matrix.type != QuantType::Q6K) {
    linear_q8_range(matrix, input, output, samples, 0, matrix.output);
    return;
  }
  if (samples == 1) {
    linear_q8(matrix, input, output, threads);
    return;
  }
  const int blocks = matrix.input / 256;
  std::vector<Q8KBlock> activations(static_cast<std::size_t>(samples) * blocks);
  for (int sample = 0; sample < samples; ++sample)
    quantize_q8k_into(input + static_cast<std::size_t>(sample) * matrix.input,
                      activations.data() +
                          static_cast<std::size_t>(sample) * blocks,
                      matrix.input);
  const bool vector = avx2_available();
  constexpr int row_block = 16;
  const int row_blocks = (matrix.output + row_block - 1) / row_block;
  parallel_rows(row_blocks, threads, [&](int block) {
    const int row_begin = block * row_block;
    const int row_end = std::min(row_begin + row_block, matrix.output);
    for (int sample_begin = 0; sample_begin < samples; sample_begin += 16) {
      const int sample_end = std::min(sample_begin + 16, samples);
      for (int row = row_begin; row < row_end; ++row) {
        const std::uint8_t *source =
            static_cast<const std::uint8_t *>(matrix.data) +
            static_cast<std::size_t>(row) * matrix.row_bytes;
        for (int sample = sample_begin; sample < sample_end; ++sample) {
          const Q8KBlock *activation =
              activations.data() + static_cast<std::size_t>(sample) * blocks;
          const float value =
              matrix.type == QuantType::Q4K
                  ? dot_q4k_q8k_blocks(source, activation, matrix.input, vector)
                  : dot_q6k_q8k_blocks(source, activation, matrix.input,
                                       vector);
          output[static_cast<std::size_t>(sample) * matrix.output + row] =
              value + (matrix.bias ? matrix.bias[row] : 0.0f);
        }
      }
    }
  });
}

void linear_q8_range(const LinearMatrix &matrix, const float *input,
                     float *output, int samples, int row_begin, int row_end) {
  validate(matrix, samples);
  require(input != nullptr && output != nullptr, "linear buffers are null");
  require(matrix.type != QuantType::F32,
          "Q8 linear requires quantized weights");
  require(row_begin >= 0 && row_end >= row_begin && row_end <= matrix.output,
          "linear row range is invalid");
  const bool vector = avx2_available();
  if (samples > 1 &&
      (matrix.type == QuantType::Q4K || matrix.type == QuantType::Q6K)) {
    const BatchQ8 batch = make_batch_q8(input, matrix.input, samples);
#if PARAKEET_POC_X86
    if (vector && matrix.type == QuantType::Q4K) {
      int row = row_begin;
      for (; row + 1 < row_end; row += 2) {
        const std::uint8_t *source0 =
            static_cast<const std::uint8_t *>(matrix.data) +
            static_cast<std::size_t>(row) * matrix.row_bytes;
        const std::uint8_t *source1 =
            static_cast<const std::uint8_t *>(matrix.data) +
            static_cast<std::size_t>(row + 1) * matrix.row_bytes;
        q4k_rows2_avx2(source0, source1, batch, output, row, row + 1,
                       matrix.output, matrix.bias);
      }
      if (row < row_end) {
        const std::uint8_t *source =
            static_cast<const std::uint8_t *>(matrix.data) +
            static_cast<std::size_t>(row) * matrix.row_bytes;
        q4k_batch_row_scalar(source, batch, output, row, matrix.output,
                             matrix.bias);
      }
      return;
    }
    if (vector && matrix.type == QuantType::Q6K) {
      int row = row_begin;
      for (; row + 1 < row_end; row += 2) {
        const std::uint8_t *source0 =
            static_cast<const std::uint8_t *>(matrix.data) +
            static_cast<std::size_t>(row) * matrix.row_bytes;
        const std::uint8_t *source1 =
            static_cast<const std::uint8_t *>(matrix.data) +
            static_cast<std::size_t>(row + 1) * matrix.row_bytes;
        q6k_rows2_avx2(source0, source1, batch, output, row, row + 1,
                       matrix.output, matrix.bias);
      }
      if (row < row_end) {
        const std::uint8_t *source =
            static_cast<const std::uint8_t *>(matrix.data) +
            static_cast<std::size_t>(row) * matrix.row_bytes;
        q6k_batch_row_scalar(source, batch, output, row, matrix.output,
                             matrix.bias);
      }
      return;
    }
#endif
    for (int row = row_begin; row < row_end; ++row) {
      const std::uint8_t *source =
          static_cast<const std::uint8_t *>(matrix.data) +
          static_cast<std::size_t>(row) * matrix.row_bytes;
      if (matrix.type == QuantType::Q4K) {
#if PARAKEET_POC_X86
        if (vector)
          q4k_batch_row_avx2_fast2(source, batch, output, row, matrix.output,
                                   matrix.bias);
        else
#endif
          q4k_batch_row_scalar(source, batch, output, row, matrix.output,
                               matrix.bias);
      } else {
#if PARAKEET_POC_X86
        if (vector)
          q6k_batch_row_avx2(source, batch, output, row, matrix.output,
                             matrix.bias);
        else
#endif
          q6k_batch_row_scalar(source, batch, output, row, matrix.output,
                               matrix.bias);
      }
    }
    return;
  }
  for (int sample = 0; sample < samples; ++sample) {
    const float *source_input =
        input + static_cast<std::size_t>(sample) * matrix.input;
    float *destination =
        output + static_cast<std::size_t>(sample) * matrix.output;
    if (matrix.type == QuantType::Q4K || matrix.type == QuantType::Q6K) {
      const std::vector<Q8KBlock> activation =
          quantize_q8k(source_input, matrix.input);
      for (int row = row_begin; row < row_end; ++row) {
        const std::uint8_t *source =
            static_cast<const std::uint8_t *>(matrix.data) +
            static_cast<std::size_t>(row) * matrix.row_bytes;
        const float value =
            matrix.type == QuantType::Q4K
                ? dot_q4k_q8k(source, activation, matrix.input, vector)
                : dot_q6k_q8k(source, activation, matrix.input, vector);
        destination[row] = value + (matrix.bias ? matrix.bias[row] : 0.0f);
      }
    } else {
      const std::vector<Q8Block> activation =
          quantize_q8(source_input, matrix.input);
      for (int row = row_begin; row < row_end; ++row) {
        const std::uint8_t *source =
            static_cast<const std::uint8_t *>(matrix.data) +
            static_cast<std::size_t>(row) * matrix.row_bytes;
        destination[row] = dot_q8_q8(source, activation, matrix.input, vector) +
                           (matrix.bias ? matrix.bias[row] : 0.0f);
      }
    }
  }
}

void linear_q8_batch(const LinearMatrix &matrix, const float *input,
                     float *output, int samples, int threads) {
  validate(matrix, samples);
  require(input != nullptr && output != nullptr, "linear buffers are null");
  require(matrix.type != QuantType::F32,
          "Q8 linear requires quantized weights");
  if (samples <= 1 ||
      (matrix.type != QuantType::Q4K && matrix.type != QuantType::Q6K)) {
    linear_q8_range(matrix, input, output, samples, 0, matrix.output);
    return;
  }
#if PARAKEET_POC_X86
  if (avx2_available()) {
    linear_q8_exact(matrix, input, output, samples, threads);
    return;
  }
#endif
#if PARAKEET_POC_X86
  if (matrix.type == QuantType::Q4K && avx2_available() &&
      matrix.output % 2 == 0) {
    const BatchQ8 batch = make_batch_q8(input, matrix.input, samples);
    parallel_rows(matrix.output / 2, threads, [&](int pair) {
      const int row0 = pair * 2;
      const int row1 = row0 + 1;
      const std::uint8_t *source0 =
          static_cast<const std::uint8_t *>(matrix.data) +
          static_cast<std::size_t>(row0) * matrix.row_bytes;
      const std::uint8_t *source1 =
          static_cast<const std::uint8_t *>(matrix.data) +
          static_cast<std::size_t>(row1) * matrix.row_bytes;
      q4k_rows2_avx2(source0, source1, batch, output, row0, row1, matrix.output,
                     matrix.bias);
    });
    return;
  }
  if (matrix.type == QuantType::Q6K && avx2_available() &&
      matrix.output % 2 == 0) {
    const BatchQ8 batch = make_batch_q8(input, matrix.input, samples);
    parallel_rows(matrix.output / 2, threads, [&](int pair) {
      const int row0 = pair * 2;
      const int row1 = row0 + 1;
      const std::uint8_t *source0 =
          static_cast<const std::uint8_t *>(matrix.data) +
          static_cast<std::size_t>(row0) * matrix.row_bytes;
      const std::uint8_t *source1 =
          static_cast<const std::uint8_t *>(matrix.data) +
          static_cast<std::size_t>(row1) * matrix.row_bytes;
      q6k_rows2_avx2(source0, source1, batch, output, row0, row1, matrix.output,
                     matrix.bias);
    });
    return;
  }
#endif
  if (matrix.type == QuantType::Q6K && matrix.packed && avx2_available() &&
      matrix.packed->type == matrix.type) {
    const BatchQ8 batch = make_batch_q8(input, matrix.input, samples);
    const auto *storage =
        reinterpret_cast<const std::uint8_t *>(matrix.packed->words.data());
    const std::size_t block_bytes =
        matrix.packed->row_bytes / static_cast<std::size_t>(matrix.input / 256);
    parallel_rows(matrix.packed->groups, threads, [&](int group) {
      if (matrix.type == QuantType::Q4K) {
        const auto *weights = storage + static_cast<std::size_t>(group) *
                                            matrix.packed->row_bytes;
        packed_q4k8_group(weights, block_bytes, batch, output, matrix.output,
                          group, matrix.bias);
      } else {
        const auto *weights = storage + static_cast<std::size_t>(group) *
                                            matrix.packed->row_bytes;
        packed_q6k8_group(weights, block_bytes, batch, output, matrix.output,
                          group, matrix.bias);
      }
    });
    return;
  }
  const BatchQ8 batch = make_batch_q8(input, matrix.input, samples);
  const bool vector = avx2_available();
  parallel_rows(matrix.output, threads, [&](int row) {
    const std::uint8_t *source =
        static_cast<const std::uint8_t *>(matrix.data) +
        static_cast<std::size_t>(row) * matrix.row_bytes;
    if (matrix.type == QuantType::Q4K) {
#if PARAKEET_POC_X86
      if (vector)
        q4k_batch_row_avx2_fast2(source, batch, output, row, matrix.output,
                                 matrix.bias);
      else
#endif
        q4k_batch_row_scalar(source, batch, output, row, matrix.output,
                             matrix.bias);
    } else {
#if PARAKEET_POC_X86
      if (vector)
        q6k_batch_row_avx2(source, batch, output, row, matrix.output,
                           matrix.bias);
      else
#endif
        q6k_batch_row_scalar(source, batch, output, row, matrix.output,
                             matrix.bias);
    }
  });
}

const char *backend_name() { return avx2_available() ? "avx2" : "scalar"; }

}
}
