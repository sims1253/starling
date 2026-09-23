#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace parakeet_poc {
namespace cpu {

enum class QuantType {
  F32,
  Q4K,
  Q6K,
  Q8_0,
};

struct PackedMatrix {
  QuantType type = QuantType::F32;
  int input = 0;
  int output = 0;
  int groups = 0;
  std::size_t row_bytes = 0;
  std::vector<std::uint64_t> words;
};

struct LinearMatrix {
  const void *data = nullptr;
  const float *bias = nullptr;
  QuantType type = QuantType::F32;
  int input = 0;
  int output = 0;
  std::size_t row_bytes = 0;
  std::shared_ptr<const PackedMatrix> packed;
};

void linear_f32(const LinearMatrix &matrix, const float *input, float *output,
                int samples, int threads);
void linear_f32_range(const LinearMatrix &matrix, const float *input,
                      float *output, int samples, int row_begin, int row_end);
void linear_q8(const LinearMatrix &matrix, const float *input, float *output,
               int threads);
void linear_q8_range(const LinearMatrix &matrix, const float *input,
                     float *output, int samples, int row_begin, int row_end);
void linear_q8_batch(const LinearMatrix &matrix, const float *input,
                     float *output, int samples, int threads);
void linear_q8_exact(const LinearMatrix &matrix, const float *input,
                     float *output, int samples, int threads);
std::shared_ptr<PackedMatrix> pack_matrix(const LinearMatrix &matrix);
const char *backend_name();

}
}
