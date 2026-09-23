#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace direct {

struct Tensor {
    std::array<int64_t, 4> shape{1, 1, 1, 1};
    int rank = 0;
    uint32_t type = 0;
    uint64_t offset = 0;
    const uint8_t* data = nullptr;
    int64_t cols() const { return shape[0]; }
    int64_t rows() const { return shape[1] * shape[2] * shape[3]; }
};

class Model {
public:
    explicit Model(const std::string& path);
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    const Tensor& at(const std::string& name) const;
    const Tensor* find(const std::string& name) const;
    int64_t integer(const std::string& name, int64_t fallback) const;
    double real(const std::string& name, double fallback) const;
    std::vector<int64_t> integers(const std::string& name) const;
    std::vector<std::string> strings(const std::string& name) const;
private:
    void* mapping_ = nullptr;
    size_t file_size_ = 0;
    std::unordered_map<std::string, Tensor> tensors_;
    std::unordered_map<std::string, int64_t> ints_;
    std::unordered_map<std::string, double> floats_;
    std::unordered_map<std::string, std::vector<int64_t>> int_arrays_;
    std::unordered_map<std::string, std::vector<std::string>> string_arrays_;
};

float half(uint16_t bits);
float scalar(const Tensor& t, size_t index);
void dequant_row(const Tensor& t, int64_t row, float* out);

} // namespace direct
