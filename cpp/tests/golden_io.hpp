#pragma once
#include <cstdint>
#include <string>
#include <vector>
namespace starling::ggml::test {
bool read_f32(const std::string&,std::vector<float>&,std::string&);
bool read_i64(const std::string&,std::vector<int64_t>&,std::string&);
bool read_shape(const std::string&,std::vector<int64_t>&,std::string&);
}
