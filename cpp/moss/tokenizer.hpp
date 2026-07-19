#pragma once
#include "loader.hpp"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
namespace starling::ggml::moss {
class Tokenizer {
 public:
  bool load(const ModelLoader&,const Config&,std::string& err);
  std::string decode(const std::vector<int32_t>&,bool skip_special_tokens=true) const;
 private:
  std::vector<std::string> tokens_; std::vector<int64_t> types_;
  std::unordered_map<uint32_t,uint8_t> byte_decoder_;
};
}
