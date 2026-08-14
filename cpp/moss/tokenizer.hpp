#pragma once
#include "loader.hpp"
#include "lib/bpe_tokenizer.hpp"
namespace starling::ggml::moss {
// Qwen BPE byte-decoder: thin adapter over the shared lib::BpeTokenizer.
class Tokenizer : public lib::BpeTokenizer {
 public:
  bool load(const ModelLoader& m, const Config&, std::string& e) {
      return lib::BpeTokenizer::load(m, e);
  }
  std::string decode(const std::vector<int32_t>& ids, bool skip_special_tokens = true) const {
      return lib::BpeTokenizer::decode(ids, skip_special_tokens);
  }
};
}
