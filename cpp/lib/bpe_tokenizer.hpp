// bpe_tokenizer.hpp — shared decode-only GPT-2/Qwen BPE tokenizer.
//
// Extracted from the four verbatim copies in moss/ark/higgs/hojo
// tokenizer.{hpp,cpp} (higgs and hojo were verbatim copies of ark; moss the
// same code in dense style). Reads the standard GGUF GPT-2 tokenizer table
// (tokenizer.ggml.model=="gpt2", .tokens, .token_type) and decodes token ids
// to UTF-8 text via the GPT-2 byte-to-unicode inverse map. The map and the
// table layout are model-agnostic; per-model Tokenizer types are thin
// adapters over this class.
#pragma once

#include "runtime/model_loader.hpp"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace starling::ggml::lib {

class BpeTokenizer {
public:
    bool load(const ModelLoader& m, std::string& e);
    std::string decode(const std::vector<int32_t>& ids, bool skip_special) const;

private:
    std::vector<std::string> tokens_;
    std::vector<int32_t> types_;  // 3 == unused/special, skipped when skip_special
    std::unordered_map<uint32_t, uint8_t> byte_decoder_;
};

} // namespace starling::ggml::lib
