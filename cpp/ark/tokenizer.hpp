#pragma once
#include "runtime/model_loader.hpp"
#include "config.hpp"
#include <string>
#include <unordered_map>
#include <vector>
namespace starling::ggml::ark {
// Qwen2.5 BPE byte-decoder (same GPT-2 scheme as moss). The GGUF carries the
// full 151936-entry tokenizer table; decode maps token -> UTF-8 text via the
// GPT-2 byte-to-unicode inverse.
class Tokenizer {
public:
    bool load(const ModelLoader& m, const Config& c, std::string& e);
    std::string decode(const std::vector<int32_t>& ids, bool skip_special) const;
private:
    std::vector<std::string> tokens_;
    std::vector<int32_t> types_;  // 3 == unused/special, skipped when skip_special
    std::unordered_map<uint32_t, uint8_t> byte_decoder_;
};
} // namespace starling::ggml::ark
