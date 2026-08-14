#pragma once
#include "runtime/model_loader.hpp"
#include "config.hpp"
#include "lib/bpe_tokenizer.hpp"
namespace starling::ggml::higgs {
// Qwen3 BPE byte-decoder: thin adapter over the shared lib::BpeTokenizer
// (this model's tokenizer is a standard Qwen3 BPE table in the GGUF).
class Tokenizer : public lib::BpeTokenizer {
public:
    bool load(const ModelLoader& m, const Config&, std::string& e) {
        return lib::BpeTokenizer::load(m, e);
    }
};
} // namespace starling::ggml::higgs
