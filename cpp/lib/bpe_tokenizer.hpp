// bpe_tokenizer.hpp — decode-only GPT-2/Qwen BPE tokenizer. Reads the
// standard GGUF GPT-2 tokenizer table (tokenizer.ggml.model=="gpt2",
// .tokens, .token_type) and decodes token ids to UTF-8 text via the GPT-2
// byte-to-unicode inverse map. Model-agnostic: every LLM engine's tokenizer
// (moss/ark/higgs/hojo) is an alias of this class — the BPE table is
// self-describing in the GGUF, so there is no per-model behavior.
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

    // Overload for the per-model engine shells, which call
    // tokenizer.load(model.loader, model.config, err) for symmetry with the
    // other component loaders. The config is not consulted.
    template <typename ConfigT>
    bool load(const ModelLoader& m, const ConfigT&, std::string& e) {
        return load(m, e);
    }

    std::string decode(const std::vector<int32_t>& ids,
                       bool skip_special = true) const;

private:
    std::vector<std::string> tokens_;
    std::vector<int32_t> types_;  // 3 == unused/special, skipped when skip_special
    std::unordered_map<uint32_t, uint8_t> byte_decoder_;
};

} // namespace starling::ggml::lib
