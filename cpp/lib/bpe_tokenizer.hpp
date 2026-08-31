// bpe_tokenizer.hpp — GPT-2/Qwen BPE tokenizer: decode AND encode.
//
// Reads the standard GGUF GPT-2 tokenizer table (tokenizer.ggml.model=="gpt2",
// .tokens, .token_type, .merges) and decodes token ids to UTF-8 text via the
// GPT-2 byte-to-unicode inverse map. Encoding (text -> ids) implements the
// Qwen pre-tokenizer regex + byte-level BPE with merge ranks — the encoder
// s1 needs (every other engine only decodes). Model-agnostic: every LLM
// engine's tokenizer is an alias of this class; encoding requires .merges in
// the GGUF and is exact for ASCII input (see bpe_tokenizer.cpp for the
// non-ASCII classification approximation).
#pragma once

#include "runtime/model_loader.hpp"
#include <cstdint>
#include <map>
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

    // Encode UTF-8 text to token ids, replicating the HF byte-level BPE
    // path: special-token longest-match first, then the Qwen pre-tokenizer
    // regex, then BPE merges by rank. Adds no special tokens of its own.
    // Requires .merges in the GGUF (the s1 converter writes them); engines
    // that only decode keep working without.
    bool encode(const std::string& text, std::vector<int32_t>& ids,
                std::string& e) const;

private:
    std::vector<std::string> tokens_;
    std::vector<int32_t> types_;  // 3 == unused/special, skipped when skip_special
    std::unordered_map<uint32_t, uint8_t> byte_decoder_;
    // Encoder-side tables (built when tokenizer.ggml.merges is present).
    std::string byte_encoder_utf8_[256];  // byte -> byte-level unicode char (UTF-8)
    std::unordered_map<std::string, int32_t> token_ids_;  // byte-level piece -> id
    std::map<std::pair<std::string, std::string>, int32_t> merge_ranks_;  // pair -> rank
    std::unordered_map<std::string, int32_t> special_ids_;  // special string -> id
};

} // namespace starling::ggml::lib
