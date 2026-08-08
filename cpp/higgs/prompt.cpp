// prompt.cpp — Higgs ChatML prompt construction + audio-embedding scatter.
//
// The prompt layout (src/starling/higgs/pipeline.py _build_input_tokens):
//   <|im_start|>user\n + instruction + " " + <|audio_bos|> +
//     (<|AUDIO|> * N) +
//   <|audio_eos|> + \n + <|im_end|> + \n + <|im_start|> + assistant\n
// The single <|AUDIO|> placeholder in the upstream string template EXPANDS to N
// audio tokens here (N = audio_token_count(mel_frames)); each expanded slot is
// then clobbered by a projector audio embedding in build_inputs_embeds, matching
// merge_input_ids_with_audio_features. The prefix/suffix text (including the
// instruction, whitespace, and the ChatML role markers) is PRE-TOKENIZED by the
// GGUF converter (higgs.prompt_prefix / higgs.prompt_suffix), since the C++ side
// has a decode-only tokenizer.
#include "prompt.hpp"
#include "runtime/backend.hpp"
#include "runtime/graph.hpp"
#include "ggml.h"
namespace starling::ggml::higgs {

Prompt build_transcribe_prompt(const Config& c, int64_t mel_frames) {
    Prompt p;
    // Default prefix/suffix if the GGUF didn't bake them: the minimal special-token
    // skeleton (<|im_start|>user ... <|audio_bos|> + ... + <|audio_eos|> ...
    // <|im_start|>assistant). Byte-exact ASR requires the converter to bake the
    // full pre-tokenized prompt (the instruction + whitespace tokenization depends
    // on the BPE merge table the C++ decoder does not have).
    const std::vector<int32_t> pre = c.prompt_prefix.empty()
        ? std::vector<int32_t>{c.im_start_id, c.audio_bos_id} : c.prompt_prefix;
    const std::vector<int32_t> suf = c.prompt_suffix.empty()
        ? std::vector<int32_t>{c.audio_eos_id, c.im_end_id, c.im_start_id} : c.prompt_suffix;
    const int64_t n = audio_token_count(mel_frames);
    p.ids.reserve(pre.size() + (size_t) n + suf.size());
    p.audio_mask.reserve(p.ids.capacity());
    for (auto x : pre) { p.ids.push_back(x); p.audio_mask.push_back(0); }
    for (int64_t i = 0; i < n; ++i) {
        p.ids.push_back(c.audio_placeholder_id);
        p.audio_mask.push_back(1);
    }
    for (auto x : suf) { p.ids.push_back(x); p.audio_mask.push_back(0); }
    return p;
}

bool build_inputs_embeds(const HiggsModel& m, const Prompt& p, const AudioEncoding& a,
                         InputsEmbeds& out, std::string& e) {
    if (p.ids.size() != p.audio_mask.size()) { e = "invalid Higgs prompt mask"; return false; }
    size_t slots = 0;
    for (auto x : p.audio_mask) slots += x != 0;
    if (a.width != (int64_t) m.config.llm.hidden || a.data.size() % (size_t) a.width != 0) {
        e = "Higgs audio/prompt scatter size mismatch";
        return false;
    }
    const int64_t sa = a.n_tokens;
    ensure_weights_realized(m.loader);
    std::vector<int32_t> ids = p.ids;
    std::vector<ggml_bf16_t> ah;
    if (sa > 0) {
        ah.resize((size_t) sa * (size_t) a.width);
        for (size_t i = 0; i < ah.size(); ++i) ah[i] = ggml_fp32_to_bf16(a.data[i]);
    }
    std::vector<float> emb;
    bool ok = run_graph([&](ggml_context* c) {
        int64_t ne[1] = {(int64_t) ids.size()};
        auto* it = graph_input_tensor(c, GGML_TYPE_I32, 1, ne, ids.data(),
                                      ids.size() * sizeof(ids[0]));
        return ggml_cast(c, ggml_get_rows(c, clone_weight(c, m.loader, "llm.embed.weight"), it),
                         GGML_TYPE_F32);
    }, emb);
    if (!ok) { e = "Higgs embedding lookup failed"; return false; }
    size_t row = 0;
    for (size_t i = 0; i < p.ids.size(); ++i) {
        if (!p.audio_mask[i]) continue;
        for (size_t d = 0; d < (size_t) a.width; ++d) {
            // row indexes into the (possibly truncated) feature stream. When the
            // projector emits fewer features than audio slots (long audio, mel
            // capped), the overflow slots must be ZEROED — matching HF
            // merge_input_ids_with_audio_features (final_embedding zero-init then
            // scatter), which overwrites the slot with a literal zero, NOT the
            // embedded audio_placeholder_id value the lookup left there.
            if (row < (size_t) sa)
                emb[i * (size_t) a.width + d] = ggml_bf16_to_fp32(ah[row * (size_t) a.width + d]);
            else
                emb[i * (size_t) a.width + d] = 0.0f;
        }
        ++row;
    }
    out.data = std::move(emb);
    out.n_tokens = (int64_t) p.ids.size();
    out.width = m.config.llm.hidden;
    return true;
}
} // namespace starling::ggml::higgs
