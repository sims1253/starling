// prompt.cpp — Higgs ChatML prompt construction + audio-embedding scatter.
//
// The prompt layout (src/starling/higgs/pipeline.py _build_input_tokens, then
// the collator's HiggsAudioSampleCollator expands per chunk):
//   <|im_start|>user\n + instruction + <|audio_bos|> + (<|AUDIO|> * N_k) +
//     <|audio_eos|>   [repeated once per audio chunk k] +
//   <|im_end|>\n<|im_start|>assistant\n
//
// The eager collator splits each clip into ceil(n_samples / chunk_size_samples)
// chunks (chunk_size_seconds = 4.0 for higgs-audio-v3-stt) and inserts ONE
// <|audio_bos|>...<|audio_eos|> segment per chunk. The C++ replicates this:
// build_transcribe_prompt takes the per-chunk audio-token counts and emits one
// segment per chunk.
//
// The prefix/suffix text (including the instruction, whitespace, and the ChatML
// role markers) is PRE-TOKENIZED by the GGUF converter (higgs.prompt_prefix /
// higgs.prompt_suffix). The converter bakes the prefix ending in <|audio_bos|>
// and the suffix starting with <|audio_eos|>; build_transcribe_prompt splits
// those audio boundary tokens out of the baked arrays so it can re-emit them
// once per chunk (and drops the spurious separator token the converter inserted
// alongside the boundary, so the merged ids match _build_input_tokens exactly).
#include "prompt.hpp"
#include "runtime/backend.hpp"
#include "runtime/graph.hpp"
#include "lib/embed_scatter.hpp"
#include "ggml.h"

#include <algorithm>
namespace starling::ggml::higgs {

namespace {
// Split the baked prefix at its trailing <|audio_bos|>, dropping it (and any
// single whitespace token immediately before it) so build_transcribe_prompt can
// re-emit <|audio_bos|> once per chunk. Returns the head ids (everything before
// the audio segment). The reference (_build_input_tokens) has the instruction
// end directly in <|audio_bos|> with no separating space, so a trailing 220
// (space) before the bos is a converter artifact and is dropped here.
std::vector<int32_t> split_head(const std::vector<int32_t>& prefix, int32_t audio_bos) {
    std::vector<int32_t> head = prefix;
    // Drop the trailing <|audio_bos|>.
    while (!head.empty() && head.back() == audio_bos) head.pop_back();
    // Drop one trailing whitespace token (220 = " ") the converter may have
    // inserted between the instruction and <|audio_bos|>.
    if (!head.empty() && head.back() == 220) head.pop_back();
    return head;
}
// Split the baked suffix at its leading <|audio_eos|>, dropping it (and any
// single newline token immediately after it) so build_transcribe_prompt can
// re-emit <|audio_eos|> once per chunk. The reference has <|audio_eos|> flow
// directly into <|im_end|> with no separating newline.
std::vector<int32_t> split_tail(const std::vector<int32_t>& suffix, int32_t audio_eos) {
    size_t start = 0;
    while (start < suffix.size() && suffix[start] == audio_eos) ++start;
    // Drop one leading newline token (198 = "\n") the converter may have
    // inserted between <|audio_eos|> and <|im_end|>.
    if (start < suffix.size() && suffix[start] == 198) ++start;
    return std::vector<int32_t>(suffix.begin() + start, suffix.end());
}
} // namespace

Prompt build_transcribe_prompt(const Config& c, const std::vector<int64_t>& chunk_tokens) {
    Prompt p;
    const std::vector<int32_t> pre = c.prompt_prefix.empty()
        ? std::vector<int32_t>{c.im_start_id} : c.prompt_prefix;
    const std::vector<int32_t> suf = c.prompt_suffix.empty()
        ? std::vector<int32_t>{c.im_end_id, c.im_start_id} : c.prompt_suffix;
    // Head = prefix with the trailing <|audio_bos|> (and spurious space) removed;
    // tail = suffix with the leading <|audio_eos|> (and spurious newline) removed.
    const std::vector<int32_t> head = c.prompt_prefix.empty()
        ? pre : split_head(pre, c.audio_bos_id);
    const std::vector<int32_t> tail = c.prompt_suffix.empty()
        ? suf : split_tail(suf, c.audio_eos_id);

    // Reserve: head + per-chunk [audio_bos + tokens + audio_eos] + tail.
    size_t total = head.size() + tail.size();
    for (int64_t n : chunk_tokens) total += (size_t) n + 2;
    p.ids.reserve(total);
    p.audio_mask.reserve(total);
    for (auto x : head) { p.ids.push_back(x); p.audio_mask.push_back(0); }
    for (int64_t n : chunk_tokens) {
        p.ids.push_back(c.audio_bos_id); p.audio_mask.push_back(0);
        for (int64_t i = 0; i < n; ++i) {
            p.ids.push_back(c.audio_placeholder_id);
            p.audio_mask.push_back(1);
        }
        p.ids.push_back(c.audio_eos_id); p.audio_mask.push_back(0);
    }
    for (auto x : tail) { p.ids.push_back(x); p.audio_mask.push_back(0); }
    return p;
}

bool build_inputs_embeds(const HiggsModel& m, const Prompt& p, const AudioEncoding& a,
                         InputsEmbeds& out, std::string& err) {
    std::vector<float> emb;
    if (!lib::embed_and_scatter_audio(m.loader, m.config.llm.hidden, p.ids,
                                      p.audio_mask, a.data.data(), a.data.size(),
                                      a.width, a.n_tokens, emb, "Higgs", err))
        return false;
    out.data = std::move(emb);
    out.n_tokens = (int64_t) p.ids.size();
    out.width = m.config.llm.hidden;
    return true;
}
} // namespace starling::ggml::higgs
