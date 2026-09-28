// #313 desktop feasibility: full native MOSS audio path, with an already-known
// Parakeet text as the only draft source. No oracle target tokens are exposed.
#include "moss_preview_copy.hpp"

#include "moss/audio_encoder.hpp"
#include "moss/llm.hpp"
#include "moss/mel.hpp"
#include "moss/prompt.hpp"
#include "moss/tokenizer.hpp"
#include "serve/audio.hpp"
#include "starling_ggml.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace {
namespace moss = starling::ggml::moss;
using starling::bench::moss_preview::PreviewCopyDrafter;
using starling::bench::moss_preview::normalize_preview;
using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::string ids_fingerprint(const std::vector<int32_t>& ids) {
    uint64_t hash = 14695981039346656037ull;
    for (int32_t id : ids)
        for (int shift = 0; shift < 32; shift += 8) {
            hash ^= uint8_t(uint32_t(id) >> shift);
            hash *= 1099511628211ull;
        }
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << hash;
    return out.str();
}

const char* stop_name(starling::ggml::lib::GenStopReason reason) {
    using Reason = starling::ggml::lib::GenStopReason;
    switch (reason) {
    case Reason::kEos: return "eos";
    case Reason::kBudgetExhausted: return "budget";
    case Reason::kCancelled: return "cancelled";
    }
    return "unknown";
}

struct Run {
    moss::GenerateResult output;
    starling::ggml::lib::SpeculativeStats stats;
    std::string text;
    size_t source_tokens = 0;
    double mel_ms = 0, audio_ms = 0, prompt_ms = 0;
    double source_ms = 0, generate_ms = 0, detokenize_ms = 0, total_ms = 0;
};

bool run(const moss::MossModel& model, const moss::Tokenizer& tokenizer,
         const std::vector<float>& pcm, const std::string& preview,
         int max_k, Run& out, std::string& err) {
    const auto t0 = Clock::now();
    moss::MelFeatures mel;
    if (!moss::compute_log_mel(model.config, model.loader, pcm.data(),
                               pcm.size(), mel, err)) return false;
    const auto t1 = Clock::now();
    moss::AudioEncoding audio;
    if (!moss::encode_audio_and_adapt(model, mel, audio, err)) return false;
    const auto t2 = Clock::now();
    const moss::Prompt prompt = moss::build_transcribe_prompt(model.config,
                                                              mel.n_frames);
    moss::InputsEmbeds input;
    if (!moss::build_inputs_embeds(model, prompt, audio, input, err)) return false;
    const auto t3 = Clock::now();

    moss::GenerateOptions options;
    options.max_new_tokens = int32_t(model.config.max_new_tokens);
    options.max_cache_len = int32_t(model.config.llm.max_cache);
    options.eos_token_id = model.config.eos_token_id;
    if (max_k == 0) {
        if (!moss::greedy_generate(model, input, options, out.output, err))
            return false;
    } else {
        // The preview string is already in memory in the live-preview use
        // case. Include BPE encoding and proposer construction in the timed
        // target call, but not a fresh Parakeet inference or file read.
        std::vector<int32_t> source_ids;
        if (!tokenizer.encode(normalize_preview(preview), source_ids, err))
            return false;
        out.source_tokens = source_ids.size();
        PreviewCopyDrafter drafter(std::move(source_ids), max_k);
        const auto ts = Clock::now();
        out.source_ms = elapsed_ms(t3, ts);
        const bool diagnostic_no_draft =
            std::getenv("MOSS_PREVIEW_DIAGNOSTIC_NO_DRAFT") != nullptr;
        auto propose = [&drafter, diagnostic_no_draft](
                           const std::vector<int32_t>& prefix, int cap) {
            if (diagnostic_no_draft) return std::vector<int32_t>{};
            return drafter.propose(prefix, cap);
        };
        if (!moss::speculative_generate(model, input, options, max_k, propose,
                                        {}, out.output, out.stats, err))
            return false;
    }
    const auto t4 = Clock::now();
    out.text = tokenizer.decode(out.output.ids, true);
    const auto t5 = Clock::now();
    out.mel_ms = elapsed_ms(t0, t1);
    out.audio_ms = elapsed_ms(t1, t2);
    out.prompt_ms = elapsed_ms(t2, t3);
    out.generate_ms = elapsed_ms(t3, t4) - out.source_ms;
    out.detokenize_ms = elapsed_ms(t4, t5);
    out.total_ms = elapsed_ms(t0, t5);
    return true;
}

bool load_wav(const std::string& path, std::vector<float>& pcm) {
    const std::string bytes = read_file(path);
    int sample_rate = 0;
    return !bytes.empty() &&
           starling::serve::audio::wav_bytes_to_float32(bytes, pcm,
                                                         sample_rate) &&
           sample_rate == 16000;
}

void print_pair(const char* tier, int k, int repeat, const Run& greedy,
                const Run& draft, bool golden_match) {
    const bool parity = greedy.output.ids == draft.output.ids &&
                        greedy.output.stop_reason == draft.output.stop_reason &&
                        greedy.text == draft.text;
    std::printf("{\"case\":\"%s\",\"k\":%d,\"repeat\":%d,"
                "\"order\":\"%s\",\"parity\":%s,\"golden_text\":%s,"
                "\"stop\":\"%s\",\"tokens\":%zu,\"ids_fnv64\":\"%s\","
                "\"draft_stop\":\"%s\",\"draft_tokens\":%zu,"
                "\"draft_ids_fnv64\":\"%s\","
                "\"source_tokens\":%zu,\"greedy_ms\":%.3f,\"draft_ms\":%.3f,"
                "\"greedy_mel_ms\":%.3f,\"greedy_audio_ms\":%.3f,"
                "\"greedy_prompt_ms\":%.3f,\"greedy_gen_ms\":%.3f,"
                "\"draft_mel_ms\":%.3f,\"draft_audio_ms\":%.3f,"
                "\"draft_prompt_ms\":%.3f,\"draft_source_ms\":%.3f,"
                "\"draft_gen_ms\":%.3f,\"accepted\":%d,\"proposed\":%d,"
                "\"verify_calls\":%d,\"fallback_steps\":%d,"
                "\"proposal_ms\":%.3f,\"verify_ms\":%.3f,\"prefill_ms\":%.3f}"
                "\n",
                tier, k, repeat, repeat % 2 == 0 ? "greedy-first" : "draft-first",
                parity ? "true" : "false", golden_match ? "true" : "false",
                stop_name(greedy.output.stop_reason), greedy.output.ids.size(),
                ids_fingerprint(greedy.output.ids).c_str(),
                stop_name(draft.output.stop_reason), draft.output.ids.size(),
                ids_fingerprint(draft.output.ids).c_str(), draft.source_tokens,
                greedy.total_ms, draft.total_ms,
                greedy.mel_ms, greedy.audio_ms, greedy.prompt_ms, greedy.generate_ms,
                draft.mel_ms, draft.audio_ms, draft.prompt_ms, draft.source_ms,
                draft.generate_ms, draft.stats.accepted, draft.stats.proposed,
                draft.stats.verify_calls, draft.stats.fallback_steps,
                draft.stats.proposal_ms, draft.stats.verify_ms, draft.stats.prefill_ms);
    std::fflush(stdout);
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::fprintf(stderr,
            "usage: %s moss.gguf golden-dir short.wav medium.wav repeats\n", argv[0]);
        return 2;
    }
    const int repeats = std::atoi(argv[5]);
    if (repeats < 1 || repeats > 5) return 2;
    int diagnostic_k = 0;
    if (const char* value = std::getenv("MOSS_PREVIEW_DIAGNOSTIC_K")) {
        diagnostic_k = std::atoi(value);
        if (diagnostic_k < 1 || diagnostic_k > 4) return 2;
    }
    moss::MossModel model;
    std::string err;
    if (!model.load(argv[1], err)) {
        std::fprintf(stderr, "MOSS model load: %s\n", err.c_str());
        return 2;
    }
    moss::Tokenizer tokenizer;
    if (!tokenizer.load(model.loader, model.config, err)) {
        std::fprintf(stderr, "MOSS tokenizer load: %s\n", err.c_str());
        return 2;
    }
    std::printf("{\"backend\":\"%s\",\"repeats\":%d,"
                "\"row_attention\":%s,\"diagnostic_k\":%d}\n",
                starling_ggml_backend_name(), repeats,
                std::getenv("STARLING_MOSS_VERIFY_ROW_ATTN") ? "true" : "false",
                diagnostic_k);
    std::fflush(stdout);

    for (int case_i = 0; case_i < 2; ++case_i) {
        const char* tier = case_i == 0 ? "short" : "medium";
        const char* diagnostic_tier = std::getenv("MOSS_PREVIEW_DIAGNOSTIC_TIER");
        if (diagnostic_tier && std::string(diagnostic_tier) != tier) continue;
        const std::string root = argv[2];
        const std::string preview = read_file(root + "/parakeet_tdt_" + tier +
                                              "_text.txt");
        const std::string golden_text = read_file(root + "/moss_" + tier +
                                                  "_text.txt");
        std::vector<float> pcm;
        if (preview.empty() || golden_text.empty() ||
            !load_wav(argv[case_i == 0 ? 3 : 4], pcm)) {
            std::fprintf(stderr, "%s: missing preview, golden or 16 kHz WAV\n", tier);
            return 2;
        }
        // Warm each arm after loading the model; exclude warmup from pairs.
        const std::vector<int> warm_ks = diagnostic_k
            ? std::vector<int>{0, diagnostic_k} : std::vector<int>{0, 2, 4};
        for (int k : warm_ks) {
            Run warm;
            if (!run(model, tokenizer, pcm, preview, k, warm, err)) {
                std::fprintf(stderr, "%s warmup k=%d: %s\n", tier, k, err.c_str());
                return 2;
            }
        }
        const std::vector<int> measured_ks = diagnostic_k
            ? std::vector<int>{diagnostic_k} : std::vector<int>{2, 4};
        for (int k : measured_ks) for (int rep = 0; rep < repeats; ++rep) {
            Run greedy, draft;
            if (rep % 2 == 0) {
                if (!run(model, tokenizer, pcm, preview, 0, greedy, err) ||
                    !run(model, tokenizer, pcm, preview, k, draft, err)) {
                    std::fprintf(stderr, "%s pair k=%d rep=%d: %s\n",
                                 tier, k, rep, err.c_str());
                    return 2;
                }
            } else {
                if (!run(model, tokenizer, pcm, preview, k, draft, err) ||
                    !run(model, tokenizer, pcm, preview, 0, greedy, err)) {
                    std::fprintf(stderr, "%s pair k=%d rep=%d: %s\n",
                                 tier, k, rep, err.c_str());
                    return 2;
                }
            }
            const bool golden_match = greedy.text ==
                golden_text.substr(0, golden_text.find_last_not_of("\r\n") + 1);
            print_pair(tier, k, rep, greedy, draft, golden_match);
            if (greedy.output.ids != draft.output.ids ||
                greedy.output.stop_reason != draft.output.stop_reason ||
                greedy.text != draft.text ||
                greedy.output.stop_reason !=
                    starling::ggml::lib::GenStopReason::kEos) {
                size_t first = 0;
                while (first < greedy.output.ids.size() &&
                       first < draft.output.ids.size() &&
                       greedy.output.ids[first] == draft.output.ids[first]) ++first;
                const int32_t greedy_id = first < greedy.output.ids.size()
                    ? greedy.output.ids[first] : -1;
                const int32_t draft_id = first < draft.output.ids.size()
                    ? draft.output.ids[first] : -1;
                std::fprintf(stderr,
                    "%s k=%d rep=%d mismatch index=%zu greedy_id=%d draft_id=%d "
                    "greedy_count=%zu draft_count=%zu greedy_stop=%s draft_stop=%s\n",
                    tier, k, rep, first, greedy_id, draft_id,
                    greedy.output.ids.size(), draft.output.ids.size(),
                    stop_name(greedy.output.stop_reason),
                    stop_name(draft.output.stop_reason));
                return 1;
            }
        }
    }
    return 0;
}
