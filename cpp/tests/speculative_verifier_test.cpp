// Exercise the real batched Qwen verifier on a tiny Granite decoder.
#include "granite/llm.hpp"
#include "granite/loader.hpp"
#include "moss/llm.hpp"
#include "moss/loader.hpp"
#include "runtime/backend.hpp"
#include "runtime/graph.hpp"
#include "golden_io.hpp"
#include "tiny_granite_fixture.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace {
using namespace starling::ggml;
using namespace starling::ggml::granite;
using starling::ggml::lib::GenStopReason;
using starling::ggml::lib::SpeculativeStats;

int failures = 0;
void check(bool ok, const char* what, const std::string& detail = {}) {
    std::printf("[%s] %s%s%s\n", ok ? "PASS" : "FAIL", what,
                ok || detail.empty() ? "" : ": ", ok ? "" : detail.c_str());
    if (!ok) ++failures;
}

std::string ids_string(const std::vector<int32_t>& ids) {
    std::string out;
    for (int32_t id : ids) out += std::to_string(id) + " ";
    return out;
}

bool same_output(const GenerateResult& a, const GenerateResult& b) {
    return a.ids == b.ids && a.stop_reason == b.stop_reason;
}

bool real_moss_checks(const char* gguf, const char* embeddings) {
    moss::MossModel model;
    std::string err;
    if (!model.load(gguf, err)) {
        std::fprintf(stderr, "real MOSS load failed: %s\n", err.c_str());
        return false;
    }
    moss::InputsEmbeds input;
    if (!test::read_f32(embeddings, input.data, err)) {
        std::fprintf(stderr, "real MOSS embeddings failed: %s\n", err.c_str());
        return false;
    }
    input.width = model.config.llm.hidden;
    if (input.width <= 0 || input.data.size() % (size_t)input.width != 0) {
        std::fprintf(stderr, "real MOSS embeddings size %zu is invalid for width %lld\n",
                     input.data.size(), (long long)input.width);
        return false;
    }
    input.n_tokens = (int64_t)(input.data.size() / (size_t)input.width);
    moss::GenerateOptions op;
    op.max_new_tokens = 12;
    op.max_cache_len = model.config.llm.max_cache;
    op.eos_token_id = -1;
    const auto now = [] { return std::chrono::steady_clock::now(); };
    moss::GenerateResult warmup, greedy, perfect_out, rejected_out;
    lib::SpeculativeStats perfect_stats, rejected_stats;
    if (!moss::greedy_generate(model, input, op, warmup, err)) {
        std::fprintf(stderr, "real MOSS warmup failed: %s\n", err.c_str());
        return false;
    }
    auto t0 = now();
    if (!moss::greedy_generate(model, input, op, greedy, err)) {
        std::fprintf(stderr, "real MOSS greedy failed: %s\n", err.c_str());
        return false;
    }
    auto t1 = now();
    auto perfect = [&](const std::vector<int32_t>& prefix, int cap) {
        const size_t offset = std::min(prefix.size(), greedy.ids.size());
        const size_t count = std::min((size_t)cap, greedy.ids.size() - offset);
        const auto begin = greedy.ids.begin() + (ptrdiff_t)offset;
        return std::vector<int32_t>(begin, begin + (ptrdiff_t)count);
    };
    if (!moss::speculative_generate(model, input, op, 4, perfect, {},
                                    perfect_out, perfect_stats, err)) {
        std::fprintf(stderr, "real MOSS perfect verify failed: %s\n", err.c_str());
        return false;
    }
    auto t2 = now();
    int calls = 0;
    auto rejected = [&](const std::vector<int32_t>& prefix, int cap) {
        auto draft = perfect(prefix, cap);
        if (calls++ == 0 && draft.size() > 2)
            draft[2] = (draft[2] + 1) % (int32_t)model.config.llm.vocab;
        return draft;
    };
    if (!moss::speculative_generate(model, input, op, 4, rejected, {},
                                    rejected_out, rejected_stats, err)) {
        std::fprintf(stderr, "real MOSS rejection verify failed: %s\n", err.c_str());
        return false;
    }
    auto t3 = now();
    const auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    std::printf("real MOSS CPU (warmed): prompt=%lld output=%zu greedy=%.1fms "
                "perfect=%.1fms rejected=%.1fms perfect_accept=%d/%d "
                "rejected_accept=%d/%d proposal=%.3f/%.3fms "
                "verify=%.1f/%.1fms prefill=%.1f/%.1fms "
                "total=%.1f/%.1fms\n",
                (long long)input.n_tokens, greedy.ids.size(), ms(t0, t1),
                ms(t1, t2), ms(t2, t3), perfect_stats.accepted,
                perfect_stats.proposed, rejected_stats.accepted,
                rejected_stats.proposed, perfect_stats.proposal_ms,
                rejected_stats.proposal_ms, perfect_stats.verify_ms,
                rejected_stats.verify_ms, perfect_stats.prefill_ms,
                rejected_stats.prefill_ms, perfect_stats.total_ms,
                rejected_stats.total_ms);
    check(same_output(warmup, greedy), "real MOSS warmup matches measured greedy");
    check(same_output(perfect_out, greedy), "real MOSS perfect oracle matches greedy",
          ids_string(perfect_out.ids));
    check(same_output(rejected_out, greedy), "real MOSS rejection and rollback match greedy",
          ids_string(rejected_out.ids));
    return true;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 1 && argc != 3) {
        std::fprintf(stderr, "usage: %s [moss.gguf moss_inputs_embeds.f32]\n", argv[0]);
        return 2;
    }
    const auto path = std::filesystem::temp_directory_path() /
                      "starling-speculative-verifier-tiny.gguf";
    TinyGraniteFixture fixture(path, /*token_chain=*/true);
    if (!fixture.wrote()) return 2;
    GraniteModel model;
    std::string err;
    if (!model.load(path.string().c_str(), err)) {
        std::fprintf(stderr, "load failed: %s\n", err.c_str());
        return 2;
    }
    InputsEmbeds input;
    input.width = model.config.llm.hidden;
    input.n_tokens = 4;
    input.data.assign((size_t)input.width * (size_t)input.n_tokens, 0.0f);
    GenerateOptions op;
    op.max_new_tokens = 10;
    op.max_cache_len = 14;  // exactly prompt + output budget
    op.eos_token_id = -1;
    GenerateResult greedy;
    if (!greedy_generate(model, input, op, greedy, err)) {
        std::fprintf(stderr, "greedy failed: %s\n", err.c_str());
        return 2;
    }
    check(greedy.ids == std::vector<int32_t>({0, 1, 2, 3, 4, 0, 1, 2, 3, 4}),
          "token-chain fixture yields changing greedy tokens", ids_string(greedy.ids));

    // The second cancellation check runs after prefill. Neither a one-token
    // budget nor first-token EOS may commit the in-flight prefill result.
    for (int eos : {-1, 0}) {
        GenerateOptions first_op = op;
        first_op.max_new_tokens = 1;
        first_op.eos_token_id = eos;
        GenerateResult prefill_cancelled;
        SpeculativeStats prefill_cancel_stats;
        int prefill_checks = 0;
        err.clear();
        const bool prefill_cancel_ok = speculative_generate(model, input,
            first_op, 4,
            [](const std::vector<int32_t>&, int) { return std::vector<int32_t>{}; },
            [&] { return ++prefill_checks == 2; },
            prefill_cancelled, prefill_cancel_stats, err);
        check(prefill_cancel_ok && prefill_checks == 2 &&
                  prefill_cancelled.stop_reason == GenStopReason::kCancelled &&
                  prefill_cancelled.ids.empty(),
              eos < 0 ? "cancel after one-token prefill hides output"
                      : "cancel after first-token EOS prefill hides output", err);
    }

    // A perfect proposer verifies multiple candidates per target pass.
    auto perfect = [&](const std::vector<int32_t>& prefix, int cap) {
        const size_t offset = std::min(prefix.size(), greedy.ids.size());
        const size_t count = std::min((size_t)cap, greedy.ids.size() - offset);
        const auto begin = greedy.ids.begin() + (ptrdiff_t)offset;
        return std::vector<int32_t>(begin, begin + (ptrdiff_t)count);
    };
    GenerateResult all;
    SpeculativeStats all_stats;
    err.clear();
    const bool all_ok = speculative_generate(model, input, op, 4, perfect, {},
                                             all, all_stats, err);
    check(all_ok && same_output(all, greedy) && all_stats.accepted > 0 &&
              all_stats.verify_calls > 0 && all_stats.fallback_steps == 0,
          "accept-all matches greedy at the cache boundary", err);

    GenerateOptions over_limit = op;
    over_limit.max_cache_len = 13;
    GenerateResult rejected_config;
    SpeculativeStats rejected_config_stats;
    err.clear();
    check(!speculative_generate(model, input, over_limit, 4, perfect, {},
                                rejected_config, rejected_config_stats, err) &&
              !err.empty(), "cache boundary rejects one extra output token");

    GenerateResult empty_draft;
    SpeculativeStats empty_stats;
    err.clear();
    const bool empty_ok = speculative_generate(model, input, op, 4,
        [](const std::vector<int32_t>&, int) { return std::vector<int32_t>{}; },
        {}, empty_draft, empty_stats, err);
    check(empty_ok && same_output(empty_draft, greedy) &&
              empty_stats.verify_calls == 0 && empty_stats.fallback_steps == 9,
          "empty draft falls back to greedy steps", err);

    // Cancellation during a proposer must discard its tentative draft before
    // building a verification graph.
    bool stop_after_propose = false;
    GenerateResult proposal_cancelled;
    SpeculativeStats proposal_cancel_stats;
    err.clear();
    const bool proposal_cancel_ok = speculative_generate(model, input, op, 4,
        [&](const std::vector<int32_t>& prefix, int cap) {
            auto draft = perfect(prefix, cap);
            stop_after_propose = true;
            return draft;
        }, [&] { return stop_after_propose; },
        proposal_cancelled, proposal_cancel_stats, err);
    check(proposal_cancel_ok &&
              proposal_cancelled.stop_reason == GenStopReason::kCancelled &&
              proposal_cancelled.ids == std::vector<int32_t>{0} &&
              proposal_cancel_stats.verify_calls == 0,
          "cancel after proposal skips verification", err);

    // A cancellation triggered by the fallback graph must not emit its
    // tentative token, just as with a multi-row verify graph.
    int fallback_checks = 0;
    GenerateResult fallback_cancelled;
    SpeculativeStats fallback_cancel_stats;
    err.clear();
    const bool fallback_cancel_ok = speculative_generate(model, input, op, 4,
        [](const std::vector<int32_t>&, int) { return std::vector<int32_t>{}; },
        [&] { return ++fallback_checks == 5; },
        fallback_cancelled, fallback_cancel_stats, err);
    check(fallback_cancel_ok &&
              fallback_cancelled.stop_reason == GenStopReason::kCancelled &&
              fallback_cancelled.ids == std::vector<int32_t>{0},
          "cancel after fallback hides tentative output", err);

    // Reject at each draft position. Later steps must ignore the stale KV
    // rows written for the rejected suffix and produce the greedy sequence.
    for (int reject_at = 0; reject_at < 4; ++reject_at) {
        int proposals = 0;
        auto proposer = [&](const std::vector<int32_t>& prefix, int cap) {
            if (proposals++) return std::vector<int32_t>{};
            auto draft = perfect(prefix, cap);
            draft[(size_t)reject_at] =
                (draft[(size_t)reject_at] + 1) % (int32_t)model.config.llm.vocab;
            return draft;
        };
        GenerateResult got;
        SpeculativeStats stats;
        err.clear();
        const bool ok = speculative_generate(model, input, op, 4, proposer, {},
                                              got, stats, err);
        check(ok && same_output(got, greedy) && stats.accepted == reject_at &&
                  stats.verify_calls == 1 && stats.fallback_steps > 0,
              ("reject draft position " + std::to_string(reject_at)).c_str(),
              err.empty() ? ids_string(got.ids) : err);
    }

    // EOS generated inside a verified draft commits only the prefix through
    // EOS, even though the graph computed later tentative rows.
    GenerateOptions eos_op = op;
    eos_op.eos_token_id = 2;
    GenerateResult eos_greedy, eos_spec;
    SpeculativeStats eos_stats;
    err.clear();
    const bool eos_ok = greedy_generate(model, input, eos_op, eos_greedy, err) &&
        speculative_generate(model, input, eos_op, 4,
            [](const std::vector<int32_t>&, int cap) {
                std::vector<int32_t> draft{1, 2, 3, 4};
                draft.resize(std::min(draft.size(), (size_t)cap));
                return draft;
            }, {}, eos_spec, eos_stats, err);
    check(eos_ok && same_output(eos_spec, eos_greedy) &&
              eos_spec.ids == std::vector<int32_t>({0, 1, 2}) &&
              eos_spec.stop_reason == GenStopReason::kEos && eos_stats.accepted == 2,
          "EOS inside draft stops at verified token", err);

    // The fifth cancellation check runs after the verify graph. Tentative
    // tokens must stay invisible; a fresh run must still match greedy.
    int checks = 0;
    GenerateResult cancelled;
    SpeculativeStats cancel_stats;
    err.clear();
    const bool cancel_ok = speculative_generate(model, input, op, 4, perfect,
        [&] { return ++checks == 5; }, cancelled, cancel_stats, err);
    check(cancel_ok && cancelled.stop_reason == GenStopReason::kCancelled &&
              cancelled.ids == std::vector<int32_t>{0},
          "cancel after verification hides tentative output", err);
    GenerateResult resumed;
    SpeculativeStats resumed_stats;
    err.clear();
    const bool resumed_ok = speculative_generate(model, input, op, 4, perfect, {},
                                                  resumed, resumed_stats, err);
    check(resumed_ok && same_output(resumed, greedy),
          "new run after cancellation matches greedy", err);

    const bool moss_ok = argc != 3 || real_moss_checks(argv[1], argv[2]);

    shutdown_backend();
    if (!moss_ok) return 2;
    return failures ? 1 : 0;
}
