#include "granite/ctc_proposer.hpp"
#include "granite/speculative.hpp"

#include <cstdio>
#include <string>
#include <vector>

int main() {
    using starling::ggml::granite::CtcProposer;
    int failed = 0;
    const auto check = [&](bool pass, const char* label) {
        std::printf("[%s] %s\n", pass ? "PASS" : "FAIL", label);
        failed += !pass;
    };
    CtcProposer aligned({10, 11, 12, 13, 14}, 4);
    check(aligned.propose({5}, 4) == std::vector<int32_t>({10, 11}),
          "prefill sees only CTC draft");
    check(aligned.propose({5, 10, 11, 12}, 4) == std::vector<int32_t>({13, 14}),
          "full accept and target bonus realign ahead");
    check(aligned.propose({5, 10, 11, 12, 13, 14, 99}, 4).empty(),
          "exhausted draft falls back to greedy");

    CtcProposer rejected({10, 11, 12, 13, 14}, 4);
    check(rejected.propose({5}, 4) == std::vector<int32_t>({10, 11}),
          "rejection case initial proposal");
    check(rejected.propose({5, 10, 12}, 4) == std::vector<int32_t>({13}),
          "target correction realigns after partial accept");
    check(rejected.propose({5, 10, 12, 13, 14}, 4).empty(),
          "accepted short proposal and bonus exhaust draft");

    CtcProposer stalled({10, 11, 12}, 4);
    check(stalled.propose({5}, 1) == std::vector<int32_t>({10}),
          "verifier cap bounds proposal");
    std::vector<int32_t> prefix{5};
    for (int i = 0; i < 8; ++i) {
        prefix.push_back(50 + i);
        auto next = stalled.propose(prefix, 1);
        check(next == std::vector<int32_t>({i == 7 ? 11 : 10}),
              "stalled formatting boundary eventually advances");
    }
    starling::ggml::granite::GraniteModel unloaded;
    starling::ggml::granite::MelFeatures mel;
    starling::ggml::granite::InputsEmbeds input;
    starling::ggml::granite::GenerateOptions options;
    starling::ggml::granite::GenerateResult cancelled_output;
    starling::ggml::granite::CtcSpeculativeStats cancelled_stats;
    std::string err;
    check(starling::ggml::granite::ctc_speculative_generate(
              unloaded, mel, input, options, 2, [] { return true; },
              cancelled_output, cancelled_stats, err) &&
              cancelled_output.ids.empty() &&
              cancelled_output.stop_reason == starling::ggml::lib::GenStopReason::kCancelled &&
              cancelled_stats.draft_count == 0,
          "cancel before CTC extraction publishes no output");
    check(!starling::ggml::granite::ctc_speculative_generate(
              unloaded, mel, input, options, 17, {},
              cancelled_output, cancelled_stats, err),
          "invalid maximum K is rejected before extraction");
    return failed ? 1 : 0;
}
