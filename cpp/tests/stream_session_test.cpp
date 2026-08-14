// stream_session_test.cpp — unit tests for the C++ streaming session port.
//
// Verifies stitch_words, ChunkStreamer boundary advancement, and
// StreamSession buffer management against expected behavior from the
// Python reference (src/starling/stream_chunk.py).

#include "serve/stream_session.hpp"
#include "serve/server.hpp"

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

using namespace starling::serve;

// ---- test harness ---------------------------------------------------------
static int g_tests = 0;
static int g_passed = 0;

#define CHECK(cond) do { \
    ++g_tests; \
    if (cond) { ++g_passed; } \
    else { std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, #cond); } \
} while(0)

#define CHECK_EQ(a, b) do { \
    ++g_tests; \
    if ((a) == (b)) { ++g_passed; } \
    else { std::fprintf(stderr, "FAIL: %s:%d: %s != %s (got %s)\n", \
        __FILE__, __LINE__, #a, #b, std::to_string(a).c_str()); } \
} while(0)

// ---- stitch_words tests ---------------------------------------------------
static void test_stitch_basic() {
    // No overlap: simple concatenation.
    auto r = stitch_words({"hello", "world"}, {"foo", "bar"});
    CHECK(r.size() == 4);

    // Overlap dedup: "the" is shared at the boundary.
    r = stitch_words({"hello", "the", "world"}, {"the", "world", "foo"});
    // Should dedup "the world" → ["hello", "the", "world", "foo"]
    CHECK(r.size() == 4);
    CHECK(r[0] == "hello");
    CHECK(r[1] == "the");
    CHECK(r[2] == "world");
    CHECK(r[3] == "foo");
}

static void test_stitch_empty() {
    auto r = stitch_words({}, {"a", "b"});
    CHECK(r.size() == 2);

    r = stitch_words({"a", "b"}, {});
    CHECK(r.size() == 2);
}

static void test_stitch_no_match() {
    // No common words → concatenation.
    auto r = stitch_words({"alpha", "beta"}, {"gamma", "delta"});
    CHECK(r.size() == 4);
}

static void test_stitch_punctuation() {
    // Punctuation should be stripped for matching (min_match=2 needs 2+ words).
    auto r = stitch_words({"hello", "the", "world."}, {"the", "world,", "foo"});
    // "the world." and "the world," normalize to "the world" → dedup (size=2 >= min_match).
    CHECK(r.size() == 4);  // ["hello", "the", "world.", "foo"]
    CHECK(r[0] == "hello");
    CHECK(r[1] == "the");
    CHECK(r[2] == "world.");
    CHECK(r[3] == "foo");
}

// ---- norm_word test -------------------------------------------------------
static void test_norm_word() {
    CHECK(norm_word("Hello") == "hello");
    CHECK(norm_word("World!") == "world");
    CHECK(norm_word("it's") == "it's");
    CHECK(norm_word("") == "");
}

// ---- split/join tests -----------------------------------------------------
static void test_split_join() {
    auto words = split_words("hello world foo");
    CHECK(words.size() == 3);
    CHECK(words[0] == "hello");
    CHECK(words[2] == "foo");

    CHECK(join_words({"a", "b", "c"}) == "a b c");

    // Empty string.
    words = split_words("");
    CHECK(words.empty());

    // Multiple spaces.
    words = split_words("  a   b  ");
    CHECK(words.size() == 2);
}

// ---- ChunkStreamer test ---------------------------------------------------
static void test_chunk_streamer_basic() {
    // Create a streamer with small chunks for testing.
    // sr=16000, chunk=1s, overlap=0.25s, min=0.5s, partial_interval=0
    ChunkStreamer cs(16000, 1.0, 0.25, 0.5, 0.0);

    // Simulate transcribe function that returns the number of samples as text.
    int call_count = 0;
    TranscribeFn tx = [&](const float* s, int64_t n) -> std::optional<std::string> {
        call_count++;
        return std::to_string(n);
    };

    // Feed exactly one chunk of audio.
    std::vector<float> samples(16000, 0.0f);
    auto result = cs.step(samples, 1.0, tx);
    CHECK(result.has_value());
    CHECK(cs.boundary() == 12000);  // advance = 16000 - 4000 = 12000
}

static void test_chunk_streamer_partial() {
    // Small tail (< min) should not emit a partial.
    ChunkStreamer cs(16000, 1.0, 0.25, 0.5, 0.0);
    int call_count = 0;
    TranscribeFn tx = [&](const float* s, int64_t n) -> std::optional<std::string> {
        call_count++;
        return "text";
    };

    // 0.3s of audio (less than min=0.5s).
    std::vector<float> samples(4800, 0.0f);
    auto result = cs.step(samples, 1.0, tx);
    // Should not emit (too short).
    CHECK(!result.has_value());
}

static void test_chunk_streamer_flush() {
    ChunkStreamer cs(16000, 1.0, 0.25, 0.5, 0.0);
    TranscribeFn tx = [&](const float* s, int64_t n) -> std::optional<std::string> {
        return "flushed";
    };

    // 0.5s of audio → flush should return the text.
    std::vector<float> samples(8000, 0.0f);
    std::string text = cs.flush(samples, tx);
    CHECK(text == "flushed");
}

// ---- model slug mapping tests ---------------------------------------------
static void test_model_mapping() {
    CHECK(slug_to_model("parakeet") == STARLING_GGML_PARAKEET_TDT);
    CHECK(slug_to_model("moss") == STARLING_GGML_MOSS);
    CHECK(slug_to_model("ark") == STARLING_GGML_ARK);
    CHECK(slug_to_model("higgs") == STARLING_GGML_HIGGS);
    CHECK(slug_to_model("unknown") == (starling_ggml_model)0);

    CHECK(is_supported_model("parakeet") == true);
    CHECK(is_supported_model("unknown") == false);

    CHECK(std::string(model_to_slug(STARLING_GGML_PARAKEET_TDT)) == "parakeet");
}

// ---- main -----------------------------------------------------------------
int main() {
    test_stitch_basic();
    test_stitch_empty();
    test_stitch_no_match();
    test_stitch_punctuation();
    test_norm_word();
    test_split_join();
    test_chunk_streamer_basic();
    test_chunk_streamer_partial();
    test_chunk_streamer_flush();
    test_model_mapping();

    std::printf("stream_session_test: %d/%d passed\n", g_passed, g_tests);
    return g_passed == g_tests ? 0 : 1;
}
