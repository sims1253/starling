// stream_session_test.cpp — unit tests for the C++ streaming session port.
//
// Verifies stitch_words, ChunkStreamer boundary advancement, and
// StreamSession buffer management against expected behavior from the
// Python reference (src/starling/stream_chunk.py).

#include "serve/stream_session.hpp"
#include "serve/server.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
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
    // Exactly one transcribe: the window finalize. The 0.25 s tail is below
    // min_ (0.5 s) so no partial attempt is made on it.
    CHECK(call_count == 1);
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
    // Nothing was transcribed: below chunk_ (no finalize window) and below
    // min_ (no partial tail attempt).
    CHECK(call_count == 0);
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
    CHECK(slug_to_model("hojo") == STARLING_GGML_HOJO);
    CHECK(slug_to_model("unknown") == (starling_ggml_model)0);

    CHECK(is_supported_model("parakeet") == true);
    CHECK(is_supported_model("hojo") == true);
    CHECK(is_supported_model("unknown") == false);

    CHECK(std::string(model_to_slug(STARLING_GGML_PARAKEET_TDT)) == "parakeet");
    CHECK(std::string(model_to_slug(STARLING_GGML_HOJO)) == "hojo");

    // Exact equality pins the registry-derived ordering (a reordered or
    // duplicated row would change it).
    CHECK(supported_models_str() == "parakeet moss ark higgs hojo granite qwen3 audex");
}

// ---- ChunkStreamer::rebase -------------------------------------------------
// rebase() keeps the boundary valid after the session drops `dropped` samples
// from the front of its buffer (the PR #9 review fix; previously untested).
static void test_chunk_streamer_rebase() {
    ChunkStreamer cs(16000, 1.0, 0.25, 0.5, 0.0);
    TranscribeFn tx = [](const float*, int64_t) -> std::optional<std::string> {
        return "w";
    };

    // Finalize two windows: boundary 0 -> 12000 -> 24000.
    std::vector<float> samples(30000, 0.0f);
    cs.step(samples, 1.0, tx);
    cs.step(samples, 2.0, tx);
    CHECK(cs.boundary() == 24000);

    // The session trims the first 24000 samples; the boundary must follow.
    samples.erase(samples.begin(), samples.begin() + 24000);
    cs.rebase(24000);
    CHECK(cs.boundary() == 0);

    // Rebase never goes negative (clamps at 0).
    cs.rebase(100);
    CHECK(cs.boundary() == 0);

    // After rebase the streamer continues from the shifted origin: the next
    // full window uses samples[0..16000) of the trimmed buffer.
    samples.resize(16000, 0.0f);
    int called = 0;
    int64_t last_n = -1;
    const float* last_p = nullptr;
    TranscribeFn tx2 = [&](const float* p, int64_t n) -> std::optional<std::string> {
        called++;
        last_n = n;
        last_p = p;
        return "w";
    };
    cs.step(samples, 3.0, tx2);
    CHECK(called == 1);
    CHECK(last_n == 16000);
    CHECK(last_p == samples.data());  // window starts at the shifted origin
    CHECK(cs.boundary() == 12000);
}

// ---- StreamSession tests ---------------------------------------------------
// These drive StreamSession itself with an injected fake transcribe fn
// (set_transcribe_fn) so the rolling-buffer trim + busy-retry logic is tested
// without a model. The StarlingServer is constructed (never loaded) purely to
// supply the session's config.

static ServerConfig test_cfg() {
    ServerConfig cfg;                       // model/gguf unused (never loaded)
    cfg.stream_chunk_seconds = 1.0;         // 16000-sample windows
    cfg.stream_overlap_seconds = 0.25;      // advance = 12000
    cfg.min_chunk_seconds = 0.5;            // 8000-sample partial minimum
    cfg.partial_interval = 0.0;             // never throttle in tests
    cfg.max_stream_seconds = 0.0;           // unlimited (cap tested via HTTP)
    return cfg;
}

// Encode position into the audio: int16 value of sample i is (i%30000)-15000,
// so a window's first sample reveals which absolute index it starts at.
static std::string pcm_for_range(int64_t start, int64_t n) {
    std::string bytes(n * 2, '\0');
    for (int64_t i = 0; i < n; ++i) {
        int16_t v = static_cast<int16_t>((start + i) % 30000 - 15000);
        std::memcpy(&bytes[static_cast<size_t>(i) * 2], &v, 2);
    }
    return bytes;
}

static void test_stream_session_buffer_trim() {
    StarlingServer server(test_cfg());
    StreamSession session(&server);

    // Record the first sample + length of every transcribe call.
    std::vector<std::pair<int16_t, int64_t>> calls;
    TranscribeFn tx = [&](const float* p, int64_t n) -> std::optional<std::string> {
        calls.emplace_back(static_cast<int16_t>(p[0] * 32768.0f), n);
        return "w";
    };
    session.set_transcribe_fn(tx);

    // 2.5 s of audio, then step: three full windows finalize
    // (boundaries 0/12000/24000, each 16000 samples), boundary -> 36000.
    session.append_pcm(pcm_for_range(0, 40000));
    auto r1 = session.stream_step(1.0);
    CHECK(r1.has_value());
    CHECK(calls.size() == 3);
    CHECK(session.buffered_seconds() == 2.5);
    CHECK(session.live_seconds() == 2.5);   // nothing trimmed yet

    // The next append triggers the trim: boundary (36000) >= kStreamTrimMin
    // (16000), so the first 36000 samples drop and the chunker rebases.
    session.append_pcm(pcm_for_range(40000, 1600));
    CHECK(session.buffered_seconds() == 2.6);  // 41600 samples total
    CHECK(session.live_seconds() == 0.35);     // 5600 live after trim

    // Rebase correctness: top up the live buffer to exactly one window and
    // step. The window must be samples[0..16000) of the TRIMMED buffer, i.e.
    // absolute samples [36000, 52000): first value = 36000%30000-15000 = -9000.
    session.append_pcm(pcm_for_range(41600, 10400));
    size_t calls_before = calls.size();
    auto r2 = session.stream_step(2.0);
    CHECK(r2.has_value());
    CHECK(calls.size() == calls_before + 1);
    if (calls.size() == calls_before + 1) {
        CHECK(calls.back().second == 16000);        // full window
        CHECK(calls.back().first == -9000);         // starts at abs index 36000
    }

    // Commit finalizes the remaining tail (4000 live samples past the
    // boundary, starting at abs index 36000+12000=48000 -> value 3000).
    calls.clear();
    std::string text = session.stream_flush();
    // Five transcribes total (4 windows + tail), each returning "w";
    // single-word texts never dedup (min_match=2) so they space-join.
    CHECK(text == "w w w w w");
    CHECK(calls.size() == 1);
    if (calls.size() == 1) {
        CHECK(calls[0].second == 4000);             // tail past the boundary
        CHECK(calls[0].first == 48000 % 30000 - 15000);
    }
    CHECK(session.buffered_seconds() == 3.25);       // nothing lost overall
}

static void test_stream_session_busy_retry() {
    // TranscribeFn returning nullopt means "transcriber busy": the chunker
    // must retry WITHOUT advancing state (boundary unchanged, no commit).
    ChunkStreamer cs(16000, 1.0, 0.25, 0.5, 0.0);
    int call_count = 0;
    TranscribeFn busy = [&](const float*, int64_t) -> std::optional<std::string> {
        call_count++;
        return std::nullopt;
    };

    std::vector<float> samples(24000, 0.0f);  // 1.5 s: one window + 0.5 s tail
    auto r = cs.step(samples, 1.0, busy);
    CHECK(!r.has_value());          // busy: nothing emitted
    CHECK(cs.boundary() == 0);      // boundary unchanged (retry later)
    // step made exactly two attempts: the finalize window + the live tail
    // (tail >= min_ with partial_interval 0).
    CHECK(call_count == 2);

    // flush retries the tail kMaxRetries (5) times, then drops it with the
    // boundary still unchanged — the audio stays buffered for a later retry.
    std::string text = cs.flush(samples, busy);
    CHECK(text.empty());
    CHECK(cs.boundary() == 0);
    // 1 finalize attempt + 5 tail retries.
    CHECK(call_count == 2 + 1 + 5);

    // End-to-end through StreamSession: busy transcriber → no emission, and
    // the buffered audio is neither trimmed nor lost.
    StarlingServer server(test_cfg());
    StreamSession session(&server);
    session.set_transcribe_fn(busy);
    session.append_pcm(pcm_for_range(0, 24000));
    auto rs = session.stream_step(1.0);
    CHECK(!rs.has_value());
    CHECK(session.buffered_seconds() == 1.5);
    CHECK(session.live_seconds() == 1.5);
    std::string fs = session.stream_flush();
    CHECK(fs.empty());
    CHECK(session.buffered_seconds() == 1.5);
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
    test_chunk_streamer_rebase();
    test_stream_session_buffer_trim();
    test_stream_session_busy_retry();
    test_model_mapping();

    std::printf("stream_session_test: %d/%d passed\n", g_passed, g_tests);
    return g_passed == g_tests ? 0 : 1;
}
