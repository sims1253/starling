// stream_session_test.cpp — unit tests for the C++ streaming session port.
//
// Verifies stitch_words, ChunkStreamer boundary advancement, and
// StreamSession buffer management against expected behavior from the
// Python reference (src/starling/stream_chunk.py).

#include "serve/stream_pump.hpp"
#include "serve/stream_session.hpp"
#include "serve/server.hpp"
#include "runtime/call_abort.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
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

// Seconds accessors return doubles (sample_count / kSampleRate): compare
// with a tolerance instead of exact == — a count whose quotient happens to be
// binary-exact today need not stay that way. The tolerance is relative to
// the expected magnitude (1e-9, with a 1.0 floor so near-zero expectations
// keep an absolute bound): a fixed epsilon that is fine for seconds-scale
// values would grow too tight for large durations and too loose for tiny
// ones.
#define CHECK_NEAR(a, b) do { \
    ++g_tests; \
    const double got_ = static_cast<double>(a); \
    const double want_ = static_cast<double>(b); \
    const double tol_ = 1e-9 * std::max(1.0, std::fabs(want_)); \
    if (std::fabs(got_ - want_) <= tol_) { ++g_passed; } \
    else { std::fprintf(stderr, "FAIL: %s:%d: %s != %s (got %s)\n", \
        __FILE__, __LINE__, #a, #b, std::to_string(got_).c_str()); } \
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

// ---- non-ASCII stitch keys (issue #118) ------------------------------------
// std::isalnum is false for every byte >= 0x80 in the default C locale, so the
// old byte filter mapped Cyrillic/CJK words to "" and runs of empty keys then
// matched as an overlap, deleting the new chunk's words at every window
// boundary. These tests pin the conservative UTF-8-aware keys.
static void test_stitch_unicode_disjoint() {
    // Disjoint Cyrillic chunks must keep ALL words (the empty-key bug
    // returned just the committed half: ["привет", "мир"]).
    auto r = stitch_words({u8"привет", u8"мир"}, {u8"совсем", u8"другое"});
    CHECK(r.size() == 4);
    CHECK(r[0] == u8"привет");
    CHECK(r[1] == u8"мир");
    CHECK(r[2] == u8"совсем");
    CHECK(r[3] == u8"другое");

    // Disjoint CJK chunks keep all words too.
    r = stitch_words({u8"你好", u8"世界"}, {u8"再见", u8"朋友"});
    CHECK(r.size() == 4);
    CHECK(r[0] == u8"你好");
    CHECK(r[3] == u8"朋友");
}

static void test_stitch_unicode_overlap() {
    // A genuine Cyrillic overlap still dedupes across the window boundary.
    auto r = stitch_words({u8"привет", u8"мир", u8"тут"}, {u8"мир", u8"тут", u8"ок"});
    CHECK(r.size() == 4);  // ["привет", "мир", "тут", "ок"]
    CHECK(r[0] == u8"привет");
    CHECK(r[2] == u8"тут");
    CHECK(r[3] == u8"ок");

    // CJK overlap dedupes.
    r = stitch_words({u8"今天", u8"天气", u8"很好"}, {u8"天气", u8"很好", u8"吗"});
    CHECK(r.size() == 4);
    CHECK(r[3] == u8"吗");

    // Mixed-script boundary: a shared Cyrillic run inside ASCII text dedupes.
    r = stitch_words({"hello", u8"мир", u8"тут"}, {u8"мир", u8"тут", "ok"});
    CHECK(r.size() == 4);  // ["hello", "мир", "тут", "ok"]
    CHECK(r[0] == "hello");
    CHECK(r[3] == "ok");

    // ASCII punctuation around non-ASCII words is still stripped for keys.
    r = stitch_words({u8"привет,", u8"мир."}, {u8"привет", u8"мир", "!"});
    CHECK(r.size() == 3);  // committed spelling kept: ["привет,", "мир.", "!"]
    CHECK(r[0] == u8"привет,");
    CHECK(r[2] == "!");
}

static void test_stitch_empty_keys_never_match() {
    // Words that normalize to "" (pure ASCII punctuation) must never count as
    // an overlap run — defense in depth for issue #118: matching empty keys
    // would drop the new chunk's leading words.
    auto r = stitch_words({"--", ";;"}, {"..", ",,", "word"});
    CHECK(r.size() == 5);
    CHECK(r[4] == "word");
}

// ---- norm_word test -------------------------------------------------------
static void test_norm_word() {
    CHECK(norm_word("Hello") == "hello");
    CHECK(norm_word("World!") == "world");
    CHECK(norm_word("it's") == "it's");
    CHECK(norm_word("") == "");

    // Non-ASCII words keep their UTF-8 bytes: no more empty keys (issue #118).
    CHECK(norm_word(u8"привет") == u8"привет");
    CHECK(norm_word(u8"你好") == u8"你好");
    CHECK(norm_word(u8"café") == u8"café");
    // ASCII inside UTF-8 text: punctuation stripped, ASCII lowercased,
    // non-ASCII bytes kept verbatim (no Unicode case folding — the key only
    // has to be deterministic and identical across chunk boundaries).
    CHECK(norm_word(u8"Привет, World!") == u8"Приветworld");
    // Pure-ASCII punctuation still normalizes to the empty string.
    CHECK(norm_word("--") == "");
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
    // Two transcribes: the window finalize, then a preview of the 0.25 s
    // overlap tail. The first-partial minimum (0.5 s) gates the take, not
    // the tail (issue #357), and the take already holds 1 s.
    CHECK(call_count == 2);
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
    TranscribeFn tx = [](const float*, int64_t) -> std::optional<std::string> {
        return "flushed";
    };

    // 0.5s of audio → flush should return the text.
    std::vector<float> samples(8000, 0.0f);
    auto text = cs.flush(samples, tx);
    CHECK(text == "flushed");
}

static void test_chunk_streamer_unicode_boundary() {
    // Cyrillic transcripts on two overlapping windows (issue #118): the
    // shared boundary words dedupe AND every unique word survives. With the
    // old empty-key normalization the second window's words were dropped
    // entirely at the boundary.
    ChunkStreamer cs(16000, 1.0, 0.5, 0.5, 0.0);
    int call = 0;
    TranscribeFn tx = [&](const float*, int64_t) -> std::optional<std::string> {
        ++call;
        return call == 1 ? u8"привет мир сегодня" : u8"мир сегодня хорошая";
    };

    // 1 s of audio: one full window [0,16000) finalizes, then the tail
    // [8000,16000) is transcribed for the partial — the tail's transcript
    // overlaps the window's by two words.
    std::vector<float> samples(16000, 0.0f);
    auto result = cs.step(samples, 1.0, tx);
    CHECK(result.has_value());
    CHECK(*result == u8"привет мир сегодня хорошая");
    CHECK(call == 2);
    CHECK(cs.boundary() == 8000);
}

// ---- ChunkStreamer::stable_words -------------------------------------------
// The stable prefix is a promise to streaming clients (#297): the first
// stable_words() words of every later partial and of the final never change.
static void test_chunk_streamer_stable_prefix_never_changes() {
    // chunk 1 s, overlap 0.25 s: max_overlap_words = 8.
    ChunkStreamer cs(16000, 1.0, 0.25, 0.5, 0.0);
    int call = 0;
    // Window 1 commits twelve words (a0..a3 frozen). Window 2 re-hears the
    // two words right after the frozen prefix and little else, so the
    // stitch keeps only up to them and the text shrinks to seven words.
    // Window 3 then aligns on "a1 a2": a plain stitch over the last eight
    // words would keep a0..a2 and drop the frozen "a3".
    TranscribeFn tx = [&](const float*, int64_t) -> std::optional<std::string> {
        ++call;
        switch (call) {
            case 1: return std::string("a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11");
            case 2: return std::string("a4 a5 b0");
            case 3: return std::string("a1 a2 c0 c1");
            default: return std::string("d0 d1 d2 d3 d4 d5 d6 d7 d8 d9");
        }
    };
    std::vector<std::string> prefix;
    int64_t stable = 0;
    std::vector<float> samples;
    auto check = [&](const std::optional<std::string>& text) {
        CHECK(cs.stable_words() >= stable);
        if (!text) return;
        auto words = split_words(*text);
        CHECK(static_cast<int64_t>(words.size()) >= cs.stable_words());
        for (size_t i = 0; i < prefix.size() && i < words.size(); ++i) {
            CHECK(words[i] == prefix[i]);
        }
        stable = cs.stable_words();
        prefix.assign(words.begin(), words.begin() + stable);
    };
    for (int step = 1; step <= 5; ++step) {
        samples.resize(static_cast<size_t>(12000 * step + 4000), 0.0f);
        check(cs.step(samples, static_cast<double>(step), tx));
    }
    check(cs.flush(samples, tx));
    CHECK(stable >= 4);
    CHECK(prefix.size() >= 4 && prefix[3] == "a3");
    cs.reset();
    CHECK(cs.stable_words() == 0);
}

// ---- model slug mapping tests ---------------------------------------------
static void test_model_mapping() {
    CHECK(slug_to_model("parakeet") == STARLING_GGML_PARAKEET_TDT);
    CHECK(slug_to_model("moss") == STARLING_GGML_MOSS);
    CHECK(slug_to_model("ark") == STARLING_GGML_ARK);
    CHECK(slug_to_model("higgs") == STARLING_GGML_HIGGS);
    CHECK(slug_to_model("hojo") == STARLING_GGML_HOJO);
    CHECK(slug_to_model("s1") == STARLING_GGML_S1);
    CHECK(slug_to_model("unknown") == (starling_ggml_model)0);

    CHECK(is_supported_model("parakeet") == true);
    CHECK(is_supported_model("hojo") == true);
    CHECK(is_supported_model("s1") == true);
    CHECK(is_supported_model("unknown") == false);

    CHECK(std::string(model_to_slug(STARLING_GGML_PARAKEET_TDT)) == "parakeet");
    CHECK(std::string(model_to_slug(STARLING_GGML_HOJO)) == "hojo");
    CHECK(std::string(model_to_slug(STARLING_GGML_S1)) == "s1");

    // Exact equality pins the registry-derived ordering (a reordered or
    // duplicated row would change it).
    CHECK(supported_models_str() == "parakeet moss ark higgs hojo granite qwen3 s1 audex ark06 voxtral qwen3_06");
}

// ---- ChunkStreamer::rebase -------------------------------------------------
// rebase() keeps the boundary valid after the session drops `dropped` samples
// from the front of its buffer (the PR #9 review fix; previously untested).
static void test_chunk_streamer_rebase() {
    // First-partial minimum 100 s: previews never run, so every transcribe
    // below is a window and the offsets are the point of the test.
    ChunkStreamer cs(16000, 1.0, 0.25, 100.0, 0.0);
    TranscribeFn tx = [](const float*, int64_t) -> std::optional<std::string> {
        return "w";
    };

    // Finalize two windows: boundary 0 -> 12000 -> 24000. The first step
    // emits the stitched text; the second is a no-op (no new window, and
    // previews are off) and returns nullopt — both returns are checked, not
    // discarded.
    std::vector<float> samples(30000, 0.0f);
    CHECK(cs.step(samples, 1.0, tx).has_value());
    CHECK(!cs.step(samples, 2.0, tx).has_value());
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
    // Previews off (100 s first-partial minimum): every transcribe below is a
    // window or the flush tail, and their offsets are the point of the test.
    session.set_preview_policy(100.0, 0.0);

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
    CHECK_NEAR(session.buffered_seconds(), 2.5);
    CHECK_NEAR(session.live_seconds(), 2.5);   // nothing trimmed yet

    // The next append triggers the trim: boundary (36000) >= kStreamTrimMin
    // (16000), so the first 36000 samples drop and the chunker rebases.
    session.append_pcm(pcm_for_range(40000, 1600));
    CHECK_NEAR(session.buffered_seconds(), 2.6);  // 41600 samples total
    CHECK_NEAR(session.live_seconds(), 0.35);     // 5600 live after trim

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
    auto text = session.stream_flush();
    // Five transcribes total (4 windows + tail), each returning "w";
    // single-word texts never dedup (min_match=2) so they space-join.
    CHECK(text == "w w w w w");
    CHECK(calls.size() == 1);
    if (calls.size() == 1) {
        CHECK(calls[0].second == 4000);             // tail past the boundary
        CHECK(calls[0].first == 48000 % 30000 - 15000);
    }
    CHECK_NEAR(session.buffered_seconds(), 3.25);   // nothing lost overall
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
    CHECK(call_count == 1);

    auto text = cs.flush(samples, busy);
    CHECK(!text.has_value());
    CHECK(cs.boundary() == 0);
    CHECK(call_count == 1 + 5);

    // End-to-end through StreamSession: busy transcriber → no emission, and
    // the buffered audio is neither trimmed nor lost.
    StarlingServer server(test_cfg());
    StreamSession session(&server);
    session.set_transcribe_fn(busy);
    session.append_pcm(pcm_for_range(0, 24000));
    auto rs = session.stream_step(1.0);
    CHECK(!rs.has_value());
    CHECK_NEAR(session.buffered_seconds(), 1.5);
    CHECK_NEAR(session.live_seconds(), 1.5);
    auto fs = session.stream_flush();
    CHECK(!fs.has_value());
    CHECK_NEAR(session.buffered_seconds(), 1.5);
    session.set_transcribe_fn([](const float*, int64_t n) -> std::optional<std::string> {
        CHECK(n <= 16000);
        return "retained audio";
    });
    CHECK(session.stream_flush() == "retained audio");
    CHECK_NEAR(session.buffered_seconds(), 1.5);
}

// Build a minimal mono PCM16 RIFF/WAVE container around raw little-endian
// sample bytes (the same shape audio_parser_test's make_wav produces;
// StreamSession decodes it via audio::wav_bytes_to_float32).
static std::string make_wav_bytes(int sample_rate, const std::string& pcm) {
    auto le32 = [](uint32_t v) {
        std::string s(4, '\0');
        s[0] = static_cast<char>(v & 0xff);
        s[1] = static_cast<char>((v >> 8) & 0xff);
        s[2] = static_cast<char>((v >> 16) & 0xff);
        s[3] = static_cast<char>((v >> 24) & 0xff);
        return s;
    };
    auto le16 = [](uint16_t v) {
        std::string s(2, '\0');
        s[0] = static_cast<char>(v & 0xff);
        s[1] = static_cast<char>((v >> 8) & 0xff);
        return s;
    };
    const uint32_t data_size = static_cast<uint32_t>(pcm.size());
    std::string h = "RIFF";
    h += le32(36 + data_size);
    h += "WAVE";
    h += "fmt ";
    h += le32(16);       // fmt chunk size
    h += le16(1);        // PCM
    h += le16(1);        // mono
    h += le32(static_cast<uint32_t>(sample_rate));
    h += le32(static_cast<uint32_t>(sample_rate) * 2);  // byte rate
    h += le16(2);        // block align
    h += le16(16);       // bits per sample
    h += "data";
    h += le32(data_size);
    h += pcm;
    return h;
}

static void test_stream_session_append_rejection() {
    // issue #145: a refused binary frame (malformed WAV, non-16 kHz WAV,
    // odd-length PCM) must be observable to the caller as a typed outcome
    // and must invalidate the take: further audio is refused (TakeInvalid)
    // until reset(), so an incomplete capture is never mistaken for a whole
    // one. This pins the session half of the policy; the WS transport half
    // (error frames + refused commit) is covered by test_native_serve.py.
    StarlingServer server(test_cfg());
    StreamSession session(&server);
    session.set_transcribe_fn([](const float*, int64_t) -> std::optional<std::string> {
        return "ok";
    });

    // (1) Malformed RIFF/WAVE frame: decoder refuses, take invalidated.
    // (Explicit length: the literal embeds NULs, so the const char*
    // constructor would truncate it to "RIFF".)
    const std::string malformed("RIFF\x00\x00\x00\x00WAVEjunk", 16);
    CHECK(session.append_wav(malformed) == AppendOutcome::MalformedWav);
    CHECK(session.take_invalid());
    CHECK(session.invalid_reason() == "malformed_wav");
    CHECK(session.buffered_seconds() == 0.0);

    // (2) Further audio is refused with TakeInvalid (same reason).
    CHECK(session.append_pcm(pcm_for_range(0, 1600)) == AppendOutcome::TakeInvalid);
    CHECK(session.buffered_seconds() == 0.0);
    CHECK(session.append_wav(malformed) == AppendOutcome::TakeInvalid);

    // (3) reset() clears the invalidation and re-enables audio.
    session.reset();
    CHECK(!session.take_invalid());
    CHECK(session.invalid_reason().empty());
    CHECK(session.append_pcm(pcm_for_range(0, 1600)) == AppendOutcome::Accepted);
    CHECK_NEAR(session.buffered_seconds(), 0.1);

    // (4) Valid -> invalid -> valid (the issue's regression sequence):
    // audio before the rejection is retained, audio after it is refused.
    session.reset();
    CHECK(session.append_pcm(pcm_for_range(0, 8000)) == AppendOutcome::Accepted);
    const std::string wav8k = make_wav_bytes(8000, std::string(16000, '\0'));
    CHECK(session.append_wav(wav8k) == AppendOutcome::RateMismatch);
    CHECK(session.invalid_reason() == "sample_rate_mismatch");
    CHECK_NEAR(session.buffered_seconds(), 0.5);  // pre-rejection audio kept
    CHECK(session.append_pcm(pcm_for_range(8000, 8000)) == AppendOutcome::TakeInvalid);
    CHECK_NEAR(session.buffered_seconds(), 0.5);  // post-rejection audio refused
    session.reset();
    CHECK(session.append_pcm(pcm_for_range(0, 8000)) == AppendOutcome::Accepted);
    CHECK_NEAR(session.buffered_seconds(), 0.5);
    CHECK(session.stream_flush() == "ok");     // clean take finalizes normally

    // (5) 48 kHz WAV is refused the same way as 8 kHz.
    session.reset();
    const std::string wav48k = make_wav_bytes(48000, std::string(96000, '\0'));
    CHECK(session.append_wav(wav48k) == AppendOutcome::RateMismatch);
    CHECK(session.invalid_reason() == "sample_rate_mismatch");
    CHECK(session.buffered_seconds() == 0.0);

    // (6) Odd-length raw PCM (a split int16 sample): the whole frame is
    // refused and the take invalidated — the dangling byte is not silently
    // dropped.
    session.reset();
    std::string odd = pcm_for_range(0, 800);
    odd.push_back('\x7f');
    CHECK(session.append_pcm(odd) == AppendOutcome::OddPcmLength);
    CHECK(session.invalid_reason() == "odd_pcm_length");
    CHECK(session.buffered_seconds() == 0.0);
    // Even-length frames stay accepted no-ops/append as before.
    session.reset();
    CHECK(session.append_pcm(std::string()) == AppendOutcome::Accepted);
    CHECK(!session.take_invalid());

    // (7) A 16 kHz WAV still appends (the rejection is about validity,
    // not the WAV container itself).
    session.reset();
    const std::string wav16k =
        make_wav_bytes(16000, pcm_for_range(0, 8000));
    CHECK(session.append_wav(wav16k) == AppendOutcome::Accepted);
    CHECK(!session.take_invalid());
    CHECK_NEAR(session.buffered_seconds(), 0.5);
}

static void test_bounded_retry_recovery() {
    for (bool flush : {false, true}) {
        ChunkStreamer cs(1, 12, 2, 5, 0);
        std::vector<float> samples(30);
        for (int i = 0; i < 30; ++i) samples[i] = static_cast<float>(i);
        std::vector<std::pair<int, int>> calls;
        TranscribeFn tx = [&](const float* data, int64_t n) -> std::optional<std::string> {
            calls.emplace_back(static_cast<int>(*data), static_cast<int>(n));
            return calls.size() == 1 ? std::nullopt
                                    : std::optional<std::string>("hello world");
        };
        if (flush) {
            CHECK(cs.flush(samples, tx) == "hello world");
            CHECK(cs.boundary() == 30);
        } else {
            CHECK(!cs.step(samples, 1, tx).has_value());
            CHECK(calls.size() == 1);
            CHECK(cs.step(samples, 2, tx) == "hello world");
        }
        const std::vector<std::pair<int, int>> expected = {{0, 12}, {0, 12}, {10, 12}, {20, 10}};
        CHECK(calls == expected);
    }

    ChunkStreamer cs(1, 12, 2, 5, 0);
    std::vector<float> samples(30);
    for (int i = 0; i < 30; ++i) samples[i] = static_cast<float>(i);
    int first_window_calls = 0;
    bool busy = true;
    TranscribeFn tx = [&](const float* data, int64_t n) -> std::optional<std::string> {
        CHECK(n <= 12);
        if (*data == 0) ++first_window_calls;
        if (busy && *data == 10) return std::nullopt;
        return "hello world";
    };
    CHECK(!cs.flush(samples, tx).has_value());
    CHECK(cs.boundary() == 10);
    busy = false;
    CHECK(cs.flush(samples, tx) == "hello world");
    CHECK(first_window_calls == 1);
    CHECK(cs.boundary() == 30);
}

// ---- exact streaming-tail reuse (S11) ---------------------------------------
// All of these drive StreamSession with an injected fake engine
// (set_transcribe_fn) and count engine invocations. Scenario geometry: with
// test_cfg() (1 s chunks, 0.25 s overlap, 0.5 s partial minimum) a 0.7 s
// (11200-sample) append is a pure tail — below one chunk window, above the
// partial minimum — so a preview transcribes exactly [0, 11200) and a commit
// with no further audio used to transcribe the identical window a second
// time. The retained exact-tail entry must turn that second call into a
// replay, and must NEVER answer anything whose identity differs in the
// slightest (one sample, one option, one model id, one callback swap, one
// invalidation).

// Shared setup for the scenarios below (PR #199 test hygiene): the repetitive
// server + session + fake-engine wiring in one place. The engine counts its
// calls and can be steered from the test body — canned `text`, `busy`, or a
// window-keyed `hook` that replaces both. prime() runs the common prologue:
// append the 0.7 s pure-tail audio and run one successful preview (exactly
// one engine call, the entry retained under it).
struct TailFixture {
    StarlingServer server;
    StreamSession session;
    int engine_calls = 0;
    bool busy = false;                // the engine reports busy (nullopt)
    std::string text = "alpha beta";  // canned success text
    // Window-keyed behavior override (receives the window length): when set
    // it replaces the canned text/busy logic entirely.
    std::function<std::optional<std::string>(int64_t)> hook;

    // The session's fake engine captures `this`: a copied fixture would
    // silently count its engine calls into the ORIGINAL (and read the
    // original's text/busy/hook), so copies are deleted rather than trusted
    // to be noticed. The moves are deleted too (R29, issue #236): deleted
    // copies already SUPPRESS implicit move generation, but stating the
    // rule-of-five in full keeps a future member change from silently
    // re-enabling moves on a this-capturing type.
    TailFixture(const TailFixture&) = delete;
    TailFixture& operator=(const TailFixture&) = delete;
    TailFixture(TailFixture&&) = delete;
    TailFixture& operator=(TailFixture&&) = delete;

    TailFixture() : server(test_cfg()), session(&server) {
        session.set_transcribe_fn([this](const float*, int64_t n)
                                      -> std::optional<std::string> {
            ++engine_calls;
            if (hook) return hook(n);
            if (busy) return std::nullopt;
            return text;
        });
    }

    void prime() {
        CHECK(session.append_pcm(pcm_for_range(0, 11200))
              == AppendOutcome::Accepted);
        auto partial = session.stream_step(1.0);
        // The preview IS the canned result (the empty string too, when
        // text == ""): has_value alone would not pin what was previewed.
        CHECK(partial == std::optional<std::string>(text));
        CHECK(engine_calls == 1);
    }
};

static void test_tail_reuse_preview_then_commit() {
    TailFixture fx;
    fx.prime();  // the preview's one engine call

    // Stop arrives with no new audio: the flush tail is byte-identical to the
    // previewed window → the retained result answers it. Two engine calls
    // become one, with identical output.
    auto final_ = fx.session.stream_flush();
    CHECK(final_.has_value());
    CHECK(*final_ == "alpha beta");
    CHECK(fx.engine_calls == 1);
    CHECK(fx.session.tail_cache_hits() == 1);

    // Duplicate commit: everything is finalized (boundary == buffer end), so
    // the second flush returns the committed text without any engine call —
    // and without touching the retained entry.
    auto again = fx.session.stream_flush();
    CHECK(again.has_value());
    CHECK(*again == "alpha beta");
    CHECK(fx.engine_calls == 1);
    CHECK(fx.session.tail_cache_hits() == 1);
}

static void test_tail_reuse_one_appended_sample_forces_recompute() {
    TailFixture fx;
    fx.prime();

    // One more sample arrives before Stop: the committed tail is longer than
    // the previewed window, so the preview's result is not an exact answer.
    // The engine must run again.
    fx.session.append_pcm(pcm_for_range(11200, 1));
    auto final_ = fx.session.stream_flush();
    CHECK(final_.has_value());
    CHECK(*final_ == "alpha beta");
    CHECK(fx.engine_calls == 2);
    CHECK(fx.session.tail_cache_hits() == 0);
}

static void test_tail_reuse_engine_identity_change_forces_recompute() {
    TailFixture fx;
    fx.prime();

    // The engine identity is part of the key: a reloaded/re-configured
    // engine (here simulated by a quant swap in the identity string) can
    // answer the same bytes differently → the retained result is dropped and
    // the very same window must be recomputed.
    fx.session.set_engine_identity(fx.session.engine_identity() + "|q8");
    CHECK(fx.session.stream_step(2.0) == std::optional<std::string>("alpha beta"));
    CHECK(fx.engine_calls == 2);
    // The recomputation is retained under the new identity going forward:
    // commit on unchanged audio reuses it.
    auto final_ = fx.session.stream_flush();
    CHECK(final_ == std::optional<std::string>("alpha beta"));
    CHECK(fx.engine_calls == 2);
    CHECK(fx.session.tail_cache_hits() == 1);
}

static void test_tail_reuse_mid_step_engine_identity_never_rekeys_in_flight_calls() {
    // PR #199 batch-4 (medium): engine_id had the same construction-vs-
    // invocation gap the generation had — a set_engine_identity() issued
    // from inside an earlier window's callback would re-key the in-flight
    // wrapper's tail retention under the NEW identity. The fix snapshots
    // engine_id with the callback and generation at construction.
    TailFixture fx;
    fx.hook = [&](int64_t n) -> std::optional<std::string> {
        if (n == 16000) {
            fx.hook = nullptr;
            // Mid-step identity change (a re-configured engine answering
            // differently): the in-flight wrapper must stay keyed under the
            // identity it was built with, and the retained entry must not
            // survive the change for the NEW wrappers either.
            fx.session.set_engine_identity(fx.session.engine_identity() + "|q8");

            return "full window old-id";
        }
        return "tail from old-id";
    };
    fx.text = "tail from old-id";
    fx.session.append_pcm(pcm_for_range(0, 24000));
    CHECK(fx.session.stream_step(1.0)
          == std::optional<std::string>("full window old-id tail from old-id"));
    CHECK(fx.engine_calls == 2);
    CHECK(fx.session.tail_cache_hits() == 0);

    // Under the new identity the same tail must recompute (no stale match),
    // and the new result is retained for the unchanged-audio commit.
    CHECK(fx.session.stream_step(2.0)
          == std::optional<std::string>("full window old-id tail from old-id"));
    CHECK(fx.engine_calls == 3);
    CHECK(fx.session.tail_cache_hits() == 0);
    CHECK(fx.session.stream_flush()
          == std::optional<std::string>("full window old-id tail from old-id"));
    CHECK(fx.engine_calls == 3);  // commit reuses the retained entry
    CHECK(fx.session.tail_cache_hits() == 1);
}

static void test_tail_reuse_callback_swap_forces_recompute() {
    // PR #199 (medium): set_transcribe_fn() used to swap custom_tx_ without
    // invalidating the retained exact-tail entry, so byte-identical audio
    // under a NEW callback replayed the OLD callback's cached text. The
    // transcribe-callback generation is part of the retention key: callback
    // B must run on the same window, and B's result must then be retained.
    TailFixture fx;
    fx.prime();  // one call under callback A ("alpha beta"), entry retained

    int b_calls = 0;
    fx.session.set_transcribe_fn([&](const float*, int64_t)
                                     -> std::optional<std::string> {
        ++b_calls;
        return "callback b text";
    });

    // Same audio window, no append: A's retained result must not answer for
    // B (no stale replay).
    auto partial = fx.session.stream_step(2.0);
    CHECK(partial == std::optional<std::string>("callback b text"));
    CHECK(b_calls == 1);
    CHECK(fx.engine_calls == 1);               // A ran once, in the prime only
    CHECK(fx.session.tail_cache_hits() == 0);  // no stale replay

    // B's result is retained: the commit on unchanged audio reuses it.
    auto final_ = fx.session.stream_flush();
    CHECK(final_ == std::optional<std::string>("callback b text"));
    CHECK(b_calls == 1);
    CHECK(fx.session.tail_cache_hits() == 1);
}

static void test_tail_reuse_mid_step_swap_never_rekeys_in_flight_calls() {
    // PR #199 batch-2 (medium): active_tx() used to capture the callback at
    // construction but read the generation inside the lambda at invocation.
    // One wrapper is invoked more than once per step (the full windows, then
    // the tail), so a set_transcribe_fn() issued from INSIDE an earlier
    // window's callback — between this wrapper's construction and its tail
    // invocation — paired the OLD callback with the NEW generation: the old
    // callback's tail result was retained under the new generation, where
    // the new callback's next identical window replayed it without running
    // (the stale match R24's keying must forbid). The fix snapshots the
    // callback and its generation together, so the in-flight call stays
    // keyed as the pair that existed at its construction.
    TailFixture fx;
    // Geometry: 24000 samples = one full window [0,16000) plus a 12000-sample
    // tail (>= the 8000 partial minimum): one stream_step invokes the wrapper
    // twice — the window first, the tail second.
    int b_calls = 0;
    fx.hook = [&](int64_t n) -> std::optional<std::string> {
        if (n == 16000) {
            // The full-window call: swap in callback B mid-step. The wrapper
            // built for THIS step must keep running callback A — under A's
            // generation — for the tail call that follows.
            fx.hook = nullptr;  // A's tail call takes the canned text below
            fx.session.set_transcribe_fn([&](const float*, int64_t)
                                             -> std::optional<std::string> {
                ++b_calls;
                return "tail from b";
            });
            return "full window a";
        }
        return "tail from a";
    };
    fx.text = "tail from a";
    fx.session.append_pcm(pcm_for_range(0, 24000));
    CHECK(fx.session.stream_step(1.0)
          == std::optional<std::string>("full window a tail from a"));
    CHECK(fx.engine_calls == 2);             // A ran both the window and the tail
    CHECK(b_calls == 0);                     // B has not run yet
    CHECK(fx.session.tail_cache_hits() == 0);

    // The next step runs under B (a fresh wrapper, the new generation): the
    // tail window is unchanged, but A's retained result must NOT answer it —
    // no stale match; B recomputes, and B's own result is what gets retained.
    CHECK(fx.session.stream_step(2.0)
          == std::optional<std::string>("full window a tail from b"));
    CHECK(b_calls == 1);
    CHECK(fx.session.tail_cache_hits() == 0);

    // Commit on unchanged audio now reuses B's own retained result.
    CHECK(fx.session.stream_flush()
          == std::optional<std::string>("full window a tail from b"));
    CHECK(b_calls == 1);
    CHECK(fx.session.tail_cache_hits() == 1);
}

static void test_tail_reuse_empty_text_is_a_valid_result() {
    TailFixture fx;
    fx.text = "";  // a successful empty transcription (silence)
    fx.prime();    // a partial WAS produced (empty string, not nullopt)

    auto final_ = fx.session.stream_flush();
    CHECK(final_.has_value());
    CHECK(final_->empty());
    CHECK(fx.engine_calls == 1);
    CHECK(fx.session.tail_cache_hits() == 1);
}

static void test_tail_reuse_busy_preview_recomputes_at_commit() {
    // A busy/cancelled preview retains nothing (nullopt is never a reusable
    // result): the commit must call the engine and produce the full text.
    TailFixture fx;
    fx.busy = true;
    fx.text = "recovered words";
    fx.session.append_pcm(pcm_for_range(0, 11200));
    auto partial = fx.session.stream_step(1.0);
    CHECK(!partial.has_value());
    CHECK(fx.engine_calls == 1);

    fx.busy = false;
    auto final_ = fx.session.stream_flush();
    CHECK(final_.has_value());
    CHECK(*final_ == "recovered words");
    CHECK(fx.engine_calls == 2);
    CHECK(fx.session.tail_cache_hits() == 0);
}

static void test_tail_reuse_cancel_after_success_not_reused() {
    // A successful preview, then new frames arrive ("during inference" in a
    // paced session) and the next preview is cancelled/busy: the stale
    // success must not answer the grown commit window.
    TailFixture fx;
    // Window-keyed fake engine: the 11200-sample window succeeds; the grown
    // 12800-sample window is busy ONCE — keyed by window length (like
    // test_tail_reuse_survives_overlap_rebasing), not by call order (R29,
    // issue #236): an overlap-policy change that shifts how many windows the
    // chunker emits must not silently move WHICH call is treated as
    // cancelled — then succeeds with overlapping text.
    bool cancelled_grown_window = false;
    fx.hook = [&](int64_t n) -> std::optional<std::string> {
        if (n == 11200) return "alpha beta";
        if (n == 12800 && !cancelled_grown_window) {
            cancelled_grown_window = true;
            return std::nullopt;  // cancelled preview of the grown tail
        }
        return "alpha beta gamma";
    };

    fx.prime();
    fx.session.append_pcm(pcm_for_range(11200, 1600));
    CHECK(!fx.session.stream_step(2.0).has_value());  // busy on the grown tail

    auto final_ = fx.session.stream_flush();
    CHECK(final_.has_value());
    // Stitching dedupes the 2-word overlap: "alpha beta" + "alpha beta gamma".
    CHECK(*final_ == "alpha beta gamma");
    CHECK(fx.engine_calls == 3);              // preview, cancelled preview, commit
    CHECK(fx.session.tail_cache_hits() == 0); // nothing was reused
}

static void test_tail_reuse_commit_completes_while_engine_busy() {
    // The commit path is never dropped or delayed by reuse/coalescing: with
    // the exact answer already retained, a commit succeeds even while the
    // engine is busy (nothing about the final audio is skipped — it was all
    // transcribed by the preview on identical bytes).
    TailFixture fx;
    fx.prime();

    fx.busy = true;  // the engine goes busy before Stop
    auto final_ = fx.session.stream_flush();
    CHECK(final_.has_value());
    CHECK(*final_ == "alpha beta");
    CHECK(fx.engine_calls == 1);             // answered by the retained result
    CHECK(fx.session.tail_cache_hits() == 1);
}

static void test_tail_reuse_invalidated_take_not_reused() {
    // An invalidated take (a refused frame, issue #145) must never be answered
    // from the retained entry. The WS layer refuses commit outright for an
    // invalidated take; this pins the session-level defense for the case
    // where stream_flush is reached anyway.
    TailFixture fx;
    fx.prime();

    std::string odd = pcm_for_range(11200, 800);
    odd.push_back('\x7f');
    CHECK(fx.session.append_pcm(odd) == AppendOutcome::OddPcmLength);
    CHECK(fx.session.take_invalid());

    CHECK(fx.session.stream_flush().has_value());
    CHECK(fx.engine_calls == 2);             // recomputed, NOT reused
    CHECK(fx.session.tail_cache_hits() == 0);
}

static void test_tail_reuse_duplicate_snapshots_coalesced() {
    // The WS layer calls stream_step after EVERY binary frame, including
    // frames that append no new audio: duplicate step snapshots used to
    // re-run the engine on the identical window. Coalescing: the stale
    // generation is answered from the retained exact result instead.
    TailFixture fx;
    fx.prime();

    for (int i = 0; i < 5; ++i) {
        CHECK(fx.session.append_pcm(std::string()) == AppendOutcome::Accepted);
        auto partial = fx.session.stream_step(2.0 + i);
        CHECK(partial == std::optional<std::string>("alpha beta"));
    }
    CHECK(fx.engine_calls == 1);             // five stale generations coalesced
    CHECK(fx.session.tail_cache_hits() == 5);

    // The final commit still produces the full text with zero engine calls.
    CHECK(fx.session.stream_flush() == std::optional<std::string>("alpha beta"));
    CHECK(fx.engine_calls == 1);
    CHECK(fx.session.tail_cache_hits() == 6);
}

static void test_tail_reuse_newer_snapshot_wins() {
    // Audio grows between previews: each preview reflects the NEWEST window
    // only and each distinct window is transcribed exactly once — the
    // superseded (stale) snapshot is never re-run, and the commit always
    // uses the newest audio.
    TailFixture fx;
    // Window-keyed fake engine: the returned text reveals which window ran.
    fx.hook = [](int64_t n) -> std::optional<std::string> {
        return std::to_string(n);
    };

    fx.session.append_pcm(pcm_for_range(0, 11200));
    CHECK(fx.session.stream_step(1.0) == std::optional<std::string>("11200"));
    CHECK(fx.engine_calls == 1);
    fx.session.append_pcm(pcm_for_range(11200, 1600));
    // Growing the tail supersedes the stale 11200-sample snapshot: the newest
    // preview reflects the 12800-sample window (preview tails are not
    // committed — each partial is the current tail stitched onto committed).
    CHECK(fx.session.stream_step(2.0) == std::optional<std::string>("12800"));

    // The commit always uses the newest audio: the flush tail IS the newest
    // 12800-sample window, answered from its own retained result.
    auto final_ = fx.session.stream_flush();
    CHECK(final_ == std::optional<std::string>("12800"));
    CHECK(fx.engine_calls == 2);             // one call per distinct window
    CHECK(fx.session.tail_cache_hits() == 1);  // the commit reused the newest
}

static void test_tail_reuse_reset_clears_entry() {
    // reset() starts a new take: a window at the same absolute indices with
    // the same length must not inherit the previous take's result. reset()
    // drops the entry AND bumps the audio revision (enforced monotonicity),
    // so the pre- and post-reset keys cannot collide even though absolute
    // indices restart at 0.
    TailFixture fx;
    fx.text = "take one";
    fx.prime();

    fx.session.reset();
    fx.text = "take two";
    fx.session.append_pcm(pcm_for_range(0, 11200));  // same length, fresh take
    CHECK(fx.session.stream_step(2.0) == std::optional<std::string>("take two"));
    CHECK(fx.engine_calls == 2);
    CHECK(fx.session.stream_flush() == std::optional<std::string>("take two"));
    CHECK(fx.engine_calls == 2);
    CHECK(fx.session.tail_cache_hits() == 1);  // only the second take's commit
}

static void test_tail_reuse_aba_across_sessions() {
    // A/B/A request pattern: each session owns exactly one retained entry,
    // so nothing crosses sessions; every session independently gets the
    // preview→commit reuse with its own engine's output.
    struct Result { int calls; int64_t hits; std::string final_text; };
    auto run = [](const std::string& slug, const std::string& text) {
        ServerConfig cfg = test_cfg();
        cfg.model_slug = slug;  // different model id → different identity
        StarlingServer server(cfg);
        StreamSession session(&server);
        int calls = 0;
        session.set_transcribe_fn([&](const float*, int64_t)
                                      -> std::optional<std::string> {
            ++calls;
            return text;
        });
        (void)session.stream_step(0.0);  // no audio yet: no call
        session.append_pcm(pcm_for_range(0, 11200));
        CHECK(session.stream_step(1.0) == std::optional<std::string>(text));
        auto final_ = session.stream_flush();
        CHECK(final_ == std::optional<std::string>(text));
        return Result{calls, session.tail_cache_hits(), *final_};
    };

    const Result a1 = run("parakeet", "alpha words");
    const Result b = run("moss", "beta words");
    const Result a2 = run("parakeet", "alpha words");
    CHECK(a1.calls == 1 && a1.hits == 1 && a1.final_text == "alpha words");
    CHECK(b.calls == 1 && b.hits == 1 && b.final_text == "beta words");
    CHECK(a2.calls == 1 && a2.hits == 1 && a2.final_text == "alpha words");

    // Different model slugs really do produce different engine identities.
    ServerConfig ca = test_cfg();
    ca.model_slug = "parakeet";
    ServerConfig cb = test_cfg();
    cb.model_slug = "moss";
    StarlingServer sa(ca), sb(cb);
    StreamSession session_a(&sa), session_b(&sb);
    CHECK(session_a.engine_identity() != session_b.engine_identity());
}

static void test_tail_reuse_survives_overlap_rebasing() {
    // Overlap rebasing/trim interaction: the key uses ABSOLUTE sample indices
    // (live offset + trimmed prefix), so a preview after a trim and the
    // commit on unchanged audio still name the same window and reuse, while
    // the window handed to the engine provably starts at the right absolute
    // sample (position-encoded audio).
    StarlingServer server(test_cfg());
    StreamSession session(&server);
    // First partial at 2.6 s of take audio: the 2.5 s step below only
    // commits windows, and the first preview follows the trim.
    session.set_preview_policy(2.6, 0.0);
    std::vector<std::pair<int16_t, int64_t>> calls;  // first sample, length
    TranscribeFn tx = [&](const float* p, int64_t n)
                          -> std::optional<std::string> {
        calls.emplace_back(static_cast<int16_t>(p[0] * 32768.0f), n);
        return calls.size() <= 3 ? "w" : "tail words";
    };
    session.set_transcribe_fn(tx);

    // 2.5 s: three full windows finalize (boundary → 36000).
    session.append_pcm(pcm_for_range(0, 40000));
    CHECK(session.stream_step(1.0) == std::optional<std::string>("w w w"));
    CHECK(calls.size() == 3);
    CHECK_NEAR(session.buffered_seconds(), 2.5);

    // The next append trims the 36000 finalized samples and rebases; then
    // top the take up past the 2.6 s first-partial minimum.
    session.append_pcm(pcm_for_range(40000, 1600));
    CHECK_NEAR(session.live_seconds(), 0.35);  // 5600 live after the trim
    session.append_pcm(pcm_for_range(41600, 2400));

    // Preview on the rebased tail: live [0, 8000) = absolute [36000, 44000).
    // The first sample encodes the absolute start: 36000 % 30000 - 15000.
    CHECK(session.stream_step(2.0)
          == std::optional<std::string>("w w w tail words"));
    CHECK(calls.size() == 4);
    if (calls.size() == 4) {
        CHECK(calls[3].second == 8000);
        CHECK(calls[3].first == 36000 % 30000 - 15000);
    }

    // Commit with no further audio: the flush names the same absolute window
    // → reuse. Audio accounting is unchanged (nothing dropped or shortened).
    auto final_ = session.stream_flush();
    CHECK(final_ == std::optional<std::string>("w w w tail words"));
    CHECK(calls.size() == 4);
    CHECK(session.tail_cache_hits() == 1);
    CHECK_NEAR(session.buffered_seconds(), 2.75);
}

// ---- stream window config validation (issue #146) --------------------------
// Negative/NaN/oversized stream window values used to reach the chunker, whose
// member-init list computed advance_ from the unclamped overlap_: a negative
// --stream-overlap-seconds made windows skip audio and let flush() call the
// transcriber with a negative sample count. Validation now rejects these at
// the CLI (before model load) and at construction, and every callback must
// stay within (0, chunk].

static void test_strict_number_parsing() {
    // Full-string, finite-only parses for CLI flags (std::stod/std::stoi
    // alone accept partial parses and non-finite tokens).
    CHECK(parse_double_strict("3.5") == 3.5);
    CHECK(parse_double_strict(" 2.5 ") == 2.5);
    CHECK(parse_double_strict("-0.25") == -0.25);
    CHECK(!parse_double_strict("3abc").has_value());   // trailing junk
    CHECK(!parse_double_strict("nan").has_value());
    CHECK(!parse_double_strict("inf").has_value());
    CHECK(!parse_double_strict("-infinity").has_value());
    CHECK(!parse_double_strict("").has_value());
    CHECK(parse_int_strict("42") == 42);
    CHECK(parse_int_strict(" -7 ") == -7);
    CHECK(!parse_int_strict("8181abc").has_value());   // trailing junk
    CHECK(!parse_int_strict("3.5").has_value());
    CHECK(!parse_int_strict("99999999999999").has_value());  // out of int range
    CHECK(!parse_int_strict("").has_value());
}

static void test_stream_window_config_error() {
    auto err = [](double chunk, double overlap, double min, double partial) {
        return stream_window_config_error(16000, chunk, overlap, min, partial);
    };
    // The shipped defaults and the legacy whole-buffer switch are valid.
    CHECK(err(12.0, 3.0, 5.0, 3.0).empty());
    CHECK(err(0.0, 3.0, 5.0, 3.0).empty());
    // Everything else must be rejected with an error message.
    CHECK(!err(-12.0, 3.0, 5.0, 3.0).empty());            // negative chunk
    CHECK(!err(12.0, -3.0, 5.0, 3.0).empty());            // negative overlap
    CHECK(!err(12.0, 12.0, 5.0, 3.0).empty());            // overlap == chunk
    CHECK(!err(12.0, 13.0, 5.0, 3.0).empty());            // overlap > chunk
    CHECK(!err(12.0, 3.0, -5.0, 3.0).empty());            // negative min
    CHECK(!err(12.0, 3.0, 5.0, -3.0).empty());            // negative partial
    CHECK(err(12.0, 3.0, 5.0, 0.0).empty());              // zero partial is fine
    CHECK(!err(1e-9, 3.0, 5.0, 3.0).empty());             // sub-sample window
    CHECK(!err(1e12, 3.0, 5.0, 3.0).empty());             // sample-count overflow
    CHECK(!err(12.0, 3.0, 1e12, 3.0).empty());            // min-count overflow
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    CHECK(!err(nan, 3.0, 5.0, 3.0).empty());
    CHECK(!err(12.0, nan, 5.0, 3.0).empty());
    CHECK(!err(12.0, 3.0, inf, 3.0).empty());
    CHECK(!err(12.0, 3.0, 5.0, nan).empty());
    CHECK(!stream_window_config_error(0, 12.0, 3.0, 5.0, 3.0).empty());  // bad rate
}

static void test_chunk_streamer_rejects_invalid_config() {
    auto ctor_throws = [](int sr, double chunk, double overlap,
                          double min, double partial) {
        try {
            ChunkStreamer(sr, chunk, overlap, min, partial);
        } catch (const std::invalid_argument&) {
            return true;
        } catch (...) {
            return true;
        }
        return false;
    };
    // The issue #146 repro (12 s window, -3 s overlap) and its neighbors.
    CHECK(ctor_throws(16000, 12.0, -3.0, 5.0, 0.0));
    CHECK(ctor_throws(16000, 12.0, 12.0, 5.0, 0.0));
    CHECK(ctor_throws(16000, 12.0, 13.0, 5.0, 0.0));
    CHECK(ctor_throws(16000, 0.0, 3.0, 5.0, 0.0));   // legacy mode is CLI-only
    CHECK(ctor_throws(16000, -12.0, 3.0, 5.0, 0.0));
    CHECK(ctor_throws(16000, 1e-9, 0.0, 5.0, 0.0));  // sub-sample window
    CHECK(ctor_throws(16000, 1e12, 3.0, 5.0, 0.0));  // sample-count overflow
    CHECK(ctor_throws(16000, 12.0, 3.0, -5.0, 0.0));
    CHECK(ctor_throws(16000, 12.0, 3.0, 5.0, -1.0));
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    CHECK(ctor_throws(16000, nan, 3.0, 5.0, 0.0));
    CHECK(ctor_throws(16000, 12.0, inf, 5.0, 0.0));
    CHECK(ctor_throws(16000, 12.0, 3.0, nan, 0.0));
    CHECK(ctor_throws(0, 12.0, 3.0, 5.0, 0.0));
    // Valid configurations still construct.
    CHECK(!ctor_throws(16000, 12.0, 3.0, 5.0, 3.0));
    CHECK(!ctor_throws(1, 12.0, 2.0, 5.0, 0.0));     // existing sr=1 tests
}

static void test_chunk_streamer_overlap_clamped_to_half_chunk() {
    // Overlap above half the chunk keeps its documented clamp (half the
    // chunk): 1 s chunks with 0.75 s requested overlap advance 0.5 s.
    ChunkStreamer cs(16000, 1.0, 0.75, 0.5, 0.0);
    TranscribeFn tx = [](const float*, int64_t) -> std::optional<std::string> {
        return "w";
    };
    std::vector<float> samples(16000, 0.0f);
    auto result = cs.step(samples, 1.0, tx);
    CHECK(result.has_value());
    CHECK(cs.boundary() == 8000);  // advance = 16000 - 8000
}

static void test_chunk_streamer_windows_stay_in_range() {
    // The issue #146 invariant: every transcriber callback receives a
    // nonempty window of at most one chunk, inside the buffered samples.
    ChunkStreamer cs(16000, 12.0, 3.0, 5.0, 0.0);
    std::vector<float> samples(16000 * 30, 0.0f);
    const float* base = samples.data();
    const float* end = base + samples.size();
    int calls = 0;
    TranscribeFn tx = [&](const float* data, int64_t n) -> std::optional<std::string> {
        ++calls;
        CHECK(n > 0);
        CHECK(n <= 16000 * 12);
        CHECK(data >= base);
        CHECK(data + n <= end);
        return "w";
    };
    auto partial = cs.step(samples, 1.0, tx);
    CHECK(partial.has_value());
    auto final_text = cs.flush(samples, tx);
    CHECK(final_text.has_value());
    // Three 12 s windows, a preview of the 3 s overlap tail (the take is past
    // the 5 s first-partial minimum), and the 3 s flush tail.
    CHECK(calls == 5);
}

static void test_chunk_streamer_never_transcribes_empty_window() {
    // min=0 with an exactly-consumed buffer: the partial-tail branch must
    // not call the transcriber with a zero-length window.
    ChunkStreamer cs(16000, 1.0, 0.0, 0.0, 0.0);
    int calls = 0;
    TranscribeFn tx = [&](const float*, int64_t n) -> std::optional<std::string> {
        ++calls;
        CHECK(n > 0);
        return "w";
    };
    std::vector<float> samples(16000, 0.0f);  // exactly one window
    auto result = cs.step(samples, 1.0, tx);
    CHECK(result.has_value());
    CHECK(calls == 1);  // the full window only; no empty partial
}

static void test_stream_session_rejects_invalid_window_config() {
    // StreamSession builds its ChunkStreamer from the server config: an
    // invalid stream window config must fail at construction, never reach a
    // live transcription session (the CLI rejects it even earlier).
    ServerConfig cfg = test_cfg();
    cfg.stream_overlap_seconds = -3.0;
    StarlingServer server(cfg);
    bool threw = false;
    try {
        StreamSession session(&server);
        (void)session;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

// ---- stream call ledger (issue #226) ----------------------------------------
// The ledger records every transcribe call with its kind and the absolute
// audio span it covered, so overlap and repeated previews are visible and the
// stop-time work can be checked against the unfinalized tail.

static void test_trace_ledger_kinds_and_spans() {
    StarlingServer server(test_cfg());
    StreamSession session(&server);
    session.set_transcribe_fn([](const float*, int64_t) -> std::optional<std::string> {
        return std::string("w");
    });

    // 2.5 s: three full windows (advance 12000) and a preview of the
    // 4000-sample tail.
    session.append_pcm(pcm_for_range(0, 40000));
    CHECK(session.stream_step(1.0).has_value());
    // 6000 more samples: a 10000-sample tail preview.
    session.append_pcm(pcm_for_range(40000, 6000));
    CHECK(session.stream_step(2.0).has_value());

    const auto& calls = session.calls();
    CHECK(calls.size() == 5);
    if (calls.size() == 5) {
        const int64_t starts[] = {0, 12000, 24000, 36000, 36000};
        const int64_t lens[] = {16000, 16000, 16000, 4000, 10000};
        for (size_t i = 0; i < 5; ++i) {
            CHECK(std::string(calls[i].kind) == (i < 3 ? "window" : "preview"));
            CHECK(calls[i].abs_start == starts[i]);
            CHECK(calls[i].length == lens[i]);
            CHECK(std::string(calls[i].result) == "ok");
            CHECK(calls[i].t1_ms >= calls[i].t0_ms);
        }
    }
    // Overlap and repeated previews count as work: 62000 engine samples for
    // 46000 recorded.
    CHECK(session.totals().engine_calls == 5);
    CHECK(session.totals().engine_samples == 62000);

    // Stop with no new audio: the preview's exact result answers the tail.
    auto final_ = session.stream_flush();
    CHECK(final_.has_value());
    CHECK(std::string(session.final_path()) == "reused");
    CHECK(session.flush_totals().engine_calls == 0);
    CHECK(session.flush_totals().reused == 1);
    const std::string tj = session.trace_final_json();
    CHECK(tj.find("\"path\":\"reused\"") != std::string::npos);
    CHECK(tj.find("\"unfinalized_s\":0.625") != std::string::npos);
    CHECK(tj.find("\"kind\":\"flush_tail\",\"start_s\":2.250,\"end_s\":2.875")
          != std::string::npos);
    CHECK(session.trace_partial_json().find("\"covered_s\":2.875")
          != std::string::npos);

    session.reset();
    CHECK(session.calls().empty());
    CHECK(session.totals().calls == 0);
    CHECK(std::string(session.final_path()).empty());
}

static void test_trace_new_final_samples_run_the_tail() {
    StarlingServer server(test_cfg());
    StreamSession session(&server);
    session.set_transcribe_fn([](const float*, int64_t) -> std::optional<std::string> {
        return std::string("w");
    });
    session.append_pcm(pcm_for_range(0, 11200));
    CHECK(session.stream_step(1.0).has_value());  // preview [0, 11200)
    // Samples arriving after the latest preview must be finalized: the stale
    // preview is never the final.
    session.append_pcm(pcm_for_range(11200, 800));
    CHECK(session.stream_flush().has_value());
    CHECK(std::string(session.final_path()) == "tail");
    CHECK(session.flush_totals().engine_calls == 1);
    CHECK(session.flush_totals().engine_samples == 12000);
    const auto& last = session.calls().back();
    CHECK(std::string(last.kind) == "flush_tail");
    CHECK(last.abs_start == 0 && last.length == 12000);
}

static void test_trace_flush_bounded_by_tail_not_take() {
    // A 60 s take streamed in 0.1 s frames: the work after commit covers only
    // the unfinalized remainder (less than one window), never the take.
    StarlingServer server(test_cfg());
    StreamSession session(&server);
    session.set_transcribe_fn([](const float*, int64_t) -> std::optional<std::string> {
        return std::string("w");
    });
    const int64_t frame = 1600, total = 60 * 16000;
    double now = 0.0;
    for (int64_t at = 0; at < total; at += frame) {
        session.append_pcm(pcm_for_range(at, frame));
        session.stream_step(now += 0.1);
    }
    session.append_pcm(pcm_for_range(total, 500));  // new final samples
    CHECK(session.stream_flush().has_value());
    CHECK(std::string(session.final_path()) == "tail");
    CHECK(session.flush_totals().engine_samples < 16000);
    CHECK(session.flush_totals().engine_calls == 1);
    CHECK(session.totals().engine_samples > total);  // overlap + previews
    for (const auto& c : session.calls()) {
        CHECK(c.length <= 16000);
        CHECK(c.abs_start + c.length <= total + 500);
    }
}

// ---- preview cadence and coalescing (issue #357) -----------------------------

static void test_preview_minimum_gates_the_take_not_the_tail() {
    ChunkStreamer cs(16000, 1.0, 0.25, 0.5, 0.0);
    std::vector<int64_t> lens;
    TranscribeFn tx = [&](const float*, int64_t n) -> std::optional<std::string> {
        lens.push_back(n);
        return std::string("w");
    };
    std::vector<float> samples(7999, 0.0f);
    CHECK(!cs.step(samples, 1.0, tx).has_value());  // take below 0.5 s
    CHECK(lens.empty());
    samples.resize(8000);
    CHECK(cs.step(samples, 2.0, tx).has_value());   // first partial
    CHECK(lens.size() == 1 && lens.back() == 8000);
    // A window commit emits its committed text and restarts the interval
    // (it covers the whole buffer), so no preview runs right behind it.
    samples.resize(16000 + 1600);
    lens.clear();
    CHECK(cs.step(samples, 3.0, tx).has_value());
    CHECK(lens.size() == 1 && lens[0] == 16000);
    // The tail is the 0.25 s overlap plus 0.1 s, below the minimum, yet
    // still previewed once the interval passes: commits never stall previews.
    CHECK(cs.step(samples, 4.0, tx).has_value());
    CHECK(lens.size() == 2);
    if (lens.size() == 2) CHECK(lens[1] == 16000 + 1600 - 12000);
}

static void test_preview_coalesced_when_newer_audio_is_queued() {
    ChunkStreamer cs(16000, 1.0, 0.25, 0.5, 0.0);
    std::vector<int64_t> lens;
    TranscribeFn tx = [&](const float*, int64_t n) -> std::optional<std::string> {
        lens.push_back(n);
        return std::string("w x");
    };
    bool pending = true;
    PendingFn newer = [&] { return pending; };
    // A full window plus tail with newer audio queued: the window commits
    // (required work), the preview is skipped.
    std::vector<float> samples(20000, 0.0f);
    CHECK(!cs.step(samples, 1.0, tx, newer).has_value());
    CHECK(lens.size() == 1 && lens[0] == 16000);
    CHECK(cs.boundary() == 12000);
    CHECK(cs.coalesced_previews() == 1);
    // The newer audio arrives; nothing is queued now. This step previews
    // the newest tail and carries the owed committed-text update along.
    samples.resize(22000);
    pending = false;
    auto out = cs.step(samples, 1.1, tx, newer);
    CHECK(out.has_value());
    CHECK(lens.size() == 2 && lens[1] == 10000);
    // The flush still covers every sample.
    CHECK(cs.flush(samples, tx).has_value());
    CHECK(lens.back() == 22000 - 12000);
}

static void test_window_commit_does_not_force_a_throttled_preview() {
    // A 10 s interval: a window commit inside it emits the committed text
    // at once but does not decode the tail (the preview duty bound holds).
    ChunkStreamer cs(16000, 1.0, 0.25, 0.5, 10.0);
    std::vector<int64_t> lens;
    TranscribeFn tx = [&](const float*, int64_t n) -> std::optional<std::string> {
        lens.push_back(n);
        return std::string("w x");
    };
    std::vector<float> samples(8000, 0.0f);
    CHECK(cs.step(samples, 20.0, tx).has_value());  // first preview
    CHECK(lens.size() == 1 && lens[0] == 8000);
    samples.resize(20000);
    auto out = cs.step(samples, 21.0, tx);
    CHECK(out == std::optional<std::string>("w x"));  // committed text only
    CHECK(lens.size() == 2 && lens[1] == 16000);
    samples.resize(21000);
    CHECK(!cs.step(samples, 21.5, tx).has_value());  // still throttled
    CHECK(lens.size() == 2);
    // The commit restarted the interval: 10 s after the commit, not after
    // the last preview (20.0).
    CHECK(!cs.step(samples, 30.5, tx).has_value());
    CHECK(lens.size() == 2);
    CHECK(cs.step(samples, 31.5, tx).has_value());   // interval over
    CHECK(lens.size() == 3 && lens[2] == 21000 - 12000);
}

static void test_preview_interval_adapts_to_preview_cost() {
    // A 30 ms preview with a zero interval: previews may take at most half
    // of wall time, so the next one waits >= 60 ms after the last started.
    ChunkStreamer cs(16000, 1.0, 0.25, 0.1, 0.0);
    int calls = 0;
    TranscribeFn tx = [&](const float*, int64_t) -> std::optional<std::string> {
        ++calls;
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        return std::string("w");
    };
    std::vector<float> samples(4000, 0.0f);
    CHECK(cs.step(samples, 10.0, tx).has_value());
    CHECK(calls == 1);
    CHECK(cs.effective_interval() >= 0.06);
    CHECK(cs.effective_interval() < 1.0);
    samples.resize(5000);
    CHECK(!cs.step(samples, 10.01, tx).has_value());  // throttled
    CHECK(calls == 1);
    CHECK(cs.step(samples, 10.001 + cs.effective_interval(), tx).has_value());
    CHECK(calls == 2);
    cs.reset();
    CHECK(cs.effective_interval() == 0.0);
}

static void test_preview_policy_validation() {
    CHECK(preview_policy_error(16000, 1.0, 0.5).empty());
    CHECK(preview_policy_error(16000, 0.0, 0.0).empty());
    CHECK(!preview_policy_error(16000, -1.0, 0.5).empty());
    CHECK(!preview_policy_error(16000, 1.0, -0.5).empty());
    CHECK(!preview_policy_error(16000, std::nan(""), 0.5).empty());
    CHECK(!preview_policy_error(16000, 1.0, INFINITY).empty());
    CHECK(!preview_policy_error(16000, 1e9, 0.5).empty());
    ChunkStreamer cs(16000, 1.0, 0.25, 0.5, 0.0);
    bool threw = false;
    try { cs.set_preview_policy(-1.0, 0.0); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    cs.set_preview_policy(2.0, 0.25);
    CHECK_NEAR(cs.min_preview_seconds(), 2.0);
    CHECK_NEAR(cs.preview_interval(), 0.25);
}

static void test_preview_policy_is_not_result_provenance() {
    // The cadence decides when a window is transcribed, not what the engine
    // returns: changing it keeps the exact-tail entry.
    TailFixture fx;
    const std::string id = fx.session.engine_identity();
    fx.prime();
    fx.session.set_preview_policy(3.0, 1.0);
    CHECK(fx.session.engine_identity() == id);
    CHECK(fx.session.stream_flush() == std::optional<std::string>("alpha beta"));
    CHECK(fx.engine_calls == 1);
}

// ---- StreamPump (issue #357) ---------------------------------------------------
// The pump runs the session on its own worker; these tests drive it like the
// WS transport does and read back the frames it sends.

struct PumpFixture {
    StarlingServer server;
    StreamSession session;
    std::mutex mu;
    std::vector<std::string> sent;
    std::condition_variable sent_cv;
    std::vector<std::pair<int64_t, int64_t>> windows;  // (abs start, length)

    explicit PumpFixture(ServerConfig cfg = test_cfg())
        : server(cfg), session(&server) {}

    StreamPump::SendFn sender() {
        return [this](const std::string& m) {
            std::lock_guard<std::mutex> lk(mu);
            sent.push_back(m);
            sent_cv.notify_all();
            return true;
        };
    }
    // Wait until a frame containing `needle` arrives; returns its index.
    size_t wait_for(const std::string& needle) {
        std::unique_lock<std::mutex> lk(mu);
        size_t found = SIZE_MAX;
        sent_cv.wait_for(lk, std::chrono::seconds(10), [&] {
            for (size_t i = 0; i < sent.size(); ++i)
                if (sent[i].find(needle) != std::string::npos) { found = i; return true; }
            return false;
        });
        return found;
    }
    size_t count(const std::string& needle) {
        std::lock_guard<std::mutex> lk(mu);
        size_t n = 0;
        for (const auto& m : sent) n += m.find(needle) != std::string::npos;
        return n;
    }
};

// Decode the absolute start of a position-encoded window (pcm_for_range).
static int64_t window_start(const float* p, int64_t hint) {
    int64_t v = static_cast<int64_t>(std::lround(p[0] * 32768.0f)) + 15000;
    return hint - (hint % 30000) + v;  // exact while windows start < 30000
}

static void test_pump_stop_during_inference_finalizes_all_audio() {
    PumpFixture fx;
    std::mutex gate_mu;
    std::condition_variable gate_cv;
    bool release = false;
    std::atomic<int> in_preview{0};
    fx.session.set_transcribe_fn([&](const float* p, int64_t n)
                                     -> std::optional<std::string> {
        {
            std::lock_guard<std::mutex> lk(fx.mu);
            fx.windows.emplace_back(window_start(p, 0), n);
        }
        if (in_preview.fetch_add(1) == 0) {  // block the first preview
            std::unique_lock<std::mutex> lk(gate_mu);
            gate_cv.wait(lk, [&] { return release; });
        }
        return std::string("w") + std::to_string(n);
    });
    StreamPump::Options opt;
    opt.trace = true;
    StreamPump pump(fx.session, opt, fx.sender());
    pump.push_audio(pcm_for_range(0, 9600));      // 0.6 s: preview starts
    while (in_preview.load() == 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    // Stop arrives while that preview is running, after more audio.
    pump.push_audio(pcm_for_range(9600, 3200));
    pump.push_commit();
    {
        std::lock_guard<std::mutex> lk(gate_mu);
        release = true;
    }
    gate_cv.notify_all();
    const size_t fin = fx.wait_for("\"type\":\"final\"");
    CHECK(fin != SIZE_MAX);
    pump.drain();
    std::lock_guard<std::mutex> lk(fx.mu);
    // The stale preview never becomes the final: the flush transcribed the
    // whole 12800-sample tail, including the audio sent during inference.
    CHECK(fx.windows.size() == 2);
    if (fx.windows.size() == 2) {
        CHECK(fx.windows[0].second == 9600);
        CHECK(fx.windows[1].first == 0 && fx.windows[1].second == 12800);
    }
    if (fin != SIZE_MAX) {
        CHECK(fx.sent[fin].find("\"text\":\"w12800\"") != std::string::npos);
        CHECK(fx.sent[fin].find("\"duration_s\":0.8") != std::string::npos);
        CHECK(fx.sent[fin].find("\"path\":\"tail\"") != std::string::npos);
        CHECK(fin == fx.sent.size() - 1);
    }
}

static void test_pump_coalesces_previews_and_keeps_every_sample() {
    // A slow engine (20 ms per call) fed a 3 s burst of 0.05 s frames: the
    // worker drains whatever queued during each call, so it previews far
    // fewer times than frames arrive, and the final covers every sample.
    PumpFixture fx;
    std::atomic<int> previews{0};
    fx.session.set_transcribe_fn([&](const float*, int64_t n)
                                     -> std::optional<std::string> {
        if (n < 16000) previews.fetch_add(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        return std::string("w");
    });
    StreamPump::Options opt;
    opt.trace = true;
    StreamPump pump(fx.session, opt, fx.sender());
    const int frames = 60, frame = 800;
    for (int i = 0; i < frames; ++i)
        pump.push_audio(pcm_for_range(static_cast<int64_t>(i) * frame, frame));
    pump.push_commit();
    const size_t fin = fx.wait_for("\"type\":\"final\"");
    CHECK(fin != SIZE_MAX);
    CHECK(previews.load() < frames / 4);
    std::lock_guard<std::mutex> lk(fx.mu);
    if (fin != SIZE_MAX) {
        CHECK(fx.sent[fin].find("\"duration_s\":3") != std::string::npos);
        CHECK(fx.sent[fin].find("\"audio_s\":3.000") != std::string::npos);
        CHECK(fx.sent[fin].find("\"covered_s\":3.000") != std::string::npos);
    }
}

static void test_window_decode_time_counts_toward_the_preview_gap() {
    // A 150 ms window decode at a 0.1 s interval: the step clock moves past
    // the window, so the next step 0.2 s (fake) after the step began is
    // still inside the gap the commit restarted.
    ChunkStreamer cs(16000, 1.0, 0.25, 0.1, 0.1);
    std::vector<int64_t> lens;
    TranscribeFn tx = [&](const float*, int64_t n) -> std::optional<std::string> {
        lens.push_back(n);
        if (n == 16000) std::this_thread::sleep_for(std::chrono::milliseconds(150));
        return std::string("w");
    };
    std::vector<float> samples(16000 + 800, 0.0f);
    CHECK(cs.step(samples, 10.0, tx).has_value());  // window commit only
    CHECK(lens.size() == 1);
    CHECK(!cs.step(samples, 10.2, tx).has_value()); // 0.05 s after the window
    CHECK(lens.size() == 1);
    CHECK(cs.step(samples, 10.3, tx).has_value());  // gap over: preview
    CHECK(lens.size() == 2 && lens[1] == 16800 - 12000);
}

// A fake preview that runs until the engine's abort hook fires, polling it
// the way the Parakeet engines do between decoder frames; 10 s otherwise.
static std::optional<std::string> abortable_preview(std::atomic<int>& started) {
    started.fetch_add(1);
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - t0 < std::chrono::seconds(10)) {
        if (starling::ggml::call_abort_requested()) return std::nullopt;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return std::string("stale preview");
}

static void wait_until(const std::atomic<int>& v, int at_least) {
    const auto t0 = std::chrono::steady_clock::now();
    while (v.load() < at_least
           && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(10))
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

static void test_pump_stop_preempts_running_preview() {
    // Stop after newer audio while a preview runs: the preview is cancelled
    // at its next checkpoint and the flush starts at once, instead of the
    // final waiting for the preview to finish.
    PumpFixture fx;
    std::atomic<int> previews{0};
    fx.session.set_transcribe_fn([&](const float* p, int64_t n)
                                     -> std::optional<std::string> {
        {
            std::lock_guard<std::mutex> lk(fx.mu);
            fx.windows.emplace_back(window_start(p, 0), n);
        }
        if (std::strcmp(fx.session.chunker()->call_kind(), "preview") == 0)
            return abortable_preview(previews);
        return std::string("w") + std::to_string(n);
    });
    StreamPump::Options opt;
    opt.trace = true;
    StreamPump pump(fx.session, opt, fx.sender());
    pump.push_audio(pcm_for_range(0, 9600));
    wait_until(previews, 1);
    const auto t_stop = std::chrono::steady_clock::now();
    pump.push_audio(pcm_for_range(9600, 3200));
    pump.push_commit();
    const size_t fin = fx.wait_for("\"type\":\"final\"");
    const double stop_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_stop).count();
    CHECK(fin != SIZE_MAX);
    CHECK(stop_s < 2.0);  // not the 10 s the preview would take
    pump.drain();
    std::lock_guard<std::mutex> lk(fx.mu);
    CHECK(fx.windows.size() == 2);
    if (fx.windows.size() == 2) {
        CHECK(fx.windows[0].second == 9600);
        CHECK(fx.windows[1].first == 0 && fx.windows[1].second == 12800);
    }
    if (fin != SIZE_MAX) {
        const std::string& f = fx.sent[fin];
        CHECK(f.find("\"text\":\"w12800\"") != std::string::npos);
        CHECK(f.find("stale") == std::string::npos);
        CHECK(f.find("\"result\":\"preempted\"") != std::string::npos);
        CHECK(f.find("\"preempted\":1") != std::string::npos);
        CHECK(f.find("\"path\":\"tail\"") != std::string::npos);
    }
    // No partial from the cancelled preview.
    for (const auto& m : fx.sent) CHECK(m.find("stale") == std::string::npos);
}

static void test_pump_commit_without_new_audio_reuses_running_preview() {
    // Stop with no audio after the preview started: the preview is exactly
    // the flush tail, so it is not cancelled and the final reuses it.
    PumpFixture fx;
    std::atomic<int> previews{0};
    std::atomic<int> engine{0};
    std::mutex gate_mu;
    std::condition_variable gate_cv;
    bool release = false;
    fx.session.set_transcribe_fn([&](const float*, int64_t n)
                                     -> std::optional<std::string> {
        engine.fetch_add(1);
        if (std::strcmp(fx.session.chunker()->call_kind(), "preview") == 0) {
            previews.fetch_add(1);
            std::unique_lock<std::mutex> lk(gate_mu);
            while (!release) {
                if (starling::ggml::call_abort_requested()) return std::nullopt;
                gate_cv.wait_for(lk, std::chrono::milliseconds(1));
            }
        }
        return std::string("w") + std::to_string(n);
    });
    StreamPump::Options opt;
    opt.trace = true;
    StreamPump pump(fx.session, opt, fx.sender());
    pump.push_audio(pcm_for_range(0, 9600));
    wait_until(previews, 1);
    pump.push_commit();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    {
        std::lock_guard<std::mutex> lk(gate_mu);
        release = true;
    }
    gate_cv.notify_all();
    const size_t fin = fx.wait_for("\"type\":\"final\"");
    CHECK(fin != SIZE_MAX);
    pump.drain();
    CHECK(engine.load() == 1);  // the preview; the flush reused it
    std::lock_guard<std::mutex> lk(fx.mu);
    if (fin != SIZE_MAX) {
        CHECK(fx.sent[fin].find("\"text\":\"w9600\"") != std::string::npos);
        CHECK(fx.sent[fin].find("\"path\":\"reused\"") != std::string::npos);
        CHECK(fx.sent[fin].find("\"preempted\":0") != std::string::npos);
    }
}

static void test_pump_window_audio_preempts_running_preview() {
    // Audio that completes a window queues behind a running preview: the
    // preview is cancelled and the window commits, with no audio lost.
    PumpFixture fx;
    std::atomic<int> previews{0};
    std::atomic<int> windows{0};
    fx.session.set_transcribe_fn([&](const float*, int64_t n)
                                     -> std::optional<std::string> {
        const std::string kind = fx.session.chunker()->call_kind();
        if (kind == "preview" && previews.load() == 0)
            return abortable_preview(previews);
        if (kind == "window") windows.fetch_add(1);
        return std::string("w") + std::to_string(n);
    });
    StreamPump::Options opt;
    opt.trace = true;
    StreamPump pump(fx.session, opt, fx.sender());
    pump.push_audio(pcm_for_range(0, 9600));
    wait_until(previews, 1);
    // Short of the window: the preview keeps running.
    pump.push_audio(pcm_for_range(9600, 3200));
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    CHECK(windows.load() == 0);
    // Completes the 16000-sample window: preempts.
    const auto t0 = std::chrono::steady_clock::now();
    pump.push_audio(pcm_for_range(12800, 4000));
    wait_until(windows, 1);
    CHECK(windows.load() == 1);
    CHECK(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(2));
    pump.push_commit();
    const size_t fin = fx.wait_for("\"type\":\"final\"");
    CHECK(fin != SIZE_MAX);
    pump.drain();
    std::lock_guard<std::mutex> lk(fx.mu);
    if (fin != SIZE_MAX) {
        CHECK(fx.sent[fin].find("\"duration_s\":1.05") != std::string::npos);
        CHECK(fx.sent[fin].find("\"result\":\"preempted\"") != std::string::npos);
    }
}

static void test_pump_reset_preempts_running_preview() {
    PumpFixture fx;
    std::atomic<int> previews{0};
    fx.session.set_transcribe_fn([&](const float*, int64_t) -> std::optional<std::string> {
        return abortable_preview(previews);
    });
    StreamPump pump(fx.session, {}, fx.sender());
    pump.push_audio(pcm_for_range(0, 9600));
    wait_until(previews, 1);
    const auto t0 = std::chrono::steady_clock::now();
    pump.push_reset();
    CHECK(fx.wait_for("reset_ack") != SIZE_MAX);
    CHECK(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(2));
    CHECK(fx.count("stale") == 0);
}

static void test_pump_empty_commit_reports_committed_stop() {
    PumpFixture fx;
    StreamPump::Options opt;
    opt.trace = true;
    StreamPump pump(fx.session, opt, fx.sender());
    pump.push_commit();
    const size_t fin = fx.wait_for("\"type\":\"final\"");
    CHECK(fin != SIZE_MAX);
    std::lock_guard<std::mutex> lk(fx.mu);
    if (fin != SIZE_MAX) {
        const std::string& f = fx.sent[fin];
        CHECK(f.find("\"stop\":{\"path\":\"committed\",\"t0_ms\":0.000,\"t1_ms\":0.000")
              != std::string::npos);
    }
}

static void test_pump_stalled_peer_releases_the_transport() {
    // A peer that stops reading: the first partial's write stalls until the
    // transport's write timeout (here 100 ms) and fails, while the transport
    // thread keeps pushing audio into a 3200-byte queue. The failed send
    // closes the pump: the worker stops engine work and every later push
    // returns at once instead of waiting for queue space.
    PumpFixture fx;
    std::atomic<int> engine{0};
    std::atomic<int> sends{0};
    fx.session.set_transcribe_fn([&](const float*, int64_t) -> std::optional<std::string> {
        engine.fetch_add(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        return std::string("w");
    });
    StreamPump::Options opt;
    opt.max_pending_bytes = 3200;
    StreamPump pump(fx.session, opt, [&](const std::string&) {
        sends.fetch_add(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        return false;  // write timed out
    });
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 200; ++i) pump.push_audio(pcm_for_range(i * 1600, 1600));
    pump.push_commit();
    const double pushed_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    CHECK(pump.is_closed());
    CHECK(pushed_s < 2.0);  // not one 100 ms stall per frame
    const int engine_at_close = engine.load();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    CHECK(engine.load() == engine_at_close);  // no engine work after the close
    CHECK(sends.load() == 1);                 // nothing written after the failure
}

static void test_pump_ping_follows_earlier_audio() {
    PumpFixture fx;
    fx.session.set_transcribe_fn([](const float*, int64_t) -> std::optional<std::string> {
        return std::string("hello");
    });
    StreamPump pump(fx.session, {}, fx.sender());
    pump.push_audio(pcm_for_range(0, 11200));
    pump.push_ping();
    const size_t pong = fx.wait_for("\"type\":\"pong\"");
    CHECK(pong != SIZE_MAX);
    std::lock_guard<std::mutex> lk(fx.mu);
    CHECK(fx.sent.size() == 2);
    if (fx.sent.size() == 2) {
        CHECK(fx.sent[0].find("\"type\":\"partial\"") != std::string::npos);
        CHECK(fx.sent[0].find("trace") == std::string::npos);  // opt-in only
        CHECK(pong == 1);
    }
}

static void test_pump_backpressure_never_drops_audio() {
    // A 3200-byte queue bound and a slow engine: the transport blocks
    // instead of dropping, and the final still covers all audio.
    PumpFixture fx;
    fx.session.set_transcribe_fn([](const float*, int64_t) -> std::optional<std::string> {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        return std::string("w");
    });
    StreamPump::Options opt;
    opt.max_pending_bytes = 3200;
    StreamPump pump(fx.session, opt, fx.sender());
    for (int i = 0; i < 40; ++i) pump.push_audio(pcm_for_range(i * 1600, 1600));
    pump.push_commit();
    const size_t fin = fx.wait_for("\"type\":\"final\"");
    CHECK(fin != SIZE_MAX);
    std::lock_guard<std::mutex> lk(fx.mu);
    if (fin != SIZE_MAX)
        CHECK(fx.sent[fin].find("\"duration_s\":4") != std::string::npos);
}

static void test_pump_reset_discards_queued_take() {
    PumpFixture fx;
    std::atomic<int> calls{0};
    fx.session.set_transcribe_fn([&](const float*, int64_t) -> std::optional<std::string> {
        calls.fetch_add(1);
        return std::string("w");
    });
    StreamPump pump(fx.session, {}, fx.sender());
    pump.push_audio(pcm_for_range(0, 11200));
    pump.push_reset();
    pump.push_commit();
    CHECK(fx.wait_for("\"type\":\"final\"") != SIZE_MAX);
    pump.drain();
    CHECK(fx.count("reset_ack") == 1);
    CHECK(fx.count("\"duration_s\":0}") == 1);  // nothing left after the reset
    // If the audio and the reset landed in one batch, no preview ran at all.
    CHECK(calls.load() <= 1);
}

static void test_pump_ping_does_not_coalesce_the_preview() {
    // A ping queued while a window commits carries no audio: the preview of
    // the newest tail still runs (it would otherwise stay owed until more
    // audio arrived), and the pong follows it.
    PumpFixture fx;
    StreamPump* pump_ptr = nullptr;
    std::atomic<int> previews{0};
    fx.session.set_transcribe_fn([&](const float*, int64_t n)
                                     -> std::optional<std::string> {
        if (n == 16000) pump_ptr->push_ping();  // arrives during the commit
        else previews.fetch_add(1);
        return std::string("w");
    });
    StreamPump pump(fx.session, {}, fx.sender());
    pump_ptr = &pump;
    pump.push_audio(pcm_for_range(0, 17600));
    const size_t pong = fx.wait_for("\"type\":\"pong\"");
    CHECK(pong != SIZE_MAX);
    pump.drain();
    CHECK(previews.load() == 1);
    CHECK(fx.session.chunker()->coalesced_previews() == 0);
    std::lock_guard<std::mutex> lk(fx.mu);
    CHECK(pong == 1 && fx.sent[0].find("\"type\":\"partial\"") != std::string::npos);
}

static void test_pump_backlog_commits_windows_before_the_cap() {
    // A 3 s live-buffer cap and 6 s of audio queued behind a blocked
    // preview: the worker commits (and trims) each full window as the
    // backlog is appended, so the backlog never trips the cap.
    ServerConfig cfg = test_cfg();
    cfg.max_stream_seconds = 3.0;
    PumpFixture fx(cfg);
    std::mutex gate_mu;
    std::condition_variable gate_cv;
    bool release = false;
    std::atomic<int> calls{0};
    fx.session.set_transcribe_fn([&](const float*, int64_t)
                                     -> std::optional<std::string> {
        if (calls.fetch_add(1) == 0) {
            std::unique_lock<std::mutex> lk(gate_mu);
            gate_cv.wait(lk, [&] { return release; });
        }
        return std::string("w");
    });
    StreamPump pump(fx.session, {}, fx.sender());
    pump.push_audio(pcm_for_range(0, 8000));  // 0.5 s: the preview blocks
    while (calls.load() == 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    for (int i = 0; i < 22; ++i)
        pump.push_audio(pcm_for_range(8000 + static_cast<int64_t>(i) * 4000, 4000));
    pump.push_commit();
    {
        std::lock_guard<std::mutex> lk(gate_mu);
        release = true;
    }
    gate_cv.notify_all();
    const size_t fin = fx.wait_for("\"type\":\"final\"");
    CHECK(fin != SIZE_MAX);
    pump.drain();
    CHECK(fx.count("\"type\":\"error\"") == 0);
    CHECK(!fx.session.overflowed());
    std::lock_guard<std::mutex> lk(fx.mu);
    if (fin != SIZE_MAX)
        CHECK(fx.sent[fin].find("\"duration_s\":6") != std::string::npos);
}

static void test_pump_event_bound_blocks_control_frames() {
    // The event bound covers control frames too: with room for one queued
    // event and the worker busy, a second ping waits instead of growing
    // the queue, and every ping is still answered.
    PumpFixture fx;
    std::mutex gate_mu;
    std::condition_variable gate_cv;
    bool release = false;
    std::atomic<int> calls{0};
    fx.session.set_transcribe_fn([&](const float*, int64_t)
                                     -> std::optional<std::string> {
        if (calls.fetch_add(1) == 0) {
            std::unique_lock<std::mutex> lk(gate_mu);
            gate_cv.wait(lk, [&] { return release; });
        }
        return std::string("w");
    });
    StreamPump::Options opt;
    opt.max_pending_events = 1;
    StreamPump pump(fx.session, opt, fx.sender());
    pump.push_audio(pcm_for_range(0, 8000));
    while (calls.load() == 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    pump.push_ping();  // fills the queue
    std::atomic<bool> second_queued{false};
    std::thread t([&] { pump.push_ping(); second_queued = true; });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    CHECK(!second_queued.load());
    {
        std::lock_guard<std::mutex> lk(gate_mu);
        release = true;
    }
    gate_cv.notify_all();
    t.join();
    CHECK(second_queued.load());
    pump.drain();
    CHECK(fx.count("\"type\":\"pong\"") == 2);
}

// A pump fixture whose first engine call blocks until release(), so a test
// can queue a whole batch behind it.
struct GatedEngine {
    std::mutex mu;
    std::condition_variable cv;
    bool released = false;
    std::atomic<int> calls{0};
    std::atomic<int> previews{0};  // calls shorter than a window
    TranscribeFn fn() {
        return [this](const float*, int64_t n) -> std::optional<std::string> {
            if (n < 16000) previews.fetch_add(1);
            if (calls.fetch_add(1) == 0) {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return released; });
            }
            return std::string("w");
        };
    }
    void wait_first() {
        while (calls.load() == 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    void release() {
        { std::lock_guard<std::mutex> lk(mu); released = true; }
        cv.notify_all();
    }
};

static void test_pump_ping_preview_coalesced_by_later_batch_audio() {
    // Batch [audio, ping, audio, ping]: the first ping's preview would miss
    // the audio queued behind it, so only the last ping previews. The worker
    // is held in its first send (not in the engine, whose cost would stretch
    // the preview interval) while the batch queues.
    PumpFixture fx;
    std::atomic<int> previews{0};
    fx.session.set_transcribe_fn([&](const float*, int64_t n) -> std::optional<std::string> {
        if (n < 16000) previews.fetch_add(1);
        return std::string("w");
    });
    std::mutex gate_mu;
    std::condition_variable gate_cv;
    bool released = false;
    std::atomic<bool> in_send{false};
    auto inner = fx.sender();
    StreamPump pump(fx.session, {}, [&](const std::string& m) {
        if (!in_send.exchange(true)) {
            std::unique_lock<std::mutex> lk(gate_mu);
            gate_cv.wait(lk, [&] { return released; });
        }
        return inner(m);
    });
    pump.push_audio(pcm_for_range(0, 8000));  // preview, then the send blocks
    while (!in_send.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    pump.push_audio(pcm_for_range(8000, 1600));
    pump.push_ping();
    pump.push_audio(pcm_for_range(9600, 1600));
    pump.push_ping();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));  // past the interval
    {
        std::lock_guard<std::mutex> lk(gate_mu);
        released = true;
    }
    gate_cv.notify_all();
    pump.drain();
    CHECK(fx.count("\"type\":\"pong\"") == 2);
    CHECK(previews.load() == 2);  // the first one + one for the whole batch
    CHECK(fx.session.chunker()->coalesced_previews() == 1);
}

static void test_pump_ping_after_rejected_frame_sends_no_partial() {
    // Batch [valid audio, odd-length frame, ping]: the refused frame ends
    // partials for the take, so the ping only answers with a pong.
    PumpFixture fx;
    GatedEngine eng;
    fx.session.set_transcribe_fn(eng.fn());
    StreamPump pump(fx.session, {}, fx.sender());
    pump.push_audio(pcm_for_range(0, 8000));  // preview blocks
    eng.wait_first();
    pump.push_audio(pcm_for_range(8000, 1600));
    pump.push_audio(std::string(3, '\0'));    // odd PCM length
    pump.push_ping();
    eng.release();
    pump.drain();
    CHECK(eng.calls.load() == 1);
    std::lock_guard<std::mutex> lk(fx.mu);
    size_t err = SIZE_MAX;
    for (size_t i = 0; i < fx.sent.size(); ++i)
        if (fx.sent[i].find("\"type\":\"error\"") != std::string::npos) { err = i; break; }
    CHECK(err != SIZE_MAX);
    for (size_t i = err; i < fx.sent.size() && err != SIZE_MAX; ++i)
        CHECK(fx.sent[i].find("\"type\":\"partial\"") == std::string::npos);
    CHECK(!fx.sent.empty() && fx.sent.back() == "{\"type\":\"pong\"}");
}

// ---- main -----------------------------------------------------------------
int main() {
    test_stitch_basic();
    test_stitch_empty();
    test_stitch_no_match();
    test_stitch_punctuation();
    test_stitch_unicode_disjoint();
    test_stitch_unicode_overlap();
    test_stitch_empty_keys_never_match();
    test_norm_word();
    test_split_join();
    test_chunk_streamer_basic();
    test_chunk_streamer_partial();
    test_chunk_streamer_flush();
    test_chunk_streamer_unicode_boundary();
    test_chunk_streamer_rebase();
    test_chunk_streamer_stable_prefix_never_changes();
    test_stream_session_buffer_trim();
    test_stream_session_busy_retry();
    test_stream_session_append_rejection();
    test_bounded_retry_recovery();
    test_tail_reuse_preview_then_commit();
    test_tail_reuse_one_appended_sample_forces_recompute();
    test_tail_reuse_engine_identity_change_forces_recompute();
    test_tail_reuse_callback_swap_forces_recompute();
    test_tail_reuse_mid_step_swap_never_rekeys_in_flight_calls();
    test_tail_reuse_mid_step_engine_identity_never_rekeys_in_flight_calls();
    test_tail_reuse_empty_text_is_a_valid_result();
    test_tail_reuse_busy_preview_recomputes_at_commit();
    test_tail_reuse_cancel_after_success_not_reused();
    test_tail_reuse_commit_completes_while_engine_busy();
    test_tail_reuse_invalidated_take_not_reused();
    test_tail_reuse_duplicate_snapshots_coalesced();
    test_tail_reuse_newer_snapshot_wins();
    test_tail_reuse_reset_clears_entry();
    test_tail_reuse_aba_across_sessions();
    test_tail_reuse_survives_overlap_rebasing();
    test_strict_number_parsing();
    test_stream_window_config_error();
    test_chunk_streamer_rejects_invalid_config();
    test_chunk_streamer_overlap_clamped_to_half_chunk();
    test_chunk_streamer_windows_stay_in_range();
    test_chunk_streamer_never_transcribes_empty_window();
    test_stream_session_rejects_invalid_window_config();
    test_model_mapping();
    test_trace_ledger_kinds_and_spans();
    test_trace_new_final_samples_run_the_tail();
    test_trace_flush_bounded_by_tail_not_take();
    test_preview_minimum_gates_the_take_not_the_tail();
    test_preview_coalesced_when_newer_audio_is_queued();
    test_window_commit_does_not_force_a_throttled_preview();
    test_preview_interval_adapts_to_preview_cost();
    test_preview_policy_validation();
    test_preview_policy_is_not_result_provenance();
    test_pump_stop_during_inference_finalizes_all_audio();
    test_pump_coalesces_previews_and_keeps_every_sample();
    test_pump_ping_follows_earlier_audio();
    test_window_decode_time_counts_toward_the_preview_gap();
    test_pump_stop_preempts_running_preview();
    test_pump_commit_without_new_audio_reuses_running_preview();
    test_pump_window_audio_preempts_running_preview();
    test_pump_reset_preempts_running_preview();
    test_pump_empty_commit_reports_committed_stop();
    test_pump_stalled_peer_releases_the_transport();
    test_pump_backpressure_never_drops_audio();
    test_pump_reset_discards_queued_take();
    test_pump_ping_does_not_coalesce_the_preview();
    test_pump_backlog_commits_windows_before_the_cap();
    test_pump_event_bound_blocks_control_frames();
    test_pump_ping_preview_coalesced_by_later_batch_audio();
    test_pump_ping_after_rejected_frame_sends_no_partial();

    std::printf("stream_session_test: %d/%d passed\n", g_passed, g_tests);
    return g_passed == g_tests ? 0 : 1;
}
