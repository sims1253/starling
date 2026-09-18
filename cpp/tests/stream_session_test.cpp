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
    CHECK(supported_models_str() == "parakeet moss ark higgs hojo granite qwen3 s1 audex ark06 voxtral");
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
    auto text = session.stream_flush();
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
    CHECK(session.buffered_seconds() == 1.5);
    CHECK(session.live_seconds() == 1.5);
    auto fs = session.stream_flush();
    CHECK(!fs.has_value());
    CHECK(session.buffered_seconds() == 1.5);
    session.set_transcribe_fn([](const float*, int64_t n) -> std::optional<std::string> {
        CHECK(n <= 16000);
        return "retained audio";
    });
    CHECK(session.stream_flush() == "retained audio");
    CHECK(session.buffered_seconds() == 1.5);
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
    CHECK(session.buffered_seconds() == 0.1);

    // (4) Valid -> invalid -> valid (the issue's regression sequence):
    // audio before the rejection is retained, audio after it is refused.
    session.reset();
    CHECK(session.append_pcm(pcm_for_range(0, 8000)) == AppendOutcome::Accepted);
    const std::string wav8k = make_wav_bytes(8000, std::string(16000, '\0'));
    CHECK(session.append_wav(wav8k) == AppendOutcome::RateMismatch);
    CHECK(session.invalid_reason() == "sample_rate_mismatch");
    CHECK(session.buffered_seconds() == 0.5);  // pre-rejection audio kept
    CHECK(session.append_pcm(pcm_for_range(8000, 8000)) == AppendOutcome::TakeInvalid);
    CHECK(session.buffered_seconds() == 0.5);  // post-rejection audio refused
    session.reset();
    CHECK(session.append_pcm(pcm_for_range(0, 8000)) == AppendOutcome::Accepted);
    CHECK(session.buffered_seconds() == 0.5);
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
    CHECK(session.buffered_seconds() == 0.5);
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
    test_stream_session_buffer_trim();
    test_stream_session_busy_retry();
    test_stream_session_append_rejection();
    test_bounded_retry_recovery();
    test_model_mapping();

    std::printf("stream_session_test: %d/%d passed\n", g_passed, g_tests);
    return g_passed == g_tests ? 0 : 1;
}
