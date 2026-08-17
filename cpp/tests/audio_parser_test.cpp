// audio_parser_test.cpp — pure unit tests for the serve audio parsers.
//
// Covers the input-hardening regressions from issue #12:
//   * WAV bounds: the header's frame count is untrusted; the decoder derives
//     the maximum from the actual payload size and decodes in bounded batches.
//   * Multipart: the header/body separator must be skipped in full (the
//     historical bug left a stray "\r\n" on every payload, breaking WAV RIFF
//     sniffing), and the trailing-CRLF guard must be relative to the part
//     start.
//
// Everything here is pure parsing — no model, no GPU, safe for CPU CI.

#include "serve/audio.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

using namespace starling::serve::audio;

// ---- test harness ---------------------------------------------------------
static int g_tests = 0;
static int g_passed = 0;

#define CHECK(cond) do { \
    ++g_tests; \
    if (cond) { ++g_passed; } \
    else { std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, #cond); } \
} while(0)

// ---- WAV construction helpers ----------------------------------------------

// Build a minimal RIFF/WAVE PCM16 header. data_size_override lets tests lie
// about the data-chunk size the way a crafted upload does (dr_wav derives the
// frame count from it).
static std::string make_wav(uint32_t sample_rate, uint16_t channels,
                            const std::string& data,
                            uint32_t data_size_override = 0) {
    uint32_t data_size = data_size_override ? data_size_override
                                            : static_cast<uint32_t>(data.size());
    uint32_t byte_rate = sample_rate * channels * 2;
    uint16_t block_align = static_cast<uint16_t>(channels * 2);
    std::string h = "RIFF";
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
    h += le32(36 + data_size);          // riff chunk size
    h += "WAVE";
    h += "fmt ";
    h += le32(16);                       // fmt chunk size
    h += le16(1);                        // PCM
    h += le16(channels);
    h += le32(sample_rate);
    h += le32(byte_rate);
    h += le16(block_align);
    h += le16(16);                       // bits per sample
    h += "data";
    h += le32(data_size);
    h += data;
    return h;
}

static std::string pcm16_samples(const std::vector<int16_t>& s) {
    std::string out(s.size() * 2, '\0');
    std::memcpy(out.data(), s.data(), out.size());
    return out;
}

// ---- WAV decode tests ------------------------------------------------------

static void test_wav_mono_decode() {
    std::vector<int16_t> samples = {0, 16384, -16384, 32767, -32768};
    std::string wav = make_wav(16000, 1, pcm16_samples(samples));
    std::vector<float> out;
    int sr = 0;
    bool ok = wav_bytes_to_float32(wav, out, sr);
    CHECK(ok);
    CHECK(sr == 16000);
    CHECK(out.size() == samples.size());
    if (out.size() == samples.size()) {
        CHECK(out[0] == 0.0f);
        CHECK(out[1] > 0.4f && out[1] < 0.6f);    // 16384/32768
        CHECK(out[2] < -0.4f && out[2] > -0.6f);  // -16384/32768
        CHECK(out[3] <= 1.0f && out[3] > 0.9f);
        CHECK(out[4] >= -1.0f && out[4] < -0.9f);
    }
}

static void test_wav_stereo_mixdown() {
    // Stereo frames: (L,R) = (x, x) → mono x.
    std::vector<int16_t> interleaved = {8192, 8192, -8192, -8192, 8192, -8192};
    std::string wav = make_wav(16000, 2, pcm16_samples(interleaved));
    std::vector<float> out;
    int sr = 0;
    bool ok = wav_bytes_to_float32(wav, out, sr);
    CHECK(ok);
    CHECK(out.size() == 3);
    if (out.size() == 3) {
        CHECK(out[0] > 0.2f && out[0] < 0.3f);   // 8192/32768 = 0.25
        CHECK(out[1] < -0.2f && out[1] > -0.3f);
        CHECK(out[2] == 0.0f);                   // (8192 + -8192)/2
    }
}

static void test_wav_larger_than_one_batch() {
    // More frames than one decode batch (16384) exercises the batched loop.
    std::vector<int16_t> samples(40000, 1234);
    std::string wav = make_wav(16000, 1, pcm16_samples(samples));
    std::vector<float> out;
    int sr = 0;
    bool ok = wav_bytes_to_float32(wav, out, sr);
    CHECK(ok);
    CHECK(out.size() == 40000);
    bool all_same = out.size() == 40000;
    for (float v : out) {
        if (v < 0.0376f || v > 0.0377f) { all_same = false; break; }
    }
    CHECK(all_same);  // 1234/32768 ≈ 0.03765
}

static void test_wav_rejects_huge_claimed_frames() {
    // Crafted RF64 header: the data-chunk sentinel 0xFFFFFFFF defers sizes to
    // the ds64 chunk, whose sampleCount claims 0x100000000 samples (~4.3e9
    // frames) over an 8-byte payload. dr_wav reports that count verbatim —
    // the decoder must derive the maximum from the payload and fail fast
    // instead of scaling any allocation with the claim (issue #12).
    auto put32 = [](std::string& s, uint32_t v) {
        char b[4]; std::memcpy(b, &v, 4); s.append(b, 4);
    };
    auto put64 = [](std::string& s, uint64_t v) {
        char b[8]; std::memcpy(b, &v, 8); s.append(b, 8);
    };
    auto put16 = [](std::string& s, uint16_t v) {
        char b[2]; std::memcpy(b, &v, 2); s.append(b, 2);
    };
    std::string w = "RF64";
    put32(w, 0xFFFFFFFF); w += "WAVE";
    // ds64: riffSize, dataSize (real, tiny), sampleCount (the lie), tableLen.
    w += "ds64"; put32(w, 28);
    put64(w, 100); put64(w, 8); put64(w, 0x100000000ULL); put32(w, 0);
    w += "fmt "; put32(w, 16); put16(w, 1); put16(w, 1); put32(w, 16000);
    put32(w, 32000); put16(w, 2); put16(w, 16);
    w += "data"; put32(w, 0xFFFFFFFF);
    w += pcm16_samples({1, 2, 3, 4});

    std::vector<float> out;
    int sr = 0;
    bool ok = wav_bytes_to_float32(w, out, sr);
    CHECK(!ok);
    CHECK(out.empty());
}

static void test_wav_riff_size_lie_stays_bounded() {
    // A plain RIFF header whose data-chunk size claims 0x7fffffff bytes over
    // an 8-byte payload. dr_wav clamps this one itself (memory containers can
    // seek to the end), so the contract here is bounded behavior: the decode
    // must not blow up or allocate per the claim; it decodes the real payload.
    std::string data = pcm16_samples({1, 2, 3, 4});
    std::string wav = make_wav(16000, 1, data, 0x7fffffff);
    std::vector<float> out;
    int sr = 0;
    bool ok = wav_bytes_to_float32(wav, out, sr);
    CHECK(ok);
    CHECK(out.size() == 4);
}

static void test_wav_truncated_decodes_available() {
    // A truncated upload (data-chunk size claims 1000 frames, payload carries
    // 10): dr_wav clamps to what's present and the decoder returns those
    // frames — bounded, no crash, no garbage.
    std::vector<int16_t> samples(10, 100);
    std::string wav = make_wav(16000, 1, pcm16_samples(samples), 1000 * 2);
    std::vector<float> out;
    int sr = 0;
    bool ok = wav_bytes_to_float32(wav, out, sr);
    CHECK(ok);
    CHECK(out.size() == 10);
}

static void test_wav_rejects_garbage() {
    std::vector<float> out;
    int sr = 0;
    CHECK(!wav_bytes_to_float32("", out, sr));
    CHECK(!wav_bytes_to_float32("not a wav at all", out, sr));
    // RIFF magic but truncated before the fmt chunk.
    CHECK(!wav_bytes_to_float32("RIFF\x00\x00\x00\x00WAVEjunk", out, sr));
}

static void test_pcm16_conversion() {
    auto out = pcm16_to_float32(pcm16_samples({0, 16384, -16384}));
    CHECK(out.size() == 3);
    if (out.size() == 3) {
        CHECK(out[0] == 0.0f);
        CHECK(out[1] > 0.4f && out[1] < 0.6f);
        CHECK(out[2] < -0.4f && out[2] > -0.6f);
    }
    // Odd trailing byte dropped.
    auto out2 = pcm16_to_float32(pcm16_samples({1, 2}) + "x");
    CHECK(out2.size() == 2);
    CHECK(pcm16_to_float32("").empty());
}

// ---- multipart tests -------------------------------------------------------

// Build a one-part multipart body with full control over the part headers.
static std::string multipart(const std::string& boundary,
                             const std::string& part_headers,
                             const std::string& payload) {
    std::string body = "--" + boundary + "\r\n";
    body += part_headers;
    body += "\r\n\r\n";
    body += payload;
    body += "\r\n--" + boundary + "--\r\n";
    return body;
}

static void test_multipart_basic_extraction() {
    std::string payload = "RIFFxxxxWAVEfake-audio-bytes";
    std::string body = multipart("XX",
        "Content-Disposition: form-data; name=\"audio\"; filename=\"a.wav\"",
        payload);
    std::string ct = "multipart/form-data; boundary=XX";
    std::string got = extract_multipart_payload(body, ct);
    // Regression (issue #12): the payload must be byte-identical — the old
    // separator skip left a stray "\r\n" prefix that broke WAV RIFF sniffing.
    CHECK(got == payload);
}

static void test_multipart_quoted_boundary() {
    std::string payload = "AUDIO";
    std::string body = multipart("abc123", "Content-Disposition: form-data; name=\"file\"", payload);
    std::string ct = "multipart/form-data; boundary=\"abc123\"";
    CHECK(extract_multipart_payload(body, ct) == payload);
}

static void test_multipart_selection_priority() {
    // "audio"/"file" named part wins over a filename'd part.
    std::string b = "--B\r\n"
        "Content-Disposition: form-data; name=\"metadata\"\r\n\r\n"
        "meta\r\n"
        "--B\r\n"
        "Content-Disposition: form-data; name=\"audio\"; filename=\"x.wav\"\r\n\r\n"
        "AUDIOPART\r\n"
        "--B\r\n"
        "Content-Disposition: form-data; filename=\"other.bin\"\r\n\r\n"
        "FILEPART\r\n"
        "--B--\r\n";
    std::string ct = "multipart/form-data; boundary=B";
    CHECK(extract_multipart_payload(b, ct) == "AUDIOPART");
}

static void test_multipart_fallback_last_part() {
    // No scoring part matches → the last non-empty payload is returned.
    std::string b = "--B\r\n"
        "Content-Disposition: form-data; name=\"foo\"\r\n\r\n"
        "first\r\n"
        "--B\r\n"
        "Content-Disposition: form-data; name=\"bar\"\r\n\r\n"
        "second\r\n"
        "--B--\r\n";
    std::string ct = "multipart/form-data; boundary=B";
    CHECK(extract_multipart_payload(b, ct) == "second");
}

static void test_multipart_empty_part_underflow() {
    // Regression for the relative part_end guard (issue #12): a zero-width
    // part ("--B\r\n" immediately followed by the next delimiter) made the
    // OLD absolute check (part_end >= 2) strip a CRLF *before* part_start,
    // underflowing part_end - part_start and letting substr() swallow the
    // rest of the body. The real payload must still win.
    std::string payload = "AUDIO";
    std::string b = "--B\r\n"
        "--B\r\n"
        "Content-Disposition: form-data; name=\"audio\"\r\n\r\n"
        + payload + "\r\n"
        "--B--\r\n";
    std::string ct = "multipart/form-data; boundary=B";
    std::string got = extract_multipart_payload(b, ct);
    CHECK(got == payload);
}

static void test_multipart_blank_part() {
    // A part with an EMPTY payload next to the real one, plus a headerless
    // blank part at the end. Must not crash or return a body-sized blob.
    std::string payload = "AUDIO";
    std::string b = "--B\r\n"
        "Content-Disposition: form-data; name=\"audio\"\r\n\r\n"
        "\r\n"
        "--B\r\n"
        "Content-Disposition: form-data; name=\"file\"\r\n\r\n"
        + payload + "\r\n"
        "--B\r\n"
        "\r\n"
        "--B--\r\n";
    std::string ct = "multipart/form-data; boundary=B";
    CHECK(extract_multipart_payload(b, ct) == payload);
}

static void test_multipart_binary_safe() {
    // Payloads with embedded NULs and header-like bytes must pass through
    // verbatim (audio bytes are not text).
    std::string payload;
    payload += static_cast<char>(0x00);
    payload += static_cast<char>(0xff);
    payload += "\r\n--almost-a-boundary--";
    payload += "RIFF";
    std::string body = multipart("B",
        "Content-Disposition: form-data; name=\"audio\"", payload);
    std::string ct = "multipart/form-data; boundary=B";
    CHECK(extract_multipart_payload(body, ct) == payload);
}

static void test_multipart_not_multipart() {
    // No boundary parameter → body returned unchanged (raw WAV path).
    std::string body = "RIFF....WAVE....";
    CHECK(extract_multipart_payload(body, "application/octet-stream") == body);
    CHECK(extract_multipart_payload(body, "multipart/form-data") == body);
}

// ---- main -----------------------------------------------------------------
int main() {
    test_wav_mono_decode();
    test_wav_stereo_mixdown();
    test_wav_larger_than_one_batch();
    test_wav_rejects_huge_claimed_frames();
    test_wav_riff_size_lie_stays_bounded();
    test_wav_truncated_decodes_available();
    test_wav_rejects_garbage();
    test_pcm16_conversion();
    test_multipart_basic_extraction();
    test_multipart_quoted_boundary();
    test_multipart_selection_priority();
    test_multipart_fallback_last_part();
    test_multipart_empty_part_underflow();
    test_multipart_blank_part();
    test_multipart_binary_safe();
    test_multipart_not_multipart();

    std::printf("audio_parser_test: %d/%d passed\n", g_passed, g_tests);
    return g_passed == g_tests ? 0 : 1;
}
