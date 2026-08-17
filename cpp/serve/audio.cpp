// audio.cpp — WAV/PCM audio decoding implementation.
//
// NOTE: dr_wav.h with DR_WAV_IMPLEMENTATION is already included in
// cpp/runtime/audio_io.cpp (part of starling_ggml_core, which starling-serve
// links). We only need the declarations here, not a second implementation.

#include "audio.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <sstream>

#include "dr_wav.h"

namespace starling::serve::audio {

// Decode in bounded batches so no single allocation scales with the header's
// (untrusted) frame count.
constexpr drwav_uint64 kWavDecodeBatchFrames = 16384;
// No real-world upload carries more channels than this; a fmt chunk claiming
// more is crafted (it would also scale the batch buffer by channels).
constexpr uint32_t kWavMaxChannels = 64;

bool wav_bytes_to_float32(const std::string& wav_bytes,
                          std::vector<float>& out, int& sr) {
    drwav wav;
    if (!drwav_init_memory(&wav, wav_bytes.data(), wav_bytes.size(), nullptr))
        return false;

    sr = static_cast<int>(wav.sampleRate);
    uint64_t total_frames = wav.totalPCMFrameCount;
    uint32_t channels = wav.channels;

    // The header's totalPCMFrameCount is derived from the data-chunk size
    // field, which is untrusted input: a crafted header claiming millions of
    // frames must not drive the allocation. Derive the maximum frame count
    // from the actual payload length and reject headers that overclaim (a
    // truncated or lying data-chunk size is a malformed file).
    uint32_t bytes_per_frame =
        ((wav.bitsPerSample + 7) / 8) * channels;
    if (channels == 0 || channels > kWavMaxChannels || total_frames == 0
        || bytes_per_frame == 0) {
        drwav_uninit(&wav);
        return false;
    }
    uint64_t max_frames = wav_bytes.size() / bytes_per_frame;
    if (total_frames > max_frames) {
        drwav_uninit(&wav);
        return false;
    }

    out.clear();
    out.reserve(static_cast<size_t>(total_frames));

    // Decode as interleaved int16 in fixed-size batches, then convert to mono
    // float32.
    std::vector<drwav_int16> interleaved;
    interleaved.resize(static_cast<size_t>(kWavDecodeBatchFrames) * channels);
    uint64_t decoded_frames = 0;
    while (decoded_frames < total_frames) {
        uint64_t want = std::min<uint64_t>(kWavDecodeBatchFrames,
                                           total_frames - decoded_frames);
        drwav_uint64 read = drwav_read_pcm_frames_s16(
            &wav, want, interleaved.data());
        if (read == 0) break;  // payload exhausted before the header's count
        size_t old = out.size();
        out.resize(old + static_cast<size_t>(read));
        if (channels == 1) {
            for (drwav_uint64 i = 0; i < read; ++i)
                out[old + i] = static_cast<float>(interleaved[i]) / 32768.0f;
        } else {
            // Mix down to mono by averaging channels.
            for (drwav_uint64 i = 0; i < read; ++i) {
                int sum = 0;
                for (uint32_t c = 0; c < channels; ++c)
                    sum += interleaved[i * channels + c];
                out[old + i] = static_cast<float>(sum) / (channels * 32768.0f);
            }
        }
        decoded_frames += read;
    }
    drwav_uninit(&wav);

    if (out.empty()) return false;
    return true;
}

std::vector<float> pcm16_to_float32(const std::string& bytes) {
    size_t nbytes = bytes.size();
    if (nbytes == 0) return {};
    if (nbytes % 2 == 1) nbytes--;  // drop odd trailing byte
    size_t nsamples = nbytes / 2;
    std::vector<float> out(nsamples);
    const auto* src = reinterpret_cast<const int16_t*>(bytes.data());
    for (size_t i = 0; i < nsamples; ++i)
        out[i] = static_cast<float>(src[i]) / 32768.0f;
    return out;
}

// ---- multipart extraction -------------------------------------------------
// Minimal multipart/form-data parser. Finds the boundary, splits parts, and
// returns the best-scoring audio payload. Selection order:
//   1. a part named "audio" or "file"
//   2. a part with a filename
//   3. a part whose content-type starts with "audio/"
//   4. the last non-empty part
std::string extract_multipart_payload(const std::string& body,
                                      const std::string& content_type) {
    // Find the boundary parameter.
    std::string boundary;
    {
        size_t pos = 0;
        while (pos < content_type.size()) {
            size_t semi = content_type.find(';', pos);
            std::string tok = (semi == std::string::npos)
                ? content_type.substr(pos)
                : content_type.substr(pos, semi - pos);
            // trim whitespace
            size_t s = tok.find_first_not_of(" \t");
            size_t e = tok.find_last_not_of(" \t");
            if (s != std::string::npos) tok = tok.substr(s, e - s + 1);
            else tok = "";
            if (tok.size() > 9 && tok.substr(0, 9) == "boundary=") {
                boundary = tok.substr(9);
                // Strip quotes.
                if (!boundary.empty() && boundary.front() == '"')
                    boundary.erase(0, 1);
                if (!boundary.empty() && boundary.back() == '"')
                    boundary.pop_back();
                break;
            }
            if (semi == std::string::npos) break;
            pos = semi + 1;
        }
    }
    if (boundary.empty()) return body;  // not multipart

    std::string delim = "--" + boundary;
    std::vector<std::string> parts;
    std::string last_payload;

    // Split on boundary delimiter.
    size_t pos = 0;
    while (true) {
        size_t found = body.find(delim, pos);
        if (found == std::string::npos) break;
        size_t part_start = found + delim.size();
        // Skip CRLF after delimiter.
        if (part_start < body.size() && body[part_start] == '\r') part_start++;
        if (part_start < body.size() && body[part_start] == '\n') part_start++;
        // Find the next delimiter.
        size_t next = body.find(delim, part_start);
        if (next == std::string::npos) break;
        // The part is body[part_start..next), but strip the trailing CRLF.
        size_t part_end = next;
        if (part_end >= part_start + 2
            && body[part_end - 2] == '\r' && body[part_end - 1] == '\n')
            part_end -= 2;
        std::string raw_part = body.substr(part_start, part_end - part_start);
        // Split headers from body (blank line separates them).
        size_t header_end = raw_part.find("\r\n\r\n");
        size_t sep_len = 4;  // "\r\n\r\n"
        if (header_end == std::string::npos) {
            header_end = raw_part.find("\n\n");
            sep_len = 2;  // "\n\n"
        }
        std::string headers, payload;
        if (header_end == std::string::npos) {
            payload = raw_part;
        } else {
            headers = raw_part.substr(0, header_end);
            // Skip the FULL separator: the header/body blank line is 4 bytes
            // for CRLFCRLF (2 for LFLF). Skipping only 2 left every payload
            // prefixed with a stray "\r\n", which broke WAV RIFF sniffing.
            size_t body_start = std::min(header_end + sep_len, raw_part.size());
            payload = raw_part.substr(body_start);
        }

        if (!payload.empty()) last_payload = payload;
        parts.push_back(headers + "\x01" + payload);  // separator
        pos = next;
    }

    // Score each part.
    auto get_header_value = [](const std::string& headers,
                               const std::string& key) -> std::string {
        size_t pos = 0;
        std::string lkey = key;
        for (auto& c : lkey) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        while (pos < headers.size()) {
            size_t eol = headers.find('\n', pos);
            std::string line = (eol == std::string::npos)
                ? headers.substr(pos) : headers.substr(pos, eol - pos);
            // Find colon.
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                std::string hkey = line.substr(0, colon);
                for (auto& c : hkey)
                    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                // trim
                size_t s = hkey.find_first_not_of(" \t");
                if (s != std::string::npos) hkey = hkey.substr(s);
                if (hkey == lkey) {
                    std::string val = line.substr(colon + 1);
                    s = val.find_first_not_of(" \t");
                    if (s != std::string::npos) val = val.substr(s);
                    return val;
                }
            }
            if (eol == std::string::npos) break;
            pos = eol + 1;
        }
        return "";
    };

    auto get_disp_param = [](const std::string& cd_value,
                              const std::string& key) -> std::string {
        std::string lkey = key;
        for (auto& c : lkey) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        size_t pos = 0;
        while (pos < cd_value.size()) {
            size_t semi = cd_value.find(';', pos);
            std::string tok = (semi == std::string::npos)
                ? cd_value.substr(pos) : cd_value.substr(pos, semi - pos);
            size_t s = tok.find_first_not_of(" \t");
            if (s != std::string::npos) tok = tok.substr(s);
            size_t eq = tok.find('=');
            if (eq != std::string::npos) {
                std::string k = tok.substr(0, eq);
                for (auto& c : k) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                std::string v = tok.substr(eq + 1);
                if (!v.empty() && v.front() == '"') v.erase(0, 1);
                if (!v.empty() && v.back() == '"') v.pop_back();
                if (k == lkey) return v;
            }
            if (semi == std::string::npos) break;
            pos = semi + 1;
        }
        return "";
    };

    auto score = [&](const std::string& headers, const std::string& /*payload*/) -> int {
        std::string cd = get_header_value(headers, "content-disposition");
        std::string name = get_disp_param(cd, "name");
        if (!name.empty()) {
            std::string lname = name;
            for (auto& c : lname) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            if (lname == "audio" || lname == "file") return 4;
        }
        std::string filename = get_disp_param(cd, "filename");
        if (!filename.empty()) return 3;
        std::string ctype = get_header_value(headers, "content-type");
        for (auto& c : ctype) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (ctype.size() >= 6 && ctype.substr(0, 6) == "audio/") return 2;
        return 0;
    };

    int best_score = -1;
    std::string best_payload = body;
    for (const auto& p : parts) {
        size_t sep = p.find('\x01');
        std::string headers = p.substr(0, sep);
        std::string payload = p.substr(sep + 1);
        if (payload.empty()) continue;  // an empty field can't be the upload
        int sc = score(headers, payload);
        if (sc > best_score) {
            best_score = sc;
            best_payload = payload;
        }
    }
    if (best_score <= 0) {
        return last_payload.empty() ? body : last_payload;
    }
    return best_payload;
}

} // namespace starling::serve::audio
