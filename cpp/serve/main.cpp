// main.cpp — starling-serve entry point.
//
// Parses CLI args, loads the model, and serves the HTTP/WebSocket API.
// Drop-in replacement for `python -m starling.server`:
//
//   starling-serve --model <slug> --gguf <path> [--port 8181] [--warmup]
//
// The HTTP/WS transport uses cpp-httplib (vendored header-only).

// cpp-httplib uses std::thread; on some platforms we need pthread.
#define CPPHTTPLIB_THREAD_POOL_ENQUEUE 1

#include "server.hpp"
#include "stream_session.hpp"
#include "audio.hpp"
#include "starling_ggml.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// cpp-httplib
#include "httplib.h"

namespace serve = starling::serve;

// ---- version / build info -------------------------------------------------
// These are overridden at build time via -D compile definitions so clients
// can verify compatibility.
#ifndef STARLING_SERVE_VERSION
#define STARLING_SERVE_VERSION "0.1.0"
#endif

static void print_version() {
    std::printf("starling-serve %s\n", STARLING_SERVE_VERSION);
    std::printf("abi-version: %d\n", starling_ggml_abi_version());
    std::printf("backend: %s\n", starling_ggml_backend_name());
    std::printf("supported-models: %s\n", serve::supported_models_str().c_str());
}

// ---- usage ----------------------------------------------------------------
static void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s --model <slug> --gguf <path> [options]\n"
        "\n"
        "Required:\n"
        "  --model <slug>     Model slug: %s\n"
        "  --gguf <path>      Path to the GGUF model file\n"
        "\n"
        "Serving:\n"
        "  --host <addr>      Bind address (default 127.0.0.1)\n"
        "  --port <n>         Bind port (default 8181)\n"
        "  --warmup           Warm up the model on startup\n"
        "  --no-eager-load    Defer model load to first request\n"
        "  --idle-timeout <s> Shut down after N seconds idle (0 = never, default 0)\n"
        "  --request-timeout-seconds <s> Fail queued requests after N s waiting for\n"
        "                     the engine (0 = never, default 600; same flag as the\n"
        "                     Python server)\n"
        "\n"
        "Streaming:\n"
        "  --stream-chunk-seconds <s>    Fixed stream window (default 12.0)\n"
        "  --stream-overlap-seconds <s>  Overlap between windows (default 3.0)\n"
        "  --min-chunk-seconds <s>       Min audio before first partial (default 5.0)\n"
        "  --partial-interval-seconds <s> Min gap between partials (default 3.0)\n"
        "  --max-stream-seconds <s>      Per-connection audio buffer cap in s\n"
        "                     (0 = unlimited, default 60). A frame that would\n"
        "                     exceed it is rejected with an error frame and the\n"
        "                     session ignores audio until it is reset\n"
        "\n"
        "Introspection:\n"
        "  --version          Print version + ABI version + backend, then exit\n"
        "  --abi-version      Print just the ABI version integer, then exit\n"
        "  --help             Show this help\n",
        prog, serve::supported_models_str().c_str());
}

// ---- simple arg parser ----------------------------------------------------
struct Args {
    std::string model;
    std::string gguf;
    std::string host = "127.0.0.1";
    int port = 8181;
    bool warmup = false;
    bool eager_load = true;
    double idle_timeout = 0.0;
    double request_timeout = 600.0;
    double max_stream_seconds = 60.0;
    double stream_chunk = 12.0;
    double stream_overlap = 3.0;
    double min_chunk = 5.0;
    double partial_interval = 3.0;
    bool show_version = false;
    bool show_abi = false;
    bool show_help = false;
    bool error = false;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: %s requires a value\n", name);
                a.error = true;
                return "";
            }
            return argv[++i];
        };
        auto next_int = [&](const char* name) -> int {
            std::string v = next(name);
            if (a.error) return 0;
            // Strict parse: bare std::stoi accepts partial parses like "3abc"
            // and silently truncates out-of-range values (issue #146).
            auto parsed = serve::parse_int_strict(v);
            if (!parsed.has_value()) {
                std::fprintf(stderr, "error: %s requires an integer, got '%s'\n", name, v.c_str());
                a.error = true;
                return 0;
            }
            return *parsed;
        };
        auto next_double = [&](const char* name) -> double {
            std::string v = next(name);
            if (a.error) return 0.0;
            // Strict parse: bare std::stod accepts partial parses like "3abc"
            // and non-finite tokens like "nan"/"inf" (issue #146).
            auto parsed = serve::parse_double_strict(v);
            if (!parsed.has_value()) {
                std::fprintf(stderr, "error: %s requires a finite number, got '%s'\n", name, v.c_str());
                a.error = true;
                return 0.0;
            }
            return *parsed;
        };
        if (arg == "--model")          a.model = next("--model");
        else if (arg == "--gguf")      a.gguf = next("--gguf");
        else if (arg == "--host")      a.host = next("--host");
        else if (arg == "--port")      a.port = next_int("--port");
        else if (arg == "--warmup")    a.warmup = true;
        else if (arg == "--no-eager-load") a.eager_load = false;
        else if (arg == "--idle-timeout")  a.idle_timeout = next_double("--idle-timeout");
        else if (arg == "--request-timeout-seconds") a.request_timeout = next_double("--request-timeout-seconds");
        else if (arg == "--max-stream-seconds") a.max_stream_seconds = next_double("--max-stream-seconds");
        else if (arg == "--stream-chunk-seconds")   a.stream_chunk = next_double("--stream-chunk-seconds");
        else if (arg == "--stream-overlap-seconds") a.stream_overlap = next_double("--stream-overlap-seconds");
        else if (arg == "--min-chunk-seconds")      a.min_chunk = next_double("--min-chunk-seconds");
        else if (arg == "--partial-interval-seconds") a.partial_interval = next_double("--partial-interval-seconds");
        else if (arg == "--version")   a.show_version = true;
        else if (arg == "--abi-version") a.show_abi = true;
        else if (arg == "--help" || arg == "-h") a.show_help = true;
        else {
            std::fprintf(stderr, "error: unknown argument %s\n", arg.c_str());
            a.error = true;
        }
    }
    return a;
}

// ---- JSON response helper -------------------------------------------------
static void send_json(httplib::Response& res, const std::string& body, int status = 200) {
    res.status = status;
    res.set_content(body, "application/json");
}

// ---- JSON string escape (shared by WS handler) ---------------------------
static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
        case '"':  out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n";  break;
        case '\r': out += "\\r";  break;
        case '\t': out += "\\t";  break;
        case '\b': out += "\\b";  break;
        case '\f': out += "\\f";  break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                out += buf;
            } else {
                out += c;
            }
        }
    }
    return out;
}

// ---- WS /stream binary-frame rejection --------------------------------------
// Describe a refused binary audio frame as a structured WS error frame,
// mirroring the buffer-cap error style. A refused frame invalidates the take
// (or trips the buffer cap); the session then ignores audio until reset.
static std::string ws_append_error(serve::AppendOutcome outcome,
                                   const serve::StreamSession& session,
                                   double max_stream_seconds) {
    std::ostringstream ss;
    ss << "{\"type\":\"error\",\"message\":\"";
    switch (outcome) {
    case serve::AppendOutcome::MalformedWav:
        ss << "malformed WAV frame rejected; audio ignored until reset";
        break;
    case serve::AppendOutcome::RateMismatch:
        ss << "WAV sample rate mismatch: expected " << serve::kSampleRate
           << "; audio ignored until reset";
        break;
    case serve::AppendOutcome::OddPcmLength:
        ss << "odd-length PCM frame rejected (split sample);"
           << " audio ignored until reset";
        break;
    case serve::AppendOutcome::Overflowed:
        ss << "stream buffer limit reached (" << max_stream_seconds
           << " s live buffer); audio ignored until reset";
        break;
    case serve::AppendOutcome::TakeInvalid:
        // Only reached on a frame AFTER the invalidating one (whose own
        // outcome carried the reason); repeat that reason, not a generic.
        // invalid_reason_ is an internal [a-z_] code: safe to embed raw.
        ss << "take invalidated (" << session.invalid_reason()
           << "); audio ignored until reset";
        break;
    case serve::AppendOutcome::Accepted:
        break;
    }
    ss << "\"}";
    return ss.str();
}

// ---- flat JSON string-field extraction (POST /normalize) ------------------
// Extracts a top-level "key": "value" string field from a flat JSON object
// (the /normalize request shape: string fields, no nesting). A single pass
// first marks every byte's in-string state and nesting depth, so a `"key"`
// sequence INSIDE a string value (e.g. a transcript that literally contains
// "context": "email") or inside a nested object can never be mistaken for a
// top-level field. A candidate only qualifies when it starts outside any
// string at depth 1, the previous non-whitespace char is `{` or `,`, and the
// next non-whitespace char is `:`. Escapes (\", \\, \n, \t, \uXXXX) are
// decoded; a \u escape encodes UTF-8, with a valid surrogate pair combining
// into one 4-byte sequence (issue #123).
// Returns false when the field is absent or malformed.

// Parse exactly four hex digits at body[start..start+4) into *out.
// Returns false on a non-hex byte (malformed \u escape).
static bool json_hex4(const std::string& body, size_t start, unsigned& out) {
    if (start + 4 > body.size()) return false;
    unsigned v = 0;
    for (int i = 0; i < 4; ++i) {
        char h = body[start + i];
        v <<= 4;
        if (h >= '0' && h <= '9') v |= (unsigned)(h - '0');
        else if (h >= 'a' && h <= 'f') v |= (unsigned)(h - 'a' + 10);
        else if (h >= 'A' && h <= 'F') v |= (unsigned)(h - 'A' + 10);
        else return false;
    }
    out = v;
    return true;
}

// Append code point cp (<= 0x10FFFF) to out as UTF-8.
static void json_append_utf8(std::string& out, unsigned cp) {
    if (cp < 0x80) {
        out += (char)cp;
    } else if (cp < 0x800) {
        out += (char)(0xc0 | (cp >> 6));
        out += (char)(0x80 | (cp & 63));
    } else if (cp < 0x10000) {
        out += (char)(0xe0 | (cp >> 12));
        out += (char)(0x80 | ((cp >> 6) & 63));
        out += (char)(0x80 | (cp & 63));
    } else {
        out += (char)(0xf0 | (cp >> 18));
        out += (char)(0x80 | ((cp >> 12) & 63));
        out += (char)(0x80 | ((cp >> 6) & 63));
        out += (char)(0x80 | (cp & 63));
    }
}

static bool json_get_string(const std::string& body, const std::string& key,
                            std::string& out) {
    // Pass 1: in-string/escape state + enclosing depth per byte.
    std::vector<uint8_t> in_str(body.size(), 0);
    std::vector<int> depth(body.size(), 0);
    bool str = false, esc = false;
    int d = 0;
    for (size_t i = 0; i < body.size(); ++i) {
        in_str[i] = str ? 1 : 0;
        depth[i] = d;
        char c = body[i];
        if (str) {
            if (esc) esc = false;
            else if (c == '\\') esc = true;
            else if (c == '"') str = false;
            continue;
        }
        if (c == '"') str = true;
        else if (c == '{' || c == '[') ++d;
        else if (c == '}' || c == ']') --d;
    }

    const std::string needle = "\"" + key + "\"";
    size_t pos = body.find(needle);
    while (pos != std::string::npos) {
        // Qualify: outside any string, top-level object, after '{' or ','.
        if (in_str[pos] || depth[pos] != 1) {
            pos = body.find(needle, pos + 1);
            continue;
        }
        // find_last_not_of starts AT the given index; pass pos-1 so the
        // needle's own opening quote is not the "previous" char.
        size_t prev = pos > 0 ? body.find_last_not_of(" \t\r\n", pos - 1)
                              : std::string::npos;
        if (prev == std::string::npos ||
            (body[prev] != '{' && body[prev] != ',')) {
            pos = body.find(needle, pos + 1);
            continue;
        }
        size_t q = body.find_first_not_of(" \t\r\n", pos + needle.size());
        if (q == std::string::npos || body[q] != ':') {
            pos = body.find(needle, pos + 1);
            continue;
        }
        q = body.find_first_not_of(" \t\r\n", q + 1);
        if (q == std::string::npos || body[q] != '"') return false;
        ++q;
        // Pass 2: decode the value string with escapes.
        out.clear();
        while (q < body.size()) {
            char c = body[q];
            if (c == '"') return true;
            if (c != '\\') { out += c; ++q; continue; }
            if (++q >= body.size()) return false;
            char e = body[q];
            switch (e) {
            case '"':  out += '"';  break;
            case '\\': out += '\\'; break;
            case '/':  out += '/';  break;
            case 'n':  out += '\n'; break;
            case 'r':  out += '\r'; break;
            case 't':  out += '\t'; break;
            case 'b':  out += '\b'; break;
            case 'f':  out += '\f'; break;
            case 'u': {
                unsigned cp = 0;
                if (!json_hex4(body, q + 1, cp)) return false;
                // Surrogate pairs (issue #123): "\ud83d\ude00" is ONE code
                // point and must decode to one 4-byte UTF-8 sequence, not to
                // two 3-byte sequences (surrogate code points have no UTF-8
                // encoding — that output was invalid UTF-8 while the raw
                // character decoded correctly). An unpaired surrogate half
                // decodes to U+FFFD (REPLACEMENT CHARACTER): JSON only
                // permits paired escapes, but rejecting the whole request
                // over one stray half would 400 transcripts that common
                // serializers emit and other parsers accept; replacement is
                // the WHATWG encoding standard's interoperable policy.
                if (cp >= 0xd800 && cp <= 0xdbff) {
                    unsigned lo = 0;
                    if (q + 10 < body.size() && body[q + 5] == '\\'
                        && body[q + 6] == 'u'
                        && json_hex4(body, q + 7, lo)
                        && lo >= 0xdc00 && lo <= 0xdfff) {
                        cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
                        q += 6;  // consume the low escape's "\u" too
                    } else {
                        cp = 0xfffd;  // unpaired high surrogate
                    }
                } else if (cp >= 0xdc00 && cp <= 0xdfff) {
                    cp = 0xfffd;      // unpaired low surrogate
                }
                json_append_utf8(out, cp);
                q += 4;
                break;
            }
            default: return false;
            }
            ++q;
        }
        return false;  // unterminated string
    }
    return false;  // key absent
}


// ---- idle-timeout monitor -------------------------------------------------
static std::atomic<bool> g_should_exit{false};
static std::atomic<time_t> g_last_activity{0};
static std::atomic<bool> g_warmup_running{false};

static void idle_timeout_thread(serve::StarlingServer* server, double timeout_s) {
    if (timeout_s <= 0.0) return;
    while (!g_should_exit.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        time_t now = std::time(nullptr);
        if (g_last_activity.load() > 0
            && server->loaded()
            && !server->busy()
            && now - g_last_activity.load() > static_cast<time_t>(timeout_s)) {
            std::fprintf(stderr,
                "[starling-serve] idle timeout (%.0fs) reached, shutting down\n",
                timeout_s);
            g_should_exit.store(true);
            // httplib doesn't expose a clean stop from a thread; we exit.
            std::exit(0);
        }
    }
}

// ---- main -----------------------------------------------------------------
int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    if (args.show_help) {
        usage(argv[0]);
        return 0;
    }
    if (args.show_version) {
        print_version();
        return 0;
    }
    if (args.show_abi) {
        std::printf("%d\n", starling_ggml_abi_version());
        return 0;
    }
    if (args.error) {
        usage(argv[0]);
        return 1;
    }
    // Validate the stream window configuration BEFORE the model is loaded
    // (issue #146): out-of-range values used to surface much later, when the
    // chunker derived a negative-length transcription window mid-session.
    // (The stream flags already passed the strict finite-number parse above.)
    if (auto err = serve::stream_window_config_error(
            serve::kSampleRate, args.stream_chunk, args.stream_overlap,
            args.min_chunk, args.partial_interval);
        !err.empty()) {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    if (args.max_stream_seconds < 0.0) {
        std::fprintf(stderr,
            "error: --max-stream-seconds must be nonnegative (0 = unlimited)\n");
        return 1;
    }
    if (args.model.empty() || args.gguf.empty()) {
        std::fprintf(stderr, "error: --model and --gguf are required\n");
        usage(argv[0]);
        return 1;
    }
    if (!serve::is_supported_model(args.model)) {
        std::fprintf(stderr,
            "error: unsupported model '%s'. Supported: %s\n",
            args.model.c_str(), serve::supported_models_str().c_str());
        return 1;
    }

    // Verify GGUF exists.
    {
        std::ifstream f(args.gguf, std::ios::binary);
        if (!f) {
            std::fprintf(stderr, "error: cannot open GGUF file: %s\n",
                         args.gguf.c_str());
            return 1;
        }
    }

    if (!args.eager_load && args.warmup) {
        std::fprintf(stderr,
            "error: --warmup requires eager loading (incompatible with --no-eager-load)\n");
        return 1;
    }

    // Build config.
    serve::ServerConfig cfg;
    cfg.model_slug = args.model;
    cfg.gguf_path = args.gguf;
    cfg.host = args.host;
    cfg.port = args.port;
    cfg.warmup = args.warmup;
    cfg.eager_load = args.eager_load;
    cfg.stream_chunk_seconds = args.stream_chunk;
    cfg.stream_overlap_seconds = args.stream_overlap;
    cfg.min_chunk_seconds = args.min_chunk;
    cfg.partial_interval = args.partial_interval;
    cfg.max_stream_seconds = args.max_stream_seconds;
    cfg.request_timeout_seconds = args.request_timeout;

    // shared_ptr: the detached /warmup worker captures it and must outlive
    // main()'s reference if a warmup is still running at shutdown.
    auto server = std::make_shared<serve::StarlingServer>(cfg);

    // Eager-load the model.
    if (cfg.eager_load) {
        server->load();
        if (!server->loaded()) {
            std::fprintf(stderr, "error: failed to load model\n");
            return 1;
        }
        if (cfg.warmup) {
            server->warmup();
        }
    }

    // Start idle-timeout monitor (only if timeout > 0).
    std::thread idle_thread;
    if (args.idle_timeout > 0.0) {
        idle_thread = std::thread(idle_timeout_thread, server.get(), args.idle_timeout);
    }

    g_last_activity.store(std::time(nullptr));

    // Build HTTP server.
    httplib::Server svr;

    // Enforce upload size limit at the HTTP layer.
    svr.set_payload_max_length(
        static_cast<size_t>(server->config().max_upload_mb) * 1024 * 1024);

    // ---- GET /health ----
    svr.Get("/health",
        [&server](const httplib::Request&, httplib::Response& res) {
            g_last_activity.store(std::time(nullptr));
            send_json(res, server->health_json());
        });

    // ---- GET / (health alias) ----
    svr.Get("/",
        [&server](const httplib::Request&, httplib::Response& res) {
            g_last_activity.store(std::time(nullptr));
            send_json(res, server->health_json());
        });

    // ---- POST /warmup ----
    svr.Post("/warmup",
        [&server](const httplib::Request&, httplib::Response& res) {
            g_last_activity.store(std::time(nullptr));
            // Fire warmup asynchronously (it's idempotent — deduped
            // internally). One worker at a time: a client spamming /warmup
            // must not spawn unbounded threads.
            if (!g_warmup_running.exchange(true)) {
                // Capture the shared_ptr by value: the detached worker must
                // keep the server alive for the whole warmup() call even if
                // main() returns and resets its own reference.
                std::thread([server]() {
                    server->warmup();
                    g_warmup_running.store(false);
                }).detach();
            }
            std::ostringstream ss;
            ss << "{\"status\":\"warmup started\",\"phase\":\""
               << (server->phase() == serve::Phase::Ready ? "ready" : "loading")
               << "\"}";
            send_json(res, ss.str(), 202);
        });

    // ---- shared transcribe handler ----
    auto handle_transcribe = [&server](
            const httplib::Request& req, httplib::Response& res) {
        g_last_activity.store(std::time(nullptr));

        // Get request ID from header.
        std::string rid = req.get_header_value("x-request-id");
        if (rid.empty()) rid = req.get_header_value("x-correlation-id");
        if (rid.empty()) {
            // Generate a UUID-like ID.
            std::ostringstream ss;
            ss << std::hex << std::time(nullptr) << "-"
               << std::this_thread::get_id();
            rid = ss.str();
        } else if (rid[0] == '#') {
            // '#' prefixes the server's internal anonymous queue tickets;
            // a client using one could collide with them.
            send_json(res, "{\"error\":\"invalid request id\"}", 400);
            return;
        }

        // Extract the audio payload FIRST: cpp-httplib parses
        // multipart/form-data bodies itself into req.form (req.body stays
        // empty for them), so the body-empty check below must not run before
        // the form has been consulted. Selection order mirrors
        // audio::extract_multipart_payload (the manual parser in audio.cpp —
        // kept as the Python-parity reference; this is its production twin):
        // a part named "audio", then one named "file", then any file part,
        // then a filename-less field named
        // "audio"/"file" (httplib routes parts without a filename to
        // form.fields; the Python server accepts those too). Raw bodies
        // carry the bytes directly.
        std::string payload;
        bool is_multipart = req.is_multipart_form_data();
        if (is_multipart) {
            // Every branch requires non-empty content: an empty "audio" part
            // must not shadow a populated "file" part (same rule as the
            // parity parser, which skips empty parts when scoring).
            if (req.form.has_file("audio")
                && !req.form.get_file("audio").content.empty()) {
                payload = req.form.get_file("audio").content;
            } else if (req.form.has_file("file")
                       && !req.form.get_file("file").content.empty()) {
                payload = req.form.get_file("file").content;
            } else {
                for (const auto& [name, file] : req.form.files) {
                    (void)name;
                    if (!file.content.empty()) {
                        payload = file.content;
                        break;
                    }
                }
            }
            // Filename-less parts ("audio"/"file" sent as plain form fields,
            // e.g. curl -F 'audio=<clip.wav' or files={"audio": (None,
            // data)}) live in form.fields, not form.files.
            if (payload.empty()) {
                if (!req.form.get_field("audio").empty()) {
                    payload = req.form.get_field("audio");
                } else if (!req.form.get_field("file").empty()) {
                    payload = req.form.get_field("file");
                }
            }
        } else {
            payload = req.body;
        }

        // Check payload size.
        size_t max_bytes = static_cast<size_t>(server->config().max_upload_mb) * 1024 * 1024;
        if (payload.size() > max_bytes) {
            send_json(res, "{\"error\":\"request body too large\"}", 413);
            return;
        }
        if (payload.empty()) {
            send_json(res, "{\"error\":\"empty request body\"}", 400);
            return;
        }

        // Decode audio.
        std::vector<float> samples;
        int sr = 0;
        bool looks_like_wav = payload.size() >= 12
            && (payload.compare(0, 4, "RIFF") == 0
                || payload.compare(0, 4, "RF64") == 0
                || payload.compare(0, 4, "RIFX") == 0)
            && payload.compare(8, 4, "WAVE") == 0;
        bool decoded = serve::audio::wav_bytes_to_float32(payload, samples, sr);
        if (!decoded) {
            // A payload with a RIFF/WAVE magic that fails WAV decoding (e.g. a
            // header claiming more frames than the payload holds, or a
            // truncated data chunk) is malformed: fail fast with 400 rather
            // than reinterpreting header bytes as raw PCM16.
            if (looks_like_wav) {
                send_json(res, "{\"error\":\"malformed audio payload\",\"text\":\"\"}", 400);
                return;
            }
            // Try raw PCM16.
            samples = serve::audio::pcm16_to_float32(payload);
            sr = 16000;
            if (samples.empty()) {
                send_json(res, "{\"error\":\"malformed audio payload\",\"text\":\"\"}", 400);
                return;
            }
        }
        if (sr != 0 && sr != serve::kSampleRate) {
            // The engine expects 16 kHz; there is no C++ resampler (the
            // Python server resamples via scipy). Reject non-16 kHz uploads.
            std::ostringstream ss;
            ss << "{\"error\":\"sample rate mismatch: expected "
               << serve::kSampleRate << " got " << sr << "\"}";
            send_json(res, ss.str(), 400);
            return;
        }

        // Register for cancellation.
        auto* ctx = server->register_request(rid);
        if (!ctx) {
            send_json(res,
                R"({"error":"request id already active","text":"","request_id":")"
                + json_escape(rid) + "\"}",
                409);
            return;
        }

        // Run transcription.
        std::string err;
        auto result = server->transcribe_pcm(
            samples.data(), static_cast<int64_t>(samples.size()), ctx, &err);

        server->finish_request(ctx);

        if (err == "server busy") {
            std::ostringstream ss;
            ss << "{\"error\":\"server busy\",\"text\":\"\",\"queue_depth\":"
               << server->queue_depth() << ",\"request_id\":\"" << json_escape(rid) << "\"}";
            send_json(res, ss.str(), 503);
            return;
        }
        if (err == "cancelled") {
            send_json(res,
                R"({"error":"cancelled","text":"","request_id":")" + json_escape(rid) + "\"}",
                499);
            return;
        }
        if (err == "request timed out") {
            send_json(res,
                R"({"error":"request timed out","text":"","request_id":")" + json_escape(rid) + "\"}",
                504);
            return;
        }
        if (err == "model not loaded") {
            send_json(res,
                R"({"error":"model not loaded","text":"","request_id":")" + json_escape(rid) + "\"}",
                503);
            return;
        }
        if (!err.empty()) {
            std::ostringstream ss;
            ss << "{\"error\":\"" << json_escape(err) << "\",\"text\":\"\",\"request_id\":\""
               << json_escape(rid) << "\"}";
            send_json(res, ss.str(), 500);
            return;
        }

        // Success.
        std::string json = result.to_json();
        // Insert request_id before closing brace.
        json = json.substr(0, json.size() - 1) +
               ",\"request_id\":\"" + json_escape(rid) + "\"}";
        send_json(res, json, 200);
    };

    svr.Post("/transcribe", handle_transcribe);
    svr.Post("/inference", handle_transcribe);

    // OpenAI-compatible batch transcription subset. Keep validation separate
    // from legacy routes: unsupported options must never look as if they worked.
    svr.Get("/v1/models", [&server](const httplib::Request&, httplib::Response& res) {
        g_last_activity.store(std::time(nullptr));
        send_json(res, "{\"object\":\"list\",\"data\":[{\"id\":\""
            + json_escape(server->model_slug())
            + "\",\"object\":\"model\",\"created\":0,\"owned_by\":\"starling\"}]}");
    });
    svr.Get("/v1/starling/capabilities", [&server](const httplib::Request&, httplib::Response& res) {
        g_last_activity.store(std::time(nullptr));
        send_json(res, "{\"schema_version\":1,\"model\":\"" + json_escape(server->model_slug())
            + "\",\"audio_transcription\":" + (server->is_text_model() ? "false" : "true")
            + ",\"audio_formats\":[\"wav\"],\"sample_rate_hz\":16000,"
              "\"response_formats\":[\"json\",\"text\"],\"prompt\":false,"
              "\"language_selection\":false,\"word_timestamps\":false,"
              "\"streaming_transcriptions\":false,\"legacy_websocket_path\":\"/stream\"}");
    });
    svr.Post("/v1/audio/transcriptions", [&server, &handle_transcribe](
            const httplib::Request& req, httplib::Response& res) {
        g_last_activity.store(std::time(nullptr));
        auto fail = [&res](const std::string& message, const std::string& param, int status = 400) {
            send_json(res, "{\"error\":{\"message\":\"" + json_escape(message)
                + "\",\"type\":\"" + (status >= 500 ? "server_error" : "invalid_request_error")
                + "\",\"param\":" + (param.empty() ? "null" : "\"" + json_escape(param) + "\"")
                + ",\"code\":null}}", status);
        };
        if (!req.is_multipart_form_data()) {
            fail("Expected multipart/form-data with file and model", "file");
            return;
        }
        if (req.form.get_field_count("model") != 1 || req.form.get_field("model").empty()) {
            fail("Exactly one model field is required; use GET /v1/models", "model");
            return;
        }
        if (req.form.get_field("model") != server->model_slug()) {
            fail("Requested model is not served by this process; use GET /v1/models", "model", 404);
            return;
        }
        if (server->is_text_model()) {
            fail("This model accepts text, not audio", "model");
            return;
        }
        for (const auto& [name, field] : req.form.fields) {
            const auto& value = field.content;
            if (req.form.get_field_count(name) != 1) {
                fail("Duplicate field", name);
                return;
            }
            if (name == "model" || name == "response_format") continue;
            if (name == "stream" && value == "false") continue;
            if (name == "temperature" && (value == "0" || value == "0.0")) continue;
            fail("Unsupported transcription option; see /v1/starling/capabilities", name);
            return;
        }
        const std::string format = req.form.get_field("response_format");
        if (!format.empty() && format != "json" && format != "text") {
            fail("Supported response formats are json and text", "response_format");
            return;
        }
        if (req.form.files.size() != 1 || !req.form.has_file("file")) {
            fail("Exactly one audio file named file is required", "file");
            return;
        }
        const auto& payload = req.form.get_file("file").content;
        if (payload.size() < 12 || payload.compare(8, 4, "WAVE") != 0
            || (payload.compare(0, 4, "RIFF") != 0 && payload.compare(0, 4, "RF64") != 0)) {
            fail("This backend accepts 16 kHz WAV files; convert compressed audio before upload", "file");
            return;
        }
        handle_transcribe(req, res);
        if (res.status != 200) {
            std::string message;
            if (!json_get_string(res.body, "error", message)) message = "Transcription failed";
            fail(message, "", res.status);
            return;
        }
        std::string text, request_id;
        if (!json_get_string(res.body, "text", text)) {
            fail("Invalid internal transcription response", "", 500);
            return;
        }
        if (json_get_string(res.body, "request_id", request_id)) res.set_header("X-Request-Id", request_id);
        if (format == "text") res.set_content(text, "text/plain; charset=utf-8");
        else send_json(res, "{\"text\":\"" + json_escape(text) + "\"}");
    });

    // ---- POST /normalize (text models: s1) ----
    // Body: JSON {"transcript": "...", "styling": "...?, "structure": "...?,
    //             "context": "...?"} — control fields optional (trained
    // defaults). Response: {"text": "...", "request_id": "..."}.
    svr.Post("/normalize", [&server](
            const httplib::Request& req, httplib::Response& res) {
        g_last_activity.store(std::time(nullptr));
        if (!server->is_text_model()) {
            send_json(res,
                "{\"error\":\"model has no text path (audio models use /transcribe)\"}",
                400);
            return;
        }
        if (req.body.empty()) {
            send_json(res, "{\"error\":\"empty request body\"}", 400);
            return;
        }
        size_t max_bytes = static_cast<size_t>(server->config().max_upload_mb) * 1024 * 1024;
        if (req.body.size() > max_bytes) {
            send_json(res, "{\"error\":\"request body too large\"}", 413);
            return;
        }
        std::string transcript, styling, structure, context;
        if (!json_get_string(req.body, "transcript", transcript)) {
            send_json(res, "{\"error\":\"missing or malformed 'transcript' field\"}", 400);
            return;
        }
        json_get_string(req.body, "styling", styling);    // optional
        json_get_string(req.body, "structure", structure);
        json_get_string(req.body, "context", context);

        std::string rid = req.get_header_value("x-request-id");
        if (rid.empty()) rid = req.get_header_value("x-correlation-id");
        if (rid.empty()) {
            std::ostringstream ss;
            ss << std::hex << std::time(nullptr) << "-"
               << std::this_thread::get_id();
            rid = ss.str();
        } else if (rid[0] == '#') {
            send_json(res, "{\"error\":\"invalid request id\"}", 400);
            return;
        }

        auto* ctx = server->register_request(rid);
        if (!ctx) {
            send_json(res,
                R"({"error":"request id already active","text":"","request_id":")"
                + json_escape(rid) + "\"}",
                409);
            return;
        }

        std::string err;
        auto text = server->normalize_text(
            transcript, styling, structure, context, ctx, &err);
        server->finish_request(ctx);

        if (err == "server busy") {
            std::ostringstream ss;
            ss << "{\"error\":\"server busy\",\"text\":\"\",\"queue_depth\":"
               << server->queue_depth() << ",\"request_id\":\"" << json_escape(rid) << "\"}";
            send_json(res, ss.str(), 503);
            return;
        }
        if (err == "cancelled") {
            send_json(res,
                R"({"error":"cancelled","text":"","request_id":")" + json_escape(rid) + "\"}",
                499);
            return;
        }
        if (err == "request timed out") {
            send_json(res,
                R"({"error":"request timed out","text":"","request_id":")"
                + json_escape(rid) + "\"}",
                504);
            return;
        }
        if (err == "model not loaded") {
            send_json(res,
                R"({"error":"model not loaded","text":"","request_id":")"
                + json_escape(rid) + "\"}",
                503);
            return;
        }
        if (!err.empty()) {
            std::ostringstream ss;
            ss << "{\"error\":\"" << json_escape(err)
               << "\",\"text\":\"\",\"request_id\":\"" << json_escape(rid) << "\"}";
            send_json(res, ss.str(), 400);
            return;
        }

        std::ostringstream ss;
        ss << "{\"text\":\"" << json_escape(text)
           << "\",\"request_id\":\"" << json_escape(rid) << "\"}";
        send_json(res, ss.str(), 200);
    });


    // ---- DELETE /inference/<id> ----
    svr.Delete(R"(/inference/(.*))",
        [&server](const httplib::Request& req, httplib::Response& res) {
            g_last_activity.store(std::time(nullptr));
            std::string rid = req.matches.size() > 1
                ? std::string(req.matches[1]) : "";
            if (rid.empty()) {
                send_json(res, "{\"error\":\"missing request id\"}", 400);
                return;
            }
            bool cancelled = server->cancel_request(rid);
            std::ostringstream ss;
            ss << "{\"status\":\"" << (cancelled ? "cancelled" : "not_found")
               << "\",\"request_id\":\"" << json_escape(rid) << "\"}";
            send_json(res, ss.str(), cancelled ? 200 : 404);
        });

    // ---- WS /stream ----
    svr.WebSocket("/stream",
        [&server, &cfg](const httplib::Request&,
                        httplib::ws::WebSocket& ws) {
            serve::StreamSession session(server.get());
            // Sent once when a binary frame is refused (buffer cap, malformed
            // WAV, sample-rate mismatch, odd PCM length); re-armed on reset
            // so a fresh dictation gets a fresh error if it is refused.
            bool reject_error_sent = false;
            std::fprintf(stderr, "[starling-serve] WS /stream client connected\n");

            std::string msg;
            while (ws.is_open()) {
                auto rr = ws.read(msg);
                if (rr == httplib::ws::ReadResult::Fail) break;
                // Every received message counts as activity so the idle
                // timeout can't fire mid-dictation (it only checks between
                // transcribes).
                g_last_activity.store(std::time(nullptr));

                if (rr == httplib::ws::ReadResult::Text) {
                    // Parse JSON command.
                    // Minimal JSON parsing: look for "type":"<value>".
                    // (A full JSON parser is overkill for 3 message types.)
                    std::string type;
                    {
                        // Find "type" key.
                        size_t pos = msg.find("\"type\"");
                        if (pos != std::string::npos) {
                            size_t colon = msg.find(':', pos);
                            if (colon != std::string::npos) {
                                size_t q1 = msg.find('"', colon + 1);
                                if (q1 != std::string::npos) {
                                    size_t q2 = msg.find('"', q1 + 1);
                                    if (q2 != std::string::npos)
                                        type = msg.substr(q1 + 1, q2 - q1 - 1);
                                }
                            }
                        }
                    }

                    if (type == "commit") {
                        // An invalidated take (a rejected binary frame) holds
                        // incomplete audio: committing it as an ordinary
                        // successful final would silently miss speech, so the
                        // commit is refused and the client falls back to its
                        // authoritative local WAV after a reset (issue #145).
                        // The busy-retry path below is untouched: it retains
                        // VALID audio, while this path refuses INVALID audio.
                        if (session.take_invalid()) {
                            std::ostringstream ss;
                            ss << "{\"type\":\"error\",\"message\":\"take "
                               << "invalidated (" << session.invalid_reason()
                               << "); reset and resend\"}";
                            ws.send(ss.str());
                            continue;
                        }
                        double dur = session.buffered_seconds();
                        std::string text;
                        if (dur > 0.0) {
                            auto final = session.stream_flush();
                            if (!final.has_value()) {
                                ws.send("{\"type\":\"error\",\"message\":\"server busy\"}");
                                continue;
                            }
                            text = *final;
                        }
                        std::string safe_text = json_escape(text);
                        std::ostringstream ss;
                        ss << "{\"type\":\"final\",\"text\":\""
                           << safe_text << "\",\"segments\":[{\"text\":\""
                           << safe_text << "\",\"start_s\":0.0,\"end_s\":"
                           << dur << "}],\"duration_s\":" << dur << "}";
                        ws.send(ss.str());
                        session.reset();
                        // reset() re-enables audio (clears the buffer cap and
                        // any take invalidation); re-arm the one-shot error
                        // frame with it.
                        reject_error_sent = false;
                        continue;
                    } else if (type == "ping") {
                        ws.send("{\"type\":\"pong\"}");
                        continue;
                    } else if (type == "reset") {
                        session.reset();
                        reject_error_sent = false;
                        ws.send("{\"type\":\"reset_ack\"}");
                        continue;
                    } else {
                        std::ostringstream ss;
                        ss << "{\"type\":\"error\",\"message\":\"unknown type '"
                           << json_escape(type) << "'\"}";
                        ws.send(ss.str());
                        continue;
                    }
                }

                if (rr == httplib::ws::ReadResult::Binary) {
                    // Audio data. Enforce the per-connection buffer cap
                    // (--max-stream-seconds) and the frame-validity policy
                    // (issue #145): a refused frame is reported once as an
                    // error frame, and the session stops accepting audio
                    // until it is reset.
                    serve::AppendOutcome outcome = serve::AppendOutcome::Accepted;
                    if (!session.overflowed() && !session.take_invalid()) {
                        if (msg.size() >= 12 && msg.substr(0, 4) == "RIFF"
                            && msg.substr(8, 4) == "WAVE") {
                            outcome = session.append_wav(msg);
                        } else {
                            outcome = session.append_pcm(msg);
                        }
                    } else if (session.take_invalid()) {
                        outcome = serve::AppendOutcome::TakeInvalid;
                    } else {
                        outcome = serve::AppendOutcome::Overflowed;
                    }
                    if (outcome != serve::AppendOutcome::Accepted) {
                        if (!reject_error_sent) {
                            reject_error_sent = true;
                            ws.send(ws_append_error(
                                outcome, session, cfg.max_stream_seconds));
                        }
                        continue;
                    }

                    double now = static_cast<double>(
                        std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now().time_since_epoch()
                        ).count()) / 1000.0;

                    auto text_opt = session.stream_step(now);
                    if (text_opt.has_value()) {
                        std::string safe_text = json_escape(*text_opt);
                        double dur = session.buffered_seconds();
                        std::ostringstream ss;
                        ss << "{\"type\":\"partial\",\"text\":\""
                           << safe_text << "\",\"segments\":[{\"text\":\""
                           << safe_text << "\",\"start_s\":0.0,\"end_s\":"
                           << dur << "}],\"start_s\":0.0,\"end_s\":" << dur << "}";
                        ws.send(ss.str());
                    }
                }
            }
            std::fprintf(stderr,
                "[starling-serve] WS /stream client disconnected\n");
        });

    // ---- WebSocket heartbeat (detect dead connections) ----
    svr.set_websocket_ping_interval(30);  // send a ping every 30s
    svr.set_websocket_max_missed_pongs(3);  // close after 3 missed pongs (90s)

    // ---- start serving ----
    std::fprintf(stderr,
        "[starling-serve] starting on %s:%d (model=%s, backend=%s, abi=%d)\n",
        cfg.host.c_str(), cfg.port, cfg.model_slug.c_str(),
        starling_ggml_backend_name(), starling_ggml_abi_version());

    if (cfg.host != "127.0.0.1" && cfg.host != "localhost" && cfg.host != "::1") {
        std::fprintf(stderr,
            "[starling-serve] WARNING: binding unauthenticated ASR endpoints to %s\n",
            cfg.host.c_str());
    }

    if (!svr.listen(cfg.host.c_str(), cfg.port)) {
        std::fprintf(stderr, "[starling-serve] failed to bind %s:%d\n",
                     cfg.host.c_str(), cfg.port);
        return 1;
    }

    g_should_exit.store(true);
    if (idle_thread.joinable()) idle_thread.join();
    server.reset();  // destroy StarlingServer (calls starling_ggml_free + shutdown)
    return 0;
}
