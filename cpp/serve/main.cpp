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
        "  --model <slug>     Model slug: parakeet, moss, ark, higgs, hojo, granite\n"
        "  --gguf <path>      Path to the GGUF model file\n"
        "\n"
        "Serving:\n"
        "  --host <addr>      Bind address (default 127.0.0.1)\n"
        "  --port <n>         Bind port (default 8181)\n"
        "  --warmup           Warm up CUDA graphs on startup\n"
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
        prog);
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
            try { return std::stoi(v); }
            catch (...) {
                std::fprintf(stderr, "error: %s requires an integer, got '%s'\n", name, v.c_str());
                a.error = true;
                return 0;
            }
        };
        auto next_double = [&](const char* name) -> double {
            std::string v = next(name);
            if (a.error) return 0.0;
            try { return std::stod(v); }
            catch (...) {
                std::fprintf(stderr, "error: %s requires a number, got '%s'\n", name, v.c_str());
                a.error = true;
                return 0.0;
            }
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
            // Sent once when the per-connection buffer cap trips; re-armed on
            // reset so a fresh dictation gets a fresh error if it overflows.
            bool cap_error_sent = false;
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
                        double dur = session.buffered_seconds();
                        std::string text;
                        if (dur > 0.0) {
                            text = session.stream_flush();
                        }
                        std::string safe_text = json_escape(text);
                        std::ostringstream ss;
                        ss << "{\"type\":\"final\",\"text\":\""
                           << safe_text << "\",\"segments\":[{\"text\":\""
                           << safe_text << "\",\"start_s\":0.0,\"end_s\":"
                           << dur << "}],\"duration_s\":" << dur << "}";
                        ws.send(ss.str());
                        session.reset();
                        // reset() re-enables audio (clears the buffer cap);
                        // re-arm the one-shot error frame with it.
                        cap_error_sent = false;
                        continue;
                    } else if (type == "ping") {
                        ws.send("{\"type\":\"pong\"}");
                        continue;
                    } else if (type == "reset") {
                        session.reset();
                        cap_error_sent = false;
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
                    // (--max-stream-seconds): a frame that would exceed it is
                    // refused, reported once as an error frame, and the
                    // session stops accepting audio until it is reset.
                    if (!session.overflowed()) {
                        if (msg.size() >= 12 && msg.substr(0, 4) == "RIFF"
                            && msg.substr(8, 4) == "WAVE") {
                            session.append_wav(msg);
                        } else {
                            session.append_pcm(msg);
                        }
                    }
                    if (session.overflowed()) {
                        if (!cap_error_sent) {
                            cap_error_sent = true;
                            std::ostringstream ss;
                            ss << "{\"type\":\"error\",\"message\":\"stream buffer"
                               << " limit reached (" << cfg.max_stream_seconds
                               << " s live buffer); audio ignored until"
                               << " reset\"}";
                            ws.send(ss.str());
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
