# Server capability descriptor and cross-client conformance fixtures (E10)

`capability.json` is the canonical, versioned descriptor of what the CURRENT
native `starling-serve` actually implements; `capability.schema.json` is its
JSON Schema; `fixtures/` holds request/response conformance pairs for the
OpenAI-compatible subset and the native routes, including the representative
error ladder. Nothing here is aspirational: an unimplemented capability is
absent or explicitly `false`, and deployment-tunable defaults are modelled as
`{value, cli_flag}` settings so a client can tell a fixed limit from a default.

The executable check is `tests/test_capabilities_conformance.py` (CPU-only;
schema validation via `jsonschema` when installed, else the repo's
`tests/minischema.py` fallback). It validates the descriptor and fixtures,
enforces cross-field consistency (every endpoint has fixtures, every fixture
route is a documented server route, audio limits agree), pins the mapping to
the live capability route, and re-verifies the adoption map below against the
actual client code.

## The contract in one paragraph

One model is resident per process; inference is serial with a bounded waiter
queue; batch transcription accepts 16 kHz WAV (OpenAI subset: multipart
`file`+`model`, `json`/`text` responses) plus raw PCM16 on the legacy routes;
live dictation is Starling's own `WS /stream` protocol (fixed overlapping
windows, commit/reset/ping controls, live-buffer cap, take invalidation) and
is NOT OpenAI Realtime. Unsupported options (prompt, language, word
timestamps, nonzero temperature, streaming on the OpenAI route, verbose_json,
SRT/VTT, diarization) are refused with 400, never silently ignored. Errors
carry the OpenAI envelope on `/v1/*` and a flat string envelope on legacy
routes; the HTTP parser's own non-JSON 413 is possible, so clients must check
status first.

## Evidence base (where each field's truth comes from)

All paths relative to the repo root; line numbers verified on branch
`program/wave-a` (baseline of E10: `bec01da`).

| Descriptor field | Truth source |
| --- | --- |
| `schema_version: 1` | live route body, cpp/serve/main.cpp:748-756; openapi.json `/v1/starling/capabilities` |
| endpoints (all 11 routes) | cpp/serve/main.cpp:529 (GET /health), 536 (GET / alias), 543 (POST /warmup), 737-738 (POST /transcribe, /inference), 742 (GET /v1/models), 748 (GET /v1/starling/capabilities), 757 (POST /v1/audio/transcriptions), 830 (POST /normalize), 926 (DELETE /inference/{id}), 943 (WS /stream) |
| `audio.sample_rates_hz: [16000]` | cpp/serve/server.hpp:42 (`kSampleRate`); rejection at main.cpp:669-677 |
| `audio.resampling: false` | main.cpp:669-677 comment ("no C++ resampler"); docs/api.md:84-87 |
| `audio.channel_handling` (64-ch mixdown) | cpp/serve/audio.cpp:23 (`kWavMaxChannels`), 69-80 |
| `audio.max_upload_mb: 256` | server.hpp:50 (`kMaxUploadMB`); enforced main.cpp:525-526, 633-637; no native CLI override flag (main.cpp Args) |
| `audio.batch` (wav magic per route, pcm16 fallback, `file` part name) | main.cpp:646-667 (legacy), 804-808 (OpenAI subset), 799-802 (file part) |
| `audio.live` (wav/pcm16 frames, whole samples) | main.cpp:1039-1057; stream_session.hpp:139-148 |
| `models` (one resident, created 0, 404 on other slug) | main.cpp:742-747, 774-777; docs/api.md:17-19 |
| `responses` (json/text; legacy shape) | main.cpp:794-798, 822-823; server.cpp:107-119; openapi.json |
| `streaming.chunking` defaults (12/3/5/3 s) | main.cpp:97-100 (Args defaults), server.hpp:46-47 |
| `streaming.finalization` (busy retains audio; commit refused on invalid take) | main.cpp:1002-1006, 991-998; stream_session.hpp:100-101 |
| `streaming.live_buffer_cap` (60 s, one error frame, reset) | main.cpp:96, 222-225, 1046-1065; server.hpp:53; tests/test_native_serve.py:test_websocket_stream_cap |
| `streaming.take_invalidation` reasons | stream_session.hpp:168-170; main.cpp:205-238 (error frames); docs/native-serving.md#ws-stream |
| `streaming.heartbeat` (30 s ping, 3 missed pongs) | main.cpp:1090-1091 |
| `limits.max_queue_waiters: 8` | server.hpp:49 (`kMaxWaiters`); 503 at main.cpp:696-701 |
| `limits.request_timeout_seconds` (600 default, 0 = never) | main.cpp:66-67, 95; server.hpp:51; 504 at main.cpp:709-714 |
| `limits.serial_inference` | server.cpp:263-412 (serial queue); server.hpp:176-180 |
| `request_ids` (header, fallback, `#` reserved, echo, cancel) | main.cpp:571-584, 821, 926-940 |
| `error_contract` (envelopes, status ladder, non-JSON 413) | main.cpp:760-765 (envelope), 635-727 (ladder); docs/api.md:53-56; openapi.json responses |
| `features.*` (all false flags) | live route main.cpp:752-755 (prompt/language/word_timestamps/streaming); docs/api.md:47-51 (verbose_json, SRT/VTT, diarization, temperature) |
| `features.authentication: false` | docs/api.md:30-32 ("the local Starling server does not check it") |
| `features.audio_transcription` / `text_normalization` split | main.cpp:750-751, 778-781, 833-837 |

Fixture bodies come from the live contract tests, not invention: the
deterministic fixture-engine transcript and the full OpenAI error ladder from
`backends/native/tests/test_openai_api.py`; transport behaviors (health,
warmup, 400/404/409/413/499/503/504, PCM16 fallback, WS controls, buffer
cap, invalid-audio rejection, disconnect) from `tests/test_native_serve.py`;
exact JSON strings from the handler code cited above.

## Mapping to the live `/v1/starling/capabilities` route

The server already serves a partial capability shape (schema_version 1). The
descriptor is the superset; the projection below is enforced by
`tests/test_capabilities_conformance.py` against
`fixtures/starling_capabilities_route.json`:

| Live route field | Descriptor field |
| --- | --- |
| `schema_version` | `schema_version` |
| `model` | `models.served[0].id` |
| `audio_transcription` | `features.audio_transcription` |
| `audio_formats` | `audio.batch.openai_subset.containers` |
| `sample_rate_hz` | `audio.sample_rates_hz[0]` |
| `response_formats` | `responses.openai_subset_formats` |
| `prompt` | `features.prompt` |
| `language_selection` | `features.language_selection` |
| `word_timestamps` | `features.word_timestamps` |
| `streaming_transcriptions` | `features.server_sent_transcription_events` |
| `legacy_websocket_path` | `streaming.path` |

A follow-up may extend the live route towards the full descriptor; until
then clients that need more than the subset above must keep reading
`/health`, `/v1/models`, and this package's versioned files.

## Adoption map — who consumes which field today vs should

Today-state verified by grepping the actual client code (re-verified by
`tests/test_capabilities_conformance.py`). **No client fetches
`/v1/starling/capabilities` today**; every consumer hardcodes its assumptions
instead. Wiring the clients to consume the descriptor is follow-up work (next
section), deliberately not part of this contract drop.

| Capability field | Electron (TS) today | Rust today | Python backend today | Should |
| --- | --- | --- | --- | --- |
| endpoints batch `/v1/audio/transcriptions` | hardcoded ternary: packages/dictation/src/client.ts:436-437, apps/desktop/electron/main.ts:365-366 | not present in this snapshot (no `apps/desktop-gpui`, no `.rs` outside third_party; E10's 20 Sep revision defers the Rust client to PR #193) | not served by the deprecated backend (entry point backends/python/serve.py → src/starling/server.py:1468-1673, legacy routes only; docs/api.md:70-72) | derive route from descriptor `endpoints` |
| endpoints live `WS /stream` | hardcoded default path: packages/dictation/src/streaming.ts:210 | — | serves it (src/starling/server.py:1673) | read `streaming.path` |
| `models` (one resident; created 0; 404 otherwise) | free-text model field, required for the openai protocol: packages/dictation/src/client.ts:438,460-462; hardcoded `"parakeet"` fallback: apps/desktop/electron/main.ts:369; probe via `/v1/models`: client.ts:511, main.ts:318 | — | one configured backend, no listing route | populate model picker from `/v1/models` + descriptor; state honestly that selecting a slug cannot load another model |
| `audio.sample_rates_hz` / `resampling: false` | resamples client-side to 16 kHz: packages/dictation/src/audio.ts:3,102 | — | resamples server-side via scipy (docs/native-serving.md:114-117) | surface "server rejects other rates" instead of guessing; keep client-side 16 kHz prep |
| `audio.max_upload_mb` | hardcoded duplicate cap: apps/desktop/electron/main.ts:42 (`maximumAudioBytes = 256 * 1024 * 1024`) | — | `--max-upload-mb` CLI: src/starling/server.py:1893 | read the limit; explain refusals, don't silently trim |
| `streaming.chunking` / `live_buffer_cap` / `take_invalidation` | commit/reset/ping implemented: packages/dictation/src/streaming.ts:490-509; cap and invalidation handling present (fidelity.ts/streaming.ts tests) but timings are not fetched from the server | — | same WS contract; resampling divergence (docs/python-serving.md) | disable unsupported controls with an explanation; show reset rule from `live_buffer_cap` |
| `limits` (waiters 8, timeout 600 s) | client timeout hardcoded 120 s: packages/dictation/src/client.ts:441, streaming.ts:167 | — | MAX_WAITERS 8: src/starling/server.py:109; upload default 256 MiB: :114 | align client timeouts and busy/backoff UI with server limits |
| `error_contract` (status ladder, non-JSON 413) | tolerant decode (string or nested error): packages/dictation/src/client.ts:107-113,254-269; status checked first | — | FastAPI detail errors | keep status-first handling; map 499/503/504 to user-facing retry guidance from the descriptor's `status_codes` |
| `features.*` false flags (language, prompt, timestamps, streaming) | not offered in UI; not derived from any descriptor | — | not implemented | disable with the descriptor's reason instead of hardcoding |
| `request_ids` (X-Request-Id, `#` reserved, cancel) | generated + validated client-side: client.ts:234-244; cancel DELETE: client.ts:549 | — | legacy only | unchanged once server-side; `#` rule already mirrored |

Kotlin/Swift (apps/mobile, apps/ios) use their native HTTP APIs against the
same contract; they are outside this snapshot's verified grep scope.

## Caveat and follow-up work

This package is the contract and the fixtures only. Adopting it in the
clients is separate, explicitly listed work:

1. Electron shared client: fetch and validate the capability descriptor (or
   the live subset) at startup; drive endpoint selection, the model picker,
   and the unsupported-option disable/reason UI from it
   (packages/dictation/src/client.ts).
2. Electron app: replace the hardcoded `maximumAudioBytes` and `"parakeet"`
   fallback with descriptor/`/v1/models` values
   (apps/desktop/electron/main.ts:42,369).
3. Extend the live `/v1/starling/capabilities` response towards this
   descriptor (server-side change; keep `schema_version` bump discipline and
   update the mapping table above plus its enforcing test).
4. Rust client (GPUI): adopt the same fixtures when it lands (E10's 20 Sep
   revision, PR #193); reuse these files rather than forking wire shapes.
5. Run these fixtures as behavioral conformance suites against a live server
   (extend tests/test_native_serve.py / backends/native/tests) — the checks
   here are static file/JSON conformance, not live round trips.
6. Client timeout/backoff policy derived from `limits` instead of hardcoded
   120 s (client.ts:441, streaming.ts:167).
