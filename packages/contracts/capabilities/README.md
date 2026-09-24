# Server capability descriptor and conformance fixtures

`capability.json` describes the current native `starling-serve` API. The
schemas validate the descriptor and the request/response fixtures. Run
`uv run --no-project --with pytest --with jsonschema pytest -q tests/test_capabilities_conformance.py`
to check that the routes, limits, fixtures, and live capability response agree.

One model is resident per process. Batch transcription uses
`POST /v1/audio/transcriptions` with a 16 kHz WAV `file` and `model` field.
The response is `{ "text": "..." }` or plain text. Live dictation uses
`WS /stream` with partial/final messages and commit, reset, and ping controls.
The WebSocket route is a Starling extension, separate from the batch API.
`DELETE /v1/audio/transcriptions/{id}` cancels a queued or running request.
Unsupported batch options return 400.

## Mapping to the live capabilities route

`GET /v1/starling/capabilities` serves a subset of `capability.json`.
`tests/test_capabilities_conformance.py` verifies this mapping against
`fixtures/starling_capabilities_route.json`.

| Live field | Descriptor field |
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
| `websocket_path` | `streaming.path` |

## Client adoption

No client currently fetches the capability descriptor. The apps use the
published routes directly. A later change can use the descriptor to select
models and show server limits.

| Client | Current source |
| --- | --- |
| TypeScript client | `packages/dictation/src/client.ts` |
| Desktop (gpui, Rust) | `apps/desktop-gpui/crates/dictation/src/client.rs` and `apps/desktop-gpui/crates/app/src/live_stream.rs` |
| Android | `apps/mobile/app/src/main/java/dev/starling/mobile/network/InferenceClient.kt` and `StreamClient.kt` |
| iOS | `apps/ios/Sources/Core/StarlingClient.swift` |
| Python backend | `backends/python/serve.py`, `src/starling/server.py` |

The fixtures check static JSON and route definitions. Live behavior is covered
by `backends/native/tests/test_openai_api.py` and `tests/test_native_serve.py`.
