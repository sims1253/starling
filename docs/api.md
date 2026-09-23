# Transcription API

Starling's native server implements a subset of the
[OpenAI transcription API](https://developers.openai.com/api/reference/python/resources/audio/subresources/transcriptions/methods/create)
and [model list API](https://developers.openai.com/api/reference/python/resources/models/methods/list).
This is wire compatibility for local clients; it does not call OpenAI or reuse
another project's inference backend.

## OpenAI-compatible routes

| Route | Contract |
| --- | --- |
| `GET /v1/models` | `{object: "list", data: [{id, object: "model", created, owned_by}]}`; lists the one configured model |
| `POST /v1/audio/transcriptions` | Multipart `file` and `model`; returns `{text}` by default |
| `GET /v1/starling/capabilities` | Starling extension describing supported audio and options |

Use the slug from `/v1/models` as `model`, such as `parakeet`. A name for a model
that this process does not serve returns 404. The `created` value is 0 because
Starling does not track a model creation timestamp.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8181/v1", api_key="local")
with open("recording.wav", "rb") as audio:
    transcript = client.audio.transcriptions.create(model="parakeet", file=audio)
print(transcript.text)
```

The SDK requires an API-key argument; the local Starling server does not check
it. Keep the service on loopback or use your own authenticated HTTPS reverse
proxy for access from another device. The apps do not expose the server for you.

### Supported subset

- Exactly one WAV file named `file`, sampled at 16 kHz. The apps produce mono
  PCM16 WAV for consistent behavior. Compressed formats such as MP3, M4A, and
  WebM are rejected by this route instead of being interpreted as raw PCM.
- Exactly one `model` field matching the server's configured model.
- `response_format=json` (default) or `text`. JSON contains the raw `text`;
  plain text has content type `text/plain; charset=utf-8`.
- `stream=false` and `temperature=0` or `0.0` may be sent as compatibility
  defaults. No configurable sampling behavior is exposed.
- Successful responses carry `X-Request-Id`. A caller can supply that header
  and cancel a queued request with `DELETE /v1/audio/transcriptions/{id}`.

Language selection, vocabulary prompts, nonzero temperature, `verbose_json`,
SRT/VTT, word timestamps, diarization, and server-sent transcription events are
not implemented. Requests for these options return 400 rather than silently
ignoring them. `/stream` remains Starling's own WebSocket protocol; it is not an
implementation of OpenAI Realtime. The capabilities extension reports these limits.

Application errors use `{error: {message, type, param, code}}`. A malformed or
oversized request rejected by the underlying HTTP parser can use that parser's
error response. Clients must tolerate non-JSON errors and always check status.

The [OpenAPI document](../packages/contracts/openapi.json) records this subset.

## App configuration

Configure the apps with the server root, such as `http://127.0.0.1:8181`, and
use the model slug returned by `GET /v1/models`. Send mono PCM16 WAV at 16 kHz.
The native server rejects other sample rates, while the Python server resamples
them. The apps also use Starling's `WS /stream` for live dictation where supported.
Read [native serving](native-serving.md) for queueing, cancellation, streaming,
and model loading.

## Fidelity and recovery

The response text is the recognition result, including any recognition errors.
No cleanup model runs automatically. Saved audio and prior recognition attempts
are evidence for review and retry, not proof that every utterance was recognized.

A future vocabulary API must report whether the selected engine actually accepts
hints. A future cleanup API must return a separate suggestion, preserve the raw
text, and make meaningful changes reviewable before insertion. Do not silently
renumber a list or remove words such as "never", "not", "haven't", or "like".
