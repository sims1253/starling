# Native server: starling-serve

`starling-serve` runs GGUF models behind an HTTP and WebSocket API. It uses
`libstarling_ggml` and does not require Python, PyTorch, Transformers, or Triton.
GPU builds still need the platform's driver and runtime libraries.

Start with [build](#build) and [usage](#usage), [install with npm](#install-with-npm)
for the prebuilt path, or see [release artifacts](#release-artifacts) for
binary prerequisites. The [API contract](#api-contract) covers differences
from `starling.server`.

## Install with npm

The `starling-serve` npm package ([packages/serve](../packages/serve/)) is a
launcher: it picks the release artifact for your platform and backend,
downloads it from GitHub Releases on first run, verifies both checksum layers,
caches it, and execs it with the arguments you pass. No compiler or GPU
required, and no postinstall script — see [packaging](packaging.md) for why.

```bash
npx starling-serve --model parakeet --gguf model.gguf --port 8181
pnpm dlx starling-serve --model parakeet --gguf model.gguf --port 8181
```

The launcher defaults to `metal` on Apple Silicon, `vulkan` on Linux when the
Vulkan loader is present (`cpu` otherwise), and `cpu` on Windows. Force a
backend with `--starling-backend cuda` or `STARLING_SERVE_BACKEND=cuda`. The
full matrix, cache layout, and environment overrides are documented in
[packaging](packaging.md).

## Build

Run these commands from a checkout with submodules initialized. You need
CMake 3.18 or later, a C++17 compiler, Git, and Bash (Git Bash on Windows).
CMake validates and applies the ggml patches during configuration.
GPU builds also need the matching development toolkit: CUDA, ROCm, the Vulkan
SDK, or Apple's Metal tools. Choose one backend below and start with a fresh
`build` directory. If you keep several builds, give each a separate directory.

```bash
# CPU-only (development / smoke tests):
cmake -B build -DSTARLING_SERVE=ON
cmake --build build -j --target starling-serve

# CUDA (NVIDIA production path):
cmake -B build -DSTARLING_SERVE=ON -DSTARLING_GGML_CUDA=ON
cmake --build build -j --target starling-serve

# ROCm / HIP (AMD Radeon & Instinct path on Linux):
cmake -B build -DSTARLING_SERVE=ON -DSTARLING_GGML_HIP=ON \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build build -j --target starling-serve

# Vulkan (universal Intel/AMD/ARM path):
cmake -B build -DSTARLING_SERVE=ON -DSTARLING_GGML_VULKAN=ON
cmake --build build -j --target starling-serve

# Metal (macOS):
cmake -B build -DSTARLING_SERVE=ON -DSTARLING_GGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
cmake --build build -j --target starling-serve
```

On Windows, add `-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded` to statically link
the CRT across the build. This does not bundle GPU runtime libraries.

## Usage

```bash
# Verify compatibility at startup:
starling-serve --version
starling-serve --abi-version

# Serve a model:
starling-serve --model parakeet --gguf model.gguf --port 8181 [--warmup]
```

The built executable is `build/starling-serve` (`build/Release/starling-serve.exe`
with a multi-configuration Windows generator). Replace `starling-serve` in the
examples with that path, or add its directory to `PATH`. The optional `--warmup`
flag runs a warmup at startup; omit the square brackets when using it.

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model <slug>` | (required) | Model slug: parakeet, moss, ark, ark06, higgs, hojo, granite, qwen3, qwen3_06, s1, audex, voxtral |
| `--gguf <path>` | (required) | Path to the GGUF model file |
| `--host <addr>` | `127.0.0.1` | Bind address |
| `--port <n>` | `8181` | Bind port (`0` = any free port; see [port announcement](#port-announcement-and-parent-watchdog)) |
| `--parent-pid <pid>` | (none) | Exit cleanly when that process exits (supervised sidecar) |
| `--warmup` | off | Capture CUDA graphs on startup |
| `--no-eager-load` | off | Defer model load to first request |
| `--idle-timeout <s>` | `0` (never) | Shut down after N seconds idle |
| `--request-timeout-seconds <s>` | `600` | Fail queued requests after N s waiting for the engine (`504`); same flag as the Python server |
| `--granite-chunk-fairness` | off | Opt in to one FIFO turn per Granite chunk; requires `--model granite` |
| `--stream-chunk-seconds <s>` | `12.0` | Fixed WS stream window |
| `--stream-overlap-seconds <s>` | `3.0` | Overlap between windows |
| `--min-chunk-seconds <s>` | `1.0` | Min audio before first partial |
| `--partial-interval-seconds <s>` | `0.5` | Min gap between partials |
| `--max-stream-seconds <s>` | `60.0` | Per-WS-connection LIVE buffer cap in s (0 = unlimited); see `WS /stream` |
| `--version` | n/a | Print version + ABI + backend, exit |
| `--abi-version` | n/a | Print ABI version integer, exit |

With `--granite-chunk-fairness`, a long Granite upload releases its engine
turn after each existing policy-defined chunk and re-enters the FIFO queue.
An already waiting stream is served before that upload's next chunk. Streaming
calls wait for their turn in this mode, up to the existing queue deadline;
the queue still admits at most eight tickets or reserved continuations in
total. A yielded upload keeps its admission slot until it requeues, so a new
arrival cannot evict an already accepted job. FIFO age prevents an upload from
starving under a stream of new arrivals. A stream can wait behind at most seven
older tickets (including an active chunk); this is a chunk-count bound, not a
wall-clock latency promise or preemption inside a chunk. One final response is
emitted only after every upload chunk succeeds. A queued cancellation or
timeout discards the job's partial text. The server retains the model and the
caller's PCM until that synchronous request finishes. The default and every
other model keep whole-request serial turns.

### Port announcement and parent watchdog

Supervisors (the desktop app, issue #362) launch `starling-serve` as a sidecar
and need two process-level guarantees:

- **Port announcement**: in all cases (fixed port or `--port 0` for any free
  port), once the socket is bound the server prints exactly one line to
  **stdout** — `STARLING_SERVE_LISTENING <host>:<port>` — flushes it, and only
  then serves requests, so a line on stdout always means the socket is already
  accepting connections. `--port` values outside 0..65535 are rejected with
  exit code 1.
- **Parent watchdog**: `--parent-pid <pid>` starts a watchdog thread that
  exits the process with `_Exit(0)` (after logging
  `[starling-serve] parent process <pid> exited; shutting down` to stderr)
  when that process disappears, so a crashed supervisor cannot leave an
  orphaned server holding a model in memory. POSIX watches a direct parent
  through `getppid()` (immune to PID reuse) and any other pid via
  `kill(pid,0)` polling every 500 ms; Windows waits on an `OpenProcess`
  handle.


## Standard API compatibility

The native backend also exposes `POST /v1/audio/transcriptions` and
`GET /v1/models`. Read the [API contract](api.md) for the supported
OpenAI-compatible subset and the machine-readable capabilities endpoint.

## API contract

The servers share audio routes and streaming messages. Clients must account
for these differences:

- **Inputs**: this server requires 16 kHz audio (below); the Python server
  resamples non-16 kHz WAVs via scipy instead of rejecting them. HTTP uploads
  require WAV on both servers. Both WebSocket endpoints accept PCM16
  and WAV.
- **Models**: both serve `parakeet`, `moss`, `ark`, `higgs`, `granite`, `qwen3`,
  `qwen3_06`, and `audex`. Native serving also supports `hojo` and `s1`; Python
  serving also supports `parakeet_unified` and `cohere`. Native `s1` exposes the
  additional `POST /normalize` text endpoint.
- **Request ids**: `X-Request-Id` values starting with `#` are rejected
  with `400`: the prefix is reserved for the server's internal queue
  tickets. The Python server accepts them.
- **Phase names**: `unloaded → loading → ready → busy` here; the Python
  server reports `loading_weights` and `warming_up` during startup.
- **Error responses**: the status-code and body details documented below
  differ in places from the Python server's.

### `GET /health`

```json
{"status":"ok","model":"parakeet","loaded":true,"busy":false,"phase":"ready","queue_depth":0,"backend":"CUDA0","warm":true,"load_error":null}
```

Phase drives the UI: `unloaded → loading → ready → busy`. Three additive
supervision fields (issue #362): `backend` (the runtime ggml device name once
a model is loaded; the compile-time backend family before that), `warm`
(whether a warmup has finished), and `load_error` (`null`, or the last load
failure message — a supervisor can show it and retry with `POST /warmup`).
`/health` never blocks during a load: the status fields are guarded by their
own mutex.

### `POST /v1/audio/transcriptions`

Accepts multipart/form-data with one `file` WAV part and one `model` field.
**Audio must be 16 kHz**. The native server has no resampler, so WAVs at other
sample rates are rejected with
`400` and a `sample rate mismatch` error (the Python server resamples via
scipy instead). WAV parsing is bounded by the actual payload size: a header
whose claimed frame count exceeds what the payload can hold (crafted or
truncated) is rejected with `400` and a `malformed audio payload` error: it
is never reinterpreted as raw PCM. Returns:

```json
{"text":"hello world"}
```

Uses `X-Request-Id` header for tracking. Errors map to: `400` malformed
audio / sample-rate mismatch / invalid request id, `409` duplicate active
request id, `413` request body too large, `499` cancelled, `503` busy or
model not loaded, `504` queue timeout, `500` other engine failures.

### `POST /warmup`

Idempotent warmup (CUDA graph capture): a silent clip for audio models, a
probe transcript for text models (s1). A model deferred by `--no-eager-load`
is loaded first (a no-op when resident), so a supervisor can start the process
fast and then load and warm it with one request; a failed load surfaces as
`/health`'s `load_error` and a later `/warmup` retries it. Still asynchronous:
returns `202` `{"status":"warmup started","phase":...}` — poll `/health`
(`loaded` + `warm`) for completion.

### `POST /normalize` (s1 only)

Text-in/text-out path for the normalizer. JSON body:

```json
{"transcript":"so um i need to send the the report by uh friday","styling":"semi-formal","structure":"prose","context":"general"}
```

`transcript` is required; the control fields are optional (defaults
`semi-formal`/`prose`/`general`) and must come from the trained sets. Unknown
values are rejected with `400` (the card warns off-spec controls make
the model hallucinate). Prompts over ~1,000 tokens (the trained input max)
are rejected with `400`; chunk long transcripts at sentence boundaries
first. Returns:

```json
{"text":"So I need to send the report by Friday.","request_id":"..."}
```

Audio models answer `400` ("model has no text path"): use `/v1/audio/transcriptions`.

### `DELETE /v1/audio/transcriptions/<id>`

Cancels a queued or in-flight request by request ID.

### `WS /stream`

Real-time streaming dictation. Send binary frames (raw PCM16 or WAV) and
receive JSON messages:

- `{"type":"partial","text":"...","start_s":0.0,"end_s":12.5,"stable_words":40}`: growing partial.
  The first `stable_words` whitespace-separated words of `text` are fixed:
  every later partial and the final start with exactly those words, so a
  client may treat them as final while the take is still being spoken (the
  desktop staging editor does). The count only grows, and it is 0 in the
  whole-buffer mode (`--stream-chunk-seconds 0`).
- `{"type":"final","text":"...","segments":[...],"duration_s":12.5}`: on commit
- `{"type":"error","message":"..."}`: on error
- `{"type":"pong"}`: in response to `{"type":"ping"}`
- `{"type":"reset_ack"}`: in response to `{"type":"reset"}`

**Preview cadence** (issue #357): the window geometry
(`--stream-chunk-seconds`, `--stream-overlap-seconds`) decides what is
committed. The cadence decides only when the live tail is previewed.
`--min-chunk-seconds` is the first-partial minimum for the whole take;
it does not gate the tail. Once the take holds that much audio, every
nonempty tail is previewed, including the overlap after a window commit.
`--partial-interval-seconds` is the minimum gap between previews, measured
from when the step's window work ends. The server stretches it so that
previews use at most half of wall time: the next preview waits at least
twice as long as the last one took. A window commit sends the committed
text at once and restarts the gap, because that text already covers the
audio received so far; no preview runs right behind it.
That bound adapts to slow models and devices without a per-device preset.
A client can set the cadence for one connection with query parameters:
`/stream?min_partial_seconds=1&partial_interval_seconds=0.5`. Invalid
values get an `invalid stream parameter` error frame, and the server
closes the connection. The cadence never changes which audio is committed
or finalized.

**Coalescing** (issue #357): the connection's read loop only queues frames.
One worker per connection appends every queued frame before each step.
Each preview therefore covers all audio received so far. If more audio,
a `commit` or a `reset` arrives while full windows are being committed,
the server skips that preview and previews the newer audio next. A `ping`
does not skip a preview. A preview that is already running is cancelled
when required work queues behind it: a `reset`, a `commit` after newer
audio, or queued audio that completes a window. The Parakeet engine stops
at its next checkpoint (between pipeline stages and between four slices of
the encoder graph; the fast Vulkan engine also between decoder frames),
and the audio stays in the buffer for the work that follows. Engines
without checkpoints finish the preview; its result is discarded all the
same. A `commit` with no newer audio lets the preview finish: its result
is the exact tail, and the flush reuses it. When the worker catches up on
a backlog, it commits each full window as soon as it is appended, so the backlog alone
does not hit the buffer cap. Window commits and finalization always run,
and the server never drops audio. When more than 32 MiB or 4096 frames
are queued, the server stops reading from the socket until the worker
catches up. A `ping` is answered after every earlier frame has been
processed, as before. If a frame cannot be written to the client (the
peer is gone or stalled past the write timeout), the server closes the
connection and the worker stops.

**Stream instrumentation** (issue #226): connect to `/stream?trace=1` and
every partial and final carries an extra `"trace"` object. Clients that do
not ask get the frames above unchanged. Times are milliseconds since the
take's first audio; audio positions are seconds into the take.

- Partial: `{"v":1,"t_ms":…,"audio_s":…,"covered_s":…,"totals":{…},"preview":{…}}`.
  `audio_s` is the audio received so far, and `covered_s` is the end of
  the audio that the text reflects. `preview` holds the connection's
  `min_s`, `interval_s`, the current `effective_interval_s` and the
  number of `coalesced` previews. `totals` holds `calls`,
  `engine_calls`, `engine_audio_s`, `engine_ms`, `reused`, `busy`,
  `preempted` (previews cancelled while running) and `preempted_ms`
  (the engine time they used before they stopped; not in `engine_ms`).
  `engine_audio_s` counts window overlap and each repeated preview, so
  `engine_audio_s / audio_s` is the inference work per recorded second.
  `engine_ms` is the wall time of each transcribe call. It includes a lazy
  model load on the first call and, in Granite chunk-fairness mode, the
  wait for the serial queue; other modes answer `busy` instead of waiting.
  The trace starts a commit's `stop` section when the worker begins the
  flush, so the time a commit waits for a running preview to stop is not
  counted there. The client's own stop-to-final time includes that wait.
- Final: the same fields, plus `by_kind` totals for `window` (full windows
  while recording), `preview` (live tail), `flush_window` and `flush_tail`
  (work after commit) and `redecode` (a committed window or flush tail
  decoded again from an earlier start; see "Stitching" below). It also has `stop`, `calls` and `calls_dropped`.
  `stop` covers the commit's own work. `path` is `tail` (the engine
  transcribed only the unfinalized remainder), `reused` (the exact
  tail result answered it) or `committed` (nothing was left, including
  a commit of an empty take). The
  Python server labels its whole-buffer mode `full_take`. `stop` also
  holds `unfinalized_s` (the audio past the committed boundary at
  commit), `t0_ms`, `t1_ms` and `totals`. `calls` lists each call
  with `kind`, `start_s`, `end_s`, `t0_ms`, `t1_ms` and `result`
  (`ok`, `reused`, `busy`, `timed_out` or `preempted`). A call that produced
  committed text (not a preview) on an engine with word timestamps (Parakeet)
  also has `words`: `[{"w":…,"start":…,"end":…}]`, the decode's words with
  the take seconds each was heard at. It holds at most 20,000
  entries. `calls_dropped` counts later calls, which still enter
  the totals.

`benchmarks/experiments/stream_replay.py` uses this to replay workload takes
at microphone pace and to report latency, work and stop-time metrics.

**Buffer cap** (`--max-stream-seconds`, default 60 s): a binary frame that
would push the session's live audio buffer past the cap is refused. The cap
counts the audio past the committed boundary. The session also keeps up to
0.75 x the overlap before that boundary (2.25 s by default) for re-decodes;
that audio is not counted, so memory is bounded by the cap plus it. The
server emits one error frame:

```json
{"type":"error","message":"stream buffer limit reached (60 s live buffer); audio ignored until reset"}
```

and then ignores all further audio frames for that session (no more partials,
no crash, memory bounded). The client sends `{"type":"reset"}` to clear the
session and start accepting audio again.

The cap bounds the **live** buffer (memory), not cumulative audio: finalized
windows are trimmed from the buffer as the stream advances, so a long
dictation session without commits keeps memory bounded without tripping the
cap. It only fires when the un-finalized buffer itself grows past the limit
(e.g. streaming faster than the engine finalizes, or a session where
transcription never succeeds).

**Invalid audio frames** (issue #145): a binary frame the server cannot accept
is refused with one error frame, exactly like the buffer cap:

- a malformed WAV (fails decoding),
- a WAV whose sample rate is not 16 kHz (the native server has no resampler;
  the HTTP paths reject the same inputs with `400`),
- raw PCM16 with an odd byte count: frames are sequences of whole int16
  samples, so an odd length means a sample was split at a transport boundary.
  The dangling byte is not silently dropped — the client should re-send on
  whole-sample boundaries.

```json
{"type":"error","message":"WAV sample rate mismatch: expected 16000; audio ignored until reset"}
{"type":"error","message":"malformed WAV frame rejected; audio ignored until reset"}
{"type":"error","message":"odd-length PCM frame rejected (split sample); audio ignored until reset"}
```

A refused frame also **invalidates the take**: all further binary frames are
ignored (no more partials), and `{"type":"commit"}` is refused with

```json
{"type":"error","message":"take invalidated (sample_rate_mismatch); reset and resend"}
```

instead of a successful `final` — the buffered audio is provably incomplete,
so the client must fall back to its authoritative local recording. The
machine-readable reason codes are `malformed_wav`, `sample_rate_mismatch`,
and `odd_pcm_length`. As with the buffer cap, the connection stays alive
(control frames keep working) and `{"type":"reset"}` clears the invalidation
and re-enables audio. The deprecated Python server shares this contract for
malformed WAV and odd-length PCM; its one divergence is that it resamples
non-16 kHz WAVs via scipy instead of rejecting them with `sample_rate_mismatch`
(see [divergence note](python-serving.md#invalid-stream-audio-and-the-remaining-divergence-from-the-native-server)).

Control frames (JSON text):

- `{"type":"commit"}`: finalize all buffered audio. Returns `final` on success;
  if bounded retries stay busy, returns `{"type":"error","message":"server busy"}`
  and retains the audio. Retry `commit` after a delay. A commit on an
  invalidated take (see above) is refused with an error instead.
- `{"type":"reset"}`: discard buffer without finalizing (returns reset_ack;
  also re-enables audio after a buffer-cap error or an invalid-audio
  rejection)
- `{"type":"ping"}`: heartbeat (returns pong)

## Architecture

```text
cpp/serve/
├── main.cpp            — CLI parsing, lifecycle, HTTP/WS transport (cpp-httplib)
├── server.hpp/.cpp     — StarlingServer: model lifecycle, serial queue, transcribe
├── stream_session.hpp/.cpp — Rolling buffer + ChunkStreamer (port of Python logic)
├── stream_pump.hpp/.cpp — WS /stream worker: frame queue, drain, preview coalescing
└── audio.hpp/.cpp      — WAV/PCM decoding (dr_wav) + multipart extraction
```

The C++ and Python streaming sessions use fixed overlapping windows, including
during busy retries. Successful windows advance the committed boundary;
incomplete commits preserve the remaining audio for a later retry.

**Stitching** (issue #357; the two servers share the code paths and a parity
fixture, `tests/fixtures/stream_stitch_cases.txt`):

- With word timestamps (Parakeet: the TDT decoder's frame for each token,
  80 ms per encoder frame; a word starts at its first token's frame and ends
  at its last token's frame plus duration), neighboring windows are joined at
  a word both heard at the same time: a committed word and a new word are the
  same when they match after normalization and start at most 0.3 s apart.
  The alignment is the sequence of such pairs in order with the most pairs
  inside the shared audio, then the most pairs in all (a word at its edge
  that one window heard just past it), then the smallest total start
  difference (so pairs only one window can have heard never outweigh the
  ones both did, and a word said several times in a row pairs with the same
  occurrence), and
  the cut is the pair nearest the middle of the shared audio: committed
  words up to and including it, then the new words after it. One shared word is enough, so a
  word repeated exactly at the boundary is no longer kept twice, and a
  repetition elsewhere in the text cannot match, so periodic text is not
  shortened or repeated. Without a shared word (a pause in the overlap), or
  without shared audio (`--stream-overlap-seconds 0`), the texts are
  concatenated. Not covered: one word repeated back to back faster than the
  windows' timing agrees ("yes yes yes yes" at about 3 or more per second),
  with no other shared word near the cut, can pair one repetition off and
  lose or repeat it; the times alone cannot tell the occurrences apart. On
  synthetic runs with the measured offsets, that happens for about 2 % of
  window placements at 4 repetitions per second, 1 % at 3, and none at 2.5. On the replay workload (q8_0, notebook), the same
  word in two overlapping windows starts 0.04-0.28 s apart in the middle of
  the overlap and up to 0.52 s apart at a window's edge, where the cut is not
  made.
- Without word timestamps on both sides (other engines, the Python server),
  neighboring windows are joined by aligning a suffix of the committed words
  with a prefix of the new window's words (match +2, mismatch and gap -1).
  Committed words between the aligned run and the end cost a gap, so a common
  phrase away from the boundary cannot win and drop the words between. The
  cut is the middle matched word. With no alignment scoring 3 or more (a
  pause in the overlap, or a window that dropped those words) the texts are
  concatenated. Repeated text ("one two three one two three ...") aligns
  equally well at every multiple of its period; an alignment over committed
  words that repeat is shortened by whole periods to the number of words the
  shared audio should hold (each window's share of its voiced audio in the
  overlap). Not covered: in a re-decoded window, the shared audio (the
  overlap plus the shift) can hold more words than the 24 committed words
  searched: above about 5.3 words/s for the 1.5 s shift, 6.4 for 0.75 s and
  4.6 for 2.25 s (default 3 s overlap). Ordinary text still aligns, but
  exactly periodic text can then repeat up to the words outside the search.
  Searching more committed words would delay `stable_words` for every take,
  and words alone cannot tell a shared repetition from a newly spoken one;
  the timed join above has neither limit. A single word shared at the
  boundary is not deduplicated (an alignment needs two).
- Parakeet sometimes stops emitting partway through a window, or returns
  nothing for a window full of speech, while the same audio decodes fine one
  second later. A committed window or flush tail is therefore checked against
  its audio. It is implausible when it has fewer than 1.25 words per voiced
  second (or fewer than 0.6 x the take's median over its last 16 windows)
  over at least 2 s of voiced audio, or more words than 7 per second plus 4
  (a decoding loop). An implausible window is decoded again from 0.5, 0.25
  and 0.75 x the overlap earlier (a full window moves back whole, or ends
  earlier at the take's start; the flush tail grows backwards up to one
  window) until a result is plausible. A plausible candidate replaces an
  implausible result; between two sparse results the candidate must be
  1.25 x denser, and a sparse one replaces a loop. The next window then starts
  one overlap before the end of the audio the kept text came from. If a
  re-decode finds the engine busy before a plausible result is in hand, the
  window stays pending, like a busy window, instead of committing text known
  to be wrong. "Voiced" is 20 ms frames 15 dB
  above the span's quiet floor and above -60 dBFS.
- Text over the word bound has its longest back-to-back repeated phrase cut
  to two copies, one run at a time, only until it fits, and is cut at the
  bound if it still does not. A committed window whose every re-decode loops
  gets this; text within the bound is not touched, so real repeated speech
  survives in the final. Previews (never re-decoded, replaced by the next one)
  get it too, and additionally have any phrase repeated back to back more
  than three times (8 words or more) cut to two copies, so a decoding loop
  never reaches the client.
With opt-in Granite chunk fairness, a blocking queue timeout ends the current
take with `request timed out`; it is not retried as `server busy`. Reset the
stream before sending more audio.

## Timing trace

`STARLING_TRACE=1 starling-serve …` turns on an opt-in structured timing trace
(issue #180): one JSON record per line on stderr, prefixed `[trace] `, that
correlates one transcription request's timing across the layers it touches.
With the variable unset (the default) nothing is measured, formatted, or
printed, and the replay fast path is untouched.

```text
[trace] {"v":1,"ts":1840,"tid":91255231,"ev":"queue_enter","req":"req-42","policy":"block","depth":1}
[trace] {"v":1,"ts":1841,"tid":91255231,"ev":"queue_wait","req":"req-42","dur_ms":0.612}
[trace] {"v":1,"ts":1842,"tid":91255231,"ev":"chunk","chunk":1,"dur_ms":410.2,"req":"req-42"}
[trace] {"v":1,"ts":2600,"tid":91255231,"ev":"stage","stage":"mel_enc_proj","dur_ms":180.4,"req":"req-42","chunk":1}
[trace] {"v":1,"ts":2610,"tid":91255231,"ev":"graph_replay","dur_ms":0.31,"uid":3,"nodes":2418,"out_ne":[1024,0,0,0],"device":"CUDA0","req":"req-42","chunk":1}
[trace] {"v":1,"ts":2790,"tid":91255231,"ev":"readback_sync","dur_ms":178.9,"uid":3,"nodes":2418,"out_ne":[1024,0,0,0],"device":"CUDA0","req":"req-42","chunk":1}
[trace] {"v":1,"ts":3410,"tid":91255231,"ev":"request","dur_ms":1568.3,"req":"req-42"}
[trace] {"v":1,"ts":3412,"tid":91255231,"ev":"queue_exit","req":"req-42","reason":"completed","depth":0}
[trace] {"v":1,"ts":3413,"tid":91255231,"ev":"response","dur_ms":0.02,"req":"req-42"}
```

Record kinds and fields:

| `ev` | Layer | Fields |
| --- | --- | --- |
| `queue_enter` | serving | `req`, `policy` (`block`/`skip_if_busy`), `depth` (waiters after enqueue) |
| `queue_wait` | serving | `req`, `dur_ms` (host time blocked waiting for the serial-queue turn; emitted on turn acquisition AND on abandoned departures — skip refusal, timeout, cancellation) |
| `request` | serving | `dur_ms` (engine-call wall time) |
| `queue_exit` | serving | `req`, `reason` (`completed`/`yielded`/`failed`/`cancelled`/`server_busy`/`timed_out`), `depth` (waiters after release). Terminal per ticket: every `queue_enter` balances exactly one `queue_exit` |
| `response` | serving | `dur_ms` (result marshalling; the HTTP body build and socket write are outside the trace) |
| `chunk` | engine | `chunk` (1-based), `dur_ms` |
| `stage` | engine | `stage` (`mel_enc_proj`, `prompt_embeds`, `generate`), `dur_ms` |
| `graph_build` | runtime | `dur_ms` (construction + allocation of one captured shape), `uid`, `nodes`, `out_ne`, `device`, `mem_free` (device free bytes after admission, or `unavailable`) |
| `graph_replay` | runtime | `dur_ms` (the async launch — **host enqueue**), `uid`, `nodes`, `out_ne`, `device` |
| `readback_sync` | runtime | `dur_ms` (the single trailing sync — **host blocked**), `uid`, `nodes`, `out_ne`, `device` |
| `cache` | runtime | `cache` (e.g. `granite.encoder`, `qwen.prefill`, `parakeet.encoder`), `op` (`hit`/`miss`), `evicted`, `size`, `cap`, `mem_free` |

Every record carries `v` (schema version), `ts` (µs since the first record),
`tid`, and — when active — `req` (the HTTP request id, or the synthesized
`#anon-N` ticket of an anonymous caller) and `chunk`. Engine-layer records
inherit both through the request scope, so a replay graph fired inside chunk 3
of request `req-42` carries `req` and `chunk:3`.

Reading the numbers correctly (binding rules):

- **Host enqueue vs. host blocked vs. device time are different things.**
  `graph_replay` measures the async launch (returns almost immediately on
  CUDA). `readback_sync` measures the single trailing sync and *includes
  waiting for prior GPU work* — it is **not** transfer time alone. True
  per-graph device time is not measured and never fabricated. On the CPU
  backend the launch is itself synchronous, so `graph_replay` approximates
  the full compute — check `device` before interpreting.
- **Wall times are clock-nested; never add a child into its parent.**
  Aggregate by summing sibling records of ONE kind: the three `stage` records
  of a chunk sum to that chunk's engine work. In Granite fairness mode each
  `request` record covers one chunk turn, so sum those records across the
  request id before comparing with its `chunk` records. `graph_replay` + `readback_sync`
  overlap the stage walls (they are leaves, not additional time).
- **The queue ledger balances.** Every `queue_enter` has exactly one
  terminal `queue_exit` whose `reason` says how the ticket left
  (`completed`, `yielded`, `failed`, `server_busy`, `timed_out`, `cancelled`), and `queue_wait`
  fires for abandoned waits too — the contention outcomes the trace exists
  to diagnose are never invisible.
- **No contents.** Records carry ids, indices, shape dimensions, and cache
  occupancy — never audio, transcripts, prompts, or tensor contents.
- **Scope.** Chunk/stage spans currently come from the granite engine (the
  multi-chunk reference); other engines emit the serving and runtime records
  without chunk attribution. One-shot (non-captured) computes are not traced.
  The trace is a diagnostic tool, not telemetry: nothing is sent anywhere.

The older gates remain independent: `STARLING_GRANITE_TIMING` (granite stage
summaries) and `STARLING_REPLAY_TIMING` (per-replay split). The stage records
share `STARLING_GRANITE_TIMING`'s clocks, so the two renderings cannot
disagree.

## Pre-converted GGUF files

Download Parakeet weights from
[`scholzmx/parakeet-tdt-0.6b-v3-gguf`](https://huggingface.co/scholzmx/parakeet-tdt-0.6b-v3-gguf).
The repository includes Q8_0, K-quants, IQ2_XXS, and the importance matrix.
With the Hugging Face CLI installed:

```bash
hf download scholzmx/parakeet-tdt-0.6b-v3-gguf \
  parakeet-tdt-0.6b-v3-q8_0.gguf --local-dir ./models
starling-serve --model parakeet \
  --gguf ./models/parakeet-tdt-0.6b-v3-q8_0.gguf --port 8181
```

MOSS-Transcribe weights (Q4_0 + imatrix linears, Q8_0 tied head, 1.55 GB)
are in [`scholzmx/moss-transcribe-preview-2b-gguf`](https://huggingface.co/scholzmx/moss-transcribe-preview-2b-gguf):

```bash
hf download scholzmx/moss-transcribe-preview-2b-gguf \
  moss-transcribe-preview-2b-q4e8-fullimx.gguf --local-dir ./models
starling-serve --model moss \
  --gguf ./models/moss-transcribe-preview-2b-q4e8-fullimx.gguf --port 8181
```

The planned `starling/*-gguf` repositories are not public downloads.
For other models, use the converters below with the original model weights.
The [GGUF file guide](hf-gguf-readme.md) describes filenames and metadata.

### Quantization

Choose `q8_0` for smaller weights where it is available, or `bf16-exact` for
reference comparisons. Exact transcript parity depends on the model, backend,
and input; the filename alone does not guarantee it. See the
[engine parity notes](ggml-engine.md#correctness-contract) and
[quantization guide](quantization.md) for measured results.

### GGUF converters

| Model | Converter script |
|-------|-----------------|
| parakeet | `scripts/convert_parakeet_gguf.py` |
| moss | `scripts/convert_moss_gguf.py` |
| ark | `scripts/convert_ark_gguf.py` |
| ark06 | `scripts/convert_ark06_gguf.py` |
| voxtral | `scripts/convert_voxtral_gguf.py` |
| higgs | `scripts/convert_higgs_gguf.py` |
| hojo | `scripts/convert_hojo_gguf.py` |
| granite | `scripts/convert_granite_gguf.py` |
| qwen3 | `scripts/convert_qwen3_gguf.py` |
| qwen3_06 | `scripts/convert_qwen3_06_gguf.py` |
| s1 | `scripts/convert_s1_gguf.py` |
| audex | `scripts/convert_audex_gguf.py` |

## Release artifacts

The release workflow packages nine executables with SHA-256 checksums in
`.tar.gz` archives on Linux/macOS and `.zip` archives on Windows. It links
Starling and ggml into the executable, but does not bundle accelerator runtime
libraries. These are not fully static binaries.

| Backend | Runtime prerequisites |
| --- | --- |
| CPU | None beyond the platform C/C++ runtime (`libstdc++6` and `libgomp1` on Linux; on Windows the static CRT plus the Microsoft Visual C++ Redistributable (x64) for `vcomp140.dll`, which every Windows build needs). |
| CUDA | Compatible NVIDIA driver and CUDA runtime/cuBLAS libraries. The workflow builds with CUDA 13.3. |
| ROCm / HIP | Compatible AMD driver, HIP runtime, hipBLAS, and rocBLAS libraries. The workflow builds with ROCm 7.2.4. |
| Vulkan | Vulkan loader and a compatible GPU driver. |
| Metal | Apple Silicon macOS with the system Metal frameworks. |

The Linux CUDA, Vulkan, and CPU archives are smoke-tested outside their
build environment in a fresh Ubuntu 22.04 container with only the documented
runtime packages (see the [runtime guide](release-runtime.md)); the release
workflow checks the Windows and macOS archives on their build machines only.
None of these startup checks verify GPU inference. Which artifacts have run
representative inference on real hardware, and on what, is listed in the
runtime guide's [hardware verification](release-runtime.md#hardware-verification)
ledger (tracked in [issue #57](https://github.com/sims1253/starling/issues/57));
`linux-rocm`, `macos-metal`, and `macos-cpu` have not been verified on hardware.

Choose the executable for your operating system, CPU architecture, and GPU:

| Artifact | Platform | Backend | Notes |
|----------|----------|---------|-------|
| `starling-serve-linux-cuda` | Linux x86_64 | CUDA | NVIDIA |
| `starling-serve-linux-rocm` | Linux x86_64 | ROCm / HIP | AMD Radeon & Instinct |
| `starling-serve-linux-vulkan` | Linux x86_64 | Vulkan | Intel / AMD / NVIDIA |
| `starling-serve-linux-cpu` | Linux x86_64 | CPU | No GPU required |
| `starling-serve-windows-cuda.exe` | Windows x86_64 | CUDA | NVIDIA (static application CRT) |
| `starling-serve-windows-vulkan.exe` | Windows x86_64 | Vulkan | AMD / Intel / NVIDIA (static application CRT) |
| `starling-serve-windows-cpu.exe` | Windows x86_64 | CPU | No GPU required (static application CRT) |
| `starling-serve-macos-metal` | macOS arm64 | Metal | Apple Silicon |
| `starling-serve-macos-cpu` | macOS arm64 | CPU | Apple Silicon, no GPU needed |

### GPU selection

Each release variant includes a specific GPU backend. Download the variant for
your platform and GPU; the executable cannot add a backend that was not compiled
in. Within the build, the runtime chooses the first GPU or integrated GPU.
Set `STARLING_GGML_DEVICE` to a device name such as `CUDA0`, `Vulkan0`, or `Metal`
to select it, or to `cpu` to force the CPU backend.
