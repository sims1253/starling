# Granite mixed-request contention pilot (#174 → #178)

This is a bounded real-model CPU pilot, not the full workload matrix or latency distribution required by [#174](https://github.com/sims1253/starling/issues/174). It tests the prerequisite in [#178](https://github.com/sims1253/starling/issues/178): whether a long upload materially delays an already connected short dictation session under the current serial queue. The preregistered rule is in [`benchmarks/contention/pilot_spec.json`](../benchmarks/contention/pilot_spec.json), committed as `768eaad` before any measurement. The first attempt failed; commit `d9db021` amended only the WS connection protocol before the six scored trials, and commit `d625b5f` recorded post-hoc controls afterward. The material-contention rule was not changed.

## Inputs and method

- Native `starling-serve` built fresh from `master@6b95f3477f04aef2d0ee99a95132f311fb45de61`, pinned ggml `e91ded11` plus this repo's `third_party/ggml-patches`, CPU backend, ABI 8. Binary SHA-256: `a91847c520596042e798e5acd17e14c3ada78a7b3699f317b4402d5459cf6d49`.
- [Granite Speech 4.1 2B](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) exported to `granite-speech-4.1-2b-bf16-exact.gguf` via `scripts/convert_granite_gguf.py`; GGUF SHA-256: `b72a01687daa1825894bd5ea3e4c53f6be657a9ba31c5cd65c5114c5b6b961e2`.
- Public committed source `tests/fixtures/2086-149220-0033.wav`; `tests/fixtures/make_fixtures.py` generates `short.wav` (7.435 s, SHA-256 `5fceacff0315d49cb59fcc505bcecf1ed5f2f35c2897b1e65a59f30e5d922150`) and `long.wav` (same speech repeated ten times, 74.35 s, SHA-256 `4f97080176d3623eebc9663af07dd258894ed31f74d5d229d39e77c5ab5b0925`). This repetition stresses queue behavior; it does not represent a diverse dictation corpus.
- AMD Ryzen 9 5900X, 24 logical CPUs, Linux 6.18.33.2 WSL2. The server was pinned with `taskset -c 16-23` (four physical cores with both sibling threads), `OMP_NUM_THREADS=8`, `STARLING_TRACE=1`. It was warmed once, then ran six alternating idle/mixed trials in one process. Host load averages moved from 3.40/3.33/5.56 to 7.09/5.09/5.41; the affinity limited server scheduling to the reserved cores but does not eliminate shared-cache or WSL host effects.
- The WebSocket transport handshake was connected before each historical trial. For mixed trials, the long WAV was submitted to HTTP; the short WAV was sent on the existing WS only after the long request emitted `queue_wait` (the service turn was acquired). The client committed immediately after sending short audio and retried a `server busy` commit every second until a final or 300 s. Both WAVs were delivered as whole frames, not paced in real time. The reported stop-to-final clock starts after the short frame is sent; network upload and connection setup are excluded. The native default 12 s streaming window receives the 7.435 s short take; Granite's own policy processed the long take as three chunks.

The historical scored run used harness commit `d9db021`: it completed the WS handshake before launching the long HTTP request, but did **not** wait for an application-level response. The new-session diagnostic below shows those can be separated by a model-mutex wait, so the saved results alone cannot isolate streaming retry delay from session-admission delay. The current harness now sends a WS `ping`, requires a `pong`, and records `ws_app_ready_ms` before starting each trial. New paired #178 measurements use this barrier. The historical timing numbers and preregistered threshold remain reported as measured, with this narrower mechanism claim.

To reproduce on a CPU host, initialize the submodule, apply the checked-in ggml patch series, build `starling-serve` with `-DSTARLING_SERVE=ON -DGGML_CUDA=OFF`, regenerate the public WAVs with `uv run python tests/fixtures/make_fixtures.py`, and provide the pinned Granite GGUF. The harness checks the binary, GGUF and both WAV hashes before starting:

```sh
uv run --extra server python benchmarks/contention/run_pilot.py \
  --binary build/starling-serve --gguf models/granite-speech-4.1-2b-bf16-exact.gguf \
  --short-wav tests/fixtures/short.wav --long-wav tests/fixtures/long.wav \
  --cpu-list 16-23
```

`--diagnostics-only` runs the post-hoc serial long control and new-session admission probe instead of the six-trial sequence. Each run writes a JSON summary and trace log under its chosen `outputs/` directory; those machine-local outputs are ignored by git. On another host, copy the spec to a separately committed per-run file, update its binary and artifact hashes, and pass `--spec path/to/spec.json` before measurement. The original scored run used the spec at `d9db021`; its SHA-256 was `d18fa9966e07c17b8721bdc38de106821e7cc8c365cfa07f33813ff6f606765a`.

## Scored six-trial result (2026-09-27 UTC)

| Pair | Idle short stop→final | Mixed short stop→final | Added delay / ratio | WS busy replies | Short service idle / mixed | Long service; chunks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 8.53 s | 140.57 s | 132.05 s / 16.49× | 106 | 8.48 / 12.93 s | 126.99 s; 3 |
| 1 | 13.35 s | 140.15 s | 126.80 s / 10.50× | 105 | 13.31 / 13.74 s | 126.38 s; 3 |
| 2 | 13.54 s | 142.73 s | 129.19 s / 10.54× | 108 | 13.49 / 12.56 s | 130.17 s; 3 |

All six short requests ended in a `final` with the same 122-character transcript SHA-256 `41f9e18f92ad8dfa94861d37ccb1262f07a1660c7b42f50bf51252cf5b5d0383`. All three long HTTP requests returned 200 with a 1012-character transcript SHA-256 `8bd0f0a220db9523f2cdf545de2281e9c4e1ec6bddebe6f8ad4bb133d54745dd`, three traced Granite chunks and 0 ms long queue wait. The first short commit in each mixed trial preceded long completion. The server traced 531/526/543 WS `server_busy` queue exits, respectively, during those long service turns; the client saw 106/105/108 busy replies after its own bounded internal retries. The short service durations stayed around 8–14 s, so the additional final latency came principally from waiting for the long request's single serial turn, not from short engine work. The predeclared criterion (matching final, at least 2 s and 2× delay in each pair, busy response, overlapping long request with at least two chunks) passed all three pairs.

An earlier attempt opened a *new* WS after the long HTTP service began. Its first idle short final took 7.94 s; the 74.35 s long upload completed three chunks in 123.45 s, but the WS handler was logged only after long completion and the client received close 1001 `pong timeout`. That attempt was excluded before scoring and preserved in the prereg amendment. Preconnecting the WS fixes the scored condition; the new-session behavior was investigated separately below.

## Decision and limits

This meets #178's **material-contention prerequisite** for a transport-connected dictation session on this CPU configuration. The historical run did not prove application readiness before the long upload, so a paired run with the new ping/pong barrier is needed to attribute delay to the queued streaming call alone. It justifies an opt-in chunk-boundary scheduling experiment, with serial fallback, state-isolation tests and a paired benefit/throughput comparison. It does not show that yielding between Granite chunks is safe, does not satisfy #174's single/paced/multi-session/burst/overload matrix, and cannot provide p50/p95/p99 with only three pairs. No model, scheduler, default or personal data changes were made by this pilot. The identical short hashes and successful long finals support this narrow comparison; the separate serial long control below checks long-output equivalence explicitly.

## Post-hoc controls (not part of the preregistered threshold)

An otherwise idle, warmed process decoded the same 74.35 s long WAV in 105.38 s HTTP wall / 105.36 s traced service, with 0 ms queue wait and three chunks. Its final transcript SHA-256 and 1012-character length **exactly matched all three mixed long results**. The 126–130 s mixed long service durations were 20–24% slower than this one serial control, but the experiment is too small and host load varied; this is a throughput signal to measure in #178's paired comparison, not a stable slowdown estimate.

For new-session admission, a separate 7.435 s HTTP request was already in service when a WS handshake began. The handshake completed in 10.34 ms while HTTP was still active, but an application-level WS `ping` waited 21.26 s for `pong`; the HTTP service ended at 21.28 s, and the server logged `WS /stream client connected` only after that request's `queue_exit`. This distinguishes a quick transport handshake from delayed WS handler execution. Source inspection identifies a blocking path: `StreamSession::StreamSession` builds its engine identity by calling `starling_ggml_backend_name()`, whose C API `api_call` takes the global `runtime_mutex`; the in-flight `starling_ggml_transcribe_pcm` holds the same mutex across the entire request. This lock path explains the observed handler delay and likely the earlier long-request heartbeat timeout. It deserves a separate admission fix/validation; chunk scheduling alone may shorten, but does not remove, that lock wait.
