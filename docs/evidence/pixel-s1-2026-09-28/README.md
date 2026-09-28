# Pixel 10 Pro S1-mini protected-span pilot

The [BF16 record](s1-bf16-pixel.json), [Q4 record](s1-q4-k-m-pixel.json),
and [comparison](s1-pixel-comparison.json) use the existing
`s1-quant-spans-v1` schema and `benchmarks/s1/quant_spans.py compare`.
The original runner is preserved in git history at `60ba885`. The current
[runner](run_s1.py) documents the ADB, HTTP `/normalize`, three-repeat,
memory-snapshot, and hashing procedure without publishing the device address
or host paths. A new run supplies `STARLING_ADB_SERIAL`, `--artifact-dir`,
`--binary`, and `--engine-source-commit` explicitly.
The [BF16](s1-bf16-serve.log) and [Q4](s1-q4-k-m-serve.log) server logs
record model loading, warmup, CPU backend selection, and Q4 repack messages.
The comparison checks the fixture SHA256 and recomputes each protected-span
result from its stored output.

Both arms used Android `starling-serve` SHA256
`e9d65265def7129b0505cea08a1b7749d17e7b1e79645bf880ae84c289bbb224`,
source commit `2a3bda4`, the same protected-span fixture SHA256
`ef9184b34e51f45ce179026afa675f38975eadae560af1e3a20c00f82cfdb4a5`,
the GGML CPU engine with six threads, and one fresh process per model.
The server warmed up before the three repeated requests for each of the
eight cases. Each case produced one deterministic output within its arm.
All annotated protected spans remained present. The Q4 output changed in
two cases: `negation_and_name` (“Claire yet. Send” to “Claire. Yet, send”)
and `negated_deadline` (“June 2nd” to “June 2”). This does not establish
general quality or exact-output parity.

The post-warmup `/proc` smaps PSS was 1,687,097 kB for BF16 and 1,114,109
kB for Q4, a 572,988 kB (559.6 MiB) reduction in these two processes.
There was no GL mtrack allocation. The after-case snapshots were within
444 kB of the corresponding before-case snapshots. Neither measurement is
a loading peak or a co-residency budget.

The phone was charging during this first functional pilot (AC powered,
battery status 2), with thermal status 1 and the display off; unrelated user
activity was present.
The per-case response times in the JSON are exploratory. Charge-counter
changes cannot give idle-subtracted energy during charging. The server's
internal load message precedes Q4 warmup repacking and therefore does not
measure end-to-end ready-to-serve time.

A later unplugged, screen-off, CPU-only [ABBA pilot](controlled/README.md)
records fresh BF16–Q4–Q4–BF16 processes, their protected outputs, host HTTP
latencies, and battery-counter samples. Its energy result remains
inconclusive because gauge update latency was not established.

A separate unplugged, screen-off [CPU residency snapshot](cpu-co-residency/README.md)
kept the baseline Parakeet ASR server resident while loading optional Q4 S1.
The sampled process PSS sum was 1.43 GiB with both servers ready, and Android
`MemAvailable` ranged from 2.75 to 3.26 GiB across the joint samples. This
is a shell CPU observation; it does not establish the mobile app's process
allowance, a joint peak, or foreground responsiveness.
