# Controlled Pixel S1 CPU pilot

This is a four-block BF16–Q4–Q4–BF16 pilot on one Pixel 10 Pro on
2026-09-28. Each block launched a fresh `starling-serve` process, used
`STARLING_ENGINE=ggml`, `STARLING_GGML_DEVICE=CPU`, and six GGML threads,
completed server warmup, then sent the same eight public held-out cases once
each through host HTTP `/normalize` in fixture order. The screen was off, the
phone unplugged, thermal status 0, and the pre-block cooling rule met in each
block. The raw block records and [analysis](s1_cpu-analysis.json) are here;
each block also includes the actual CPU backend log and validator result.
This is a small pilot with two blocks per arm, not a p95 sample.

| Block | Arm | Launch to ready | Process duration | Median of eight host HTTP calls | Observed active charge-counter drop |
|---|---|---:|---:|---:|---:|
| 1 | BF16 | 52.7 s | 372.7 s | 44.03 s | 45,000 µAh |
| 2 | Q4 | 17.8 s | 110.4 s | 11.43 s | 0 µAh |
| 3 | Q4 | 20.3 s | 46.3 s | 2.47 s | 0 µAh |
| 4 | BF16 | 60.4 s | 294.1 s | 29.00 s | 42,500 µAh |

The block medians are descriptive. Same-arm timing changed markedly; in the
first adjacent pair, `negation_and_name` took 10.36 s on BF16 and 23.44 s on
Q4. Generic server warmup did not warm each case shape. Host HTTP timing
includes transport, request handling, and device scheduling. A separate
[two-call Q4 diagnostic](q4-http-native-diagnostic.json), outside this ABBA
protocol, observed 36.93/7.58 s host HTTP versus 17.50/4.88 s native total
for the same output; the gap changed from 19.43 to 2.70 s. These data do not
establish a stable or native model speedup.

All 32 requests exactly reproduced their own arm's earlier functional-pilot
outputs, and all protected spans passed. BF16 and Q4 still differ in two of
the eight outputs. The energy result is **inconclusive**. The 5,000 µAh
counter stayed unchanged throughout both Q4 active windows (110.6 and
46.5 s); maximum active plateaus over all blocks were 280.9, 110.6, 46.5,
and 181.2 s. The counter's update latency was not independently bounded.
The idle-correction arithmetic in the analysis JSON is labeled illustrative;
its intervals are not measured energy bounds, and zero counter change is not
zero energy use. No battery-side or model-only energy win is claimed.

The original preregistered protocol SHA256 is
`c66f7614fa44c34884059f80b64794905b96167cada6f102d2972756b6b0e708`.
[`protocol.redacted.json`](protocol.redacted.json) omits the private phone
address and host ADB path, so its hash intentionally differs. The raw block
records retain the original protocol hash. Runner v2 SHA256 was
`2c4f35fde860016575266e08da1bee27335fb3cd4ad863e1573ab736705a6eaa`;
original validator SHA256 was
`b896c9fa0a98d7c4c04ac5ef03e0d8e4f04dd55e59066dfa4386bb079598182e`.
This published validator resolves repository-relative fixtures and checks
the ABBA arm/index against the declared schedule. Android server SHA256 was
`e9d65265def7129b0505cea08a1b7749d17e7b1e79645bf880ae84c289bbb224`.
BF16/Q4 GGUF SHA256s were
`c4fc61df01df19655d796e085d29aa41647d33cfd090824778ecadd76a2bb11c`
and `4b12084f0cbbfcfe7c88ed45e8aac28fa20e813faf77f24b9c8d46bc7f9f5834`.

From this directory, run `python3 analyze_controlled.py s1_cpu` and
`python3 -m unittest test_controlled.py`. The analyzer first validates all
four blocks: protocol hash, schedule, unplugged/thermal state, boundary
samples, CPU backend, exact response count/order, prior arm outputs, and
protected spans. Mutation tests reject missing telemetry, wrong protocol,
arm/index, response count/order/output, and backend. The Q4 diagnostic's
original JSON SHA256 was
`556bf2c62e4d6d9449a930d644d3560a9c1e3d0a38e3c59a547d4d9465a1f026`.
