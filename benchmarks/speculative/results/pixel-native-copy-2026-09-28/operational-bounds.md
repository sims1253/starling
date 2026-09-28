# Native-copy phone pilot execution bounds

Declared before any native-copy phone process at 2026-09-28 12:41 UTC. This
operational note does not change the locked workload, parity gate, or comparison
rule in `protocol-draft.json` (SHA-256
`2e497202a1dd49e1bd49282f3e918b7b04b98de40056728bfa75719045e440b9`).

The protocol's 13-minute estimate was optimistic given the earlier Pixel BF16
quant pilot. Expected wall time is roughly 5–8 minutes for two parity-preflight
processes and 15–25 minutes for four latency processes, plus bounded cooling.
The runner caps the preflight stage at 25 minutes and the performance stage at
45 minutes. Each health-readiness wait and HTTP request also caps at 180 seconds
or the remaining stage time, whichever is shorter. Any timeout leaves incomplete
raw records and forbids a directional latency claim. A 30-second gap separates
processes. A temperature 37–41.9°C after up to 180 seconds of cooling permits
only descriptive observations; >=42°C or thermal status >=2 aborts.

The user-paused Spotify session and phone settings remain untouched. The same
phone, model, six CPU threads, fixture hashes, ID capture, and native timing
fields apply to both arms. Charge counter readings are descriptive only; the
previous quant pilot showed stale intervals over 110 seconds, so this pilot has
no energy verdict.
