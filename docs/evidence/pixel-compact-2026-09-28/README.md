# Pixel 10 Pro compact Parakeet validation

This record tests the sealed
[`parakeet-compact-validation-v1.json`](../../../quants/protocols/parakeet-compact-validation-v1.json)
gate on the phone's actual fast Vulkan path. The 600 FLEURS validation clips
are the same decoded audio and references used in the CPU and CUDA records.
The phone was a Pixel 10 Pro running Android 17 build `CP3A.260905.009`.
The selected GPU was PowerVR D-Series DXT-48-1536 MC1. The Android binary
was built from `2a3bda4` with `STARLING_FAST=ON`; its SHA256 was
`e1a6b3b0f7ff04b3e5a03a62aa5c55deeb2bf0d8bbd4ab4ee80b202755044e4a`.
Both arms forced `STARLING_ENGINE=fast`. The stderr logs confirm the *selected*
fast/Vulkan device. The generic stdout `backend=cpu` field reports the
separate GGML backend and does not identify the fast engine.

The [baseline](baseline-pixel-official.json) and
[embedding](embedding-pixel-official.json) records contain all 600 clip IDs,
decoded-audio SHA256s, references, hypotheses, and unrounded per-clip WERs.
The [verdict](embedding-pixel-verdict.json) was calculated by
`benchmarks/compare_quant_wer.py` with 10,000 paired bootstrap samples per
language, Python `random.Random` seeded with the sealed protocol's `20260928`
plus the cohort index (EN `20260928`, DE `20260929`), and a linearly
interpolated 2.5th–97.5th percentile interval.
The upper bounds are 0.000 pp EN and +0.150 pp DE, both below the strict
+0.2 pp margin. The candidate saved 15,402,752 stored bytes. The
[hypothesis differences](changed-hypotheses.json) list all 13 changed clips;
five EN hypotheses changed despite equal mean EN WER. One DE clip changed WER
by +15 pp, which makes the 300-clip DE mean +0.050 pp.

The raw `*.stdout` and `*.stderr` files in this directory retain every phone
transcript and fast-engine load line. The baseline process was stopped after
the first 50 corpus clips when one decode ran much longer than its audio;
the resumed baseline process began at the next clip. Each fresh process had
one separate `short.wav` warmup. The candidate used one fresh process. The
[phone scorer](../../../benchmarks/fast_engine/score_phone_quant.py) enforces
one warmup per process, exact sorted corpus coverage, no duplicate clip IDs,
matching WAV duration, and a fast/Vulkan device line in each stderr file.
It generated the two `*-official.json` files; the standard comparator then
generated the verdict. The model, source F32, imatrix, protocol, scorer, and
benchmark hashes are in each per-arm record. The two IQ2 model SHA256s are
`bfee29b2b3419b387c8bc1b28a1772b506dd57aa9cfd879b2530bc287d93b0d3`
and `7723f275a11dd191262e43afd3027a136785ace7af7e6762b0fbd8c5d24b0a55`.

The fast engine logs report a 718.6 MiB Vulkan weight allocation for **both**
arms. The `*-monitor.jsonl` samples show process PSS, `/proc` smaps PSS/RSS,
GL mtrack, battery gauge, and thermal state during the quality sweeps. They
are not matched allocation histories: the baseline had a process restart,
while the candidate did not. `/proc` smaps PSS omits the separately reported
graphics allocation, and `dumpsys meminfo` total PSS includes graphics
mtrack; do not add them together. The battery gauge reversed as the phone
began charging; thermal state rose and the display woke. Latency and energy
comparisons from these sweeps are invalid. Their WER pairs remain valid.

The [memory-only protocol](matched-memory-protocol.json) (SHA256
`bc4bf72540753d417d7e507b7d6bdea75d11c5d0c09bad32cdff0a8baf22bc7c`)
was saved before a new baseline/candidate pair. The original runner at
`acc6b84` was used for this evidence; the current [runner](run_memory.py)
uses `STARLING_ADB_SERIAL` (and optional `STARLING_ADB` /
`STARLING_DEVICE_DIR`) and a unique process marker for PID attribution. It
also keeps partial stdout on failure. The measured run
used one fresh fast-engine process per arm, the same `short.wav` warmup, then
eight timed short and eight medium runs; it sampled memory after the first
measured short result while the process remained alive. The raw stdout and
stderr logs and [baseline](baseline-a-matched.json) and
[candidate](embedding-a-matched.json) snapshots are included. Both snapshots
had the display off, AC charging, battery temperature 38.8°C, and thermal
status 1. This pair used two GPU loads after the seven prior health/quality
loads; a third bracket load was omitted because phone activity already
prevented a controlled latency estimate.

| Warmed short-fixture snapshot | Baseline | Q8 embedding |
|---|---:|---:|
| `dumpsys meminfo` GL mtrack | 783,232 kB | 783,232 kB |
| `dumpsys meminfo` total PSS | 838,895 kB | 839,229 kB |
| `/proc` smaps PSS | 55,714 kB | 56,052 kB |
| `/proc` smaps RSS | 69,208 kB | 68,684 kB |
| Fast Vulkan weights, load log | 718.6 MiB | 718.6 MiB |

The candidate's total PSS was 334 kB higher in this pair. This small
difference gives no evidence of a resident-memory saving, but one pair cannot
estimate a distribution or a peak. The graphics allocation and fast-engine
weights were identical. No co-residency budget exists for this comparison.
Raw warm timings are retained in the `*-matched.stdout` files for inspection;
they are not a latency verdict. Charging and unrelated phone activity also
exclude an idle-subtracted energy estimate.

## Controlled latency and energy attempt

After the user paused Spotify and left the unplugged phone idle, we saved a
[four-block paired protocol](controlled-fast-attempt/protocol.json) before
running it. The planned fast-engine order was baseline, Q8 embedding, Q8
embedding, baseline, with 72 short and 72 medium requests in each fresh
process. Its SHA256 is
`c66f7614fa44c34884059f80b64794905b96167cada6f102d2972756b6b0e708`.
The [exact runner](controlled-fast-attempt/run_controlled_v1.py) has SHA256
`0c773a6351bc70b59e02cfa70810000dc5aaa87a816d28e268900ceb805f210c`.

The first baseline block never completed. Its [stdout](controlled-fast-attempt/bench.stdout)
contains only short-fixture runs 0–8, with the same transcript in every
completed run; the next result did not appear after more than two minutes.
The fast/Vulkan PowerVR engine and 718.6 MiB weight allocation are confirmed
in [stderr](controlled-fast-attempt/bench.stderr). The owned benchmark process
was then terminated without rebooting or changing phone apps. The
[result](controlled-fast-attempt/result.json) marks the block incomplete
(`bench exit 143`). The [battery and environment samples](controlled-fast-attempt/battery-state.jsonl)
show discharging, display off, and thermal status 0 at the sampled points.
The exact cause of the stall is undetermined; it recurred despite idle,
unplugged conditions. No candidate block ran, so this attempt yields **no
paired latency or energy estimate**. The earlier 600-clip WER verdict and
matched memory observations above do not depend on this incomplete run.
