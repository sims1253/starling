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
language, NumPy `default_rng(0)`, and a 2.5th–97.5th percentile interval.
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
