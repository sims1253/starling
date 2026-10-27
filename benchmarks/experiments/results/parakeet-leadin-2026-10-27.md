# Silent lead-in before the Parakeet final pass (#494): null

FluidVoice prepends 0.25 s of zeros before its final Parakeet pass and
reports lower WER on their users' dictations. They attribute the gain to
the first word, and blame ~0.17 s of dead audio at the start of macOS
capture. On our audio a lead-in shows no first-word benefit. The one WER
gain it showed on untrimmed clips did not replicate on held-out clips
(preregistered). **Not shipped.**

Numbers: `parakeet-leadin-2026-10-27.json`. Harness:
`benchmarks/experiments/leadin_ab.py` (`prep` builds the clip sets,
`score` reports WER, first-word and first-two-word errors and a paired
bootstrap CI with seed 494, 2000 resamples). Binary: `starling-bench` at
`28d1ff66`, ggml CPU, 8 threads, one process per clip set, `--runs 1`.
Models: Parakeet TDT 0.6B v3 q8_0 (desktop) and q4pw-fullimx (mobile), and
Underdog Bark 3.0 Pro q4pw (a Parakeet fine-tune). Lead-ins: 0, 0.1, 0.25
and 0.5 s of digital zeros. Plain zeros, so no timestamp shift is
involved in the scoring.

## Clip sets

- **untrimmed**: FLEURS en_us test 0–99 + validation 0–199 (300 clips, as
  exported). The energy onset has a median of 510 ms (p10 120 ms).
- **trimmed**: the same clips, cut so the onset sits 50 ms after the
  first sample. Onset = first 10 ms frame 15 dB above the clip's
  10th-percentile frame.
- **trimmed, margin 0**: onset at the first sample, a harsher stand-in
  for capture that starts late.
- **held-out**: FLEURS en_us test 100–399, untrimmed, 0 vs 0.25 s only.

No real dictation takes were used. None with reference transcripts exist
on this host.

## Results (WER %, Δ vs 0 s with 95 % CI, first-word errors of 300)

| set | model | 0 s | 0.1 s | 0.25 s | 0.5 s | 1st word 0 / 0.1 / 0.25 / 0.5 s |
|---|---|---|---|---|---|---|
| trimmed | q8_0 | 6.28 | +0.21 [−0.17, +0.55] | +0.02 [−0.39, +0.40] | +0.02 [−0.37, +0.37] | 13 / 14 / 13 / 12 |
| trimmed | q4pw | 6.22 | −0.05 [−0.44, +0.31] | −0.06 [−0.48, +0.32] | +0.14 [−0.26, +0.52] | 14 / 15 / 15 / 14 |
| trimmed | Bark Pro | 7.62 | −0.02 [−0.32, +0.27] | −0.10 [−0.39, +0.19] | +0.08 [−0.27, +0.45] | 16 / 16 / 16 / 18 |
| margin 0 | q8_0 | 6.14 | +0.16 [−0.24, +0.55] | +0.19 [−0.22, +0.56] | +0.22 [−0.23, +0.64] | 15 / 15 / 15 / 14 |
| margin 0 | q4pw | 6.28 | −0.06 [−0.49, +0.35] | +0.11 [−0.29, +0.53] | −0.16 [−0.60, +0.25] | 15 / 17 / 16 / 14 |
| untrimmed | q8_0 | 6.26 | −0.08 [−0.35, +0.20] | −0.30 [−0.60, +0.00] | −0.19 [−0.51, +0.13] | 13 / 12 / 13 / 13 |
| untrimmed | q4pw | 6.22 | −0.11 [−0.44, +0.19] | −0.25 [−0.52, +0.00] | −0.13 [−0.42, +0.16] | 13 / 13 / 13 / 14 |
| untrimmed | Bark Pro | 7.63 | −0.14 [−0.43, +0.16] | −0.38 [−0.63, −0.13] | −0.37 [−0.67, −0.07] | 17 / 16 / 17 / 16 |
| held-out | q8_0 | 6.40 | | −0.06 [−0.33, +0.21] | | 14 / – / 18 / – |
| held-out | Bark Pro | 7.42 | | +0.01 [−0.18, +0.23] | | 19 / – / 20 / – |

- **First words.** No lead-in lowers first-word errors by more than one
  in 300. On the main sets they move by at most two either way. On the
  held-out set they rise with the lead-in (q8_0 14 → 18, Bark Pro
  19 → 20). On trimmed q8_0, the first-word errors are the same 13 clips
  at 0 and 0.25 s. Most are spellings and names ("Cuomo" → "huomo",
  "Kruger National Park" → "rukhur"). Three are word errors: a dropped
  leading "the", "now" → "not", "in" → "the". They are identical with
  and without the lead-in.
- **The untrimmed gain was not first-word.** It came from words later in
  the clip, and showed on untrimmed audio only. That set already starts
  with ~0.5 s of silence. 18 comparisons were run uncorrected, so a
  held-out replication was preregistered before running: on held-out
  clips 100–399, untrimmed, 0.25 s, the CI must lie entirely below 0 for
  both q8_0 and Bark Pro. It did not (−0.06 and +0.01). Treated as
  selection noise.
- **Cost.** 0.25 s is 2.6–2.8 % more audio on these ~9–9.5 s clips. The
  decode-time column in the JSON is not a measurement of this: other jobs
  shared the host.

## What this does not cover

Real desktop or Pixel dictation takes (no references here), the CTC/TDT
models other than Parakeet v3 and Bark Pro, and Qwen3/ARK. If a capture
path that drops its first ~0.2 s shows up, rerun `leadin_ab.py` on takes
from that path.
