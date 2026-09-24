# Fast-engine research log (Pixel 10 Pro)

Baseline: branch head `94230cf`. Workload: `starling-bench` on the phone,
`STARLING_ENGINE=fast`, median of 3 runs after warm-up (see `.auto/measure.sh`).
Gates: fixture transcripts identical (G1), FLEURS WER within 0.2 pts (G2, at
milestones), `fast_weights_test` (G3), desktop RADV ≤ 10 % regression (G4).

| # | Hypothesis | Change | Before → after (median) | Gates | Verdict |
|---|---|---|---|---|---|
| 0 | Baseline | — | PK medium 2886 ms (mel 26, enc 2612, dec 248); MOSS short 7183 ms (enc+prefill 3675, decode 3280 @102.5 ms/tok); phone_ms 10069 | G1 captured | — |
| 1 | Autotuner mis-ranks tiles on PowerVR (synthetic shapes unrepresentative) | env sweep first: TILE 32,64,4,4 → PK 2907→2526 (−13 %); GEMV rows 8 already best (4/16/32/64/128 worse) | — | — | measured, see #2 |
| 2 | Same, structural | Added candidates {32,64,4,4},{48,64,4,4}; tune recording now barriers after every kernel + norm interleave; M 256→293; **vendor 0x1010 default tile 32,64,4,4 + gemv rows 8** (tuner kept for other vendors) | phone_ms 10069 → **9363** (PK 2886→2529, enc 2612→2288; MOSS 7183→6834, decode 99.2 ms/tok) | G1 ✓ | **keep** (`91dd41f`) |

Notes:
- Tuner on-device picks by isolated synthetic recordings: 3 attempts to make
  it representative (barriers, norms, odd M) all still rated 64,32,4,4 ≥
  32,64,4,4 while reality differs ~2×. Per-dispatch tuning on PowerVR is
  dominated by fixed overheads; end-to-end wall time is the only trustworthy
  signal → vendor-keyed measured defaults.
