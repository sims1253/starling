# Pixel CPU ASR and optional S1 residency snapshot

This descriptive issue #316 observation used one Pixel 10 Pro on 2026-09-28.
It compares no owned Starling process, a warmed **Parakeet IQ2 baseline ASR**
server alone, and that same ASR server with a separately warmed **S1 Q4_K_M**
server resident. Both selected the GGML CPU backend with six threads. The
[protocol](protocol.json) was saved before the run (SHA256
`2fe9df844f1b866a8dee64bdc32cee20990997ebee2168988cf5a7ec95383f73`).
The ASR warmup used the server's five-second silent clip; S1 used its built-in
probe text. No user transcript or extra workload was sent. Each stage has
three [raw samples](samples.jsonl) about five seconds apart; the
[analysis](summary.json) checks sample order, phone state, CPU backend logs,
process IDs, and cleanup.

| Stage | Combined process PSS | Android `MemAvailable` samples |
| --- | ---: | ---: |
| No Starling process | — | 4.65–4.67 GiB |
| ASR alone | 385,807 kB (0.37 GiB) | 4.24–4.27 GiB |
| ASR with optional S1 | 1,497,164–1,497,166 kB (1.43 GiB) | 2.75–3.26 GiB |
| After both exited | — | 4.11–4.30 GiB |

In the joint stage, ASR PSS was 384,376–384,377 kB and S1 PSS was
1,112,788–1,112,789 kB. Their RSS sum was 1,505,732–1,505,748 kB;
individual process `VmHWM` values were 390,344 kB for ASR and 1,120,516 kB
for S1. These are sampled resident values and individual high-water marks,
not a continuously monitored joint peak. `MemAvailable` declined across the
joint samples while those process PSS values stayed nearly flat. The cause
was not measured, so its movement cannot be wholly attributed to these
servers. `MemAvailable` includes reclaimable memory and is not `MemFree`.

The sampled combined PSS is below the user's approximate **2 GiB soft
resident-size target** for ASR with optional S1. It is one shell-process,
screen-off, CPU-backend observation, not an acceptance threshold or evidence
of foreground responsiveness, app process-limit allowance, or the Pixel
fast/Vulkan engine's co-residency. The phone was unplugged, screen off, at
27.5°C and thermal status 0 throughout the samples. The [record](record.json)
shows both servers reached ready after their built-in warmups (8.3 s ASR;
14.2 s S1) and neither survived cleanup. No mobile behavior or default was
changed.

The exact host-specific [runner](run_residency.py), [ASR log](asr.log),
[S1 log](s1.log), model/server hashes in the protocol, and
[file hashes](hashes.sha256) preserve provenance. The runner imports phone
state/PID helpers from the separate native-copy pilot workspace; replay on
another host requires adapting its local paths and ADB serial. Run
`python3 analyze.py` here to validate and regenerate the summary without
accessing the phone.
