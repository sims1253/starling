# Agent instructions — CUDA validation & merge gate (RTX 5090)

You are validating the Starling engine program's CUDA paths and, if the
evidence below satisfies you, merging the four engine PRs. You act as the
independent verification gate: nothing merges on claims — only on evidence
you produced yourself on this machine.

## Context

- The four experiments (S01 on-device MOSS KV clearing, S02 byte-budget TDT
  graph cache, S03 EOS/budget termination, S11 exact-tail reuse) are CPU- and
  Vulkan(iGPU)-verified by the coordinator, with CPU test binaries green.
  What is NOT yet verified: the CUDA device paths and timing/quality evidence.
- Baselines: master `12b6548c96ecfa1c989abb7afd39e90080cfb92f`. The PRs:
  #196 (S01), #197 (S03), #198 (S02), #199 (S11) — each targets master,
  disjoint files, any merge order.
- This branch (`program/gpu-validation`) = all four merged + VALIDATION.md
  (the evidence runbook with result templates). Read VALIDATION.md first.

## Setup

```bash
git fetch origin && git checkout program/gpu-validation
git submodule update --init third_party/ggml
cmake -B build-cuda -DSTARLING_SERVE=ON -DSTARLING_GGML_CUDA=ON
CMAKE_BUILD_PARALLEL_LEVEL=8 cmake --build build-cuda --target \
  starling-serve device_cache_clear_test tdt_graph_budget_test \
  greedy_termination_test stream_session_test
```

## Gate A — correctness (must fully pass before anything else)

```bash
./build-cuda/device_cache_clear_test    # must report backend=CUDA0
./build-cuda/tdt_graph_budget_test
./build-cuda/greedy_termination_test
./build-cuda/stream_session_test
```
Any failure = stop, record, do NOT merge; paste full output into the report.

## Gate B — S01 timing evidence (per VALIDATION.md §B)

Run starling-serve from THIS branch and from plain master with the same MOSS
GGUF (bf16-exact if it fits comfortably, else q4 — record which + SHA256),
same fixed WAV set (≥3 durations, record SHA256s), ≥5 repeats, cold+warm,
with the structured timing trace enabled (docs/native-serving.md). Deliver:
per-request trace tables for both, the KV-clear host-byte elimination on the
branch, and latency not-worse. Identical transcripts required; any diff is a
blocking finding. A no-gain result is acceptable — report it honestly.

## Gate C — S02 budget evidence (VALIDATION.md §C)

Parakeet + varied-length audio under STARLING_TDT_GRAPH_BUDGET_MB=64/128/512:
nvidia-smi VRAM plateaus, no unbounded growth across many distinct lengths,
identical transcripts across budgets.

## Gate D — S03/S11 spot evidence

S03: real MOSS A/B vs master on the Gate-B set (identical transcripts).
S11: confirm tail_cache_hits increments (log line or serve output) on
preview-then-commit with unchanged audio, and zero engine calls avoided when
one sample is appended.

## Merge decision (only if Gates A-D satisfy you)

Merge in any order; suggested: #196, #197, #198, #199 via
`gh pr merge <n> --merge` (merge commits match repo convention). If any
PR shows unresolved conversations, the coordinator's control plane
(docs/program/tasks.json on program/wave-a) records every finding as fixed
or evidence-rebutted — spot-check anything you find suspicious rather than
trusting it. Do NOT merge if any gate failed; instead write the failure up.

## Deliverable

`docs/program/evidence/gpu-validation-5090-<date>.md` per the VALIDATION.md
template (env, gates A-D with raw pastes, merge decisions with commit SHAs).
Commit it to program/gpu-validation and push (this is authorized).
