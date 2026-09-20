# Fidelity corpus contract (v1)

Audio-backed fidelity gate fixtures for the six original dictation failure
modes plus the 20 September revision (issue draft E06). Files:

* `corpus.json` — fixture manifest (validated by `corpus.schema.json`)
* `mutants.json` — seeded defects with the gate that must catch each
  (validated by `mutants.schema.json`)
* executable scorers: `tests/fidelity_scorers.py` + harness
  `tests/test_fidelity_corpus.py` in the repo test suite (stdlib + `math`
  only; no ML, no audio decoding)

## What this corpus is — and is not

**Every fixture is synthetic.** Expected content and reference outputs were
authored by hand; audio is described only as synthesis parameters
(sine/noise speech stand-ins and silent gaps, see `tests/fidelity_audio.py`).
Real consented, licensed human speech is a tracked **future gap**: adding it
requires its own consent/provenance record and a schema version bump.
Synthetic audio here is **transport and stitch evidence only** — it proves
region annotation, chunk splitting and reassembly behavior. It is never
ASR-quality evidence: no model was run against this corpus, and no WER or
accuracy number for any Starling model may be derived from it.

The `reference_outputs` are clean synthetic system outputs used to prove the
gates have no false alarms. Scoring real pipelines against the same expected
contracts is the job of the capture/transport/engine/stitch/refine pipeline
runs; this package freezes the *definition of caught* independently of
candidate code, so agent-generated changes cannot redefine success.

## Fixture categories

`resumed-numbering`, `technical-terms`, `self-correction`, `negation`,
`intentional-filler`, `short-answers`, `long-passage` (10-minute layout with
chunk boundaries, pauses and a repeated phrase),
`multilingual-code-switching`.

Each fixture declares `languages`, `duration_seconds`, `provenance`
(const `synthetic` — enforced by the schema), `synthesis` segments, and:

* **expected per-stage contracts** — `expected.capture` (speech/silence
  region annotations), `expected.asr`, `expected.authored`,
  `expected.delivered` (critical tokens, negation spans, entities, numeric
  values, list-continuation, filler and short-answer expectations,
  correction contracts, final/superseded phrases).
* **per-stage thresholds** in `stages` — the four stages
  (capture completeness, ASR content, authored intent, delivered text) are
  scored separately and never merged into a single average.

## Gates

| Gate | Catches | Rule |
|---|---|---|
| `speech_region_coverage` | dropped-tail, missing frames | Intersection of captured segments with expected speech regions, divided by total speech duration. Silence is excluded from the denominator — missing silence is **not** an omission. Timestamps or token timing claims are ignored: only real segment spans count, so declaring timestamps over a gap proves nothing. |
| `critical_token_recall` | false-overlap, truncated refinement | Every critical token must be present (recall 1.0 by default). |
| `negation_retention` | dropped negation | A negation span survives only if the negator AND its governed content are present. Finding governed words in returned text without the negator is not retention. |
| `entity_accuracy` / `numeric_accuracy` | term and number damage | Exact token match for declared entities; canonical digit-string match for numbers. |
| `list_continuation` | wrong renumbering | Numbered markers must continue from `prior_index` (5 after 4), in order; a restart at 1 or a skip is a failure. |
| `filler_retention` | disfluency over-correction | Fillers declared intentional must survive recognition (and authored output in verbatim mode). |
| `short_answer_retention` | dropped letter/one-word answers | Each declared short answer (down to single tokens) must be present. |
| `correction_resolution` | replacement vs disjunction confusion | A `replacement` correction requires the final phrase present and the retracted phrase gone; a `disjunction` requires all alternatives kept. Self-corrections must not be flattened into conjunctions. |
| `revision_freshness` | stale thread updates | The delivered text must contain `final_phrases` and must not contain `superseded_phrases`. |
| `no_duplicate_segments` | duplicate insertion | The delivered token stream must not repeat any contiguous run of 6+ tokens; short intentional repetitions (e.g. a repeated "checkpoint saved") must not be flagged. |

## Seeded mutants

`dropped-tail`, `dropped-negation`, `wrong-renumbering`, `false-overlap`,
`truncated-refinement`, `duplicate-insertion`, `stale-thread-update` —
each names a transformation applied to a clean reference output plus the gate
expected to fail. The harness asserts: the declared gate fails on the mutant,
and every gate passes on the clean output (false-alarm contract).

## Reporting rules

Outcomes against this corpus are published as measured / inconclusive /
untested per stage. Nothing in this package may be promoted into a quality
claim about recognition accuracy, device performance or end-user experience;
those require the real-audio gap to be closed and full pipeline runs with
model/quant/runtime/device/config hashes recorded (extends #50/#64/#169/#175;
held-out acceptance audio stays independent of optimization).
