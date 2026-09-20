# Insight event contract (v1)

Local-first telemetry events and metric semantics for Starling Insights
(issue draft E28). The schema is at `schema.json`; scenario fixtures at
`fixtures/`. The executable metric oracle is `tests/insight_metrics.py` in the
repo root test suite — it is the *contract*, not the production aggregator:
the production implementation is the native runtime owned by E17 and must
reproduce the semantics frozen here and in `tests/test_insight_events.py`.

## Event kinds and identity

| Event | Carries | Meaning |
|---|---|---|
| `capture_finalized` | `capture_id`, `sample_count`, `sample_rate`, `complete_audio`, `mode_id`, `reporting_timezone` | One retained take of source audio. The only event that makes a take exist for metrics. |
| `recognition_selected` | `capture_id`, `attempt_id`, `selection_seq`, `lexical_words`, `raw_words`, `tokenizer`, `post_stop_ready_ms` | A recognition attempt was selected as final for the capture. Retries carry a new `attempt_id` and higher `selection_seq`. |
| `transformation_completed` | `capture_id`, `revision_id`, `revision_seq`, `transformation_kind`, `change_counts` | An authored revision of the output (model authoring, snippet expansion, user edit, dictionary substitution). |
| `delivery_recorded` | `capture_id`, `delivery_id`, `delivery_seq`, `status`, `output_words`, `generated_words` | An output was delivered to a target. `submitted_unconfirmed` and `confirmed` are distinct; submission is never counted as confirmed insertion. |
| `capture_deleted` | `capture_id` | Tombstone. Removes all attributable events/rollups of the capture. |

Identity rules (frozen):

* **Idempotency key = `event_id`.** A retry or sync replay must resend a
  byte-identical payload under the same `event_id`; identical replays dedupe.
  The same `event_id` with a different payload is a **conflict** and an error,
  not a harmless replay.
* **Selection:** the highest `selection_seq` per capture is the selected
  recognition. A retry revises the word count; it never adds another take's
  worth of words.
* **Transformation revisions:** the highest `revision_seq` per `revision_id`
  wins; distinct `revision_id`s are distinct passes and both count.
* **Delivery:** the highest `delivery_seq` per `delivery_id` is the final
  acknowledgement state.

## Metric semantics (frozen)

* **WPM denominator.** `recognized_words_per_captured_minute` =
  sum of eligible lexical speech words / sum of the corresponding
  **silence-inclusive captured minutes** (sample frames / sample rate of
  takes that have a selected recognition). It is labeled "recognized words
  per captured minute", not a speaking-speed truth. It is a **weighted
  total** over the population — never an average of per-take rates. A
  VAD-normalized speech rate (speech-active seconds under a versioned VAD
  policy) is a distinct metric that must be reported under its own name; it
  is never silently substituted for the capture denominator. Cross-language
  rates are only comparable while the `tokenizer` set has one element;
  otherwise the rate is `null` rather than corrupted.
* **Recognized words vs generated output.** `recognized_words` counts only
  words recognized from speech (selected final attempt, `lexical_words`).
  `generated_words` on deliveries (model-authored or snippet-expanded words)
  and `output_words` are reported separately and never enter WPM, completion
  or the time proxy.
* **Completion vs delivery.** Completion = captures with a successful final
  ASR result / eligible captured takes; an empty recognition stays
  distinguishable from a nonempty success. Delivery statuses are
  `confirmed`, `submitted_unconfirmed`, `failed`, `conflict`, `cancelled`;
  only explicit `confirmed` acknowledgements count as confirmed insertion.
* **Change counts are not corrected errors.** `change_counts` reports
  structural / user / dictionary / snippet / style changes by type.
  Without a reference transcript, accuracy is unknown, so changes are never
  labeled "mistakes fixed", and fewer changes never imply more accuracy.
* **Time-saved proxy.** `typing_time_comparison_seconds` =
  recognized speech words × 60 / user typing WPM baseline − eligible captured
  seconds − non-overlapping post-Stop wait seconds (only waits that are
  actually observed; streaming compute that overlapped speech is not
  subtracted twice). Assumptions: the user baseline is set by the user; the
  estimate **excludes unobserved correction time** and says so. It is
  `null` (not zero) when the baseline is unset or any wait is unknown, and
  it **can be negative** — negative results are displayed, never clamped.
  Generated/snippet words never count toward the typing baseline.

## Deletion, reset, export, timezone

* Deleting a capture emits `capture_deleted`; all attributable events and
  derived rollups for that `capture_id` are removed. Tombstones dominate
  stale replays: a synced pre-deletion event can never resurrect the
  capture's totals (v1 has no restore operation; a future restore must be an
  explicit operation). If a user separately authorizes retaining
  de-identified historical totals, that retention mode must be stated
  explicitly at delete time; v1 fixtures model full removal.
* Reset is starting aggregation from an empty event log; every aggregate is
  plain JSON, so export is a serialization of the aggregate (tested by a
  round-trip).
* Activity-calendar grouping is by `capture_finalized.occurred_at`
  converted to a declared reporting timezone (DST-aware; the same UTC
  instant can fall on different local days in different timezones, and the
  DST fallback hour is disambiguated by offset, not naive wall time).

## Privacy

No event ever contains raw text, selections, clipboard contents, paths,
window titles, unrestricted URLs or secrets. The schema closes every event
kind (`additionalProperties: false`), so an event carrying any extra
free-form string field is structurally invalid — this is enforced by tests
that poison valid events with such fields. Provider/model IDs, mode IDs and
timezone names are constrained tokens, not free text. "Local" does not mean
"anonymous": per-app grouping requires its own setting and is not part of
this schema.

## Fixtures

| File | Freezes |
|---|---|
| `basic-session.json` | Two takes, one retry replacing 90→100 words, confirmed vs submitted delivery, model + snippet changes; WPM 132, proxy 75 s at 50 typing WPM. |
| `sync-replay.json` | Same session reversed plus exact-duplicate replays — counts identical. |
| `deletion-propagation.json` | A third take tombstoned mid-stream with a stale post-deletion replay — totals equal the basic session. |
| `timezone-shift.json` | Takes around the Oct 25 2026 Berlin DST fallback and a UTC/Berlin day boundary. |
| `negative-proxy.json` | 1000 s post-stop wait → proxy −920 s; also an incomplete capture (`complete_audio: false`). |
| `generated-output.json` | 110 speech words vs 145 generated/240 output words; transformation retry replacing change counts; unknown wait suppresses the proxy. |
