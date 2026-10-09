# Personalization contract (v1)

Training-free personalization over the local correction records (#304):
vocabulary/replacement suggestions mined from repeated corrections, and
few-shot style retrieval for processing modes. The schema is
`personalization.schema.json` (definitions only); fixtures are in `fixtures/`.
The executable oracle is `tests/personalization.py`, replayed by
`tests/test_personalization.py`.

Inputs are `correctionRecord`s: the subset of a store_v2 `correction_records`
row this contract reads. `decision` is `accepted`, `rejected`, `reverted` or
`edited`; `transform_kinds` is the row's JSON column decoded to an array.
Only `accepted` and `edited` records with non-empty `raw_text` and
`final_text` and `secure_field != true` are used by either mechanism.

## 1. Suggestions

Inputs: correction records, the active `vocabulary` (strings) and `snippets`
(`{spoken, expansion}`), `min_frequency` (default `2`) and `min_consistency`
(default `0.8`).

1. **Extraction.** Records whose `transform_kinds` contain `rewrite` or
   `translate` are skipped: those transforms change wording on purpose.
   `raw_text` and `final_text` are split on whitespace, edge punctuation
   (`,.;:!?`) is stripped, empty tokens are dropped, and the token sequences
   are aligned case-insensitively. Each replaced run yields a
   `(source, target)` pair. Sources are lowercased, targets keep their
   casing, so casing-only changes are not mined.
2. **Thresholds.** `frequency` is the pair's count; `consistency` is
   `frequency` divided by the number of occurrences of the source as a token
   sequence in the mined records' raw texts (changed or not). A pair with
   `frequency >= min_frequency` and `consistency >= min_consistency` becomes
   a suggestion.
3. **Type.** Single-word source and target → `vocabulary`; otherwise
   `replacement`.
4. **Conflict** (first match wins, else `null`):
   - `already_in_vocabulary`: the target is in the vocabulary (case-insensitive).
   - `snippet_collision`: a snippet with the source as `spoken` expands to
     something else.
   - `competing_targets`: another target of the same source also reaches
     `min_frequency`.
5. **Output.** `id` is `sug_` + the first 16 hex digits of
   SHA-256(UTF-8 `source + "\n" + target`); `consistency` is rounded to 4
   decimals; `last_seen_utc` is the latest `decision_utc` of the pair.
   Suggestions are ordered by frequency desc, consistency desc, source.
6. **Review.** Suggestions start as `suggested` and are never enabled
   silently; the user `accepted`s or `ignored`s them.

## 2. Style retrieval

Inputs: a `retrievalRequest`, the record history, and the deleted (tombstoned)
capture ids. `max_examples` defaults to `3`, `max_characters` to `500`,
`personalization_enabled` to `true`.

1. **Toggle.** With `personalization_enabled == false` the result is empty.
2. **Eligibility.** A usable record (see above) whose `capture_id` is not
   deleted, and whose `mode_id` and `language` equal the request's. A `null`
   language matches only `null`, so examples never cross languages.
3. **Selection.** Candidates are ordered by `decision_utc` desc, then `id`
   desc. Each is added while fewer than `max_examples` are selected and the
   formatted context stays within `max_characters`; a candidate that does not
   fit is skipped and later, smaller ones are still considered.
4. **Format.** Injected into `context.personal_context`:
   ```
   [Style References]
   Input: <raw_text>
   Output: <final_text>

   Input: ...
   ```
   Texts are trimmed. `examples_used` lists the selected record ids in order;
   `character_count` is the length of `formatted_context` in Unicode code
   points (`0` when empty).

## Evaluation

`fixtures/evaluation.json` is a held-out set: applying a session's
`active_suggestions` (one pass, case-insensitive whole-word matches, never
inside inline code, a path or URL (whose segments may contain spaces), an
email or an identifier) must lower the total edit
distance to `ground_truth_target` and leave every `protected_spans` entry
intact.
