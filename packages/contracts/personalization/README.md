# Personalization Without Training Contract

Executable contract and specifications for training-free personalization (#305).
Covers vocabulary suggestions and few-shot style retrieval over local correction
data (#304, #299).

## Overview

Before training personal LoRAs or acoustic adapters (#306, #307), Starling implements
two zero-training mechanisms:

1. **Vocabulary Suggestions and Replacements**:
   Repeated user corrections (`asr_fix`, post-recognition edits) are mined for
   recurrent substitutions (e.g. `"sterling"` → `"Starling"`). When a pattern
   meets frequency and consistency thresholds, it is surfaced to the user as a
   suggested vocabulary entry or replacement snippet. Suggestions are **reviewed**
   and **never enabled silently**.

2. **Few-Shot Style Retrieval**:
   For processing modes (`clean`, `format`, `rewrite`, `translate`), the engine
   deterministically retrieves a small set of the user's own accepted historical
   examples and injects them into the model prompt under `[Style References]` in
   `context.personal_context`. Retrieval is strictly isolated by language, mode,
   and project, bounded by hard example and character budgets, and tracks provenance.

---

## 1. Suggestion Engine Rules

### Inputs
- Historical correction records from store v2 (`#304`).
- Existing active `vocabulary` (list of strings).
- Existing active `snippets` (list of `{spoken, expansion}`).
- Tuning parameters:
  - `min_frequency`: minimum occurrences of a substitution (default: `2`).
  - `min_consistency`: minimum ratio `count(src -> tgt) / count(src -> *)` (default: `0.80`).

### Invariants & Behavior
1. **Extraction**:
   - Compares `raw_text` and `final_text` on accepted/edited takes where the final
     differs from raw.
   - Extracts phrase-level substitutions `(source_phrase, target_phrase)`.
   - Normalizes whitespace; matches source phrases case-insensitively while preserving
     target casing.
2. **Threshold Filtering**:
   - `frequency >= min_frequency`.
   - `consistency >= min_consistency`.
   - Low-frequency or inconsistent edits are discarded.
3. **Target Type Classification**:
   - Single-token target with phonetically or orthographically similar source → `vocabulary`.
   - Multi-token target or phrase expansion → `replacement`.
4. **Conflict Detection**:
   - `already_in_vocabulary`: Target phrase is already in the mode's or profile's vocabulary.
   - `snippet_collision`: A snippet with the same spoken phrase already exists with a different expansion.
   - `competing_targets`: The same source phrase has multiple competing candidate targets meeting frequency thresholds.
5. **Review State**:
   - Status starts as `suggested`.
   - User actions: `accepted` (promoted to active dictionary/snippets) or `ignored` (suppressed).
   - Ignored suggestions are remembered and never re-prompted.

---

## 2. Few-Shot Style Retrieval Rules

### Inputs
- Current take request: `mode_id`, `language`, `project_id`, `personalization_enabled`.
- Candidate accepted examples from local history.
- Tombstoned/deleted capture IDs.
- Budgets: `max_examples` (default: `3`), `max_characters` (default: `500`).

### Invariants & Behavior
1. **Eligible Examples**:
   - `decision == "accepted"` or user-accepted edited take.
   - `secure_field != true` (secure/incognito takes are excluded).
   - Tombstoned/deleted takes are immediately excluded.
2. **Isolation**:
   - **Language isolation**: Candidate `language` must match current request `language`. Non-matching languages are never mixed.
   - **Mode isolation**: Candidate `mode_id` must match current request `mode_id`.
   - **Project isolation**: If the request specifies `project_id`, only examples matching that `project_id` (or unassigned if allowed) are retrieved. Cross-project leakage is forbidden.
3. **Deterministic Selection & Budget**:
   - Candidates are ordered deterministically by recency (`decision_utc` descending), tie-broken by `id`.
   - Examples are added sequentially until `max_examples` is reached or adding another example would exceed `max_characters`.
4. **Formatting**:
   - Formatted into `[Style References]` block:
     ```
     [Style References]
     Input: <raw_text>
     Output: <final_text>
     ```
   - Injected into `context.personal_context`.
   - Provenance records `examples_used` (array of record/capture IDs).
5. **Toggle**:
   - When `personalization_enabled == false`, returns empty `formatted_context` and `examples_used = []`.
