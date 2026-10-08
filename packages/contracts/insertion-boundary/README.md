# Insertion-boundary contract

Deterministic boundary formatting for dictated text that lands in the middle
of existing text (#341). When an insertion target can report the text
immediately around the insertion point (an Android `InputConnection`, the
Linux IBus surrounding-text capability, or a desktop adapter that exposes it),
the delivery step adjusts **only** the boundary: a leading space and the case
of the dictated text's first letter. Nothing else is ever touched.

The raw recognition text stays unchanged and one action away: the adjustment
is recorded as a delivery-time revision whose source is the raw revision
(`provenance: "insertion-boundary"`), never an edit of it.

This folder is frozen contract data. The Python oracle
(`tests/insertion_boundary.py` + `tests/test_insertion_boundary.py`) and the
Rust port (`starling-processing`'s `boundary` module + its conformance test)
replay `fixtures/boundary-cases.json` against the same rules, so the two
implementations can never test against different copies.

## Inputs and outputs

Inputs per case: `before` (text immediately before the insertion point),
`after` (text immediately after it), `raw` (the dictated text), and the
`verbatim` flag (a verbatim mode disables every rule).

Output: the adjusted text plus the list of changes that fired, each one
`leading_space` or `first_letter_case`. When no rule fires, the output is
byte-for-byte the raw text.

## Rules (in order; each rule is independent)

1. **Verbatim.** `verbatim: true` → no changes at all.
2. **Leading space.** Prepend one U+0020 iff ALL hold:
   - `before` is non-empty;
   - its last character is not whitespace;
   - its last character is not in the opening set
     `( [ { " ' “ ‘ „ « 「 『 【 （` — straight `"` and `'` count as opening
     here (at an insertion point they are ambiguous; this pins one reading);
   - the first character of `raw` is not whitespace (never double-space).
3. **First-letter case.** Lowercase the first cased character of `raw` iff
   ALL hold:
   - the first whitespace-delimited token of `raw` is not protected (below);
   - `before`'s trailing whitespace run contains no newline or carriage
     return (a newline boundary behaves like the start of a field: case is
     kept as recognized);
   - `before`'s last non-whitespace character is mid-sentence: alphanumeric,
     or one of `, ; : ) ] } ” ’ » 」 』 】 ） 》` — sentence enders
     (`. ! ? … 。 ！ ？`) and the opening set keep the case;
   - the character has a lowercase mapping different from itself; the
     **full** mapping is applied (multi-character lowercases such as `İ`
     → `i̇` included — the Python `str.lower()` semantics).
4. **Trailing.** Never. The end of `raw` is not modified whatever `after`
   contains ("no trailing-space changes inside words"); `after` is carried in
   the contract so future rules have pinned data, and so adapters can be
   validated on what they read.

**Protected first tokens (no case change; the space rule still applies):**
the first token contains any of `_ / \ @ #`, or contains `://`, or starts
with `www.` (paths, URLs, emails, hashtags, snake_case); or contains a camel
hump (a lowercase letter immediately followed by an uppercase one —
`camelCase`, `PascalCase` with a leading lower elsewhere, `iPhone`); or is
all-uppercase with at least two alphabetic characters (`NASA`, `MY_CONST` is
already covered by `_`); or is the English pronoun `I`.

Non-cased scripts (CJK, Arabic, Hebrew) naturally record no case change; the
space rule is script-agnostic. All processing is on logical text; no
reordering, no normalization, no combining-character changes.

## Security

Surrounding text is read only where the platform exposes it without extra
permissions, and **never** for secure/password fields or fields marked
incognito (`IME_FLAG_NO_PERSONALIZED_LEARNING`); that exclusion is an
adapter-side duty (see the runtime `DeliveryAdapter::surrounding_text` seam:
capability-reported, refused for secure targets). The text is used for this
decision only — it is not stored in history and not sent to any processing
provider.

## Files

| File | Role |
|---|---|
| `boundary.schema.json` | Structure of one fixture case. |
| `fixtures/boundary-cases.json` | The pinned cases: start of field, mid-sentence, after `.`/`?`/`!`, after a newline, after an opening quote/bracket, before existing punctuation, mid-word after-text, code identifiers, non-Latin scripts, RTL, verbatim, empties. |

Run the conformance suites with:

```
uv run python -m pytest tests/test_insertion_boundary.py -q
cd apps/desktop-gpui && cargo test -p starling-processing --test boundary_conformance
```
