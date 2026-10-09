# Insertion-boundary contract

Deterministic boundary formatting for dictated text that lands in the middle
of existing text. When an insertion target can report the text around the
insertion point (an Android `InputConnection`, IBus surrounding text, a
desktop adapter that exposes it), delivery adjusts **only** the boundary: a
leading space and the case of the first letter.

The raw recognition text is never edited: the runtime delivers the
adjustment as a separate revision derived from the raw one (`provenance:
"insertion-boundary"`).

The Python oracle (`tests/insertion_boundary.py`) and the Rust port
(`starling-processing`'s `boundary` module) both replay
`fixtures/boundary-cases.json`.

## Inputs and outputs

Inputs: `before` (text immediately before the insertion point), `after`
(text immediately after it), `raw` (the dictated text) and `verbatim`.

Output: the adjusted text plus the rules that fired, each `leading_space` or
`first_letter_case`. When no rule fires, the output is byte-for-byte `raw`.
A fixture change's `detail` is an explanation for readers, not compared.

## Rules

1. **Verbatim.** `verbatim: true` → no changes at all.
2. **Leading space.** Prepend one U+0020 iff all hold:
   - `before` is non-empty;
   - its last character is neither whitespace nor in the opening set
     `( [ { " ' “ ‘ „ « 「 『 【 （` (straight `"` and `'` are ambiguous at an
     insertion point; this pins them as opening);
   - `raw` does not start with whitespace (never double-space);
   - not both the last character of `before` and the first of `raw` are
     Han, kana or CJK punctuation (U+3000–30FF, 31F0–31FF, 3400–4DBF,
     4E00–9FFF, F900–FAFF, FF01–FF0F, FF1A–FF20, FF5B–FF9F, 20000–3FFFF).
     Hangul is not in the set: Korean is spaced like Latin text.
3. **First-letter case.** Take the first cased character of `raw`; if it has
   a lowercase mapping different from itself, replace it with its **full**
   lowercase mapping (`İ` → `i̇`, Python `str.lower()` semantics), iff all
   hold:
   - the first whitespace-delimited token of `raw` is not protected (below);
   - `before`'s trailing whitespace contains no `\n` or `\r` (a new line
     behaves like the start of a field);
   - `before`'s last non-whitespace character is alphanumeric or one of
     `, ; : ) ] } ” ’ » 」 』 】 ） 》`. Sentence enders (`. ! ? … 。 ！ ？`)
     and the opening set keep the case.

   If the first cased character is already lowercase, nothing changes:
   later capitals are never touched.
4. **Trailing.** Never. The end of `raw` is not modified whatever `after`
   contains; `after` is part of the contract so future rules have pinned
   data.

**Protected first tokens** (no case change; the space rule still applies):
the token contains any of `_ / \ @ #` or starts with `www.` (paths, URLs,
emails, hashtags, snake_case); contains a camel hump (a lowercase letter
immediately followed by an uppercase one: `camelCase`, `iPhone`); is
all-uppercase with at least two letters (`NASA`); or is the English pronoun
`I`, alone or followed by a non-alphanumeric character (`I,` `I.` `I'm`
`I’ll`).

Scripts without case (CJK, Arabic, Hebrew) never get a case change.
Processing is on logical text: no reordering,
no normalization.

## Security

Surrounding text is read only where the platform exposes it without extra
permissions, and **never** for secure/password fields or fields marked
incognito (`IME_FLAG_NO_PERSONALIZED_LEARNING`); enforcing that is the
adapter's duty (`DeliveryAdapter::surrounding_text`). The text is used for
this decision only: it is not stored in history and not sent to any
processing provider.

## Files

| File | Role |
|---|---|
| `boundary.schema.json` | Structure of one fixture case. |
| `fixtures/boundary-cases.json` | The pinned cases. |

```
uv run python -m pytest tests/test_insertion_boundary.py -q
cd apps/desktop-gpui && cargo test -p starling-processing --test boundary_conformance
```
