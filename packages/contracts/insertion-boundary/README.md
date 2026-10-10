# Insertion-boundary contract

Deterministic boundary formatting for dictated text that lands in the middle
of existing text. When an insertion target can report the text around the
insertion point (an Android `InputConnection`, IBus surrounding text, a
desktop adapter that exposes it), delivery adjusts **only** the boundary: a
leading space and the case of the first letter.

The raw recognition text is never edited: the runtime delivers the
adjustment as a separate revision derived from the raw one (`provenance:
"insertion-boundary"`).

The Python oracle (`tests/insertion_boundary.py`), the Rust port
(`starling-processing`'s `boundary` module) and the Kotlin port in the
Android keyboard (`apps/mobile`, `processing/InsertionBoundary.kt`) all
replay `fixtures/boundary-cases.json`.

## Inputs and outputs

Inputs: `before` (text immediately before the insertion point), `after`
(text immediately after it), `showing_hint` (optional, default `false`),
`raw` (the dictated text) and `verbatim`.

Output: the adjusted text plus the rules that fired, each `leading_space` or
`first_letter_case`. When no rule fires, the output is byte-for-byte `raw`.
A fixture change's `detail` is an explanation for readers, not compared.

## Rules

1. **Verbatim.** `verbatim: true` → no changes at all.
2. **Hint text.** `showing_hint: true` means the field shows only its
   placeholder (Android's `isShowingHintText`): `before` and `after` are
   hint text, and the field is treated as empty (`before` reads as `""`).
3. **Leading space.** Prepend one U+0020 iff all hold:
   - `before` is non-empty;
   - its last character is neither whitespace nor in the opening set
     `( [ { " ' “ ‘ „ « 「 『 【 （` (straight `"` and `'` are ambiguous at an
     insertion point; this pins them as opening);
   - `raw` does not start with whitespace (never double-space);
   - not both the last character of `before` and the first of `raw` are
     Han, kana or CJK punctuation (U+3000–30FF, 31F0–31FF, 3400–4DBF,
     4E00–9FFF, F900–FAFF, FF01–FF0F, FF1A–FF20, FF5B–FF9F, 20000–3FFFF).
     Hangul is not in the set: Korean is spaced like Latin text.
4. **First-letter case.** Take the first cased character of `raw`; if it has
   a lowercase mapping different from itself, replace it with its **full**
   lowercase mapping (`İ` → `i̇`, Python `str.lower()` semantics), iff all
   hold:
   - the first whitespace-delimited token of `raw` is not protected (below);
   - `before`'s trailing whitespace contains no `\n` or `\r` (a new line
     behaves like the start of a field);
   - `before`'s last non-whitespace character is alphanumeric or one of
     `, ; : ) ] } ” ’ » 」 』 】 ） 》 ، ؛` (the last two are the Arabic comma
     and semicolon). Sentence enders (`. ! ? … 。 ！ ？ ؟ ।`) and the opening
     set keep the case.

   If the first cased character is already lowercase, nothing changes:
   later capitals are never touched.
5. **Trailing.** Never. The end of `raw` is not modified whatever `after`
   contains; `after` is part of the contract so future rules have pinned
   data.

**Protected first tokens** (no case change; the space rule still applies):
the token contains any of `_ / \ @ #` or starts with `www.` (paths, URLs,
emails, hashtags, snake_case); contains a camel hump (a lowercase letter
immediately followed by an uppercase one: `camelCase`, `iPhone`); is
all-uppercase with at least two letters (`NASA`); or is the English pronoun
`I`, alone or followed by a non-alphanumeric character (`I,` `I.` `I'm`
`I’ll`).

Scripts without case (CJK, Arabic, Hebrew, Devanagari) never get a case
change. Processing is on logical text (right-to-left text included): no
reordering, no normalization.

## Security

Surrounding text is read only where the platform exposes it without extra
permissions, and **never** for secure/password fields or fields marked
incognito (`IME_FLAG_NO_PERSONALIZED_LEARNING`); enforcing that is the
adapter's duty (`DeliveryAdapter::surrounding_text`, which answers
`SurroundingRead::Protected` for such fields without reading them). The
text is used for this decision only: it is not stored in history and not
sent to any processing provider.

## Delivery (desktop runtime)

- The adjusted text is a revision derived from the requested one: id
  `{revId}:boundary-{space|case|space-case}`, `provenance:
  "insertion-boundary"`, slot `derived` in `docs.get`, persisted in
  storage v2 (`disposition: "derived"`, `sources_json.derivedFrom`). It
  never advances the document head; the requested revision is not edited.
- A derived id is always a legal `msgId` (at most 128 bytes). When
  `{revId}:boundary-{rules}` would be longer (a `revId` over 108 bytes
  with `space-case`), the runtime emits
  `{revId prefix}.{digest}:boundary-{rules}`: `digest` is the 64-bit
  FNV-1a hash of the full unshortened id as 16 lowercase hex digits, and
  the prefix is the first `128 - 17 - len(":boundary-{rules}")` bytes of
  `revId`. (The runtime also bounds a rule list too long to keep whole,
  `{prefix of the full id}.{digest}`; today's three rule lists never
  need it.) See `delivery::derived_id`.
- `delivery.prepare{boundary}` chooses: `adjust` (default) applies the
  rules where the target reports surrounding text; `raw` is the explicit
  bypass that delivers the requested revision unchanged; `verbatim` is a
  verbatim mode's flag and disables every rule. Neither `raw` nor
  `verbatim` reads the surrounding text.
- The surrounding text is read at prepare and again at apply. If it
  changed in between, apply re-derives from the requested revision and
  delivers that; a stale adjustment is never delivered. A target that
  stops reporting text by apply time gets the requested text unchanged.

## Delivery (Android keyboard)

- The keyboard (`ui/BoundaryDelivery.kt`) reads `getTextBeforeCursor` and
  `getTextAfterCursor` right before its single `commitText`; password
  variations and `IME_FLAG_NO_PERSONALIZED_LEARNING` fields are never read,
  and verbatim modes skip the read. `showing_hint` is never set there: an
  `InputConnection` reports the editor's content, never its placeholder,
  so a field showing its hint already reads as `""`.
- Live composing text gets the boundary read when the composing region
  starts; the final re-reads, with the take's own composing text cut from
  the text before the cursor (when the cursor has left that region, the
  boundary is unknown and the text goes in unchanged).
- An adjusted delivery is stored on the recording as a derived revision
  (`provenance: "insertion-boundary"`, the rule kinds, the text it was
  derived from); the transcript and its revisions are not edited.

## Files

| File | Role |
|---|---|
| `boundary.schema.json` | Structure of one fixture case. |
| `fixtures/boundary-cases.json` | The pinned cases. |

```
uv run python -m pytest tests/test_insertion_boundary.py -q
cd apps/desktop-gpui && cargo test -p starling-processing --test boundary_conformance
cd apps/mobile && ./gradlew :app:testDebugUnitTest --tests '*InsertionBoundaryConformanceTest'
```
