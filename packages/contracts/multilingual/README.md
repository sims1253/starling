# Multilingual and code-switching text contract (v1)

Language/locale descriptors, a code-switching segment model over utterances,
and frozen locale-aware text transforms (issue draft E24). Files:

* `language.schema.json` — the contract (validated by both `jsonschema`
  and the repo's minimal fallback `tests/minischema.py`)
* `fixtures/code_switching.json` — mixed-language utterances, script
  mixing, an RTL run inside LTR text, and an astral character
* `fixtures/locale_text.json` — number, quotation-mark and casing locale
  expectations as exact transforms
* executable oracle: `tests/multilingual_contracts.py` + harness
  `tests/test_multilingual_conformance.py` (stdlib only)

Every fixture carries `provenance: "synthetic"` (schema-enforced `const`,
same policy as the fidelity corpus).

## What this contract guarantees — text-level correctness only

1. **Language descriptors.** Every language appearing in a fixture is
   declared with a BCP-47-shaped `tag` (same pattern as the fidelity
   corpus `languages` field), an ISO 15924 `script`, a `direction`
   (`ltr`/`rtl`) and a `display_name`. Tags are canonically cased
   (lowercase primary, Titlecase script, uppercase region), oracle-checked.
2. **Code-switching segments.** An utterance is partitioned by segments,
   each carrying a per-segment `language` and a `span`. Spans follow the
   mode-routing `decision.schema.json` convention: `span_encoding` is
   `unicode_codepoints`, half-open `[start, end)` indices from 0 into the
   code-point sequence — explicitly **not** native UTF-16 ranges. The
   oracle proves the spans are sorted, non-overlapping, non-empty, and
   cover the utterance exactly; `ml-astral-01` carries U+1F680 (one code
   point, two UTF-16 units) so a UTF-16 mis-index would fail loudly, and a
   dedicated test re-indexes the fixture with UTF-16 offsets to prove the
   oracle catches that mistake.
3. **Round-trip identity.** Concatenating the segment texts reconstructs
   the utterance byte-for-byte; each segment `text` equals its code-point
   slice. Splitting and rejoining language runs is lossless — the
   authoring layer must never need to translate or transliterate to
   round-trip text through segmentation.
4. **Stored text is NFC.** Utterances and segments are Unicode NFC so
   spans, diffs and selection ranges stay stable under editing. (NFC
   closure under concatenation is not assumed in general; each stored
   string is checked individually.)
5. **Bidi is a display concern.** `ml-rtl-in-ltr-01` stores a Hebrew run
   inside an English utterance as ordinary logical-order code points.
   Display reordering must never change stored offsets.
6. **Frozen locale rules as exact transforms** (see tables below). The
   fixtures pin exact expected strings; the oracle re-derives them from
   the rule tables and also proves the rules invert (round-trip back to
   the canonical source rendering). There is no silent English formatting
   assumption: parsing a number under the wrong locale's rule is an error,
   never a best-effort reinterpretation.

### Number rules (v1: canonical non-negative plain decimals)

| Locale | Rule id | Digits | Thousands | Decimal | Grouping |
|---|---|---|---|---|---|
| `en-US` | `latn-group3-comma-decimal-dot` | `0-9` | `,` | `.` | 3-3-3 |
| `de-DE` | `latn-group3-dot-decimal-comma` | `0-9` | `.` | `,` | 3-3-3 |
| `en-IN` | `latn-group322-comma-decimal-dot` | `0-9` | `,` | `.` | rightmost 3, then 2s (`1,23,45,678.9`) |
| `fa-IR` | `arabext-group3-thousands-decimal` | U+06F0..U+06F9 | U+066C | U+066B | 3-3-3 |

Basis: the separator/grouping conventions these locales are commonly
documented with (CLDR-derived practice), frozen here per rule id so a
consumer implements the frozen rule, not "whatever the platform locale
data does today".

### Quotation-mark rules

| Locale | Rule id | Exact code points |
|---|---|---|
| `en-US` | `double-curved` | `“` U+201C … `”` U+201D |
| `de-DE` | `low-open-high-close` | `„` U+201E … `“` U+201C |
| `fr-FR` | `guillemets-nbsp-inside` | `«` U+00AB + U+00A0 … U+00A0 + `»` U+00BB |
| `ru-RU` | `guillemets-no-inner-space` | `«` U+00AB … `»` U+00BB |
| `ja-JP` | `corner-brackets` | `「` U+300C … `」` U+300D |

### Casing rules

| Locale | Rule id | Basis |
|---|---|---|
| `tr-TR` | `unicode-specialcasing-tr-az` | Unicode `SpecialCasing.txt`, "Turkic mappings" (tr, az): upper `i`→`İ` U+0130, `ı`→`I`; lower `I`→`ı` U+0131, `İ`→plain `i` U+0069 (overriding the unconditional default `İ`→`i`+U+0307) |

Python's `str.upper`/`str.lower`/`str.casefold` are locale-independent
default Unicode case conversions and **cannot** produce the Turkish
results (`"i".upper() == "I"`, `"İ".casefold() == "i\u0307"`). `str.casefold`
is not a Turkish caser; the oracle implements the tr/az mappings as an
explicit per-code-point map, and a test freezes the difference.

## What this contract does NOT guarantee

ASR recognition quality on these fixtures is a future model-gated gap and
is not implied by these contracts. No model was run against these
utterances, no WER or accuracy number for any Starling model may be
derived from this package, and the closed schema vocabulary (closed by
`additionalProperties: false` everywhere, oracle-checked) cannot even
express a quality claim. Also out of scope, as declared gaps:

* **Recognition/decoding**: whether any engine recognizes the languages,
  code-switches, or respects a language hint. The served API today does
  not accept language selection at all (see the adoption map).
* **Real audio**: every fixture is synthetic text; consented real
  multilingual speech needs its own provenance record and a schema
  version bump (fidelity-corpus policy).
* **Runtime/UI behaviors from E24 that need live clients**: IME
  composition, selection/diff/insertion survival under bidi editing,
  per-profile locale plumbing, localized correction/snippet semantics.
  This package freezes the text-level semantics those features must
  preserve; their runtime implementations and their UI tests are future
  work.
* **Translation**: nothing here translates. Mixed-language text is
  preserved as spoken (the round-trip identity is the formal statement).
* **Full CLDR**: the rule tables are frozen per named locale, not a
  general locale-data implementation. Adding locales means adding frozen
  rules plus fixtures.

## Adoption map (who consumes language fields today)

* `packages/contracts/capabilities/capability.json` —
  `features.language_selection: false`, and the schema freezes it as
  `{"const": false}`: the served API rejects language selection (HTTP 400
  per `docs/api.md`). This contract is the future consumer-side
  vocabulary for when that flips true.
* `packages/contracts/fidelity-corpus/corpus.json` — every fixture
  carries a `languages` array (BCP-47 subtag pattern, shared with this
  schema's `languageTag`), including the `multilingual-code-switching`
  fixture `fx-multi-01` (de/en). The conformance test asserts every
  corpus language code has a descriptor here.
* `packages/contracts/mode-routing/decision.schema.json` — the
  `unicode_codepoints` span convention this package reuses for segment
  spans.
* `docs/models.md` — model cards advertise language counts
  (ARK-ASR-0.6B "19 languages", Voxtral-Mini "13 languages");
  parakeet-unified-en is English-only. These are card claims, not decoder
  language control.
* `apps/desktop-gpui` — no language selector exists in the client today (the
  E24 finding); when one lands it should be driven by these descriptors
  plus capability flags, not a hardcoded list.
* `s1-mini` normalizer (`/normalize`) — performs truecasing and number
  formatting as text operations; the frozen casing/number rules here are
  the reference semantics a locale-aware version of that stage should
  match.

## Running

```bash
uv run python -m pytest tests/test_multilingual_conformance.py -q
```
