"""Executable oracle for the trailing spoken instruction grammar (#298).

A delimiter phrase spoken near the end of a *finalized* take splits it:
everything after the delimiter is the instruction (it travels as the
transform request's ``instruction``, never as input text), everything
before it is the payload. The configuration is contract data
(``packages/contracts/mode-routing/spoken-instructions.json``); this
module is its semantics, and ``starling-processing``'s ``instructions``
module is the Rust port. Both replay ``fixtures/spoken-instructions.json``.

Rules (closed; nothing beyond them is interpreted):

- Tokens and token cores are exactly the deterministic step's
  (``spoken_commands``): maximal runs of non-whitespace over the six
  ASCII whitespace characters, cores trimmed of the punctuation a
  recognizer attaches (``,.;:!?``) and lowercased.
- A token is a delimiter occurrence when its core is one of the table's
  ``match_tokens`` (ASR spelling variants included), it sits within the
  last ``window_words`` words of the take, the previous token's core is
  not the language's literal word (``spoken-commands.json``), and what
  follows it, trimmed of whitespace, is non-empty and contains a letter
  or digit (a delimiter with no instruction after it is not an
  occurrence; the payload is untouched).
- A quoted mention never matches: an opening quote is not recognizer
  punctuation, so it stays attached to the token and the core no longer
  equals a delimiter token.
- Among the occurrences the LAST one wins; text before it (earlier
  delimiters included) is the payload, text after it is the
  instruction. Offsets are Unicode code points.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import spoken_commands as sc

REPO = Path(__file__).resolve().parents[1]
CONTRACT = REPO / "packages" / "contracts" / "mode-routing"


def load_table() -> dict[str, Any]:
    return json.loads((CONTRACT / "spoken-instructions.json").read_text(encoding="utf-8"))


def canonical(table: dict[str, Any]) -> str:
    """The delimiter spelling the UI shows."""
    return table["delimiter"]["canonical"]


def _tokens(text: str) -> list[tuple[int, int]]:
    # The deterministic step's tokenization, so both grammars split the
    # same way (spoken_commands is the authority).
    return sc._tokens(text)


def split(text: str, *, language: str | None, table: dict[str, Any],
          commands_table: dict[str, Any]) -> dict[str, Any]:
    """The semantic split record: matched, payload, instruction, spans.

    A no-fire returns the whole text as the payload with ``matched``
    false; a fire returns payload/instruction and code-point spans. The
    spans cover the delimiter token and everything after it, so the raw
    text reconstructs as ``raw == payload + raw[delimiter_span[0]:]``.
    """
    no_match = {"matched": False, "payload": text, "instruction": "",
                "delimiter_span": None, "instruction_span": None}
    delimiter = table["delimiter"]
    matches = delimiter["match_tokens"]
    window = delimiter["window_words"]
    # The literal escape word is the spoken-commands table's for this
    # language, so the two grammars stay in sync.
    commands = sc.table_for(commands_table, language)
    literal = commands["literal"] if commands else "literal"

    spans = _tokens(text)
    cores = [sc._core(text[s:e]) for s, e in spans]
    floor = max(0, len(spans) - window)
    found = None
    for i in range(floor, len(spans)):
        if cores[i] not in matches:
            continue
        if i > 0 and cores[i - 1] == literal:
            continue
        tail = text[spans[i][1]:].strip(sc.WHITESPACE)
        if not tail or not any(c.isalnum() for c in tail):
            continue
        found = i
    if found is None:
        return no_match
    start, end = spans[found]
    return {
        "matched": True,
        "payload": text[:start],
        "instruction": text[end:].strip(sc.WHITESPACE),
        "delimiter_span": [start, end],
        "instruction_span": [end, len(text)],
    }


def record(text: str, *, language: str | None, table: dict[str, Any],
          commands_table: dict[str, Any]) -> dict[str, Any]:
    """Lift a split into the spoken-instruction.schema.json v1 record."""
    result = split(text, language=language, table=table, commands_table=commands_table)
    return {
        "schema_version": 1,
        "matched": result["matched"],
        "payload": result["payload"],
        "instruction": result["instruction"],
        "delimiter_span": result["delimiter_span"],
        "instruction_span": result["instruction_span"],
        "span_encoding": "unicode_codepoints",
    }


def strip_delimiter(instruction: str, *, table: dict[str, Any]) -> str:
    """Removes a leading delimiter token from an instruction region's
    text: the draft's command region carries the delimiter, the model
    sees only the instruction. A no-op when the text does not start with
    a delimiter token.
    """
    matches = table["delimiter"]["match_tokens"]
    spans = _tokens(instruction)
    if spans:
        start, end = spans[0]
        if sc._core(instruction[start:end]) in matches:
            return instruction[end:].strip(sc.WHITESPACE)
    return instruction


def cases() -> list[dict[str, Any]]:
    return json.loads(
        (CONTRACT / "fixtures" / "spoken-instructions.json").read_text(encoding="utf-8")
    )
