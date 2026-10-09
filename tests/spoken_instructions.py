"""Executable oracle for the trailing spoken instruction grammar.

A delimiter spoken near the end of a *finalized* take splits it: the text
after it is the instruction (it travels as the transform request's
``instruction``, never as input text), the text before it is the payload.
The configuration is contract data
(``packages/contracts/mode-routing/spoken-instructions.json``);
``starling-processing``'s ``instructions`` module is the Rust port. Both
replay ``fixtures/spoken-instructions.json``.

Rules (closed; nothing beyond them is interpreted):

- Tokens and cores are the deterministic step's (``spoken_commands``):
  runs of non-whitespace, cores stripped of attached ``,.;:!?`` and
  lowercased. A quoted mention therefore never matches: the opening quote
  stays attached to the token.
- A token is a delimiter occurrence when its core is one of the table's
  ``match_tokens``, it sits within the last ``window_words`` tokens, the
  previous token's core is not the language's literal word, and the text
  after it contains a letter or digit.
- The LAST occurrence wins. Offsets are Unicode code points.
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


def split(text: str, *, language: str | None, table: dict[str, Any],
          commands_table: dict[str, Any]) -> dict[str, Any]:
    """The split record: matched, payload, instruction and code-point spans.

    The spans cover the delimiter and everything after it, so
    ``raw == payload + raw[delimiter_span[0]:]``.
    """
    delimiter = table["delimiter"]
    commands = sc.table_for(commands_table, language)
    literal = commands["literal"] if commands else "literal"

    spans = sc._tokens(text)
    cores = [sc._core(text[s:e]) for s, e in spans]
    found = None
    for i in range(max(0, len(spans) - delimiter["window_words"]), len(spans)):
        if cores[i] not in delimiter["match_tokens"]:
            continue
        if i > 0 and cores[i - 1] == literal:
            continue
        if not any(c.isalnum() for c in text[spans[i][1]:]):
            continue
        found = i
    if found is None:
        return {"matched": False, "payload": text, "instruction": "",
                "delimiter_span": None, "instruction_span": None}
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
    """A split as a spoken-instruction.schema.json v1 record."""
    result = split(text, language=language, table=table, commands_table=commands_table)
    return {"schema_version": 1, **result, "span_encoding": "unicode_codepoints"}


def strip_delimiter(instruction: str, *, table: dict[str, Any]) -> str:
    """Removes a leading delimiter token from an instruction region's text."""
    spans = sc._tokens(instruction)
    if spans:
        start, end = spans[0]
        if sc._core(instruction[start:end]) in table["delimiter"]["match_tokens"]:
            return instruction[end:].strip(sc.WHITESPACE)
    return instruction


def cases() -> list[dict[str, Any]]:
    return json.loads(
        (CONTRACT / "fixtures" / "spoken-instructions.json").read_text(encoding="utf-8")
    )
