"""Executable oracle for the deterministic processing step (#294).

Spoken punctuation/layout commands and mode snippets, applied before any
model step at zero model cost. The command vocabulary is contract data
(``packages/contracts/mode-routing/spoken-commands.json``); this module is
its semantics, and ``starling-processing``'s ``transforms`` module is the
Rust port. Both replay ``fixtures/spoken-commands.json``.

Rules:

- Tokens are maximal runs of non-whitespace, where whitespace is exactly
  space, tab, LF, CR, VT and FF (so every implementation splits the same
  way). A token's *core* is the token without the punctuation a recognizer
  attaches (``,.;:!?``), lowercased.
- A phrase matches when consecutive token cores equal its words. The
  longest phrase (in words, then characters) wins; on a tie a built-in
  command wins over a mode snippet.
- ``<literal word> <phrase>`` emits the phrase's original tokens as plain
  text and drops the literal word. A literal word before anything else is
  just a word.
- punctuation: trailing whitespace is dropped, one recognizer-placed
  punctuation mark at the end is replaced, then the value is appended.
- line_break / paragraph: spaces and tabs before are dropped, ``\\n`` /
  ``\\n\\n`` appended, whitespace after is dropped.
- bullet: like a line break, but a newline is added only when not already
  at the start of a line, then ``- ``.
- snippet: replaced by its expansion; surrounding whitespace is kept.
- All other text, including whitespace between ordinary tokens, is copied
  unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
CONTRACT = REPO / "packages" / "contracts" / "mode-routing"
WHITESPACE = " \t\n\r\x0b\x0c"
ATTACHED = ",.;:!?"


def load_table() -> dict[str, Any]:
    return json.loads((CONTRACT / "spoken-commands.json").read_text(encoding="utf-8"))


def table_for(table: dict[str, Any], language: str | None) -> dict[str, Any] | None:
    # BCP 47 tags are case-insensitive; the table keys are lowercase.
    primary = (language or "en").split("-", 1)[0].lower()
    return table["languages"].get(primary)


def _tokens(text: str) -> list[tuple[int, int]]:
    spans = []
    i = 0
    while i < len(text):
        if text[i] in WHITESPACE:
            i += 1
            continue
        start = i
        while i < len(text) and text[i] not in WHITESPACE:
            i += 1
        spans.append((start, i))
    return spans


def _core(token: str) -> str:
    return token.strip(ATTACHED).lower()


def apply(text: str, *, language: str | None, spoken_commands: bool,
          snippets: list[dict[str, str]], table: dict[str, Any]) -> str:
    lang = table_for(table, language)
    literal = lang["literal"] if lang else "literal"
    phrases: list[tuple[list[str], int, dict[str, Any]]] = []
    if spoken_commands and lang:
        for command in lang["commands"]:
            phrase = command["phrase"].lower()
            phrases.append(([phrase[s:e] for s, e in _tokens(phrase)], 0, command))
    for snippet in snippets:
        words = [snippet["spoken"].lower()[s:e] for s, e in _tokens(snippet["spoken"].lower())]
        phrases.append((words, 1,
                        {"action": "snippet", "value": snippet["expansion"]}))
    # Longest words first, then longest characters, commands before snippets.
    phrases.sort(key=lambda p: (-len(p[0]), -len(" ".join(p[0])), p[1]))

    spans = _tokens(text)
    cores = [_core(text[s:e]) for s, e in spans]

    def match_at(i: int) -> tuple[int, dict[str, Any]] | None:
        for words, _, command in phrases:
            n = len(words)
            if words and i + n <= len(spans) and cores[i:i + n] == words:
                return n, command
        return None

    out = ""
    pos = 0
    skip_ws = False
    i = 0
    while i < len(spans):
        start, end = spans[i]
        gap = "" if skip_ws else text[pos:start]
        if cores[i] == literal:
            escaped = match_at(i + 1)
            if escaped is not None:
                n = escaped[0]
                out += gap + text[spans[i + 1][0]:spans[i + n][1]]
                pos, i, skip_ws = spans[i + n][1], i + n + 1, False
                continue
        found = match_at(i)
        if found is None:
            out += gap + text[start:end]
            pos, i, skip_ws = end, i + 1, False
            continue
        n, command = found
        action = command["action"]
        if action == "punctuation":
            out = out.rstrip(WHITESPACE)
            if out and out[-1] in ATTACHED:
                out = out[:-1]
            out += command["value"]
            skip_ws = False
        elif action in ("line_break", "paragraph"):
            out = out.rstrip(" \t") + ("\n" if action == "line_break" else "\n\n")
            skip_ws = True
        elif action == "bullet":
            out = out.rstrip(" \t")
            if out and not out.endswith("\n"):
                out += "\n"
            out += "- "
            skip_ws = True
        elif action == "snippet":
            out += gap + command["value"]
            skip_ws = False
        else:  # pragma: no cover - the table's actions are closed
            raise ValueError(f"unknown action {action!r}")
        pos, i = spans[i + n - 1][1], i + n
    if not skip_ws:
        out += text[pos:]
    return out


def cases() -> list[dict[str, Any]]:
    return json.loads((CONTRACT / "fixtures" / "spoken-commands.json").read_text(encoding="utf-8"))
