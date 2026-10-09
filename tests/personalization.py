"""Reference oracle for the personalization contract
(``packages/contracts/personalization``): vocabulary/replacement suggestions
mined from correction records, and few-shot style retrieval.
"""

from __future__ import annotations

import difflib
import re
from typing import Any

ATTACHED = ",.;:!?"
HEADER = "[Style References]"
# Decisions whose final text the user endorsed.
ELIGIBLE_DECISIONS = {"accepted", "edited"}
# A whitespace token containing a path/URL/email separator, an inner dot,
# an underscore or camelCase is code-like and never rewritten.
PROTECTED = re.compile(r"[/\\@_]|\w\.\w|[a-z][A-Z]")


def extract_substitutions(raw_text: str, final_text: str) -> list[tuple[str, str]]:
    """(source, target) pairs between raw and final text, including unchanged
    tokens as (token, token) so they count against consistency. Sources are
    lowercased; targets keep their casing."""
    raw_tokens = [t.strip(ATTACHED) for t in raw_text.split()]
    final_tokens = [t.strip(ATTACHED) for t in final_text.split()]
    matcher = difflib.SequenceMatcher(
        None,
        [t.lower() for t in raw_tokens],
        [t.lower() for t in final_tokens],
        autojunk=False,
    )
    substitutions: list[tuple[str, str]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            src = " ".join(raw_tokens[i1:i2])
            tgt = " ".join(final_tokens[j1:j2])
            if src and tgt and src.lower() != tgt.lower():
                substitutions.append((src.lower(), tgt))
        elif tag == "equal":
            substitutions.extend((t.lower(), t) for t in raw_tokens[i1:i2] if t)
    return substitutions


def is_eligible(rec: dict[str, Any]) -> bool:
    return (
        rec.get("decision") in ELIGIBLE_DECISIONS
        and not rec.get("secure_field", False)
        and bool((rec.get("raw_text") or "").strip())
        and bool((rec.get("final_text") or "").strip())
    )


def suggest_vocabulary_and_replacements(
    records: list[dict[str, Any]],
    existing_vocabulary: list[str] | None = None,
    existing_snippets: list[dict[str, str]] | None = None,
    min_frequency: int = 2,
    min_consistency: float = 0.8,
) -> list[dict[str, Any]]:
    vocabulary = {v.lower() for v in existing_vocabulary or []}
    snippets = {s["spoken"].lower(): s["expansion"] for s in existing_snippets or []}

    counts: dict[str, dict[str, int]] = {}
    last_seen: dict[str, str] = {}
    for rec in filter(is_eligible, records):
        for src, tgt in extract_substitutions(rec["raw_text"], rec["final_text"]):
            targets = counts.setdefault(src, {})
            targets[tgt] = targets.get(tgt, 0) + 1
            if tgt.lower() != src:
                last_seen[src] = max(last_seen.get(src, ""), rec["decision_utc"])

    suggestions: list[dict[str, Any]] = []
    for src, targets in counts.items():
        total = sum(targets.values())
        edits = {t: f for t, f in targets.items() if t.lower() != src and f >= min_frequency}
        for tgt, freq in edits.items():
            consistency = freq / total
            if consistency < min_consistency:
                continue
            if tgt.lower() in vocabulary:
                conflict = "already_in_vocabulary"
            elif src in snippets:
                conflict = "snippet_collision" if snippets[src] != tgt else None
            elif len(edits) > 1:
                conflict = "competing_targets"
            else:
                conflict = None
            suggestions.append(
                {
                    "id": f"sug_{src.replace(' ', '_')}_{tgt.replace(' ', '_').lower()}",
                    "target_type": "replacement" if " " in src or " " in tgt else "vocabulary",
                    "source_phrase": src,
                    "target_phrase": tgt,
                    "frequency": freq,
                    "consistency": round(consistency, 4),
                    "status": "suggested",
                    "conflict": conflict,
                    "last_seen_utc": last_seen[src],
                }
            )

    suggestions.sort(key=lambda s: (-s["frequency"], -s["consistency"], s["source_phrase"]))
    return suggestions


def retrieve_style_examples(
    history: list[dict[str, Any]],
    request: dict[str, Any],
    deleted_captures: list[str] | set[str] | None = None,
) -> dict[str, Any]:
    used: list[str] = []
    entries: list[str] = []
    if request.get("personalization_enabled", True):
        deleted = set(deleted_captures or [])
        eligible = [
            rec
            for rec in history
            if is_eligible(rec)
            and rec.get("capture_id") not in deleted
            and all(
                rec.get(key) == request.get(key) for key in ("mode_id", "language", "project_id")
            )
        ]
        eligible.sort(key=lambda r: (r["decision_utc"], r["id"]), reverse=True)

        max_examples = request.get("max_examples", 3)
        max_characters = request.get("max_characters", 500)
        length = len(HEADER)
        for rec in eligible:
            if len(used) == max_examples:
                break
            entry = f"Input: {rec['raw_text'].strip()}\nOutput: {rec['final_text'].strip()}"
            # "\n" after the header, "\n\n" between entries.
            cost = len(entry) + (2 if entries else 1)
            if length + cost <= max_characters:
                used.append(rec["id"])
                entries.append(entry)
                length += cost

    formatted = HEADER + "\n" + "\n\n".join(entries) if entries else ""
    return {
        "examples_used": used,
        "character_count": len(formatted),
        "formatted_context": formatted,
    }


def apply_suggestions_to_text(text: str, suggestions: list[dict[str, Any]]) -> str:
    """Apply accepted suggestions in one pass (no chained rewrites), leaving
    code-like tokens (paths, URLs, identifiers) untouched."""
    mapping = {s["source_phrase"].lower(): s["target_phrase"] for s in suggestions}
    if not mapping:
        return text
    sources = sorted(mapping, key=len, reverse=True)
    pattern = re.compile(r"\b(?:" + "|".join(map(re.escape, sources)) + r")\b", re.IGNORECASE)

    def replace(m: re.Match[str]) -> str:
        left = re.search(r"\S*$", text[: m.start()]).group()
        right = re.match(r"\S*", text[m.end() :]).group()
        token = (left + m.group() + right).strip(ATTACHED)
        return m.group() if PROTECTED.search(token) else mapping[m.group().lower()]

    return pattern.sub(replace, text)
