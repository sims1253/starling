"""Reference oracle for the personalization contract
(``packages/contracts/personalization``): vocabulary/replacement suggestions
mined from correction records, and few-shot style retrieval.
"""

from __future__ import annotations

import difflib
import hashlib
import re
from collections import Counter
from typing import Any

ATTACHED = ",.;:!?"
HEADER = "[Style References]"
# Decisions whose final text the user endorsed.
ELIGIBLE_DECISIONS = {"accepted", "edited"}
# Transforms that change wording on purpose; their records are not mined.
UNMINED_KINDS = {"rewrite", "translate"}
PROTECTED = re.compile(
    r"(`+).+?\1"  # inline code, closed by a backtick run of the same length
    r'|"[^"\n]*[/\\][^"\n]*"'  # quoted paths, which may contain spaces anywhere
    r"|\S*[/\\](?:[\w -]*[/\\])*\S*"  # paths and URLs; inner segments may hold spaces
    r"|\S*(?:[@_]|\w\.\w|[a-z][A-Z])\S*"  # emails, identifiers, dotted names
)


def tokens(text: str) -> list[str]:
    return [t for t in (t.strip(ATTACHED) for t in text.split()) if t]


def extract_substitutions(raw: list[str], final: list[str]) -> list[tuple[str, str]]:
    """Replaced token runs as (source, target); sources are lowercased,
    targets keep their casing."""
    matcher = difflib.SequenceMatcher(
        None, [t.lower() for t in raw], [t.lower() for t in final], autojunk=False
    )
    return [
        (" ".join(raw[i1:i2]).lower(), " ".join(final[j1:j2]))
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag == "replace"
    ]


def is_eligible(rec: dict[str, Any]) -> bool:
    return (
        rec.get("decision") in ELIGIBLE_DECISIONS
        and not rec.get("secure_field", False)
        and bool((rec.get("raw_text") or "").strip())
        and bool((rec.get("final_text") or "").strip())
    )


def suggestion_id(source: str, target: str) -> str:
    digest = hashlib.sha256(f"{source}\n{target}".encode()).hexdigest()
    return f"sug_{digest[:16]}"


def suggest_vocabulary_and_replacements(
    records: list[dict[str, Any]],
    existing_vocabulary: list[str] | None = None,
    existing_snippets: list[dict[str, str]] | None = None,
    min_frequency: int = 2,
    min_consistency: float = 0.8,
) -> list[dict[str, Any]]:
    vocabulary = {v.lower() for v in existing_vocabulary or []}
    snippets = {s["spoken"].lower(): s["expansion"] for s in existing_snippets or []}

    edits: Counter[tuple[str, str]] = Counter()
    last_seen: dict[tuple[str, str], str] = {}
    raw_texts: list[list[str]] = []
    for rec in records:
        if not is_eligible(rec) or UNMINED_KINDS & set(rec.get("transform_kinds") or []):
            continue
        raw = tokens(rec["raw_text"])
        raw_texts.append([t.lower() for t in raw])
        for src, tgt in extract_substitutions(raw, tokens(rec["final_text"])):
            edits[src, tgt] += 1
            last_seen[src, tgt] = max(last_seen.get((src, tgt), ""), rec["decision_utc"])

    def occurrences(phrase: str) -> int:
        words = phrase.split()
        n = len(words)
        return sum(t[i : i + n] == words for t in raw_texts for i in range(len(t) - n + 1))

    frequent = {pair: f for pair, f in edits.items() if f >= min_frequency}
    suggestions: list[dict[str, Any]] = []
    for (src, tgt), freq in frequent.items():
        consistency = freq / occurrences(src)
        if consistency < min_consistency:
            continue
        if tgt.lower() in vocabulary:
            conflict = "already_in_vocabulary"
        elif snippets.get(src, tgt) != tgt:
            conflict = "snippet_collision"
        elif sum(s == src for s, _ in frequent) > 1:
            conflict = "competing_targets"
        else:
            conflict = None
        suggestions.append(
            {
                "id": suggestion_id(src, tgt),
                "target_type": "replacement" if " " in src or " " in tgt else "vocabulary",
                "source_phrase": src,
                "target_phrase": tgt,
                "frequency": freq,
                "consistency": round(consistency, 4),
                "status": "suggested",
                "conflict": conflict,
                "last_seen_utc": last_seen[src, tgt],
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
            and rec.get("mode_id") == request["mode_id"]
            and rec.get("language") == request.get("language")
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
    """Apply accepted suggestions in one pass (no chained rewrites), outside
    protected spans (inline code, paths, URLs, identifiers)."""
    if not suggestions:
        return text
    ordered = sorted(suggestions, key=lambda s: len(s["source_phrase"]), reverse=True)
    pattern = re.compile(
        r"\b(?:" + "|".join(f"({re.escape(s['source_phrase'])})" for s in ordered) + r")\b",
        re.IGNORECASE,
    )
    protected = [m.span() for m in PROTECTED.finditer(text)]

    def replace(m: re.Match[str]) -> str:
        if any(start < m.end() and m.start() < end for start, end in protected):
            return m.group()
        return ordered[m.lastindex - 1]["target_phrase"]

    return pattern.sub(replace, text)
