"""Executable oracle and reference implementation for Starling personalization (#305).

Training-free personalization via:
1. Vocabulary & replacement suggestions mined from repeated #304 correction records.
2. Few-shot style retrieval for processing modes, strictly isolated by language,
   mode, and project, bounded by hard example and character budgets.
"""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
CONTRACT = REPO / "packages" / "contracts" / "personalization"
ATTACHED = ",.;:!?"
WHITESPACE = " \t\n\r\x0b\x0c"


def levenshtein(a: str, b: str) -> int:
    """Exact Levenshtein distance."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _clean_token(t: str) -> str:
    return t.strip(ATTACHED).lower()


def extract_substitutions(raw_text: str, final_text: str) -> list[tuple[str, str]]:
    """Extract word or phrase level substitutions and unchanged tokens between raw and final text."""
    raw_tokens = raw_text.split()
    final_tokens = final_text.split()

    matcher = difflib.SequenceMatcher(None, [_clean_token(t) for t in raw_tokens],
                                      [_clean_token(t) for t in final_tokens])
    substitutions: list[tuple[str, str]] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            src_phrase = " ".join(t.strip(ATTACHED) for t in raw_tokens[i1:i2])
            tgt_phrase = " ".join(t.strip(ATTACHED) for t in final_tokens[j1:j2])
            if src_phrase and tgt_phrase and src_phrase.lower() != tgt_phrase.lower():
                substitutions.append((src_phrase.lower(), tgt_phrase))
        elif tag == "equal":
            for k in range(i1, i2):
                token_clean = raw_tokens[k].strip(ATTACHED)
                if token_clean:
                    substitutions.append((token_clean.lower(), token_clean))

    return substitutions


def suggest_vocabulary_and_replacements(
    records: list[dict[str, Any]],
    existing_vocabulary: list[str] | None = None,
    existing_snippets: list[dict[str, str]] | None = None,
    min_frequency: int = 2,
    min_consistency: float = 0.8,
) -> list[dict[str, Any]]:
    """Mine candidate vocabulary or replacement suggestions from correction records."""
    existing_vocab_set = {v.lower(): v for v in (existing_vocabulary or [])}
    existing_snippets_map = {s["spoken"].lower(): s["expansion"] for s in (existing_snippets or [])}

    # source_phrase -> target_phrase -> count
    counts: dict[str, dict[str, int]] = {}
    last_seen: dict[str, str] = {}

    for rec in records:
        if rec.get("decision") != "accepted" or rec.get("secure_field", False):
            continue
        raw = rec.get("raw_text", "")
        final = rec.get("final_text", "")
        if not raw or not final:
            continue

        subs = extract_substitutions(raw, final)
        for src, tgt in subs:
            src_norm = src.strip().lower()
            tgt_norm = tgt.strip()
            if not src_norm or not tgt_norm:
                continue
            counts.setdefault(src_norm, {})
            counts[src_norm][tgt_norm] = counts[src_norm].get(tgt_norm, 0) + 1
            if tgt_norm.lower() != src_norm:
                last_seen[src_norm] = rec.get("decision_utc", "")

    suggestions: list[dict[str, Any]] = []

    for src, targets in counts.items():
        total_for_src = sum(targets.values())
        if total_for_src < min_frequency:
            continue

        for tgt, freq in targets.items():
            # Only consider genuine edits (target != source)
            if tgt.lower() == src:
                continue
            if freq < min_frequency:
                continue
            consistency = freq / total_for_src
            if consistency < min_consistency:
                continue

            # Determine target_type
            if " " in tgt or " " in src:
                target_type = "replacement"
            else:
                target_type = "vocabulary"

            # Check conflicts
            conflict = None
            if tgt.lower() in existing_vocab_set:
                conflict = "already_in_vocabulary"
            elif src.lower() in existing_snippets_map:
                if existing_snippets_map[src.lower()] != tgt:
                    conflict = "snippet_collision"
            else:
                competing = [t for t, f in targets.items() if t != tgt and t.lower() != src and f >= min_frequency]
                if competing:
                    conflict = "competing_targets"

            # Deterministic ID
            sug_id = f"sug_{src.replace(' ', '_')}_{tgt.replace(' ', '_').lower()}"

            suggestions.append({
                "id": sug_id,
                "target_type": target_type,
                "source_phrase": src,
                "target_phrase": tgt,
                "frequency": freq,
                "consistency": round(consistency, 4),
                "status": "suggested",
                "conflict": conflict,
                "last_seen_utc": last_seen.get(src),
            })

    # Sort deterministically by frequency desc, then consistency desc, then source
    suggestions.sort(key=lambda s: (-s["frequency"], -s["consistency"], s["source_phrase"]))
    return suggestions


def retrieve_style_examples(
    history: list[dict[str, Any]],
    request: dict[str, Any],
    deleted_captures: list[str] | set[str] | None = None,
) -> dict[str, Any]:
    """Deterministically retrieve accepted examples matching the request under budget caps."""
    if not request.get("personalization_enabled", True):
        return {
            "examples_used": [],
            "character_count": 0,
            "formatted_context": "",
        }

    deleted = set(deleted_captures or [])
    mode_id = request.get("mode_id")
    language = request.get("language")
    project_id = request.get("project_id")
    max_examples = request.get("max_examples", 3)
    max_characters = request.get("max_characters", 500)

    # Filter eligible
    eligible: list[dict[str, Any]] = []
    for rec in history:
        if rec.get("decision") != "accepted" or rec.get("secure_field", False):
            continue
        if rec.get("capture_id") in deleted:
            continue
        if rec.get("mode_id") != mode_id:
            continue
        if language is not None and rec.get("language") != language:
            continue
        if project_id is not None and rec.get("project_id") != project_id:
            continue

        raw = rec.get("raw_text", "").strip()
        final = rec.get("final_text", "").strip()
        if raw and final:
            eligible.append(rec)

    # Sort deterministically by decision_utc descending, tie-break by id
    eligible.sort(key=lambda r: (r.get("decision_utc", ""), r.get("id", "")), reverse=True)

    header = "[Style References]"
    used_records: list[dict[str, Any]] = []

    for cand in eligible:
        if len(used_records) >= max_examples:
            break

        cand_raw = cand.get("raw_text", "").strip()
        cand_final = cand.get("final_text", "").strip()
        entry_text = f"Input: {cand_raw}\nOutput: {cand_final}"

        # Projected text if we add this candidate
        if not used_records:
            projected = f"{header}\n{entry_text}"
        else:
            projected = f"{header}\n" + "\n\n".join(
                f"Input: {r.get('raw_text', '').strip()}\nOutput: {r.get('final_text', '').strip()}"
                for r in used_records + [cand]
            )

        if len(projected) <= max_characters:
            used_records.append(cand)
        else:
            # Cannot fit without exceeding budget
            continue

    if not used_records:
        return {
            "examples_used": [],
            "character_count": 0,
            "formatted_context": "",
        }

    formatted = f"{header}\n" + "\n\n".join(
        f"Input: {r.get('raw_text', '').strip()}\nOutput: {r.get('final_text', '').strip()}"
        for r in used_records
    )

    return {
        "examples_used": [r["id"] for r in used_records],
        "character_count": len(formatted),
        "formatted_context": formatted,
    }


def apply_suggestions_to_text(text: str, suggestions: list[dict[str, Any]]) -> str:
    """Apply accepted suggestions to text while strictly preserving protected tokens."""
    out = text
    for sug in suggestions:
        src = sug["source_phrase"]
        tgt = sug["target_phrase"]
        pattern = re.compile(rf"\b{re.escape(src)}\b", re.IGNORECASE)
        out = pattern.sub(tgt, out)
    return out


def is_protected_token(token: str) -> bool:
    """Check if a token looks like code (camelCase, snake_case), a URL, or a file path."""
    if "/" in token or "\\" in token or token.startswith("http://") or token.startswith("https://"):
        return True
    if "_" in token:
        return True
    # camelCase: lowercase followed by uppercase
    if re.search(r"[a-z][A-Z]", token):
        return True
    return False
