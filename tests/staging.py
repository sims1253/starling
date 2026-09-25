"""Executable oracle for the Starling staging and processing contract (#293).

The structural half lives in ``packages/contracts/mode-routing/``
(``draft.schema.json``, ``transform-request.schema.json``,
``transform-result.schema.json``, ``provider.schema.json``); this module is
the semantic half. Stdlib-only, like ``mode_routing.py``, so the native
runtime (``starling-processing``'s ``staging`` module) and any other client
can be differentially tested against it: both replay
``fixtures/staging.json`` and must agree on every outcome and snapshot.

The draft of one take is a single text with typed regions over it. Every
code point belongs to exactly one region; offsets are Unicode code points
(``span_encoding: unicode_codepoints``), never UTF-16 units or bytes.

- ``raw``: text of a final recognition attempt. The attempt itself is
  immutable and kept apart from the draft, so editing the region never
  touches it; ``raw_text()`` rebuilds the recognition byte for byte.
- ``partial``: the live tail of one streaming segment. A newer partial or
  the segment's final replaces it in place.
- ``user``: text the user typed, and any partial the user edited (the
  segment is then *pinned*: later partials and finals for it are recorded
  as attempts but never replace the user's text).
- ``command``: a leading mode phrase or trailing instruction (#298). It is
  kept out of the transform input and travels as the request's
  ``instruction`` instead.
- ``processed``: an accepted processed revision.

``revision`` counts text changes. A transform request records the revision
it read (``base_revision``). Its result is a proposal: it is ``current``
only while the draft is still at that revision, otherwise ``stale``. A
result never changes the draft by itself; ``accept`` (the user) or
``swap_check`` (direct delivery with an unchanged target) does, and only
for a current proposal unless the user forces a stale one.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
CONTRACT = REPO / "packages" / "contracts" / "mode-routing"
FIXTURE_DIR = CONTRACT / "fixtures"

REGION_KINDS = ("raw", "partial", "user", "command", "processed")
COMMAND_KINDS = ("mode_phrase", "trailing_instruction")


class Draft:
    def __init__(self, draft_id: str, capture_id: str) -> None:
        self.draft_id = draft_id
        self.capture_id = capture_id
        self.revision = 0
        self.deleted = False
        # [kind, text, meta] where meta holds segment / attempt_id /
        # command / request_id as applicable.
        self.regions: list[list[Any]] = []
        self.attempts: list[dict[str, Any]] = []
        self.pinned: list[int] = []
        self.finalized: list[int] = []
        self.requests: list[dict[str, Any]] = []
        self.proposals: list[dict[str, Any]] = []
        self.deliveries: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Views
    # ------------------------------------------------------------------ #
    def text(self) -> str:
        return "".join(r[1] for r in self.regions)

    def payload_text(self) -> str:
        """The transform input: every region except commands."""
        return "".join(r[1] for r in self.regions if r[0] != "command")

    def instruction(self) -> str | None:
        parts = [r[1] for r in self.regions
                 if r[0] == "command" and r[2].get("command") == "trailing_instruction"]
        return "".join(parts) if parts else None

    def raw_text(self) -> str:
        """The latest final attempt per segment, in segment order: the
        recognition exactly as it arrived, whatever happened to the draft."""
        latest: dict[int, str] = {}
        for attempt in self.attempts:
            latest[attempt["segment"]] = attempt["text"]
        return "".join(latest[s] for s in sorted(latest))

    def _request(self, request_id: str) -> dict[str, Any] | None:
        return next((r for r in self.requests if r["request_id"] == request_id), None)

    def _proposal(self, request_id: str) -> dict[str, Any] | None:
        return next((p for p in self.proposals if p["request_id"] == request_id), None)

    def _partial_index(self, segment: int) -> int | None:
        for i, region in enumerate(self.regions):
            if region[0] == "partial" and region[2].get("segment") == segment:
                return i
        return None

    # ------------------------------------------------------------------ #
    # Region editing helpers
    # ------------------------------------------------------------------ #
    def _set_text(self, regions: list[list[Any]]) -> bool:
        """Install ``regions`` (dropping empty ones, merging adjacent user
        regions) and bump the revision when the text changed."""
        before = self.text()
        cleaned: list[list[Any]] = []
        for region in regions:
            if region[1] == "":
                continue
            if cleaned and region[0] == "user" and cleaned[-1][0] == "user":
                cleaned[-1][1] += region[1]
                continue
            cleaned.append(region)
        self.regions = cleaned
        changed = self.text() != before
        if changed:
            self.revision += 1
        return changed

    def _pin(self, index: int) -> None:
        region = self.regions[index]
        segment = region[2]["segment"]
        if segment not in self.pinned:
            self.pinned.append(segment)
        self.regions[index] = ["user", region[1], {}]

    def _split_at(self, at: int) -> int:
        """Split the region containing code point ``at`` so a boundary lies
        there; return the index of the first region starting at ``at``."""
        pos = 0
        for i, region in enumerate(self.regions):
            end = pos + len(region[1])
            if at == pos:
                return i
            if pos < at < end:
                left = [region[0], region[1][: at - pos], dict(region[2])]
                right = [region[0], region[1][at - pos:], dict(region[2])]
                self.regions[i:i + 1] = [left, right]
                return i + 1
            pos = end
        return len(self.regions)

    def _span_of(self, index: int) -> tuple[int, int]:
        start = sum(len(r[1]) for r in self.regions[:index])
        return start, start + len(self.regions[index][1])

    # ------------------------------------------------------------------ #
    # Operations
    # ------------------------------------------------------------------ #
    def apply(self, op: dict[str, Any]) -> str:
        kind = op["op"]
        handler = getattr(self, "_op_" + kind, None)
        if handler is None:
            raise ValueError(f"unknown staging op {kind!r}")
        if self.deleted and kind not in ("result", "crash"):
            return "ignored_deleted"
        return handler(op)

    def _op_partial(self, op: dict[str, Any]) -> str:
        segment, text = op["segment"], op["text"]
        if segment in self.pinned:
            return "ignored_pinned"
        if segment in self.finalized:
            return "ignored_final"
        regions = copy.deepcopy(self.regions)
        index = self._partial_index(segment)
        if index is None:
            regions.append(["partial", text, {"segment": segment}])
        else:
            regions[index][1] = text
        return "applied" if self._set_text(regions) else "unchanged"

    def _op_final(self, op: dict[str, Any]) -> str:
        segment, attempt_id, text = op["segment"], op["attempt_id"], op["text"]
        known = next((a for a in self.attempts if a["attempt_id"] == attempt_id), None)
        if known is not None:
            if known["segment"] == segment and known["text"] == text:
                return "duplicate"
            return "attempt_conflict"
        self.attempts.append({"attempt_id": attempt_id, "segment": segment, "text": text})
        first_final = segment not in self.finalized
        if first_final:
            self.finalized.append(segment)
        if segment in self.pinned:
            return "recorded_pinned"
        if not first_final:
            # A re-recognition of a segment the draft already shows: the
            # attempt is kept (raw recovery uses the latest), the visible
            # text is not replaced behind the user's back.
            return "recorded"
        regions = copy.deepcopy(self.regions)
        index = self._partial_index(segment)
        region = ["raw", text, {"segment": segment, "attempt_id": attempt_id}]
        if index is None:
            regions.append(region)
        else:
            regions[index] = region
        self._set_text(regions)
        return "applied"

    def _op_insert(self, op: dict[str, Any]) -> str:
        at, text = op["at"], op["text"]
        if not 0 <= at <= len(self.text()):
            return "out_of_range"
        if text == "":
            return "unchanged"
        # Typing strictly inside a live partial pins it: the user owns
        # that segment's text from now on.
        pos = 0
        for i, region in enumerate(self.regions):
            end = pos + len(region[1])
            if region[0] == "partial" and pos < at < end:
                self._pin(i)
            pos = end
        index = self._split_at(at)
        regions = copy.deepcopy(self.regions)
        regions.insert(index, ["user", text, {}])
        self._set_text(regions)
        return "applied"

    def _op_delete(self, op: dict[str, Any]) -> str:
        start, end = op["start"], op["end"]
        if not 0 <= start <= end <= len(self.text()):
            return "out_of_range"
        if start == end:
            return "unchanged"
        pos = 0
        for i, region in enumerate(self.regions):
            region_end = pos + len(region[1])
            if region[0] == "partial" and start < region_end and end > pos:
                self._pin(i)
            pos = region_end
        regions: list[list[Any]] = []
        pos = 0
        for region in self.regions:
            text = region[1]
            region_end = pos + len(text)
            if start < region_end and end > pos:
                left = text[: max(0, start - pos)]
                right = text[min(len(text), end - pos):]
                region = [region[0], left + right, region[2]]
            regions.append(region)
            pos = region_end
        self._set_text(regions)
        return "applied"

    def _op_mark_command(self, op: dict[str, Any]) -> str:
        start, end, command = op["start"], op["end"], op["command"]
        if command not in COMMAND_KINDS:
            raise ValueError(f"unknown command kind {command!r}")
        if not 0 <= start < end <= len(self.text()):
            return "out_of_range"
        first = self._split_at(start)
        last = self._split_at(end)
        span_text = "".join(r[1] for r in self.regions[first:last])
        if any(r[0] == "partial" for r in self.regions[first:last]):
            return "refused_partial"
        self.regions[first:last] = [["command", span_text, {"command": command}]]
        return "applied"

    def _op_request(self, op: dict[str, Any]) -> str:
        request_id = op["request_id"]
        if self._request(request_id) is not None:
            return "duplicate"
        if any(r[0] == "partial" for r in self.regions):
            # Processing reads finals only; a live tail would make the
            # request stale the moment the next partial lands.
            return "refused_partial"
        retry_of = op.get("retry_of")
        if retry_of is not None:
            earlier = self._request(retry_of)
            if earlier is None:
                return "unknown_request"
            if earlier["status"] in ("pending", "interrupted"):
                earlier["status"] = "superseded"
        self.requests.append({
            "request_id": request_id,
            "base_revision": self.revision,
            "retry_of": retry_of,
            "input": self.payload_text(),
            "instruction": self.instruction(),
            "status": "pending",
        })
        return "pending"

    def _op_cancel(self, op: dict[str, Any]) -> str:
        request = self._request(op["request_id"])
        if request is None:
            return "unknown_request"
        if request["status"] not in ("pending", "interrupted"):
            return "not_pending"
        request["status"] = "cancelled"
        return "cancelled"

    def _op_result(self, op: dict[str, Any]) -> str:
        request = self._request(op["request_id"])
        if request is None:
            return "unknown_request"
        if self.deleted:
            return "discarded"
        status = request["status"]
        if status == "cancelled":
            return "discarded"
        if status == "settled" or self._proposal(op["request_id"]) is not None:
            return "duplicate"
        if op["status"] != "completed" or op.get("text") is None:
            # Raw text is untouched by a failure (a completed result without
            # text is one too); the request is done.
            if status in ("pending", "interrupted"):
                request["status"] = "settled"
            return "failed"
        proposal = {
            "request_id": request["request_id"],
            "base_revision": request["base_revision"],
            "text": op["text"],
            "status": "open",
        }
        self.proposals.append(proposal)
        if status == "superseded":
            proposal["status"] = "superseded"
            return "superseded"
        request["status"] = "settled"
        return "current" if request["base_revision"] == self.revision else "stale"

    def _accept(self, proposal: dict[str, Any]) -> None:
        self._set_text([["processed", proposal["text"], {"request_id": proposal["request_id"]}]])
        proposal["status"] = "accepted"

    def _op_accept(self, op: dict[str, Any]) -> str:
        proposal = self._proposal(op["request_id"])
        if proposal is None:
            return "unknown_request"
        if proposal["status"] != "open":
            return "not_open"
        if proposal["base_revision"] != self.revision and not op.get("force", False):
            return "stale_rejected"
        self._accept(proposal)
        return "applied"

    def _op_reject(self, op: dict[str, Any]) -> str:
        proposal = self._proposal(op["request_id"])
        if proposal is None:
            return "unknown_request"
        if proposal["status"] != "open":
            return "not_open"
        proposal["status"] = "rejected"
        return "rejected"

    def _op_revert_raw(self, op: dict[str, Any]) -> str:
        latest: dict[int, dict[str, Any]] = {}
        for attempt in self.attempts:
            latest[attempt["segment"]] = attempt
        regions = [["raw", latest[s]["text"],
                    {"segment": s, "attempt_id": latest[s]["attempt_id"]}]
                   for s in sorted(latest)]
        # Live partials of unfinished segments stay at the tail.
        regions += [copy.deepcopy(r) for r in self.regions if r[0] == "partial"]
        return "applied" if self._set_text(regions) else "unchanged"

    def _op_deliver(self, op: dict[str, Any]) -> str:
        delivery_id, digest = op["delivery_id"], op["target_digest"]
        if any(d["delivery_id"] == delivery_id for d in self.deliveries):
            return "duplicate"
        if any(d["revision"] == self.revision and d["target_digest"] == digest
               for d in self.deliveries):
            return "duplicate"
        if any(r[0] == "partial" for r in self.regions):
            return "refused_partial"
        self.deliveries.append({"delivery_id": delivery_id, "revision": self.revision,
                                "target_digest": digest})
        return "delivered"

    def _op_swap_check(self, op: dict[str, Any]) -> str:
        """Direct delivery: the processed text may replace the raw text
        already in the target only if the proposal is current and the
        target still has exactly what was delivered at its base."""
        proposal = self._proposal(op["request_id"])
        if proposal is None or proposal["status"] != "open":
            return "keep"
        if proposal["base_revision"] != self.revision:
            return "keep"
        delivered = [d for d in self.deliveries if d["revision"] == proposal["base_revision"]]
        if not delivered or delivered[-1]["target_digest"] != op["target_digest"]:
            return "keep"
        self._accept(proposal)
        return "swap"

    def _op_delete_draft(self, op: dict[str, Any]) -> str:
        self.deleted = True
        for request in self.requests:
            if request["status"] in ("pending", "interrupted"):
                request["status"] = "cancelled"
        return "deleted"

    def _op_crash(self, op: dict[str, Any]) -> str:
        """Process death and restart: live partials are not durable and are
        dropped; requests in flight become interrupted (a result that still
        arrives for one is judged like any other, against its base)."""
        for request in self.requests:
            if request["status"] == "pending":
                request["status"] = "interrupted"
        if not self.deleted:
            self._set_text([r for r in self.regions if r[0] != "partial"])
        return "restarted"

    # ------------------------------------------------------------------ #
    # Snapshot (draft.schema.json)
    # ------------------------------------------------------------------ #
    def snapshot(self) -> dict[str, Any]:
        regions = []
        pos = 0
        for kind, text, meta in self.regions:
            end = pos + len(text)
            regions.append({
                "kind": kind,
                "span": [pos, end],
                "segment": meta.get("segment"),
                "attempt_id": meta.get("attempt_id"),
                "command": meta.get("command"),
                "request_id": meta.get("request_id"),
            })
            pos = end
        proposals = []
        for p in self.proposals:
            status = p["status"]
            if status == "open":
                status = "current" if p["base_revision"] == self.revision else "stale"
            proposals.append({"request_id": p["request_id"],
                              "base_revision": p["base_revision"],
                              "text": p["text"], "status": status})
        return {
            "schema_version": 1,
            "draft_id": self.draft_id,
            "capture_id": self.capture_id,
            "revision": self.revision,
            "deleted": self.deleted,
            "text": self.text(),
            "span_encoding": "unicode_codepoints",
            "regions": regions,
            "attempts": [dict(a) for a in self.attempts],
            "pinned_segments": sorted(self.pinned),
            "requests": [{"request_id": r["request_id"],
                          "base_revision": r["base_revision"],
                          "retry_of": r["retry_of"],
                          "status": r["status"]} for r in self.requests],
            "proposals": proposals,
            "deliveries": [dict(d) for d in self.deliveries],
        }


# --------------------------------------------------------------------------- #
# Invariants checked after every op
# --------------------------------------------------------------------------- #


def invariant_violations(draft: Draft, finals: list[tuple[str, int, str]]) -> list[str]:
    """``finals`` is every (attempt_id, segment, text) the case fed in, in
    order, first occurrence per attempt id."""
    found = []
    snap = draft.snapshot()
    pos = 0
    for region in snap["regions"]:
        start, end = region["span"]
        if start != pos or end <= start:
            found.append(f"regions do not tile the text at {region}")
        pos = end
    if pos != len(snap["text"]):
        found.append("regions do not cover the text")
    seen: dict[str, tuple[int, str]] = {}
    for attempt_id, segment, text in finals:
        seen.setdefault(attempt_id, (segment, text))
    for attempt in draft.attempts:
        if seen.get(attempt["attempt_id"]) != (attempt["segment"], attempt["text"]):
            found.append(f"attempt {attempt['attempt_id']} is not the recognition it came from")
    recorded = {a["attempt_id"] for a in draft.attempts}
    latest: dict[int, str] = {}
    for attempt_id in dict.fromkeys(a for a, _, _ in finals):
        if attempt_id in recorded:
            segment, text = seen[attempt_id]
            latest[segment] = text
    expected_raw = "".join(latest[s] for s in sorted(latest))
    if draft.raw_text().encode("utf-8") != expected_raw.encode("utf-8"):
        found.append("raw text does not round-trip byte for byte")
    for proposal in snap["proposals"]:
        if proposal["status"] == "current" and proposal["base_revision"] != snap["revision"]:
            found.append(f"proposal {proposal['request_id']} current on an old base")
    return found


def run_case(case: dict[str, Any]) -> tuple[Draft, list[str], list[str]]:
    """Replay one fixture case. Returns (draft, outcomes, violations)."""
    draft = Draft(case.get("draft_id", "draft-1"), case.get("capture_id", "cap-1"))
    outcomes: list[str] = []
    violations: list[str] = []
    finals: list[tuple[str, int, str]] = []
    for step, op in enumerate(case["ops"]):
        if op["op"] == "final":
            finals.append((op["attempt_id"], op["segment"], op["text"]))
        outcome = draft.apply(op)
        outcomes.append(outcome)
        if "expect" in op and op["expect"] != outcome:
            violations.append(f"step {step} {op['op']}: expected {op['expect']!r}, got {outcome!r}")
        if "expect_text" in op and op["expect_text"] != draft.text():
            violations.append(f"step {step} {op['op']}: expected text {op['expect_text']!r}, "
                              f"got {draft.text()!r}")
        violations += [f"step {step}: {v}" for v in invariant_violations(draft, finals)]
    return draft, outcomes, violations


def expectation_mismatches(draft: Draft, expect: dict[str, Any]) -> list[str]:
    snap = draft.snapshot()
    found = []
    for key in ("text", "revision", "deleted", "pinned_segments"):
        if key in expect and snap[key] != expect[key]:
            found.append(f"{key}: expected {expect[key]!r}, got {snap[key]!r}")
    if "raw_text" in expect and draft.raw_text() != expect["raw_text"]:
        found.append(f"raw_text: expected {expect['raw_text']!r}, got {draft.raw_text()!r}")
    if "regions" in expect:
        got = [[r["kind"], snap["text"][r["span"][0]:r["span"][1]]] for r in snap["regions"]]
        if got != expect["regions"]:
            found.append(f"regions: expected {expect['regions']!r}, got {got!r}")
    if "requests" in expect:
        got = {r["request_id"]: r["status"] for r in snap["requests"]}
        if got != expect["requests"]:
            found.append(f"requests: expected {expect['requests']!r}, got {got!r}")
    if "proposals" in expect:
        got = {p["request_id"]: p["status"] for p in snap["proposals"]}
        if got != expect["proposals"]:
            found.append(f"proposals: expected {expect['proposals']!r}, got {got!r}")
    if "request_inputs" in expect:
        got = {r["request_id"]: [r["input"], r["instruction"]] for r in draft.requests}
        if got != expect["request_inputs"]:
            found.append(f"request_inputs: expected {expect['request_inputs']!r}, got {got!r}")
    return found


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def staging_cases() -> list[dict[str, Any]]:
    return load_json(FIXTURE_DIR / "staging.json")
