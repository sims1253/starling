"""Executable oracle for the Starling runtime protocol contract (E17 increment I0).

This module is the semantic half of the frozen v1 contract in
``packages/contracts/runtime-protocol/`` (the JSON schemas are the structural
half). It is deliberately stdlib-only so any implementation — the I3 Rust
crate, the I4 IPC host, or a third-party client — can be differentially tested
against it:

* envelope validation (structure, version pinning, NACK semantics);
* per-machine transition validators encoding E17 design note section 2 as
  data: states, ``command -> allowed-from-states -> events`` tables, and the
  runtime-internal edges that carry no wire message;
* trace replay: apply a fixture trace message by message, asserting every
  event is legal for the current machine state, every command is legal in the
  state it arrives in, and ``seq`` is strictly monotonic per stream;
* the cross-machine ordering rule that the audio route must be frozen before
  audio can leave the runtime (a ``jobs.submit`` is the first message that
  can carry audio out).

No runtime implementation exists yet; this oracle plus the fixtures is the
contract that I3/I4 must reproduce.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator

REPO = Path(__file__).resolve().parents[1]
CONTRACT = REPO / "packages" / "contracts" / "runtime-protocol"
FIXTURE_DIR = CONTRACT / "fixtures"

SUPPORTED_VERSIONS = (1,)

ENVELOPE_REQUIRED = ("v", "id", "ts", "type", "payload")
ENVELOPE_OPTIONAL = ("corr", "seq")

TS_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(\.[0-9]+)?(Z|[+-][0-9]{2}:[0-9]{2})$"
)
ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
TYPE_RE = re.compile(r"^[a-z][a-z0-9]*\.[a-z][A-Za-z0-9]*$")

# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #


def load_schema(name: str) -> dict[str, Any]:
    return json.loads((CONTRACT / name).read_text())


ENVELOPE_SCHEMA = load_schema("envelope.schema.json")
COMMAND_SCHEMA = load_schema("commands.schema.json")
EVENT_SCHEMA = load_schema("events.schema.json")


def message_kinds(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Map message type -> envelope branch, from a oneOf command/event schema."""
    kinds: dict[str, dict[str, Any]] = {}
    for branch in schema["oneOf"]:
        type_schema = branch["properties"]["type"]
        kinds[type_schema["const"]] = branch
    return kinds


COMMAND_KINDS = message_kinds(COMMAND_SCHEMA)
EVENT_KINDS = message_kinds(EVENT_SCHEMA)

# --------------------------------------------------------------------------- #
# Envelope validation + NACK
# --------------------------------------------------------------------------- #


def envelope_errors(msg: Any) -> list[str]:
    """Structural envelope checks (mirrors envelope.schema.json)."""
    if not isinstance(msg, dict):
        return [f"message is {type(msg).__name__}, expected object"]
    errors: list[str] = []
    for key in ENVELOPE_REQUIRED:
        if key not in msg:
            errors.append(f"missing required property {key!r}")
    for key in msg:
        if key not in ENVELOPE_REQUIRED and key not in ENVELOPE_OPTIONAL:
            errors.append(f"unexpected property {key!r}")
    if "v" in msg:
        v = msg["v"]
        if not isinstance(v, int) or isinstance(v, bool):
            errors.append("v must be an integer")
        elif v not in SUPPORTED_VERSIONS:
            errors.append(f"unsupported envelope version {v}")
    if "id" in msg and not (isinstance(msg["id"], str) and ID_RE.match(msg["id"])):
        errors.append("id must match the msgId token pattern")
    if "ts" in msg and not (isinstance(msg["ts"], str) and TS_RE.match(msg["ts"])):
        errors.append("ts must be an RFC 3339 timestamp")
    if "corr" in msg and not (
        isinstance(msg["corr"], str) and ID_RE.match(msg["corr"])
    ):
        errors.append("corr must match the msgId token pattern")
    if "seq" in msg:
        seq = msg["seq"]
        if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
            errors.append("seq must be a non-negative integer")
    if "type" in msg and not (
        isinstance(msg["type"], str) and TYPE_RE.match(msg["type"])
    ):
        errors.append("type must match the machine.message naming pattern")
    if "payload" in msg and not isinstance(msg["payload"], dict):
        errors.append("payload must be an object")
    return errors


def nack_for(msg: Any) -> dict[str, Any] | None:
    """The runtime.nack a receiver must answer an unparseable version with.

    Returns None when the message's version is supported (or absent, which is
    a plain validation failure handled elsewhere). The NACK carries
    ``corr`` = the rejected message id, per envelope.schema.json.
    """
    if not isinstance(msg, dict) or "v" not in msg:
        return None
    v = msg["v"]
    if isinstance(v, int) and not isinstance(v, bool) and v in SUPPORTED_VERSIONS:
        return None
    corr = msg.get("id") if isinstance(msg.get("id"), str) else None
    return {
        "v": 1,
        "id": "nack_" + str(corr or "unknown"),
        "ts": "1970-01-01T00:00:00Z",  # filled by the emitting runtime
        "corr": corr,
        "type": "runtime.nack",
        "payload": {"reason": "unsupported_version"},
    }


# --------------------------------------------------------------------------- #
# The five state machines (E17 design note section 2, as data)
# --------------------------------------------------------------------------- #
# Command spec keys:
#   from     -- states the command is legal in ("*" = any state);
#   to       -- state entered immediately (no outcome event expected);
#   outcomes -- {event type: target state} — the command awaits exactly one of
#               these events, correlated by corr (None target = state kept).
# Event spec keys:
#   from     -- states the event is legal in (empty = only reachable as the
#               pending outcome of its command);
#   to       -- state entered (None = unchanged);
#   to_when  -- payload-conditioned target (capture.error fatal -> Interrupted).
# internal: runtime edges that carry no wire message (scheduler dispatch,
# expiry timers, journal recovery); fixtures step over them explicitly with
# {"$advance": "<state>"} directives.

MACHINES: dict[str, dict[str, Any]] = {
    "capture": {
        "initial": "Idle",
        "states": [
            "Idle",
            "Acquiring",
            "Recording",
            "Draining",
            "Persisted",
            "Interrupted",
            "Recovering",
        ],
        "commands": {
            "capture.start": {
                "from": ["Idle", "Persisted"],
                "to": "Acquiring",
            },
            "capture.stop": {"from": ["Recording"], "to": "Draining"},
            "capture.abort": {
                "from": ["Acquiring", "Recording", "Draining"],
                "to": "Idle",
            },
        },
        "events": {
            "capture.started": {"from": ["Acquiring"], "to": "Recording"},
            "capture.progress": {
                "from": ["Recording", "Draining", "Recovering"],
                "to": None,
            },
            "capture.gap": {
                "from": ["Recording", "Draining", "Recovering"],
                "to": None,
            },
            "capture.error": {
                "from": ["Acquiring", "Recording", "Draining", "Recovering"],
                "to": None,
                "to_when": {"payload": "fatal", "equals": True, "to": "Interrupted"},
            },
            "capture.stopped": {"from": ["Draining", "Recovering"], "to": "Persisted"},
        },
        # Runtime-internal edges out of Interrupted (no wire message): a
        # salvaged take replays its durable boundary through Recovering,
        # while a failure that killed only the take attempt -- the device
        # never opened, so there is nothing to salvage or replay -- settles
        # straight back to Idle so the machine can accept the next
        # capture.start (issue #211).
        "internal": [["Interrupted", "Recovering"], ["Interrupted", "Idle"]],
    },
    "jobs": {
        "initial": "Idle",
        "states": [
            "Idle",
            "Queued",
            "Dispatched",
            "Loading",
            "Recognizing",
            "Transforming",
            "Completed",
            "Failed",
            "Cancelled",
            "Rejected",
        ],
        "commands": {
            "jobs.submit": {
                "from": ["Idle", "Completed", "Failed", "Cancelled", "Rejected"],
                "outcomes": {
                    "jobs.queued": "Queued",
                    "jobs.rejected": "Rejected",
                },
            },
            "jobs.cancel": {
                "from": [
                    "Queued",
                    "Dispatched",
                    "Loading",
                    "Recognizing",
                    "Transforming",
                ],
                "to": "Cancelled",
            },
            "jobs.setLimits": {"from": "*", "to": None},
        },
        "events": {
            "jobs.queued": {"from": [], "to": "Queued"},
            "jobs.rejected": {"from": [], "to": "Rejected"},
            "jobs.progress": {
                "from": ["Loading", "Recognizing", "Transforming"],
                "to": None,
            },
            "jobs.completed": {
                "from": ["Recognizing", "Transforming"],
                "to": "Completed",
            },
            "jobs.failed": {
                "from": [
                    "Queued",
                    "Dispatched",
                    "Loading",
                    "Recognizing",
                    "Transforming",
                ],
                "to": "Failed",
            },
        },
        "internal": [
            ["Queued", "Dispatched"],
            ["Dispatched", "Loading"],
            ["Loading", "Recognizing"],
            ["Recognizing", "Transforming"],
        ],
    },
    "context": {
        "initial": "Observing",
        "states": [
            "Observing",
            "SnapshotTaken",
            "ModeDecided",
            "RouteFrozen",
            "Expired",
            "Released",
        ],
        "commands": {
            "context.snapshot": {
                "from": ["Observing"],
                "outcomes": {"context.targetSnapshot": "SnapshotTaken"},
            },
            "mode.set": {
                "from": ["SnapshotTaken", "ModeDecided"],
                "outcomes": {"mode.decision": "ModeDecided"},
            },
            "context.expire": {
                "from": ["SnapshotTaken", "ModeDecided", "RouteFrozen"],
                "to": "Expired",
            },
        },
        "events": {
            "context.targetSnapshot": {"from": [], "to": "SnapshotTaken"},
            "mode.decision": {"from": [], "to": "ModeDecided"},
            # Emitted when the audio route freezes at capture start; only
            # legal once a mode has been decided, never after (a later spoken
            # phrase cannot un-send audio).
            "mode.routeFrozen": {"from": ["ModeDecided"], "to": "RouteFrozen"},
        },
        "internal": [
            ["SnapshotTaken", "Expired"],
            ["ModeDecided", "Expired"],
            ["RouteFrozen", "Released"],
            ["Expired", "Observing"],
            ["Released", "Observing"],
        ],
    },
    "docs": {
        "initial": "Steady",
        "states": ["Steady", "Validating", "Committed", "Conflicted"],
        "commands": {
            "docs.updateHead": {
                "from": ["Steady", "Committed", "Conflicted"],
                "to": "Validating",
            },
            "docs.appendTurn": {
                "from": ["Steady", "Committed"],
                "outcomes": {"docs.turnAppended": None},
            },
            "docs.get": {"from": ["Steady", "Committed", "Conflicted"], "to": None},
        },
        "events": {
            "docs.headUpdated": {"from": ["Validating"], "to": "Committed"},
            "docs.headConflict": {"from": ["Validating"], "to": "Conflicted"},
            "docs.turnAppended": {"from": [], "to": None},
        },
        "internal": [["Committed", "Steady"]],
    },
    "delivery": {
        "initial": "Idle",
        "states": [
            "Idle",
            "Prepared",
            "Revalidating",
            "SubmittedUnconfirmed",
            "Confirmed",
            "Failed",
            "Conflict",
            "Cancelled",
        ],
        "commands": {
            "delivery.prepare": {
                "from": ["Idle", "Confirmed", "Failed", "Conflict", "Cancelled"],
                "outcomes": {"delivery.prepared": "Prepared"},
            },
            "delivery.apply": {"from": ["Prepared"], "to": "Revalidating"},
            "delivery.cancel": {
                "from": ["Prepared", "Revalidating", "SubmittedUnconfirmed"],
                "to": "Cancelled",
            },
            "delivery.copyFallback": {
                "from": ["Failed", "Conflict"],
                "to": None,
            },
        },
        "events": {
            "delivery.prepared": {"from": [], "to": "Prepared"},
            "delivery.submittedUnconfirmed": {
                "from": ["Revalidating"],
                "to": "SubmittedUnconfirmed",
            },
            "delivery.confirmed": {
                "from": ["SubmittedUnconfirmed"],
                "to": "Confirmed",
            },
            "delivery.failed": {
                "from": ["Revalidating", "SubmittedUnconfirmed"],
                "to": "Failed",
            },
            "delivery.conflict": {
                "from": ["Revalidating", "SubmittedUnconfirmed"],
                "to": "Conflict",
            },
        },
        "internal": [],
    },
}


def machine_commands(machine: str) -> set[str]:
    return set(MACHINES[machine]["commands"])


def machine_events(machine: str) -> set[str]:
    return set(MACHINES[machine]["events"])


def exits(machine: str, state: str) -> set[str]:
    """Message types that can move the machine out of ``state``."""
    spec = MACHINES[machine]
    leaving: set[str] = set()
    for cmd, rule in spec["commands"].items():
        if state in rule["from"] or rule["from"] == "*":
            if rule.get("to") not in (None, state):
                leaving.add(cmd)
            for target in rule.get("outcomes", {}).values():
                if target not in (None, state):
                    leaving.add(cmd)
    for evt, rule in spec["events"].items():
        if state in rule["from"]:
            target = rule.get("to")
            if rule.get("to_when") and rule["to_when"]["to"] != state:
                target = rule["to_when"]["to"]
            if target not in (None, state):
                leaving.add(evt)
    return leaving


def internal_targets(machine: str, state: str) -> set[str]:
    return {to for from_, to in MACHINES[machine]["internal"] if from_ == state}


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #


class ProtocolViolation(Exception):
    """A fixture trace broke the contract. ``code`` names the violation class."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def is_directive(record: Any) -> bool:
    return isinstance(record, dict) and any(
        key.startswith("$") for key in record
    )


def iter_envelopes(trace: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for record in trace["messages"]:
        if not is_directive(record):
            yield record


def _stream_key(msg: dict[str, Any], kind: str) -> str:
    corr = msg.get("corr")
    return corr if isinstance(corr, str) else f"__{kind}s__"


class MachineReplay:
    """Replays one fixture trace against one machine's transition table."""

    def __init__(self, machine: str, initial: str | None = None):
        self.machine = machine
        self.spec = MACHINES[machine]
        self.state = initial if initial is not None else self.spec["initial"]
        if self.state not in self.spec["states"]:
            raise ProtocolViolation(
                "unknown_state", f"initial state {self.state!r} not in machine {machine}"
            )
        self.visited: list[str] = [self.state]
        self.transitions: list[dict[str, Any]] = []
        self._last_seq: dict[str, int] = {}
        self._pending: dict[str, Any] | None = None

    # -- internals ---------------------------------------------------------- #

    def _enter(self, state: str | None, kind: str, msg_type: str) -> None:
        if state is not None and state != self.state:
            self.transitions.append(
                {"kind": kind, "type": msg_type, "from": self.state, "to": state}
            )
            self.state = state
            self.visited.append(state)
        else:
            self.transitions.append(
                {"kind": kind, "type": msg_type, "from": self.state, "to": self.state}
            )

    def _check_seq(self, msg: dict[str, Any], kind: str) -> None:
        if "seq" not in msg:
            return
        stream = _stream_key(msg, kind)
        seq = msg["seq"]
        last = self._last_seq.get(stream)
        if last is not None and seq <= last:
            raise ProtocolViolation(
                "seq_not_monotonic",
                f"{msg.get('type')} seq {seq} on stream {stream!r} "
                f"follows seq {last}",
            )
        self._last_seq[stream] = seq

    def _check_envelope(self, msg: dict[str, Any]) -> str:
        v = msg.get("v")
        if (
            isinstance(v, int)
            and not isinstance(v, bool)
            and "v" in msg
            and v not in SUPPORTED_VERSIONS
        ):
            raise ProtocolViolation(
                "nack_unsupported_version",
                f"envelope version {v!r} is not supported "
                f"(supported: {list(SUPPORTED_VERSIONS)}); receiver answers "
                "runtime.nack{unsupported_version}",
            )
        errors = envelope_errors(msg)
        if errors:
            raise ProtocolViolation("invalid_envelope", "; ".join(errors))
        msg_type = msg["type"]
        if msg_type in COMMAND_KINDS:
            kind = "command"
        elif msg_type in EVENT_KINDS:
            kind = "event"
        else:
            raise ProtocolViolation(
                "unknown_message_type", f"{msg_type!r} is not a v1 message type"
            )
        own = machine_commands(self.machine) | machine_events(self.machine)
        if msg_type not in own:
            raise ProtocolViolation(
                "foreign_message",
                f"{msg_type!r} does not belong to machine {self.machine!r}",
            )
        return kind

    def _apply_command(self, msg: dict[str, Any]) -> None:
        if self._pending is not None:
            raise ProtocolViolation(
                "pending_unresolved",
                f"{msg['type']} arrived while {self._pending['type']} still "
                f"awaits one of {sorted(self._pending['outcomes'])}",
            )
        rule = self.spec["commands"][msg["type"]]
        allowed = rule["from"] == "*" or self.state in rule["from"]
        if not allowed:
            raise ProtocolViolation(
                "illegal_command",
                f"{msg['type']} is not legal in state {self.state!r} "
                f"(allowed from {rule['from']})",
            )
        if "outcomes" in rule:
            self._pending = {
                "type": msg["type"],
                "corr": msg.get("corr"),
                "id": msg.get("id"),
                "outcomes": rule["outcomes"],
            }
            self.transitions.append(
                {
                    "kind": "command",
                    "type": msg["type"],
                    "from": self.state,
                    "to": self.state,
                    "pending": sorted(rule["outcomes"]),
                }
            )
        else:
            self._enter(rule.get("to"), "command", msg["type"])

    def _apply_event(self, msg: dict[str, Any]) -> None:
        msg_type = msg["type"]
        if self._pending is not None:
            outcomes = self._pending["outcomes"]
            if msg_type in outcomes:
                if msg.get("corr") != self._pending["corr"]:
                    raise ProtocolViolation(
                        "corr_mismatch",
                        f"{msg_type} resolves {self._pending['type']} but corr "
                        f"{msg.get('corr')!r} != {self._pending['corr']!r}",
                    )
                target = outcomes[msg_type]
                self._pending = None
                self._enter(target, "event", msg_type)
                return
            raise ProtocolViolation(
                "illegal_event",
                f"{msg_type} arrived while {self._pending['type']} awaits one "
                f"of {sorted(outcomes)}",
            )
        rule = self.spec["events"][msg_type]
        if self.state not in rule["from"]:
            raise ProtocolViolation(
                "illegal_event",
                f"{msg_type} is not legal in state {self.state!r} "
                f"(legal from {rule['from']})",
            )
        target = rule.get("to")
        cond = rule.get("to_when")
        if cond is not None and msg["payload"].get(cond["payload"]) == cond["equals"]:
            target = cond["to"]
        self._enter(target, "event", msg_type)

    def _apply_directive(self, record: dict[str, Any]) -> None:
        if set(record) != {"$advance"}:
            raise ProtocolViolation(
                "unknown_directive", f"unsupported trace directive {record!r}"
            )
        target = record["$advance"]
        if self._pending is not None:
            raise ProtocolViolation(
                "pending_unresolved",
                f"$advance arrived while {self._pending['type']} still awaits "
                f"one of {sorted(self._pending['outcomes'])}",
            )
        if target not in internal_targets(self.machine, self.state):
            raise ProtocolViolation(
                "internal_edge_violation",
                f"no runtime-internal edge {self.state!r} -> {target!r} in "
                f"machine {self.machine!r}",
            )
        self.transitions.append(
            {"kind": "internal", "type": None, "from": self.state, "to": target}
        )
        self.state = target
        self.visited.append(target)

    # -- public ------------------------------------------------------------- #

    def feed(self, record: dict[str, Any]) -> None:
        """Apply one trace record (an envelope or a $-directive)."""
        if is_directive(record):
            self._apply_directive(record)
            return
        kind = self._check_envelope(record)
        self._check_seq(record, kind)
        if kind == "command":
            self._apply_command(record)
        else:
            self._apply_event(record)

    def finish(self) -> "MachineReplay":
        if self._pending is not None:
            raise ProtocolViolation(
                "pending_unresolved",
                f"trace ends while {self._pending['type']} still awaits one "
                f"of {sorted(self._pending['outcomes'])}",
            )
        return self


def replay_trace(trace: dict[str, Any]) -> MachineReplay:
    """Replay a fixture trace; raises ProtocolViolation on the first break."""
    replay = MachineReplay(trace["machine"], trace.get("initial"))
    for record in trace["messages"]:
        replay.feed(record)
    return replay.finish()


def seq_violations(messages: list[dict[str, Any]]) -> list[str]:
    """Standalone per-stream monotonicity check (strictly increasing)."""
    last: dict[str, int] = {}
    found: list[str] = []
    for msg in messages:
        if is_directive(msg) or "seq" not in msg or "type" not in msg:
            continue
        kind = "command" if msg["type"] in COMMAND_KINDS else "event"
        stream = _stream_key(msg, kind)
        if stream in last and msg["seq"] <= last[stream]:
            found.append(
                f"{msg['type']} seq {msg['seq']} on stream {stream!r} "
                f"follows seq {last[stream]}"
            )
        last[stream] = msg["seq"]
    return found


# --------------------------------------------------------------------------- #
# Cross-machine ordering: freeze before audio leaves
# --------------------------------------------------------------------------- #


def route_freeze_violations(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every jobs.submit must reference a route frozen by an earlier
    mode.routeFrozen (E17: the audio route freezes before the first frame can
    leave the runtime). ``jobs.submit`` is the first message that can carry
    audio out of the runtime, so it is the audio-leave proxy in this corpus;
    messages must be passed in global (ts) order.
    """
    frozen: dict[str, str] = {}
    violations: list[dict[str, Any]] = []
    for msg in messages:
        if is_directive(msg):
            continue
        msg_type = msg.get("type")
        ts = msg.get("ts", "")
        if msg_type == "mode.routeFrozen":
            route = msg["payload"]["route"]
            if route not in frozen or ts < frozen[route]:
                frozen[route] = ts
        elif msg_type == "jobs.submit":
            route = msg["payload"]["route"]
            if route not in frozen or frozen[route] >= ts:
                violations.append(
                    {
                        "id": msg.get("id"),
                        "route": route,
                        "detail": f"jobs.submit {msg.get('id')} references route "
                        f"{route!r} before any mode.routeFrozen froze it",
                    }
                )
    return violations


# --------------------------------------------------------------------------- #
# Fixture loading
# --------------------------------------------------------------------------- #


def load_fixture(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_fixtures() -> tuple[list[tuple[Path, dict]], list[tuple[Path, dict]]]:
    """(valid traces, invalid traces) from fixtures/ and fixtures/invalid/."""
    valid: list[tuple[Path, dict]] = []
    invalid: list[tuple[Path, dict]] = []
    for path in sorted(FIXTURE_DIR.rglob("*.json")):
        trace = load_fixture(path)
        (invalid if "invalid" in path.parent.name else valid).append((path, trace))
    return valid, invalid


def valid_corpus_messages() -> list[dict[str, Any]]:
    """All envelopes of all valid fixtures in global timestamp order."""
    messages: list[tuple[str, int, dict]] = []
    for path, trace in load_fixtures()[0]:
        for index, msg in enumerate(iter_envelopes(trace)):
            messages.append((msg.get("ts", ""), index, msg))
    messages.sort(key=lambda item: (item[0], item[2].get("id", "")))
    return [msg for _, _, msg in messages]
