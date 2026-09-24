//! The v1 wire protocol: envelope, commands, events (E17-I0 contract).
//!
//! Every public type here names its frozen counterpart in
//! `packages/contracts/runtime-protocol/` (`commands.schema.json`,
//! `events.schema.json`, `envelope.schema.json`). The schemas are structural;
//! [`crate::protocol::tables`] + [`crate::protocol::replay`] are the
//! executable semantics ported from `tests/runtime_protocol.py` (the I0
//! oracle).
//!
//! Two shapes live side by side on purpose:
//!
//! * [`Envelope`] — the envelope as parsed from (or serialized to) wire JSON,
//!   with the payload kept as `serde_json::Value`. This is what
//!   [`replay`](crate::protocol::replay) consumes and what unknown versions
//!   are NACKed against.
//! * [`Command`] / [`Event`] — the typed v1 payload sets, converted from /
//!   to the envelope's `type` + `payload`. Payload structs use
//!   `deny_unknown_fields`, which is the Rust spelling of the schemas'
//!   `additionalProperties: false`: an unknown payload field is a validation
//!   failure, never a silently ignored extra.

pub mod replay;
pub mod tables;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Envelope versions this runtime understands (envelope.schema.json
/// `$defs.version` currently pins `const: 1`).
pub const SUPPORTED_VERSIONS: &[u64] = &[1];

/// The envelope every message shares (envelope.schema.json). Required:
/// `v`, `id`, `ts`, `type`, `payload`. Optional: `corr`, `seq`.
#[derive(Debug, Clone, PartialEq)]
pub struct Envelope {
    pub v: u64,
    pub id: String,
    pub ts: String,
    pub corr: Option<String>,
    pub seq: Option<u64>,
    pub type_: String,
    pub payload: Value,
}

impl Envelope {
    /// Parses the wire form, preserving unknown `type`s and payload shapes
    /// (validation is [`envelope_errors`]'s job, not the parser's).
    pub fn from_value(value: &Value) -> Option<Envelope> {
        let object = value.as_object()?;
        Some(Envelope {
            v: object.get("v")?.as_u64()?,
            id: object.get("id")?.as_str()?.to_string(),
            ts: object.get("ts")?.as_str()?.to_string(),
            corr: object
                .get("corr")
                .and_then(Value::as_str)
                .map(str::to_string),
            seq: object.get("seq").and_then(Value::as_u64),
            type_: object.get("type")?.as_str()?.to_string(),
            payload: object.get("payload").cloned().unwrap_or(Value::Null),
        })
    }

    /// Serializes back to the wire form.
    pub fn to_value(&self) -> Value {
        let mut object = serde_json::Map::new();
        object.insert("v".into(), Value::from(self.v));
        object.insert("id".into(), Value::from(self.id.clone()));
        object.insert("ts".into(), Value::from(self.ts.clone()));
        if let Some(corr) = &self.corr {
            object.insert("corr".into(), Value::from(corr.clone()));
        }
        if let Some(seq) = self.seq {
            object.insert("seq".into(), Value::from(seq));
        }
        object.insert("type".into(), Value::from(self.type_.clone()));
        object.insert("payload".into(), self.payload.clone());
        Value::Object(object)
    }
}

// ---------------------------------------------------------------------------
// Token validation (the schemas' `msgId` / `safeToken` / `timestamp` /
// `type` patterns, hand-rolled so the runtime adds no regex dependency).
// ---------------------------------------------------------------------------

/// `msgId`: `^[A-Za-z0-9_.:-]{1,128}$`.
pub fn is_msg_id(value: &str) -> bool {
    (1..=128).contains(&value.len())
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b':' | b'-'))
}

/// `safeToken`: `^[A-Za-z0-9_.:+-]{1,128}$`.
pub fn is_safe_token(value: &str) -> bool {
    (1..=128).contains(&value.len())
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b':' | b'+' | b'-'))
}

/// `type`: `^[a-z][a-z0-9]*\.[a-z][A-Za-z0-9]*$` (`machine.message`).
pub fn is_type_name(value: &str) -> bool {
    let Some((machine, message)) = value.split_once('.') else {
        return false;
    };
    let mut machine_chars = machine.chars();
    match machine_chars.next() {
        Some(first) if first.is_ascii_lowercase() => {}
        _ => return false,
    }
    if !machine_chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit()) {
        return false;
    }
    let mut message_chars = message.chars();
    match message_chars.next() {
        Some(first) if first.is_ascii_lowercase() => {}
        _ => return false,
    }
    message_chars.all(|c| c.is_ascii_alphanumeric())
}

/// `timestamp`: RFC 3339 with optional fraction and `Z` / `±HH:MM` zone
/// (envelope.schema.json `$defs.timestamp`). ASCII digits only.
///
/// The fixed positions are validated on `bytes`, not on `str` slices: a
/// multi-byte UTF-8 `ts` is out of contract and must be *rejected*, and a
/// `&value[a..b]` would panic on a non-char-boundary index instead
/// (issue #202 — that panic took down the router thread).
pub fn is_rfc3339(value: &str) -> bool {
    fn digits(slice: &[u8], count: usize) -> bool {
        slice.len() == count && slice.iter().all(|b| b.is_ascii_digit())
    }
    let bytes = value.as_bytes();
    if bytes.len() < 20 {
        return false;
    }
    if !digits(&bytes[0..4], 4) || bytes[4] != b'-' {
        return false;
    }
    if !digits(&bytes[5..7], 2) || bytes[7] != b'-' {
        return false;
    }
    if !digits(&bytes[8..10], 2) || bytes[10] != b'T' {
        return false;
    }
    if !digits(&bytes[11..13], 2) || bytes[13] != b':' {
        return false;
    }
    if !digits(&bytes[14..16], 2) || bytes[16] != b':' {
        return false;
    }
    if !digits(&bytes[17..19], 2) {
        return false;
    }
    // Byte 19 is a char boundary: bytes 17 and 18 are ASCII digits, so no
    // multi-byte char can be in flight here.
    let mut rest = &value[19..];
    if let Some(fraction) = rest.strip_prefix('.') {
        let end = fraction
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(fraction.len());
        if end == 0 {
            return false;
        }
        rest = &fraction[end..];
    }
    match rest {
        "Z" => true,
        offset => {
            let bytes = offset.as_bytes();
            bytes.len() == 6
                && (bytes[0] == b'+' || bytes[0] == b'-')
                && digits(&bytes[1..3], 2)
                && bytes[3] == b':'
                && digits(&bytes[4..6], 2)
        }
    }
}

/// Structural envelope checks — the port of the oracle's `envelope_errors`
/// (which mirrors envelope.schema.json). Returns one human-readable error per
/// violation, in the oracle's check order.
pub fn envelope_errors(msg: &Value) -> Vec<String> {
    let Value::Object(object) = msg else {
        return vec![format!(
            "message is {}, expected object",
            json_type_name(msg)
        )];
    };
    let mut errors = Vec::new();
    for key in ["v", "id", "ts", "type", "payload"] {
        if !object.contains_key(key) {
            errors.push(format!("missing required property '{key}'"));
        }
    }
    for key in object.keys() {
        if !matches!(
            key.as_str(),
            "v" | "id" | "ts" | "type" | "payload" | "corr" | "seq"
        ) {
            errors.push(format!("unexpected property '{key}'"));
        }
    }
    if let Some(v) = object.get("v") {
        match v.as_u64() {
            Some(version) => {
                if !SUPPORTED_VERSIONS.contains(&version) {
                    errors.push(format!("unsupported envelope version {version}"));
                }
            }
            None => errors.push("v must be an integer".to_string()),
        }
    }
    if let Some(id) = object.get("id") {
        if !id.as_str().is_some_and(is_msg_id) {
            errors.push("id must match the msgId token pattern".to_string());
        }
    }
    if let Some(ts) = object.get("ts") {
        if !ts.as_str().is_some_and(is_rfc3339) {
            errors.push("ts must be an RFC 3339 timestamp".to_string());
        }
    }
    if let Some(corr) = object.get("corr") {
        if !corr.as_str().is_some_and(is_msg_id) {
            errors.push("corr must match the msgId token pattern".to_string());
        }
    }
    if let Some(seq) = object.get("seq") {
        if seq.as_u64().is_none() {
            errors.push("seq must be a non-negative integer".to_string());
        }
    }
    if let Some(type_) = object.get("type") {
        if !type_.as_str().is_some_and(is_type_name) {
            errors.push("type must match the machine.message naming pattern".to_string());
        }
    }
    if let Some(payload) = object.get("payload") {
        if !payload.is_object() {
            errors.push("payload must be an object".to_string());
        }
    }
    errors
}

fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

/// The `runtime.nack` a receiver must answer an unparseable version with
/// (port of the oracle's `nack_for`). Returns `None` when the version is
/// supported (or absent — a plain validation failure handled by
/// [`envelope_errors`]). The NACK carries `corr` = the rejected message id.
pub fn nack_for(msg: &Value) -> Option<Value> {
    let object = msg.as_object()?;
    let v = object.get("v")?;
    let supported = v
        .as_u64()
        .is_some_and(|version| SUPPORTED_VERSIONS.contains(&version));
    if supported {
        return None;
    }
    let corr = object.get("id").and_then(Value::as_str).map(str::to_string);
    let mut nack = serde_json::Map::new();
    nack.insert("v".into(), Value::from(1u64));
    nack.insert(
        "id".into(),
        Value::from(format!("nack_{}", corr.as_deref().unwrap_or("unknown"))),
    );
    nack.insert("ts".into(), Value::from("1970-01-01T00:00:00Z"));
    if let Some(corr) = corr {
        nack.insert("corr".into(), Value::from(corr));
    }
    nack.insert("type".into(), Value::from("runtime.nack"));
    nack.insert(
        "payload".into(),
        serde_json::json!({ "reason": "unsupported_version" }),
    );
    Some(Value::Object(nack))
}

/// Which v1 kind a type name belongs to. `runtime.nack` is an event
/// (events.schema.json); unknown types belong to neither.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Command,
    Event,
}

// ---------------------------------------------------------------------------
// Typed payloads. Field-for-field with commands.schema.json /
// events.schema.json; `deny_unknown_fields` implements
// `additionalProperties: false`.
// ---------------------------------------------------------------------------

/// `span` (`{startOffset, endOffset}`), shared by
/// `context.targetSnapshot.selectionRange` and
/// `mode.decision.matchedPrefixSpan`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct Span {
    pub start_offset: u64,
    pub end_offset: u64,
}

/// A gap span in `capture.gap` / `capture.stopped.gaps`
/// (`{startSample, endSample}`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct SampleGap {
    pub start_sample: u64,
    pub end_sample: u64,
}

/// The immutable revision of `docs.updateHead.newRevision` — §2.4's
/// `{revId, baseRevision, sourceAttemptIds[], instructionTemplateId, text,
/// status, provenance}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct Revision {
    pub rev_id: String,
    pub base_revision: u64,
    pub source_attempt_ids: Vec<String>,
    pub instruction_template_id: String,
    pub text: String,
    pub status: String,
    pub provenance: String,
}

/// `jobs.rejected.reason` — the one enumerated rejection vocabulary in v1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RejectReason {
    QueueFull,
    ResourceLimits,
    DuplicateSubmission,
}

impl RejectReason {
    pub fn as_str(self) -> &'static str {
        match self {
            RejectReason::QueueFull => "queue_full",
            RejectReason::ResourceLimits => "resource_limits",
            RejectReason::DuplicateSubmission => "duplicate_submission",
        }
    }
}

/// `mode.decision.source` — the other enumerated vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionSource {
    Manual,
    Phrase,
    Rule,
}

impl DecisionSource {
    pub fn as_str(self) -> &'static str {
        match self {
            DecisionSource::Manual => "manual",
            DecisionSource::Phrase => "phrase",
            DecisionSource::Rule => "rule",
        }
    }
}

/// One `perRoute` entry of `jobs.setLimits`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct RouteLimit {
    pub route: String,
    pub max_concurrent: u32,
}

/// `jobs.setLimits` payload: bounded queue, max concurrent, optional
/// per-route caps (`maxQueued` + `maxConcurrent` required).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct JobLimits {
    pub max_queued: u32,
    pub max_concurrent: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub per_route: Vec<RouteLimit>,
}

/// All 16 v1 commands, variant per `commands.schema.json` branch. The enum
/// is the typed command set; [`Command::type_name`] is the wire `type`.
#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    /// `capture.start{policy}`.
    CaptureStart { policy: String },
    /// `capture.stop{drain?}`.
    CaptureStop { drain: Option<bool> },
    /// `capture.abort{}` — no v1 reply event.
    CaptureAbort,
    /// `jobs.submit{captureRef, route, budget}`.
    JobsSubmit {
        capture_ref: String,
        route: String,
        budget: String,
    },
    /// `jobs.cancel{jobId}` — no v1 reply event.
    JobsCancel { job_id: String },
    /// `jobs.setLimits{maxQueued, maxConcurrent, perRoute?}` — legal in
    /// every state.
    JobsSetLimits(JobLimits),
    /// `context.snapshot{source}`.
    ContextSnapshot { source: String },
    /// `mode.set{mode, source: manual}`.
    ModeSet { mode: String, source: Manual },
    /// `context.expire{}`.
    ContextExpire,
    /// `docs.updateHead{docId, expectedBase, newRevision}` — compare-and-swap.
    DocsUpdateHead {
        doc_id: String,
        expected_base: u64,
        new_revision: Revision,
    },
    /// `docs.appendTurn{docId, takeRef}`.
    DocsAppendTurn { doc_id: String, take_ref: String },
    /// `docs.get{docId, page}` — served via snapshot, no v1 event.
    DocsGet { doc_id: String, page: u32 },
    /// `delivery.prepare{revisionId, targetRef}`.
    DeliveryPrepare {
        revision_id: String,
        target_ref: String,
    },
    /// `delivery.apply{deliveryId}` — user-initiated only.
    DeliveryApply { delivery_id: String },
    /// `delivery.cancel{deliveryId?}` — no v1 reply event.
    DeliveryCancel { delivery_id: Option<String> },
    /// `delivery.copyFallback{deliveryId}` — closes nothing in v1.
    DeliveryCopyFallback { delivery_id: String },
}

/// `mode.set.source` is `const "manual"` in v1 (phrase/rule sources decide
/// through `mode.decision`, not through this command).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Manual;

// Payload shapes for wire parsing: `deny_unknown_fields` is the schemas'
// `additionalProperties: false` (an unknown field is a validation failure,
// never a silently ignored extra) and missing required fields fail too.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CaptureStartP {
    policy: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CaptureStopP {
    #[serde(default)]
    drain: Option<bool>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct JobsSubmitP {
    #[serde(rename = "captureRef")]
    capture_ref: String,
    route: String,
    budget: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct JobsCancelP {
    #[serde(rename = "jobId")]
    job_id: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DeliveryCancelP {
    #[serde(rename = "deliveryId")]
    delivery_id: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ContextSnapshotP {
    source: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModeSetP {
    mode: String,
    source: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DocsUpdateHeadP {
    #[serde(rename = "docId")]
    doc_id: String,
    #[serde(rename = "expectedBase")]
    expected_base: u64,
    #[serde(rename = "newRevision")]
    new_revision: Revision,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DocsAppendTurnP {
    #[serde(rename = "docId")]
    doc_id: String,
    #[serde(rename = "takeRef")]
    take_ref: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DocsGetP {
    #[serde(rename = "docId")]
    doc_id: String,
    page: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DeliveryPrepareP {
    #[serde(rename = "revisionId")]
    revision_id: String,
    #[serde(rename = "targetRef")]
    target_ref: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DeliveryApplyP {
    #[serde(rename = "deliveryId")]
    delivery_id: String,
}

impl Command {
    /// The wire `type` of this command.
    pub fn type_name(&self) -> &'static str {
        match self {
            Command::CaptureStart { .. } => "capture.start",
            Command::CaptureStop { .. } => "capture.stop",
            Command::CaptureAbort => "capture.abort",
            Command::JobsSubmit { .. } => "jobs.submit",
            Command::JobsCancel { .. } => "jobs.cancel",
            Command::JobsSetLimits(_) => "jobs.setLimits",
            Command::ContextSnapshot { .. } => "context.snapshot",
            Command::ModeSet { .. } => "mode.set",
            Command::ContextExpire => "context.expire",
            Command::DocsUpdateHead { .. } => "docs.updateHead",
            Command::DocsAppendTurn { .. } => "docs.appendTurn",
            Command::DocsGet { .. } => "docs.get",
            Command::DeliveryPrepare { .. } => "delivery.prepare",
            Command::DeliveryApply { .. } => "delivery.apply",
            Command::DeliveryCancel { .. } => "delivery.cancel",
            Command::DeliveryCopyFallback { .. } => "delivery.copyFallback",
        }
    }

    /// The machine this command belongs to (the `machine` of
    /// `machine.message`).
    pub fn machine(&self) -> &'static str {
        match self {
            Command::CaptureStart { .. } | Command::CaptureStop { .. } | Command::CaptureAbort => {
                "capture"
            }
            Command::JobsSubmit { .. } | Command::JobsCancel { .. } | Command::JobsSetLimits(_) => {
                "jobs"
            }
            Command::ContextSnapshot { .. } | Command::ContextExpire => "context",
            Command::ModeSet { .. } => "context", // mode.* events belong to the context machine
            Command::DocsUpdateHead { .. }
            | Command::DocsAppendTurn { .. }
            | Command::DocsGet { .. } => "docs",
            Command::DeliveryPrepare { .. }
            | Command::DeliveryApply { .. }
            | Command::DeliveryCancel { .. }
            | Command::DeliveryCopyFallback { .. } => "delivery",
        }
    }

    /// Serializes the payload to the wire `payload` object.
    pub fn payload_value(&self) -> Value {
        use serde_json::json;
        match self {
            Command::CaptureStart { policy } => json!({ "policy": policy }),
            Command::CaptureStop { drain } => match drain {
                Some(drain) => json!({ "drain": drain }),
                None => json!({}),
            },
            Command::CaptureAbort => json!({}),
            Command::JobsSubmit {
                capture_ref,
                route,
                budget,
            } => json!({
                "captureRef": capture_ref,
                "route": route,
                "budget": budget,
            }),
            Command::JobsCancel { job_id } => json!({ "jobId": job_id }),
            Command::JobsSetLimits(limits) => {
                serde_json::to_value(limits).unwrap_or_else(|_| json!({}))
            }
            Command::ContextSnapshot { source } => json!({ "source": source }),
            Command::ModeSet { mode, source: _ } => json!({ "mode": mode, "source": "manual" }),
            Command::ContextExpire => json!({}),
            Command::DocsUpdateHead {
                doc_id,
                expected_base,
                new_revision,
            } => json!({
                "docId": doc_id,
                "expectedBase": expected_base,
                "newRevision": new_revision,
            }),
            Command::DocsAppendTurn { doc_id, take_ref } => {
                json!({ "docId": doc_id, "takeRef": take_ref })
            }
            Command::DocsGet { doc_id, page } => json!({ "docId": doc_id, "page": page }),
            Command::DeliveryPrepare {
                revision_id,
                target_ref,
            } => json!({ "revisionId": revision_id, "targetRef": target_ref }),
            Command::DeliveryApply { delivery_id } => json!({ "deliveryId": delivery_id }),
            Command::DeliveryCancel { delivery_id } => match delivery_id {
                Some(id) => json!({ "deliveryId": id }),
                None => json!({}),
            },
            Command::DeliveryCopyFallback { delivery_id } => {
                json!({ "deliveryId": delivery_id })
            }
        }
    }

    /// Parses a `type` + `payload` pair into the typed command. Unknown
    /// types and malformed payloads are errors — the schemas'
    /// `additionalProperties: false` surfaces through
    /// `deny_unknown_fields` on every payload struct.
    pub fn from_parts(type_: &str, payload: &Value) -> Result<Command, String> {
        fn parse<T: serde::de::DeserializeOwned>(
            payload: &Value,
            type_: &str,
        ) -> Result<T, String> {
            serde_json::from_value(payload.clone())
                .map_err(|err| format!("{type_} payload is invalid: {err}"))
        }
        match type_ {
            "capture.start" => Ok(Command::CaptureStart {
                policy: parse::<CaptureStartP>(payload, type_)?.policy,
            }),
            "capture.stop" => Ok(Command::CaptureStop {
                drain: parse::<CaptureStopP>(payload, type_)?.drain,
            }),
            "capture.abort" => Ok(Command::CaptureAbort),
            "jobs.submit" => {
                let parsed: JobsSubmitP = parse(payload, type_)?;
                Ok(Command::JobsSubmit {
                    capture_ref: parsed.capture_ref,
                    route: parsed.route,
                    budget: parsed.budget,
                })
            }
            "jobs.cancel" => Ok(Command::JobsCancel {
                job_id: parse::<JobsCancelP>(payload, type_)?
                    .job_id
                    .unwrap_or_default(),
            }),
            "jobs.setLimits" => {
                let limits: JobLimits = parse(payload, type_)?;
                Ok(Command::JobsSetLimits(limits))
            }
            "context.snapshot" => Ok(Command::ContextSnapshot {
                source: parse::<ContextSnapshotP>(payload, type_)?.source,
            }),
            "mode.set" => {
                let parsed: ModeSetP = parse(payload, type_)?;
                if parsed.source != "manual" {
                    return Err("mode.set payload field 'source' must be \"manual\"".to_string());
                }
                Ok(Command::ModeSet {
                    mode: parsed.mode,
                    source: Manual,
                })
            }
            "context.expire" => Ok(Command::ContextExpire),
            "docs.updateHead" => {
                let parsed: DocsUpdateHeadP = parse(payload, type_)?;
                Ok(Command::DocsUpdateHead {
                    doc_id: parsed.doc_id,
                    expected_base: parsed.expected_base,
                    new_revision: parsed.new_revision,
                })
            }
            "docs.appendTurn" => {
                let parsed: DocsAppendTurnP = parse(payload, type_)?;
                Ok(Command::DocsAppendTurn {
                    doc_id: parsed.doc_id,
                    take_ref: parsed.take_ref,
                })
            }
            "docs.get" => {
                let parsed: DocsGetP = parse(payload, type_)?;
                Ok(Command::DocsGet {
                    doc_id: parsed.doc_id,
                    page: parsed.page,
                })
            }
            "delivery.prepare" => {
                let parsed: DeliveryPrepareP = parse(payload, type_)?;
                Ok(Command::DeliveryPrepare {
                    revision_id: parsed.revision_id,
                    target_ref: parsed.target_ref,
                })
            }
            "delivery.apply" => Ok(Command::DeliveryApply {
                delivery_id: parse::<DeliveryApplyP>(payload, type_)?.delivery_id,
            }),
            "delivery.cancel" => Ok(Command::DeliveryCancel {
                delivery_id: parse::<DeliveryCancelP>(payload, type_)?.delivery_id,
            }),
            "delivery.copyFallback" => Ok(Command::DeliveryCopyFallback {
                delivery_id: parse::<DeliveryApplyP>(payload, type_)?.delivery_id,
            }),
            other => Err(format!("{other:?} is not a v1 command type")),
        }
    }
}

/// The target data `context.targetSnapshot` carries (§2.3).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct TargetSnapshotData {
    pub descriptor: String,
    pub digest: String,
    pub capabilities: Vec<String>,
    pub selection_range: Span,
    pub offset_encoding: String,
    pub expiry: String,
}

/// The decision data `mode.decision` carries (§2.3).
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionData {
    pub mode_id: String,
    pub source: DecisionSource,
    pub matched_prefix_span: Option<Span>,
    pub explanation: String,
    pub payload_view: String,
}

/// The completed-recognition data `jobs.completed` carries (§2.2).
#[derive(Debug, Clone, PartialEq)]
pub struct CompletionData {
    pub attempt_id: String,
    pub text: String,
    pub backend: String,
    pub timing: f64,
    pub completion_evidence: String,
}

/// All 22 v1 events (21 machine events + `runtime.nack`), variant per
/// events.schema.json branch.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// `capture.started{device, actualRate, channels}`.
    CaptureStarted {
        device: String,
        actual_rate: u32,
        channels: u32,
    },
    /// `capture.progress{ackSamples, clipRatio, level}` (throttled).
    CaptureProgress {
        ack_samples: u64,
        clip_ratio: f64,
        level: f64,
    },
    /// `capture.gap{startSample, endSample}` — flagged, never repaired.
    CaptureGap { start_sample: u64, end_sample: u64 },
    /// `capture.error{code, fatal}` — `fatal: true` enters `Interrupted`.
    CaptureError { code: String, fatal: bool },
    /// `capture.stopped{finalSampleIndex, acknowledgedSamples, gaps,
    /// journalId, sampleDurationMs, wallClockMs}`.
    CaptureStopped {
        final_sample_index: u64,
        acknowledged_samples: u64,
        gaps: Vec<SampleGap>,
        journal_id: String,
        sample_duration_ms: f64,
        wall_clock_ms: f64,
    },
    /// `jobs.queued{}` — job identity travels on `corr`.
    JobsQueued,
    /// `jobs.rejected{reason}`.
    JobsRejected { reason: RejectReason },
    /// `jobs.progress{partial, stabilityHint}` — stability independent of
    /// recording completeness.
    JobsProgress {
        partial: String,
        stability_hint: String,
    },
    /// `jobs.completed{attemptId, text, backend, timing,
    /// completionEvidence}`.
    JobsCompleted(CompletionData),
    /// `jobs.failed{reason, retryable}`.
    JobsFailed { reason: String, retryable: bool },
    /// `context.targetSnapshot{…}`.
    ContextTargetSnapshot(TargetSnapshotData),
    /// `mode.decision{…}`.
    ModeDecision(DecisionData),
    /// `mode.routeFrozen{route, decidedAt}` — legal only from
    /// `ModeDecided`; freezes the audio route before any frame leaves.
    ModeRouteFrozen { route: String, decided_at: String },
    /// `docs.headUpdated{docId, headRevision}`.
    DocsHeadUpdated { doc_id: String, head_revision: u64 },
    /// `docs.headConflict{expected, actual, candidatePreserved}`.
    DocsHeadConflict {
        expected: u64,
        actual: u64,
        candidate_preserved: bool,
    },
    /// `docs.turnAppended{turnSeq}`.
    DocsTurnAppended { turn_seq: u32 },
    /// `delivery.prepared{deliveryId, compareToken}`.
    DeliveryPrepared {
        delivery_id: String,
        compare_token: String,
    },
    /// `delivery.submittedUnconfirmed{}`.
    DeliverySubmittedUnconfirmed,
    /// `delivery.confirmed{evidenceLevel}` — states its evidence level.
    DeliveryConfirmed { evidence_level: String },
    /// `delivery.failed{reason, fallbackSuggested}`.
    DeliveryFailed {
        reason: String,
        fallback_suggested: bool,
    },
    /// `delivery.conflict{expectedTarget, actualTarget}`.
    DeliveryConflict {
        expected_target: String,
        actual_target: String,
    },
    /// `runtime.nack{reason: unsupported_version}` — corr = the rejected
    /// message's id.
    RuntimeNack { reason: NackReason },
}

/// `runtime.nack.reason` (v1 knows `unsupported_version` only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NackReason;

impl NackReason {
    pub fn as_str(self) -> &'static str {
        "unsupported_version"
    }
}

impl Event {
    /// The wire `type` of this event.
    pub fn type_name(&self) -> &'static str {
        match self {
            Event::CaptureStarted { .. } => "capture.started",
            Event::CaptureProgress { .. } => "capture.progress",
            Event::CaptureGap { .. } => "capture.gap",
            Event::CaptureError { .. } => "capture.error",
            Event::CaptureStopped { .. } => "capture.stopped",
            Event::JobsQueued => "jobs.queued",
            Event::JobsRejected { .. } => "jobs.rejected",
            Event::JobsProgress { .. } => "jobs.progress",
            Event::JobsCompleted(_) => "jobs.completed",
            Event::JobsFailed { .. } => "jobs.failed",
            Event::ContextTargetSnapshot(_) => "context.targetSnapshot",
            Event::ModeDecision(_) => "mode.decision",
            Event::ModeRouteFrozen { .. } => "mode.routeFrozen",
            Event::DocsHeadUpdated { .. } => "docs.headUpdated",
            Event::DocsHeadConflict { .. } => "docs.headConflict",
            Event::DocsTurnAppended { .. } => "docs.turnAppended",
            Event::DeliveryPrepared { .. } => "delivery.prepared",
            Event::DeliverySubmittedUnconfirmed => "delivery.submittedUnconfirmed",
            Event::DeliveryConfirmed { .. } => "delivery.confirmed",
            Event::DeliveryFailed { .. } => "delivery.failed",
            Event::DeliveryConflict { .. } => "delivery.conflict",
            Event::RuntimeNack { .. } => "runtime.nack",
        }
    }

    /// The machine this event belongs to; `runtime` for the runtime-level
    /// NACK.
    pub fn machine(&self) -> &'static str {
        match self {
            Event::RuntimeNack { .. } => "runtime",
            other => other.type_name().split('.').next().unwrap_or("runtime"),
        }
    }

    /// Serializes the payload to the wire `payload` object.
    pub fn payload_value(&self) -> Value {
        use serde_json::json;
        match self {
            Event::CaptureStarted {
                device,
                actual_rate,
                channels,
            } => json!({
                "device": device,
                "actualRate": actual_rate,
                "channels": channels,
            }),
            Event::CaptureProgress {
                ack_samples,
                clip_ratio,
                level,
            } => json!({
                "ackSamples": ack_samples,
                "clipRatio": clip_ratio,
                "level": level,
            }),
            Event::CaptureGap {
                start_sample,
                end_sample,
            } => json!({ "startSample": start_sample, "endSample": end_sample }),
            Event::CaptureError { code, fatal } => {
                json!({ "code": code, "fatal": fatal })
            }
            Event::CaptureStopped {
                final_sample_index,
                acknowledged_samples,
                gaps,
                journal_id,
                sample_duration_ms,
                wall_clock_ms,
            } => json!({
                "finalSampleIndex": final_sample_index,
                "acknowledgedSamples": acknowledged_samples,
                "gaps": gaps,
                "journalId": journal_id,
                "sampleDurationMs": sample_duration_ms,
                "wallClockMs": wall_clock_ms,
            }),
            Event::JobsQueued => json!({}),
            Event::JobsRejected { reason } => json!({ "reason": reason.as_str() }),
            Event::JobsProgress {
                partial,
                stability_hint,
            } => json!({ "partial": partial, "stabilityHint": stability_hint }),
            Event::JobsCompleted(data) => json!({
                "attemptId": data.attempt_id,
                "text": data.text,
                "backend": data.backend,
                "timing": data.timing,
                "completionEvidence": data.completion_evidence,
            }),
            Event::JobsFailed { reason, retryable } => {
                json!({ "reason": reason, "retryable": retryable })
            }
            Event::ContextTargetSnapshot(data) => {
                serde_json::to_value(data).unwrap_or_else(|_| json!({}))
            }
            Event::ModeDecision(data) => json!({
                "modeId": data.mode_id,
                "source": data.source.as_str(),
                "matchedPrefixSpan": data.matched_prefix_span,
                "explanation": data.explanation,
                "payloadView": data.payload_view,
            }),
            Event::ModeRouteFrozen { route, decided_at } => {
                json!({ "route": route, "decidedAt": decided_at })
            }
            Event::DocsHeadUpdated {
                doc_id,
                head_revision,
            } => json!({ "docId": doc_id, "headRevision": head_revision }),
            Event::DocsHeadConflict {
                expected,
                actual,
                candidate_preserved,
            } => json!({
                "expected": expected,
                "actual": actual,
                "candidatePreserved": candidate_preserved,
            }),
            Event::DocsTurnAppended { turn_seq } => json!({ "turnSeq": turn_seq }),
            Event::DeliveryPrepared {
                delivery_id,
                compare_token,
            } => json!({ "deliveryId": delivery_id, "compareToken": compare_token }),
            Event::DeliverySubmittedUnconfirmed => json!({}),
            Event::DeliveryConfirmed { evidence_level } => {
                json!({ "evidenceLevel": evidence_level })
            }
            Event::DeliveryFailed {
                reason,
                fallback_suggested,
            } => json!({ "reason": reason, "fallbackSuggested": fallback_suggested }),
            Event::DeliveryConflict {
                expected_target,
                actual_target,
            } => json!({
                "expectedTarget": expected_target,
                "actualTarget": actual_target,
            }),
            Event::RuntimeNack { .. } => json!({ "reason": "unsupported_version" }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_name_pattern_matches_contract_examples() {
        for good in [
            "capture.start",
            "jobs.setLimits",
            "mode.routeFrozen",
            "context.targetSnapshot",
            "runtime.nack",
        ] {
            assert!(is_type_name(good), "{good} should match");
        }
        for bad in [
            "Capture.start",
            "capture.Start",
            "capture",
            "capture..start",
            "9capture.start",
            "capture.start.two",
        ] {
            assert!(!is_type_name(bad), "{bad} should not match");
        }
    }

    #[test]
    fn timestamp_pattern_accepts_fixture_forms() {
        for good in [
            "2026-09-20T10:00:00Z",
            "2026-09-20T10:00:00.123Z",
            "2026-09-20T10:00:00+02:00",
        ] {
            assert!(is_rfc3339(good), "{good} should match");
        }
        for bad in [
            "2026-09-20 10:00:00Z",
            "2026-09-20T10:00:00",
            "2026-9-20T10:00:00Z",
            "2026-09-20T10:00:00+0200",
            "2026-09-20T10:00:00.",
        ] {
            assert!(!is_rfc3339(bad), "{bad} should not match");
        }
    }

    #[test]
    fn timestamp_pattern_rejects_multi_byte_utf8() {
        // Issue #202: every multi-byte char here used to hit a
        // non-char-boundary `&value[a..b]` slice and panic the router
        // thread. All of them must be plain rejections.
        for bad in [
            "日日日日日日日",          // the issue's exact ts (21 bytes)
            "日",                      // shorter than 20 bytes
            "日本語",                  // still shorter, 9 bytes
            "2日26-09-20T10:00:00Z",   // multi-byte inside the year
            "2026-09-20T10:00:0日Z",   // straddles the seconds/zone boundary
            "2026-09-20T10:00:00.日Z", // multi-byte as the first fraction digit
            "2026-09-20T10:00:00+♥é",  // 6-byte zone suffix, index 3 mid-char
            "2026-09-20T10:00:00Z日",  // trailing garbage after the zone
        ] {
            assert!(!is_rfc3339(bad), "{bad:?} must be rejected, not panic");
        }
    }

    #[test]
    fn timestamp_pattern_rejects_a_multi_byte_probe_at_every_position() {
        // Property-style sweep: splice one multi-byte char (2, 3, and
        // 4-byte encodings) into every byte position of an otherwise-valid
        // timestamp. Every mutation must be rejected — whichever check sees
        // the probe — and none may panic on a non-char-boundary slice.
        let valid = "2026-09-20T10:00:00Z";
        for code_point in [0x00A9u32, 0x05D0, 0x2764, 0x4E00, 0x1F600] {
            let probe = char::from_u32(code_point).expect("valid code point");
            for position in 0..19 {
                let mut mutated = String::with_capacity(valid.len() + 4);
                mutated.push_str(&valid[..position]);
                mutated.push(probe);
                mutated.push_str(&valid[position..]);
                assert!(
                    !is_rfc3339(&mutated),
                    "U+{code_point:04X} at byte {position}: {mutated:?} must be rejected"
                );
            }
        }
    }

    #[test]
    fn envelope_errors_port_flags_the_contract_cases() {
        let base = serde_json::json!({
            "v": 1, "id": "cmd_1", "ts": "2026-09-20T10:00:00Z",
            "type": "capture.stop", "payload": {}
        });
        assert!(envelope_errors(&base).is_empty());

        let mut missing = base.clone();
        missing.as_object_mut().unwrap().remove("ts");
        assert!(envelope_errors(&missing).iter().any(|e| e.contains("ts")));

        let mut extra = base.clone();
        extra
            .as_object_mut()
            .unwrap()
            .insert("trace_id".into(), 1.into());
        assert!(envelope_errors(&extra)
            .iter()
            .any(|e| e.contains("trace_id")));

        let mut bad_seq = base.clone();
        bad_seq
            .as_object_mut()
            .unwrap()
            .insert("seq".into(), (-1).into());
        assert!(envelope_errors(&bad_seq).iter().any(|e| e.contains("seq")));

        assert!(envelope_errors(&serde_json::json!([]))
            .iter()
            .any(|e| e.contains("expected object")));
    }

    #[test]
    fn nack_for_answers_unsupported_version_with_corr() {
        let bad = serde_json::json!({
            "v": 2, "id": "cmd_900", "ts": "2026-09-20T10:09:00Z",
            "type": "capture.stop", "payload": { "drain": true }
        });
        let nack = nack_for(&bad).expect("v2 must be nacked");
        assert_eq!(nack["type"], "runtime.nack");
        assert_eq!(nack["payload"]["reason"], "unsupported_version");
        assert_eq!(nack["corr"], "cmd_900");
        assert!(
            envelope_errors(&nack).is_empty(),
            "nack must be schema-valid"
        );

        let ok = serde_json::json!({
            "v": 1, "id": "x", "ts": "2026-09-20T10:00:00Z",
            "type": "capture.stop", "payload": {}
        });
        assert!(nack_for(&ok).is_none());
    }

    #[test]
    fn typed_command_round_trips_the_wire() {
        let revision = Revision {
            rev_id: "rev-42".into(),
            base_revision: 7,
            source_attempt_ids: vec!["att-2".into()],
            instruction_template_id: "tpl-none".into(),
            text: "Hello, world.".into(),
            status: "candidate".into(),
            provenance: "recognition".into(),
        };
        let command = Command::DocsUpdateHead {
            doc_id: "notes".into(),
            expected_base: 7,
            new_revision: revision,
        };
        let payload = command.payload_value();
        let parsed = Command::from_parts(command.type_name(), &payload).expect("round trip");
        assert_eq!(parsed, command);
        assert_eq!(parsed.type_name(), "docs.updateHead");
        assert_eq!(parsed.machine(), "docs");
    }

    #[test]
    fn unknown_payload_field_is_rejected_not_ignored() {
        let mut payload = serde_json::json!({ "policy": "push-to-talk" });
        payload
            .as_object_mut()
            .unwrap()
            .insert("extra".into(), "no".into());
        let error = Command::from_parts("capture.start", &payload).unwrap_err();
        assert!(
            error.contains("unknown field") || error.contains("extra"),
            "{error}"
        );
    }
}
