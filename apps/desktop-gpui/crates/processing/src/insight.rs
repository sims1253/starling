//! The `processing_recorded` insight event (#294,
//! `packages/contracts/insight-events/schema.json`): one processing job's
//! latency and outcome, recorded now so Insights (#308) and the latency
//! work (#226) can read it later. Identity, counts and timings only;
//! never text.

use serde::{Deserialize, Serialize};

use crate::contract::{
    FailureReason, Locality, ProviderKind, ResultStatus, TransformKind, TransformRequest,
    TransformResult,
};
use crate::staging::Outcome;

/// How a result arrived against its draft.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Arrival {
    Current,
    Stale,
    Superseded,
    Discarded,
    /// Not a proposal (a failed or cancelled job).
    NotApplicable,
}

impl Arrival {
    /// The arrival a draft reported for a result.
    pub fn from_outcome(outcome: Outcome) -> Arrival {
        match outcome {
            Outcome::Current => Arrival::Current,
            Outcome::Stale => Arrival::Stale,
            Outcome::Superseded => Arrival::Superseded,
            Outcome::Discarded => Arrival::Discarded,
            _ => Arrival::NotApplicable,
        }
    }
}

/// Insight ids are `[A-Za-z0-9_.-]{1,128}`; anything else becomes `_`.
pub fn event_id(value: &str) -> String {
    value
        .chars()
        .take(128)
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessingRecorded {
    pub schema_version: u32,
    pub event_id: String,
    pub capture_id: String,
    pub occurred_at: String,
    #[serde(rename = "type")]
    pub type_: String,
    pub job_id: String,
    pub mode_id: String,
    pub mode_version: u32,
    pub provider_kind: ProviderKind,
    pub locality: Locality,
    pub transform_kinds: Vec<TransformKind>,
    pub status: ResultStatus,
    pub failure_reason: Option<FailureReason>,
    pub arrival: Arrival,
    pub queued_ms: f64,
    pub processing_ms: f64,
    pub stop_to_result_ms: Option<f64>,
    pub input_chars: u64,
    pub output_chars: Option<u64>,
}

impl ProcessingRecorded {
    /// The event for one finished job. `occurred_at` is RFC 3339 UTC; the
    /// event id is derived from the job id, so a replay of the same job's
    /// record is idempotent.
    pub fn new(
        request: &TransformRequest,
        result: &TransformResult,
        arrival: Arrival,
        occurred_at: impl Into<String>,
    ) -> ProcessingRecorded {
        ProcessingRecorded {
            schema_version: 1,
            event_id: event_id(&format!("proc-{}", request.request_id)),
            capture_id: event_id(&request.capture_id),
            occurred_at: occurred_at.into(),
            type_: "processing_recorded".to_string(),
            job_id: event_id(&request.request_id),
            mode_id: request.mode_id.clone(),
            mode_version: request.mode_version,
            provider_kind: request.provider.kind,
            locality: request.provider.locality,
            transform_kinds: request.kinds.clone(),
            status: result.status,
            failure_reason: result.failure.as_ref().map(|failure| failure.reason),
            arrival: if result.status == ResultStatus::Completed {
                arrival
            } else {
                Arrival::NotApplicable
            },
            queued_ms: result.timing.queued_ms,
            processing_ms: result.timing.processing_ms,
            stop_to_result_ms: result.timing.stop_to_result_ms,
            input_chars: request.input.chars().count() as u64,
            output_chars: result.text.as_ref().map(|text| text.chars().count() as u64),
        }
    }
}
