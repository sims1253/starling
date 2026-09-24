//! The staged draft of one take: the port of `tests/staging.py` (#293).
//!
//! One text with typed regions over it; every code point belongs to
//! exactly one region, and every offset is a Unicode code point (never a
//! byte or UTF-16 index). Raw recognition attempts are kept apart from
//! the draft and never rewritten, so [`Draft::raw_text`] rebuilds the
//! recognition byte for byte whatever happened to the visible text.
//!
//! A transform result never changes the draft by itself. It becomes a
//! proposal pinned to the revision its request read; only
//! [`Draft::accept`] (the user) or [`Draft::swap_check`] (direct delivery
//! into an unchanged target) moves the text, and only for a *current*
//! proposal unless the user forces a stale one. That is the rule that
//! keeps a late result from overwriting a newer edit.
//!
//! `tests/staging_conformance.rs` replays `fixtures/staging.json` from
//! the contract directory through this port; the Python oracle replays
//! the same file.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegionKind {
    Raw,
    Partial,
    User,
    Command,
    Processed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommandKind {
    ModePhrase,
    TrailingInstruction,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Region {
    kind: RegionKind,
    text: String,
    segment: Option<u32>,
    attempt_id: Option<String>,
    command: Option<CommandKind>,
    request_id: Option<String>,
}

impl Region {
    fn new(kind: RegionKind, text: impl Into<String>) -> Region {
        Region {
            kind,
            text: text.into(),
            segment: None,
            attempt_id: None,
            command: None,
            request_id: None,
        }
    }

    fn chars(&self) -> usize {
        self.text.chars().count()
    }
}

/// An immutable raw recognition attempt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Attempt {
    pub attempt_id: String,
    pub segment: u32,
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestStatus {
    Pending,
    Interrupted,
    Settled,
    Superseded,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestRecord {
    pub request_id: String,
    pub base_revision: u64,
    pub retry_of: Option<String>,
    /// The transform input: the draft text without command regions.
    pub input: String,
    /// The trailing instruction, when a command region carries one.
    pub instruction: Option<String>,
    pub status: RequestStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProposalState {
    Open,
    Superseded,
    Accepted,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Proposal {
    request_id: String,
    base_revision: u64,
    text: String,
    state: ProposalState,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeliveryRecord {
    pub delivery_id: String,
    pub revision: u64,
    pub target_digest: String,
}

/// What an operation did. The spellings are the oracle's outcome strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    Applied,
    Unchanged,
    IgnoredDeleted,
    IgnoredPinned,
    IgnoredFinal,
    Duplicate,
    AttemptConflict,
    RecordedPinned,
    Recorded,
    OutOfRange,
    RefusedPartial,
    UnknownRequest,
    Pending,
    Cancelled,
    NotPending,
    Discarded,
    Failed,
    Superseded,
    Current,
    Stale,
    NotOpen,
    StaleRejected,
    Rejected,
    Delivered,
    Keep,
    Swap,
    Deleted,
    Restarted,
}

impl Outcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Outcome::Applied => "applied",
            Outcome::Unchanged => "unchanged",
            Outcome::IgnoredDeleted => "ignored_deleted",
            Outcome::IgnoredPinned => "ignored_pinned",
            Outcome::IgnoredFinal => "ignored_final",
            Outcome::Duplicate => "duplicate",
            Outcome::AttemptConflict => "attempt_conflict",
            Outcome::RecordedPinned => "recorded_pinned",
            Outcome::Recorded => "recorded",
            Outcome::OutOfRange => "out_of_range",
            Outcome::RefusedPartial => "refused_partial",
            Outcome::UnknownRequest => "unknown_request",
            Outcome::Pending => "pending",
            Outcome::Cancelled => "cancelled",
            Outcome::NotPending => "not_pending",
            Outcome::Discarded => "discarded",
            Outcome::Failed => "failed",
            Outcome::Superseded => "superseded",
            Outcome::Current => "current",
            Outcome::Stale => "stale",
            Outcome::NotOpen => "not_open",
            Outcome::StaleRejected => "stale_rejected",
            Outcome::Rejected => "rejected",
            Outcome::Delivered => "delivered",
            Outcome::Keep => "keep",
            Outcome::Swap => "swap",
            Outcome::Deleted => "deleted",
            Outcome::Restarted => "restarted",
        }
    }
}

/// Whether a result's provider call succeeded (the fixture's `status`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResultKind {
    Completed,
    Failed,
    Cancelled,
}

/// One draft operation, as `fixtures/staging.json` spells it.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Op {
    Partial {
        segment: u32,
        text: String,
    },
    Final {
        segment: u32,
        attempt_id: String,
        text: String,
    },
    Insert {
        at: usize,
        text: String,
    },
    Delete {
        start: usize,
        end: usize,
    },
    MarkCommand {
        start: usize,
        end: usize,
        command: CommandKind,
    },
    Request {
        request_id: String,
        #[serde(default)]
        retry_of: Option<String>,
    },
    Cancel {
        request_id: String,
    },
    Result {
        request_id: String,
        status: ResultKind,
        #[serde(default)]
        text: Option<String>,
    },
    Accept {
        request_id: String,
        #[serde(default)]
        force: bool,
    },
    Reject {
        request_id: String,
    },
    RevertRaw,
    Deliver {
        delivery_id: String,
        target_digest: String,
    },
    SwapCheck {
        request_id: String,
        target_digest: String,
    },
    DeleteDraft,
    Crash,
}

/// The staged draft of one take.
#[derive(Debug, Clone)]
pub struct Draft {
    draft_id: String,
    capture_id: String,
    revision: u64,
    deleted: bool,
    regions: Vec<Region>,
    attempts: Vec<Attempt>,
    pinned: Vec<u32>,
    finalized: Vec<u32>,
    requests: Vec<RequestRecord>,
    proposals: Vec<Proposal>,
    deliveries: Vec<DeliveryRecord>,
}

/// Splits `text` at code point `at` (`at` <= its length).
fn split_chars(text: &str, at: usize) -> (&str, &str) {
    let byte = text
        .char_indices()
        .nth(at)
        .map_or(text.len(), |(index, _)| index);
    text.split_at(byte)
}

impl Draft {
    pub fn new(draft_id: impl Into<String>, capture_id: impl Into<String>) -> Draft {
        Draft {
            draft_id: draft_id.into(),
            capture_id: capture_id.into(),
            revision: 0,
            deleted: false,
            regions: Vec::new(),
            attempts: Vec::new(),
            pinned: Vec::new(),
            finalized: Vec::new(),
            requests: Vec::new(),
            proposals: Vec::new(),
            deliveries: Vec::new(),
        }
    }

    // ------------------------------------------------------------------
    // Views
    // ------------------------------------------------------------------

    pub fn draft_id(&self) -> &str {
        &self.draft_id
    }

    pub fn capture_id(&self) -> &str {
        &self.capture_id
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn is_deleted(&self) -> bool {
        self.deleted
    }

    pub fn text(&self) -> String {
        self.regions
            .iter()
            .map(|region| region.text.as_str())
            .collect()
    }

    /// The transform input: every region except commands.
    pub fn payload_text(&self) -> String {
        self.regions
            .iter()
            .filter(|region| region.kind != RegionKind::Command)
            .map(|region| region.text.as_str())
            .collect()
    }

    pub fn instruction(&self) -> Option<String> {
        let parts: Vec<&str> = self
            .regions
            .iter()
            .filter(|region| region.command == Some(CommandKind::TrailingInstruction))
            .map(|region| region.text.as_str())
            .collect();
        (!parts.is_empty()).then(|| parts.concat())
    }

    /// The latest final attempt per segment, in segment order: the
    /// recognition exactly as it arrived.
    pub fn raw_text(&self) -> String {
        let mut latest: Vec<(u32, &str)> = Vec::new();
        for attempt in &self.attempts {
            match latest
                .iter_mut()
                .find(|(segment, _)| *segment == attempt.segment)
            {
                Some(entry) => entry.1 = &attempt.text,
                None => latest.push((attempt.segment, &attempt.text)),
            }
        }
        latest.sort_by_key(|(segment, _)| *segment);
        latest.into_iter().map(|(_, text)| text).collect()
    }

    pub fn attempts(&self) -> &[Attempt] {
        &self.attempts
    }

    /// The ids of the final attempts the visible raw text came from.
    pub fn source_attempt_ids(&self) -> Vec<String> {
        self.attempts
            .iter()
            .map(|attempt| attempt.attempt_id.clone())
            .collect()
    }

    pub fn request(&self, request_id: &str) -> Option<&RequestRecord> {
        self.requests
            .iter()
            .find(|request| request.request_id == request_id)
    }

    fn request_mut(&mut self, request_id: &str) -> Option<&mut RequestRecord> {
        self.requests
            .iter_mut()
            .find(|request| request.request_id == request_id)
    }

    fn proposal_index(&self, request_id: &str) -> Option<usize> {
        self.proposals
            .iter()
            .position(|proposal| proposal.request_id == request_id)
    }

    /// A proposal's status as the snapshot reports it.
    pub fn proposal_status(&self, request_id: &str) -> Option<ProposalStatus> {
        self.proposal_index(request_id)
            .map(|index| self.status_of(&self.proposals[index]))
    }

    /// A proposal's text.
    pub fn proposal_text(&self, request_id: &str) -> Option<&str> {
        self.proposal_index(request_id)
            .map(|index| self.proposals[index].text.as_str())
    }

    fn status_of(&self, proposal: &Proposal) -> ProposalStatus {
        match proposal.state {
            ProposalState::Open if proposal.base_revision == self.revision => {
                ProposalStatus::Current
            }
            ProposalState::Open => ProposalStatus::Stale,
            ProposalState::Superseded => ProposalStatus::Superseded,
            ProposalState::Accepted => ProposalStatus::Accepted,
            ProposalState::Rejected => ProposalStatus::Rejected,
        }
    }

    fn has_partial(&self) -> bool {
        self.regions
            .iter()
            .any(|region| region.kind == RegionKind::Partial)
    }

    fn partial_index(&self, segment: u32) -> Option<usize> {
        self.regions.iter().position(|region| {
            region.kind == RegionKind::Partial && region.segment == Some(segment)
        })
    }

    fn len_chars(&self) -> usize {
        self.regions.iter().map(Region::chars).sum()
    }

    // ------------------------------------------------------------------
    // Region editing helpers
    // ------------------------------------------------------------------

    /// Installs `regions` (dropping empty ones, merging adjacent user
    /// regions) and bumps the revision when the text changed.
    fn set_regions(&mut self, regions: Vec<Region>) -> bool {
        let before = self.text();
        let mut cleaned: Vec<Region> = Vec::with_capacity(regions.len());
        for region in regions {
            if region.text.is_empty() {
                continue;
            }
            if region.kind == RegionKind::User {
                if let Some(last) = cleaned.last_mut() {
                    if last.kind == RegionKind::User {
                        last.text.push_str(&region.text);
                        continue;
                    }
                }
            }
            cleaned.push(region);
        }
        self.regions = cleaned;
        let changed = self.text() != before;
        if changed {
            self.revision += 1;
        }
        changed
    }

    fn pin(&mut self, index: usize) {
        let region = &mut self.regions[index];
        if let Some(segment) = region.segment {
            if !self.pinned.contains(&segment) {
                self.pinned.push(segment);
            }
        }
        let text = std::mem::take(&mut region.text);
        *region = Region::new(RegionKind::User, text);
    }

    /// Splits the region containing code point `at` so a boundary lies
    /// there; returns the index of the first region starting at `at`.
    fn split_at(&mut self, at: usize) -> usize {
        let mut pos = 0;
        for index in 0..self.regions.len() {
            let len = self.regions[index].chars();
            let end = pos + len;
            if at == pos {
                return index;
            }
            if pos < at && at < end {
                let region = self.regions[index].clone();
                let (left, right) = split_chars(&region.text, at - pos);
                let (left, right) = (left.to_string(), right.to_string());
                let mut left_region = region.clone();
                left_region.text = left;
                let mut right_region = region;
                right_region.text = right;
                self.regions
                    .splice(index..=index, [left_region, right_region]);
                return index + 1;
            }
            pos = end;
        }
        self.regions.len()
    }

    /// Pins every live partial the predicate selects by its span.
    fn pin_partials(&mut self, touches: impl Fn(usize, usize) -> bool) {
        let mut pos = 0;
        for index in 0..self.regions.len() {
            let end = pos + self.regions[index].chars();
            if self.regions[index].kind == RegionKind::Partial && touches(pos, end) {
                self.pin(index);
            }
            pos = end;
        }
    }

    // ------------------------------------------------------------------
    // Operations
    // ------------------------------------------------------------------

    /// Applies one operation. A deleted draft ignores everything except a
    /// late result (discarded) and a crash.
    pub fn apply(&mut self, op: &Op) -> Outcome {
        if self.deleted && !matches!(op, Op::Result { .. } | Op::Crash) {
            return Outcome::IgnoredDeleted;
        }
        match op {
            Op::Partial { segment, text } => self.partial(*segment, text),
            Op::Final {
                segment,
                attempt_id,
                text,
            } => self.final_attempt(*segment, attempt_id, text),
            Op::Insert { at, text } => self.insert(*at, text),
            Op::Delete { start, end } => self.delete(*start, *end),
            Op::MarkCommand {
                start,
                end,
                command,
            } => self.mark_command(*start, *end, *command),
            Op::Request {
                request_id,
                retry_of,
            } => self.request_transform(request_id, retry_of.as_deref()),
            Op::Cancel { request_id } => self.cancel(request_id),
            Op::Result {
                request_id,
                status,
                text,
            } => self.result(request_id, *status, text.as_deref()),
            Op::Accept { request_id, force } => self.accept(request_id, *force),
            Op::Reject { request_id } => self.reject(request_id),
            Op::RevertRaw => self.revert_raw(),
            Op::Deliver {
                delivery_id,
                target_digest,
            } => self.deliver(delivery_id, target_digest),
            Op::SwapCheck {
                request_id,
                target_digest,
            } => self.swap_check(request_id, target_digest),
            Op::DeleteDraft => self.delete_draft(),
            Op::Crash => self.crash(),
        }
    }

    pub fn partial(&mut self, segment: u32, text: &str) -> Outcome {
        if self.deleted {
            return Outcome::IgnoredDeleted;
        }
        if self.pinned.contains(&segment) {
            return Outcome::IgnoredPinned;
        }
        if self.finalized.contains(&segment) {
            return Outcome::IgnoredFinal;
        }
        let mut regions = self.regions.clone();
        match self.partial_index(segment) {
            Some(index) => regions[index].text = text.to_string(),
            None => {
                let mut region = Region::new(RegionKind::Partial, text);
                region.segment = Some(segment);
                regions.push(region);
            }
        }
        if self.set_regions(regions) {
            Outcome::Applied
        } else {
            Outcome::Unchanged
        }
    }

    pub fn final_attempt(&mut self, segment: u32, attempt_id: &str, text: &str) -> Outcome {
        if self.deleted {
            return Outcome::IgnoredDeleted;
        }
        if let Some(known) = self
            .attempts
            .iter()
            .find(|attempt| attempt.attempt_id == attempt_id)
        {
            return if known.segment == segment && known.text == text {
                Outcome::Duplicate
            } else {
                Outcome::AttemptConflict
            };
        }
        self.attempts.push(Attempt {
            attempt_id: attempt_id.to_string(),
            segment,
            text: text.to_string(),
        });
        let first_final = !self.finalized.contains(&segment);
        if first_final {
            self.finalized.push(segment);
        }
        if self.pinned.contains(&segment) {
            return Outcome::RecordedPinned;
        }
        if !first_final {
            // A re-recognition of a segment the draft already shows: kept
            // as an attempt, never swapped in behind the user's back.
            return Outcome::Recorded;
        }
        let mut regions = self.regions.clone();
        let mut region = Region::new(RegionKind::Raw, text);
        region.segment = Some(segment);
        region.attempt_id = Some(attempt_id.to_string());
        match self.partial_index(segment) {
            Some(index) => regions[index] = region,
            None => regions.push(region),
        }
        self.set_regions(regions);
        Outcome::Applied
    }

    /// User typing at code point `at`. Typing strictly inside a live
    /// partial pins it: the user owns that segment's text from now on.
    pub fn insert(&mut self, at: usize, text: &str) -> Outcome {
        if self.deleted {
            return Outcome::IgnoredDeleted;
        }
        if at > self.len_chars() {
            return Outcome::OutOfRange;
        }
        if text.is_empty() {
            return Outcome::Unchanged;
        }
        self.pin_partials(|start, end| start < at && at < end);
        let index = self.split_at(at);
        let mut regions = self.regions.clone();
        regions.insert(index, Region::new(RegionKind::User, text));
        self.set_regions(regions);
        Outcome::Applied
    }

    /// User deletion of code points `[start, end)`. Touching a live
    /// partial pins it like typing does.
    pub fn delete(&mut self, start: usize, end: usize) -> Outcome {
        if self.deleted {
            return Outcome::IgnoredDeleted;
        }
        if start > end || end > self.len_chars() {
            return Outcome::OutOfRange;
        }
        if start == end {
            return Outcome::Unchanged;
        }
        self.pin_partials(|region_start, region_end| start < region_end && end > region_start);
        let mut regions = Vec::with_capacity(self.regions.len());
        let mut pos = 0;
        for region in &self.regions {
            let len = region.chars();
            let region_end = pos + len;
            let mut region = region.clone();
            if start < region_end && end > pos {
                let (left, _) = split_chars(&region.text, start.saturating_sub(pos));
                let (_, right) = split_chars(&region.text, (end - pos).min(len));
                region.text = format!("{left}{right}");
            }
            regions.push(region);
            pos = region_end;
        }
        self.set_regions(regions);
        Outcome::Applied
    }

    /// Marks `[start, end)` as a command span (#298's leading phrase or
    /// trailing instruction).
    pub fn mark_command(&mut self, start: usize, end: usize, command: CommandKind) -> Outcome {
        if self.deleted {
            return Outcome::IgnoredDeleted;
        }
        if start >= end || end > self.len_chars() {
            return Outcome::OutOfRange;
        }
        let first = self.split_at(start);
        let last = self.split_at(end);
        if self.regions[first..last]
            .iter()
            .any(|region| region.kind == RegionKind::Partial)
        {
            return Outcome::RefusedPartial;
        }
        let text: String = self.regions[first..last]
            .iter()
            .map(|region| region.text.as_str())
            .collect();
        let mut region = Region::new(RegionKind::Command, text);
        region.command = Some(command);
        self.regions.splice(first..last, [region]);
        Outcome::Applied
    }

    /// Records a transform request against the current revision.
    pub fn request_transform(&mut self, request_id: &str, retry_of: Option<&str>) -> Outcome {
        if self.deleted {
            return Outcome::IgnoredDeleted;
        }
        if self.request(request_id).is_some() {
            return Outcome::Duplicate;
        }
        if self.has_partial() {
            // Processing reads finals only; a live tail would make the
            // request stale the moment the next partial lands.
            return Outcome::RefusedPartial;
        }
        if let Some(retry_of) = retry_of {
            let Some(earlier) = self.request_mut(retry_of) else {
                return Outcome::UnknownRequest;
            };
            if matches!(
                earlier.status,
                RequestStatus::Pending | RequestStatus::Interrupted
            ) {
                earlier.status = RequestStatus::Superseded;
            }
        }
        self.requests.push(RequestRecord {
            request_id: request_id.to_string(),
            base_revision: self.revision,
            retry_of: retry_of.map(str::to_string),
            input: self.payload_text(),
            instruction: self.instruction(),
            status: RequestStatus::Pending,
        });
        Outcome::Pending
    }

    pub fn cancel(&mut self, request_id: &str) -> Outcome {
        if self.deleted {
            return Outcome::IgnoredDeleted;
        }
        let Some(request) = self.request_mut(request_id) else {
            return Outcome::UnknownRequest;
        };
        if !matches!(
            request.status,
            RequestStatus::Pending | RequestStatus::Interrupted
        ) {
            return Outcome::NotPending;
        }
        request.status = RequestStatus::Cancelled;
        Outcome::Cancelled
    }

    /// A provider's answer for `request_id`. It never changes the text.
    pub fn result(&mut self, request_id: &str, status: ResultKind, text: Option<&str>) -> Outcome {
        let has_proposal = self.proposal_index(request_id).is_some();
        let deleted = self.deleted;
        let revision = self.revision;
        let Some(request) = self.request_mut(request_id) else {
            return Outcome::UnknownRequest;
        };
        if deleted || request.status == RequestStatus::Cancelled {
            return Outcome::Discarded;
        }
        if request.status == RequestStatus::Settled || has_proposal {
            return Outcome::Duplicate;
        }
        let base_revision = request.base_revision;
        let superseded = request.status == RequestStatus::Superseded;
        if status != ResultKind::Completed || text.is_none() {
            // Raw text is untouched by a failure; the request is done.
            if !superseded {
                request.status = RequestStatus::Settled;
            }
            return Outcome::Failed;
        }
        if !superseded {
            request.status = RequestStatus::Settled;
        }
        self.proposals.push(Proposal {
            request_id: request_id.to_string(),
            base_revision,
            text: text.unwrap_or_default().to_string(),
            state: if superseded {
                ProposalState::Superseded
            } else {
                ProposalState::Open
            },
        });
        if superseded {
            Outcome::Superseded
        } else if base_revision == revision {
            Outcome::Current
        } else {
            Outcome::Stale
        }
    }

    fn take_proposal(&mut self, index: usize) {
        let proposal = &self.proposals[index];
        let mut region = Region::new(RegionKind::Processed, proposal.text.clone());
        region.request_id = Some(proposal.request_id.clone());
        self.set_regions(vec![region]);
        self.proposals[index].state = ProposalState::Accepted;
    }

    /// The user takes a proposal. A stale one only with `force`.
    pub fn accept(&mut self, request_id: &str, force: bool) -> Outcome {
        if self.deleted {
            return Outcome::IgnoredDeleted;
        }
        let Some(index) = self.proposal_index(request_id) else {
            return Outcome::UnknownRequest;
        };
        let proposal = &self.proposals[index];
        if proposal.state != ProposalState::Open {
            return Outcome::NotOpen;
        }
        if proposal.base_revision != self.revision && !force {
            return Outcome::StaleRejected;
        }
        self.take_proposal(index);
        Outcome::Applied
    }

    pub fn reject(&mut self, request_id: &str) -> Outcome {
        if self.deleted {
            return Outcome::IgnoredDeleted;
        }
        let Some(index) = self.proposal_index(request_id) else {
            return Outcome::UnknownRequest;
        };
        if self.proposals[index].state != ProposalState::Open {
            return Outcome::NotOpen;
        }
        self.proposals[index].state = ProposalState::Rejected;
        Outcome::Rejected
    }

    /// Raw recovery: the latest final attempt per segment, with live
    /// partials of unfinished segments kept at the tail.
    pub fn revert_raw(&mut self) -> Outcome {
        if self.deleted {
            return Outcome::IgnoredDeleted;
        }
        let mut latest: Vec<&Attempt> = Vec::new();
        for attempt in &self.attempts {
            match latest
                .iter_mut()
                .find(|known| known.segment == attempt.segment)
            {
                Some(known) => *known = attempt,
                None => latest.push(attempt),
            }
        }
        latest.sort_by_key(|attempt| attempt.segment);
        let mut regions: Vec<Region> = latest
            .into_iter()
            .map(|attempt| {
                let mut region = Region::new(RegionKind::Raw, attempt.text.clone());
                region.segment = Some(attempt.segment);
                region.attempt_id = Some(attempt.attempt_id.clone());
                region
            })
            .collect();
        regions.extend(
            self.regions
                .iter()
                .filter(|region| region.kind == RegionKind::Partial)
                .cloned(),
        );
        if self.set_regions(regions) {
            Outcome::Applied
        } else {
            Outcome::Unchanged
        }
    }

    /// Records delivery of the current revision into a target. A delivery
    /// id or a (revision, target) pair lands at most once.
    pub fn deliver(&mut self, delivery_id: &str, target_digest: &str) -> Outcome {
        if self.deleted {
            return Outcome::IgnoredDeleted;
        }
        if self.deliveries.iter().any(|delivery| {
            delivery.delivery_id == delivery_id
                || (delivery.revision == self.revision && delivery.target_digest == target_digest)
        }) {
            return Outcome::Duplicate;
        }
        if self.has_partial() {
            return Outcome::RefusedPartial;
        }
        self.deliveries.push(DeliveryRecord {
            delivery_id: delivery_id.to_string(),
            revision: self.revision,
            target_digest: target_digest.to_string(),
        });
        Outcome::Delivered
    }

    /// Direct delivery: the processed text may replace the raw text in
    /// the target only if the proposal is current and the target still
    /// holds exactly what was delivered at its base revision.
    pub fn swap_check(&mut self, request_id: &str, target_digest: &str) -> Outcome {
        if self.deleted {
            return Outcome::IgnoredDeleted;
        }
        let Some(index) = self.proposal_index(request_id) else {
            return Outcome::Keep;
        };
        let proposal = &self.proposals[index];
        if proposal.state != ProposalState::Open || proposal.base_revision != self.revision {
            return Outcome::Keep;
        }
        let delivered = self
            .deliveries
            .iter()
            .rev()
            .find(|delivery| delivery.revision == proposal.base_revision);
        if delivered.is_none_or(|delivery| delivery.target_digest != target_digest) {
            return Outcome::Keep;
        }
        self.take_proposal(index);
        Outcome::Swap
    }

    /// The draft is deleted; its open requests are cancelled so late
    /// results land nowhere.
    pub fn delete_draft(&mut self) -> Outcome {
        self.deleted = true;
        for request in &mut self.requests {
            if matches!(
                request.status,
                RequestStatus::Pending | RequestStatus::Interrupted
            ) {
                request.status = RequestStatus::Cancelled;
            }
        }
        Outcome::Deleted
    }

    /// Process death and restart: live partials are not durable and are
    /// dropped; requests in flight become interrupted (a result that
    /// still arrives for one is judged like any other, against its base).
    pub fn crash(&mut self) -> Outcome {
        for request in &mut self.requests {
            if request.status == RequestStatus::Pending {
                request.status = RequestStatus::Interrupted;
            }
        }
        if !self.deleted {
            let regions = self
                .regions
                .iter()
                .filter(|region| region.kind != RegionKind::Partial)
                .cloned()
                .collect();
            self.set_regions(regions);
        }
        Outcome::Restarted
    }

    // ------------------------------------------------------------------
    // Snapshot (draft.schema.json)
    // ------------------------------------------------------------------

    pub fn snapshot(&self) -> DraftSnapshot {
        let mut pos = 0;
        let regions = self
            .regions
            .iter()
            .map(|region| {
                let end = pos + region.chars();
                let snapshot = RegionSnapshot {
                    kind: region.kind,
                    span: [pos, end],
                    segment: region.segment,
                    attempt_id: region.attempt_id.clone(),
                    command: region.command,
                    request_id: region.request_id.clone(),
                };
                pos = end;
                snapshot
            })
            .collect();
        let mut pinned = self.pinned.clone();
        pinned.sort_unstable();
        DraftSnapshot {
            schema_version: 1,
            draft_id: self.draft_id.clone(),
            capture_id: self.capture_id.clone(),
            revision: self.revision,
            deleted: self.deleted,
            text: self.text(),
            span_encoding: "unicode_codepoints".to_string(),
            regions,
            attempts: self.attempts.clone(),
            pinned_segments: pinned,
            requests: self
                .requests
                .iter()
                .map(|request| RequestSnapshot {
                    request_id: request.request_id.clone(),
                    base_revision: request.base_revision,
                    retry_of: request.retry_of.clone(),
                    status: request.status,
                })
                .collect(),
            proposals: self
                .proposals
                .iter()
                .map(|proposal| ProposalSnapshot {
                    request_id: proposal.request_id.clone(),
                    base_revision: proposal.base_revision,
                    text: proposal.text.clone(),
                    status: self.status_of(proposal),
                })
                .collect(),
            deliveries: self.deliveries.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProposalStatus {
    Current,
    Stale,
    Superseded,
    Accepted,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegionSnapshot {
    pub kind: RegionKind,
    pub span: [usize; 2],
    pub segment: Option<u32>,
    pub attempt_id: Option<String>,
    pub command: Option<CommandKind>,
    pub request_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestSnapshot {
    pub request_id: String,
    pub base_revision: u64,
    pub retry_of: Option<String>,
    pub status: RequestStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProposalSnapshot {
    pub request_id: String,
    pub base_revision: u64,
    pub text: String,
    pub status: ProposalStatus,
}

/// `draft.schema.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DraftSnapshot {
    pub schema_version: u32,
    pub draft_id: String,
    pub capture_id: String,
    pub revision: u64,
    pub deleted: bool,
    pub text: String,
    pub span_encoding: String,
    pub regions: Vec<RegionSnapshot>,
    pub attempts: Vec<Attempt>,
    pub pinned_segments: Vec<u32>,
    pub requests: Vec<RequestSnapshot>,
    pub proposals: Vec<ProposalSnapshot>,
    pub deliveries: Vec<DeliveryRecord>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_count_code_points_not_bytes() {
        assert_eq!(split_chars("a👩‍💻b", 1), ("a", "👩‍💻b"));
        assert_eq!(split_chars("a👩‍💻b", 2), ("a👩", "\u{200d}💻b"));
        assert_eq!(split_chars("abc", 3), ("abc", ""));
    }

    #[test]
    fn a_late_result_never_moves_the_text() {
        let mut draft = Draft::new("d", "c");
        draft.final_attempt(0, "att-1", "raw words");
        draft.request_transform("r1", None);
        draft.insert(9, " typed later");
        assert_eq!(
            draft.result("r1", ResultKind::Completed, Some("Clean.")),
            Outcome::Stale
        );
        assert_eq!(draft.text(), "raw words typed later");
        assert_eq!(draft.accept("r1", false), Outcome::StaleRejected);
        assert_eq!(draft.text(), "raw words typed later");
        assert_eq!(draft.raw_text(), "raw words");
    }
}
