//! Durable transcription intent and the per-take claim (#220).
//!
//! A take that is to be transcribed says so in the same transaction that
//! stores it ([`super::TakeMeta::transcribe`]): a `transcription_intents`
//! row beside the `captures` row. Whoever transcribes it first claims it
//! ([`StoreV2::claim_transcription`]): one write transaction checks the
//! intent is unclaimed, starts the attempt and records the attempt as the
//! claim, so two claimants — two windows, or a host and its successor —
//! can never both run it. Settling the claimed attempt
//! ([`StoreV2::finish_attempt`]) ends the intent in the same transaction,
//! completed or failed: a failure is recorded with the take, and a retry
//! is the user's explicit choice. A request made while the take is
//! claimed ([`StoreV2::request_transcription`]) is not folded into the
//! running attempt: settling that attempt leaves the intent unclaimed and
//! due again, so the newer request is transcribed in its own right.
//!
//! A claimant that dies leaves its attempt `started` with its in-flight
//! marker released (#213). The next claim finds that attempt unowned,
//! fails it as interrupted and claims the take afresh, so a crash costs
//! the attempt, never the transcription; a claim a live process holds is
//! left to it.
//!
//! An **audio hold** ([`StoreV2::hold_audio`]) keeps a take's audio from
//! being compressed or retired by any process until it is released or
//! its holder is gone: what a retry asked for in one process needs before
//! another process starts its attempt.

use rusqlite::{params, OptionalExtension, Transaction, TransactionBehavior};

use super::{
    insert_attempt_row, AttemptRecord, RecognitionOutcome, StoreV2, StoreV2Error,
};
use crate::storage::now_iso;

/// The note a claim writes on the attempt of a claimant that went away.
pub const ABANDONED_CLAIM_NOTE: &str =
    "Starling stopped while this was being transcribed; it was transcribed again.";

/// What [`StoreV2::claim_transcription`] found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TranscriptionClaim {
    /// The caller holds the take: `attempt_id` is its started attempt.
    Claimed { attempt_id: String },
    /// Nothing waits to be transcribed: the take was never meant to be,
    /// was transcribed (or failed) already, or is gone.
    NotWanted,
    /// A live claimant is transcribing it.
    Held,
}

impl StoreV2 {
    /// Claims capture `capture_id` for transcription and starts its
    /// attempt (`backend` labels it, as [`Self::begin_recognition`]),
    /// all in one write transaction. An attempt a dead claimant left is
    /// failed with [`ABANDONED_CLAIM_NOTE`] on the way. A take whose audio
    /// the retention policy removed can never be transcribed: its intent
    /// ends and the error says why.
    pub fn claim_transcription(
        &mut self,
        capture_id: &str,
        backend: &str,
        options_json: Option<&str>,
    ) -> Result<TranscriptionClaim, StoreV2Error> {
        self.refuse_open_transaction("claim a transcription")?;
        let claimed = self.claim_in_transaction(capture_id, backend, options_json);
        if !self.conn.is_autocommit() {
            if let Err(err) = self.conn.execute_batch("ROLLBACK") {
                eprintln!("Rolling back a failed transcription claim also failed: {err}");
            }
        }
        match claimed {
            Ok(Claimed::Started { attempt_id, marker }) => {
                self.attempt_locks.insert(attempt_id.clone(), marker);
                Ok(TranscriptionClaim::Claimed { attempt_id })
            }
            Ok(Claimed::Not(claim)) => Ok(claim),
            Err((err, abandoned)) => {
                if let Some(abandoned) = abandoned {
                    self.release_attempt_lock(&abandoned);
                }
                Err(err)
            }
        }
    }

    fn claim_in_transaction(
        &mut self,
        capture_id: &str,
        backend: &str,
        options_json: Option<&str>,
    ) -> Result<Claimed, (StoreV2Error, Option<String>)> {
        let tx = Transaction::new_unchecked(&self.conn, TransactionBehavior::Immediate)
            .map_err(|err| (err.into(), None))?;
        let intent: Option<Option<String>> = tx
            .query_row(
                "SELECT attempt_id FROM transcription_intents WHERE capture_id = ?1",
                params![capture_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(|err| (err.into(), None))?;
        let Some(previous) = intent else {
            return Ok(Claimed::Not(TranscriptionClaim::NotWanted));
        };
        let mut abandoned = None;
        if let Some(previous) = previous {
            let status: Option<String> = tx
                .query_row(
                    "SELECT status FROM recognition_attempts WHERE id = ?1",
                    params![previous],
                    |row| row.get(0),
                )
                .optional()
                .map_err(|err| (err.into(), None))?;
            if status.as_deref() == Some("started") {
                if self.attempt_is_owned(&previous) {
                    return Ok(Claimed::Not(TranscriptionClaim::Held));
                }
                let extra = serde_json::json!({ "error": ABANDONED_CLAIM_NOTE }).to_string();
                tx.execute(
                    "UPDATE recognition_attempts SET status = 'failed', extra_json = ?1
                     WHERE id = ?2 AND status = 'started'",
                    params![extra, previous],
                )
                .map_err(|err| (err.into(), None))?;
                abandoned = Some(previous);
            }
        }
        if let Some(utc) = self
            .audio_retired_utc(capture_id)
            .map_err(|err| (err, abandoned.clone()))?
        {
            // Never transcribable again: the intent ends here.
            let ended = tx
                .execute(
                    "DELETE FROM transcription_intents WHERE capture_id = ?1",
                    params![capture_id],
                )
                .and_then(|_| tx.commit());
            if let Err(err) = ended {
                return Err((err.into(), abandoned));
            }
            if let Some(abandoned) = &abandoned {
                self.release_attempt_lock(abandoned);
            }
            return Err((
                StoreV2Error::Invalid(format!(
                    "the audio of capture {capture_id} was removed by the retention policy on \
                     {utc}; it cannot be transcribed"
                )),
                None,
            ));
        }
        let attempt_id = format!("a_{}", uuid::Uuid::new_v4().simple());
        let marker = self
            .open_attempt_marker(&attempt_id)
            .map_err(|err| (err, abandoned.clone()))?;
        let written = insert_attempt_row(
            &tx,
            &AttemptRecord {
                id: attempt_id.clone(),
                capture_id: capture_id.to_string(),
                backend: backend.to_string(),
                model_hash: None,
                language: None,
                options_json: options_json.map(str::to_string),
                text: String::new(),
                partial_or_final: "partial".to_string(),
                status: "started".to_string(),
                timing_json: None,
                extra_json: None,
                created_utc: None,
            },
        )
        .and_then(|()| {
            tx.execute(
                // The claim answers every request made so far, a re-request
                // a dead claimant left pending included.
                "UPDATE transcription_intents SET attempt_id = ?1, rerequested_utc = NULL
                 WHERE capture_id = ?2",
                params![attempt_id, capture_id],
            )
            .map(|_| ())
            .map_err(StoreV2Error::from)
        })
        .and_then(|()| tx.commit().map_err(StoreV2Error::from));
        match written {
            Ok(()) => {
                if let Some(abandoned) = &abandoned {
                    self.release_attempt_lock(abandoned);
                }
                Ok(Claimed::Started { attempt_id, marker })
            }
            Err(err) => {
                drop(marker);
                self.remove_attempt_marker(&attempt_id);
                // Rolled back: the abandoned attempt still reads started,
                // and its marker stays for the next claim to judge.
                Err((err, None))
            }
        }
    }

    /// Settles attempt `attempt_id` — exactly that one, whichever other
    /// attempts its capture has in flight — and ends the capture's
    /// transcription intent if this attempt held it, in one transaction —
    /// unless the take was asked for again while this attempt ran: then
    /// the intent is left unclaimed and due. [`StoreV2Error::NotFound`]
    /// when the attempt is gone (its capture was deleted) or no longer in
    /// flight.
    pub fn finish_attempt(
        &mut self,
        attempt_id: &str,
        outcome: RecognitionOutcome<'_>,
    ) -> Result<(), StoreV2Error> {
        self.refuse_open_transaction("settle a recognition attempt")?;
        let settled = self.finish_attempt_in_transaction(attempt_id, outcome);
        if !self.conn.is_autocommit() {
            if let Err(err) = self.conn.execute_batch("ROLLBACK") {
                eprintln!("Rolling back a failed attempt settle also failed: {err}");
            }
        }
        // Settled or not, nothing works on the attempt any more: its
        // marker must not make it read as in flight (the stale sweep, or
        // the next claim, takes it from here).
        self.release_attempt_lock(attempt_id);
        settled
    }

    fn finish_attempt_in_transaction(
        &mut self,
        attempt_id: &str,
        outcome: RecognitionOutcome<'_>,
    ) -> Result<(), StoreV2Error> {
        let tx = Transaction::new_unchecked(&self.conn, TransactionBehavior::Immediate)?;
        let changed = match outcome {
            RecognitionOutcome::Completed { text, extra_json } => tx.execute(
                "UPDATE recognition_attempts
                 SET text = ?1, partial_or_final = 'final', status = 'completed',
                     extra_json = COALESCE(?2, extra_json)
                 WHERE id = ?3 AND status = 'started'",
                params![text, extra_json, attempt_id],
            )?,
            RecognitionOutcome::Failed { message } => {
                let extra = serde_json::json!({ "error": message });
                tx.execute(
                    "UPDATE recognition_attempts SET status = 'failed', extra_json = ?1
                     WHERE id = ?2 AND status = 'started'",
                    params![extra.to_string(), attempt_id],
                )?
            }
        };
        if changed == 0 {
            return Err(StoreV2Error::NotFound(attempt_id.to_string()));
        }
        tx.execute(
            "DELETE FROM transcription_intents
             WHERE attempt_id = ?1 AND rerequested_utc IS NULL",
            params![attempt_id],
        )?;
        tx.execute(
            "UPDATE transcription_intents
             SET attempt_id = NULL, requested_utc = rerequested_utc, rerequested_utc = NULL
             WHERE attempt_id = ?1",
            params![attempt_id],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// [`Self::finish_attempt`] with the full transcript result, kept
    /// verbatim in the attempt row (as
    /// [`Self::finish_recognition_transcript`]).
    pub fn finish_attempt_transcript(
        &mut self,
        attempt_id: &str,
        transcript: &crate::storage::TranscriptionResult,
    ) -> Result<(), StoreV2Error> {
        let extra = serde_json::to_string(transcript)
            .map_err(|err| StoreV2Error::Invalid(err.to_string()))?;
        self.finish_attempt(
            attempt_id,
            RecognitionOutcome::Completed {
                text: &transcript.text,
                extra_json: Some(&extra),
            },
        )
    }

    /// The captures still waiting to be transcribed that nobody live is
    /// transcribing, oldest intent first, and asked for at least `aged`
    /// ago: what a host starting up (`Duration::ZERO`: all of them) or
    /// rechecking (leaving takes it is about to transcribe anyway) claims.
    pub fn transcriptions_due(
        &self,
        aged: std::time::Duration,
    ) -> Result<Vec<String>, StoreV2Error> {
        let cutoff = crate::storage::iso_utc(
            time::OffsetDateTime::now_utc()
                - time::Duration::try_from(aged).unwrap_or(time::Duration::ZERO),
        );
        let mut stmt = self.conn.prepare(
            "SELECT i.capture_id, i.attempt_id, a.status FROM transcription_intents i
             LEFT JOIN recognition_attempts a ON a.id = i.attempt_id
             WHERE i.requested_utc <= ?1
             ORDER BY i.requested_utc, i.capture_id",
        )?;
        let rows = stmt.query_map(params![cutoff], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Option<String>>(1)?,
                row.get::<_, Option<String>>(2)?,
            ))
        })?;
        let mut due = Vec::new();
        for row in rows {
            let (capture_id, attempt_id, status) = row?;
            let held = match (attempt_id, status.as_deref()) {
                (Some(attempt_id), Some("started")) => self.attempt_is_owned(&attempt_id),
                _ => false,
            };
            if !held {
                due.push(capture_id);
            }
        }
        Ok(due)
    }

    /// Whether capture `capture_id` waits to be transcribed (claimed or
    /// not).
    pub fn transcription_wanted(&self, capture_id: &str) -> Result<bool, StoreV2Error> {
        Ok(self
            .conn
            .query_row(
                "SELECT 1 FROM transcription_intents WHERE capture_id = ?1",
                params![capture_id],
                |_| Ok(()),
            )
            .optional()?
            .is_some())
    }

    /// Records the intent to transcribe an already stored capture (an
    /// import, or a take stored before it could be asked for): what a
    /// commit with [`super::TakeMeta::transcribe`] does in its own
    /// transaction. A capture whose intent nobody has claimed yet keeps
    /// it (asking twice is one transcription); one asked for again while
    /// an attempt holds it is transcribed again once that attempt settles.
    /// Any stored take may be asked for, an interrupted one included: its
    /// audio is what was kept, and transcribing it is the asker's call.
    pub fn request_transcription(&mut self, capture_id: &str) -> Result<(), StoreV2Error> {
        if self.get_capture(capture_id)?.is_none() {
            return Err(StoreV2Error::NotFound(capture_id.to_string()));
        }
        self.conn.execute(
            "INSERT INTO transcription_intents(capture_id, requested_utc) VALUES (?1, ?2)
             ON CONFLICT(capture_id) DO UPDATE
             SET rerequested_utc = COALESCE(rerequested_utc, excluded.requested_utc)
             WHERE attempt_id IS NOT NULL",
            params![capture_id, now_iso()],
        )?;
        Ok(())
    }
}

impl StoreV2 {
    /// Holds capture `capture_id`'s audio for this process until
    /// [`Self::release_audio_hold`] (or this process exits): no process's
    /// upkeep compresses or retires it meanwhile. Returns the hold's id.
    pub fn hold_audio(&mut self, capture_id: &str) -> Result<String, StoreV2Error> {
        if self.get_capture(capture_id)?.is_none() {
            return Err(StoreV2Error::NotFound(capture_id.to_string()));
        }
        let id = format!("h_{}", uuid::Uuid::new_v4().simple());
        self.conn.execute(
            "INSERT INTO audio_holds(id, capture_id, holder_pid, created_utc)
             VALUES (?1, ?2, ?3, ?4)",
            params![id, capture_id, std::process::id(), now_iso()],
        )?;
        Ok(id)
    }

    /// Ends hold `hold_id` (a release of an unknown hold does nothing).
    pub fn release_audio_hold(&mut self, hold_id: &str) -> Result<(), StoreV2Error> {
        self.conn
            .execute("DELETE FROM audio_holds WHERE id = ?1", params![hold_id])?;
        Ok(())
    }

    /// Whether a live process holds capture `capture_id`'s audio. A hold
    /// whose holder is gone counts for nothing.
    pub(super) fn audio_held(&self, capture_id: &str) -> Result<bool, StoreV2Error> {
        let mut stmt = self
            .conn
            .prepare("SELECT holder_pid FROM audio_holds WHERE capture_id = ?1")?;
        let pids = stmt.query_map(params![capture_id], |row| row.get::<_, i64>(0))?;
        for pid in pids {
            let pid = pid?;
            if u32::try_from(pid).is_ok_and(super::process_is_alive) {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

impl StoreV2 {
    /// Refuses `what` on a connection a caller holds a transaction on: the
    /// claim, the settle and a recognition start each run their own
    /// top-level transaction, whose
    /// commit (or failure-path ROLLBACK) would otherwise end the caller's.
    pub(super) fn refuse_open_transaction(&self, what: &str) -> Result<(), StoreV2Error> {
        if self.conn.is_autocommit() {
            Ok(())
        } else {
            Err(StoreV2Error::Invalid(format!(
                "cannot {what} inside an open transaction"
            )))
        }
    }
}

enum Claimed {
    Started {
        attempt_id: String,
        marker: std::fs::File,
    },
    Not(TranscriptionClaim),
}
