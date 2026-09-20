//! The reviewed migration flow's decision state machine (E02 cutover
//! phase 1, e17 §4: additive, reversible, dry-run → verified import → the
//! user accepts). Pure — the GPUI handlers drive it and perform the I/O,
//! so every refusal rule ("apply needs a report on screen", "cutover
//! needs a fully verified import") is unit-testable without a window.

use starling_dictation::store_v2::MigrationReport;

/// Where the migration UI stands. One screen per phase; confirm gates sit
/// in front of every destructive or persisting step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MigrationPhase {
    /// No report yet. Dry-run is the only action.
    Idle,
    /// A dry-run report is on screen (read-only counts + skipped-damaged
    /// list). Applying is armed behind a confirm gate.
    DryRunReady,
    /// The user asked to apply; the gate is open, waiting for the explicit
    /// confirm (or cancel back to the report).
    ConfirmApply,
    /// An apply ran; its report (with per-record verification) is on
    /// screen. Cutover (only if fully verified) and rollback are offered.
    Applied,
    /// Cutover armed, awaiting explicit confirm.
    ConfirmCutover,
    /// Rollback armed, awaiting explicit confirm.
    ConfirmRollback,
}

/// The migration flow's state, owned by the app entity.
#[derive(Debug)]
pub(crate) struct MigrationUi {
    pub phase: MigrationPhase,
    /// The report currently on screen (dry-run or applied).
    pub report: Option<MigrationReport>,
    /// A background step is running; every gate refuses while set.
    pub busy: bool,
    /// The batch id of the last applied import — the rollback handle.
    pub applied_batch: Option<String>,
    /// The persisted storage choice says v2 (cutover landed); the switch
    /// itself happens on the next start.
    pub cutover_persisted: bool,
    /// A post-action note ("v2 on next start", "import discarded") shown
    /// with the phase's screen.
    pub notice: Option<String>,
}

/// Why a transition was refused — surfaces verbatim next to the button.
pub(crate) type Refused = &'static str;

impl MigrationUi {
    pub(crate) fn new() -> Self {
        Self {
            phase: MigrationPhase::Idle,
            report: None,
            busy: false,
            applied_batch: None,
            cutover_persisted: false,
            notice: None,
        }
    }

    /// Whether an applied report is fully verified: every planned record
    /// imported without error and re-verified by content hash + count.
    /// Cutover is offered only then — an import with failures stays
    /// rollback-only.
    pub(crate) fn fully_verified(report: &MigrationReport) -> bool {
        !report.dry_run
            && report
                .records
                .iter()
                .all(|record| record.error.is_none() && record.verified == Some(true))
    }

    /// A dry-run finished; its (read-only) report is the new screen.
    /// Re-running a preview is always safe (it writes nothing to v2 and
    /// never touches v1), so completing it is unconditional — the *start*
    /// of a step is what refuses while another runs.
    pub(crate) fn dry_run_finished(&mut self, report: MigrationReport) {
        self.busy = false;
        self.phase = MigrationPhase::DryRunReady;
        self.report = Some(report);
        self.notice = None;
    }

    /// Arm the apply on the dry-run report.
    pub(crate) fn arm_apply(&mut self) -> Result<(), Refused> {
        if self.busy {
            return Err("a migration step is already running");
        }
        if self.phase != MigrationPhase::DryRunReady {
            return Err("run a dry-run preview first — the report is what you are confirming");
        }
        self.phase = MigrationPhase::ConfirmApply;
        Ok(())
    }

    pub(crate) fn cancel_apply(&mut self) {
        if self.phase == MigrationPhase::ConfirmApply {
            self.phase = MigrationPhase::DryRunReady;
        }
    }

    /// The explicit confirm on the report: the caller runs the real import.
    pub(crate) fn apply_confirmed(&mut self) -> Result<(), Refused> {
        if self.busy {
            return Err("a migration step is already running");
        }
        if self.phase != MigrationPhase::ConfirmApply {
            // Covers the "apply without a report" and "apply without
            // confirming on the report" refusals.
            return Err("confirm the dry-run report before importing");
        }
        self.busy = true;
        Ok(())
    }

    /// The import ran; its verified report is the new screen. Per-record
    /// failures stay visible here — cutover refuses them, rollback does
    /// not.
    pub(crate) fn apply_finished(&mut self, report: MigrationReport) -> Result<(), Refused> {
        if !self.busy || self.phase != MigrationPhase::ConfirmApply {
            return Err("no import was confirmed");
        }
        self.applied_batch = Some(report.batch_id.clone());
        self.phase = MigrationPhase::Applied;
        self.report = Some(report);
        self.notice = None;
        self.busy = false;
        Ok(())
    }

    /// Whether cutover may be offered on the current screen.
    pub(crate) fn cutover_ready(&self) -> bool {
        !self.busy
            && self.phase == MigrationPhase::Applied
            && self
                .report
                .as_ref()
                .is_some_and(Self::fully_verified)
    }

    pub(crate) fn arm_cutover(&mut self) -> Result<(), Refused> {
        if self.busy {
            return Err("a migration step is already running");
        }
        if !self.cutover_ready() {
            return Err(
                "cutover needs a fully verified import on screen (every recording re-checked \
                 by content hash)",
            );
        }
        self.phase = MigrationPhase::ConfirmCutover;
        Ok(())
    }

    pub(crate) fn cancel_cutover(&mut self) {
        if self.phase == MigrationPhase::ConfirmCutover {
            self.phase = MigrationPhase::Applied;
        }
    }

    /// The explicit confirm: the caller persists the v2 choice.
    pub(crate) fn cutover_confirmed(&mut self) -> Result<(), Refused> {
        if self.busy {
            return Err("a migration step is already running");
        }
        if self.phase != MigrationPhase::ConfirmCutover {
            return Err("cutover needs a fully verified import, confirmed");
        }
        self.busy = true;
        Ok(())
    }

    /// The choice was persisted; the switch itself happens on restart.
    pub(crate) fn cutover_finished(&mut self) -> Result<(), Refused> {
        if !self.busy || self.phase != MigrationPhase::ConfirmCutover {
            return Err("no cutover was confirmed");
        }
        self.phase = MigrationPhase::Applied;
        self.cutover_persisted = true;
        self.notice = Some(
            "Storage v2 is saved as your storage choice and becomes the store of record on \
             the next start. The v1 originals stay on disk untouched."
                .to_string(),
        );
        self.busy = false;
        Ok(())
    }

    /// Whether rollback may be offered: only with a live import batch.
    pub(crate) fn rollback_ready(&self) -> bool {
        !self.busy
            && self.phase == MigrationPhase::Applied
            && self.applied_batch.is_some()
    }

    pub(crate) fn arm_rollback(&mut self) -> Result<(), Refused> {
        if self.busy {
            return Err("a migration step is already running");
        }
        if !self.rollback_ready() {
            return Err("nothing to roll back — no import has been applied");
        }
        self.phase = MigrationPhase::ConfirmRollback;
        Ok(())
    }

    pub(crate) fn cancel_rollback(&mut self) {
        if self.phase == MigrationPhase::ConfirmRollback {
            self.phase = MigrationPhase::Applied;
        }
    }

    /// The explicit confirm: the caller discards the imported rows (and,
    /// when the cutover had persisted the v2 choice, resets it) before
    /// reporting back.
    pub(crate) fn rollback_confirmed(&mut self) -> Result<(), Refused> {
        if self.busy {
            return Err("a migration step is already running");
        }
        if self.phase != MigrationPhase::ConfirmRollback {
            return Err("roll back needs an applied import, confirmed");
        }
        self.busy = true;
        Ok(())
    }

    /// The import was discarded. The caller has already quarantined the
    /// imported journals and reset the persisted choice; the flow returns
    /// to its start. The v1 originals were never touched by any step.
    pub(crate) fn rollback_finished(&mut self, discarded: usize) -> Result<(), Refused> {
        if !self.busy || self.phase != MigrationPhase::ConfirmRollback {
            return Err("no rollback was confirmed");
        }
        self.phase = MigrationPhase::Idle;
        self.report = None;
        self.applied_batch = None;
        self.cutover_persisted = false;
        self.notice = Some(format!(
            "Rollback complete: {discarded} imported recording{} discarded and their journals \
             quarantined. The v1 originals were never touched.",
            if discarded == 1 { " was" } else { "s were" }
        ));
        self.busy = false;
        Ok(())
    }

    /// A background step failed: end the busy state and back out of any
    /// confirm gate. The surfaced error travels separately (the app's
    /// error banner), so the flow stays on the screen it came from.
    pub(crate) fn op_failed(&mut self) {
        self.busy = false;
        match self.phase {
            MigrationPhase::ConfirmApply => self.phase = MigrationPhase::DryRunReady,
            MigrationPhase::ConfirmCutover | MigrationPhase::ConfirmRollback => {
                self.phase = MigrationPhase::Applied;
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(dry_run: bool, batch: &str) -> MigrationReport {
        MigrationReport {
            schema_version: 1,
            generated_at: "2026-09-20T00:00:00.000Z".to_string(),
            source_root: "/v1".to_string(),
            dry_run,
            batch_id: batch.to_string(),
            counts: Default::default(),
            records: Vec::new(),
            skipped_damaged: Vec::new(),
        }
    }

    fn report_with_records(dry_run: bool, outcomes: &[Option<bool>], batch: &str) -> MigrationReport {
        let records = outcomes
            .iter()
            .map(|verified| starling_dictation::store_v2::MigratedRecord {
                id: "s".to_string(),
                category: "session".to_string(),
                v1_status: "transcribed".to_string(),
                source_wav_hash: "00".to_string(),
                source_sample_hash: "00".to_string(),
                source_sample_count: 1,
                dest_journal_hash: None,
                dest_sample_hash: None,
                dest_sample_count: None,
                verified: *verified,
                error: if verified.is_some() { None } else { Some("boom".to_string()) },
            })
            .collect();
        MigrationReport {
            records,
            ..report(dry_run, batch)
        }
    }

    #[test]
    fn apply_is_refused_without_a_report() {
        let mut ui = MigrationUi::new();
        assert_eq!(
            ui.arm_apply().unwrap_err(),
            "run a dry-run preview first — the report is what you are confirming"
        );
        assert_eq!(
            ui.apply_confirmed().unwrap_err(),
            "confirm the dry-run report before importing"
        );
        // And a confirmed apply cannot land without having been confirmed.
        assert_eq!(
            ui.apply_finished(report(false, "mig_a")).unwrap_err(),
            "no import was confirmed"
        );
    }

    #[test]
    fn apply_requires_the_confirm_not_just_the_report() {
        let mut ui = MigrationUi::new();
        ui.dry_run_finished(report(true, "mig_dry"));
        assert_eq!(
            ui.apply_confirmed().unwrap_err(),
            "confirm the dry-run report before importing"
        );
        ui.arm_apply().expect("arm");
        ui.cancel_apply();
        assert_eq!(ui.phase, MigrationPhase::DryRunReady);
        assert_eq!(
            ui.apply_confirmed().unwrap_err(),
            "confirm the dry-run report before importing"
        );
    }

    #[test]
    fn the_happy_path_runs_dry_run_apply_cutover_rollback() {
        let mut ui = MigrationUi::new();
        ui.dry_run_finished(report_with_records(true, &[Some(true)], "mig_dry"));
        ui.arm_apply().expect("arm apply");
        ui.apply_confirmed().expect("confirm apply");
        assert!(ui.busy);
        ui.apply_finished(report_with_records(false, &[Some(true)], "mig_app"))
            .expect("apply finished");
        assert_eq!(ui.phase, MigrationPhase::Applied);
        assert_eq!(ui.applied_batch.as_deref(), Some("mig_app"));
        assert!(ui.cutover_ready(), "fully verified import on screen");

        ui.arm_cutover().expect("arm cutover");
        ui.cutover_confirmed().expect("confirm cutover");
        ui.cutover_finished().expect("cutover finished");
        assert!(ui.cutover_persisted);
        assert!(ui.notice.as_deref().unwrap().contains("next start"));

        // Rollback stays available after the cutover.
        assert!(ui.rollback_ready());
        ui.arm_rollback().expect("arm rollback");
        ui.cancel_rollback();
        ui.arm_rollback().expect("arm rollback again");
        ui.rollback_confirmed().expect("confirm rollback");
        ui.rollback_finished(3).expect("rollback finished");
        assert_eq!(ui.phase, MigrationPhase::Idle);
        assert!(ui.applied_batch.is_none());
        assert!(!ui.cutover_persisted);
        assert!(ui.notice.as_deref().unwrap().contains("never touched"));
    }

    #[test]
    fn cutover_is_refused_without_a_fully_verified_apply() {
        let mut ui = MigrationUi::new();
        // A dry-run report, however clean, never arms cutover.
        ui.dry_run_finished(report_with_records(true, &[Some(true)], "mig_dry"));
        assert!(!ui.cutover_ready());
        assert!(ui.arm_cutover().is_err());

        // An apply with an unverified record neither.
        ui.arm_apply().expect("arm");
        ui.apply_confirmed().expect("confirm");
        ui.apply_finished(report_with_records(false, &[Some(true), None], "mig_x"))
            .expect("applied");
        assert!(!ui.cutover_ready(), "one record failed to verify");
        assert_eq!(
            ui.arm_cutover().unwrap_err(),
            "cutover needs a fully verified import on screen (every recording re-checked by \
             content hash)"
        );

        // Rollback is offered regardless: discarding a partial import is
        // always safe (originals were never touched).
        assert!(ui.rollback_ready());
    }

    #[test]
    fn rollback_is_refused_with_no_applied_batch() {
        let mut ui = MigrationUi::new();
        assert!(!ui.rollback_ready());
        assert_eq!(
            ui.arm_rollback().unwrap_err(),
            "nothing to roll back — no import has been applied"
        );

        // After a rollback, a second rollback refuses.
        ui.dry_run_finished(report(true, "m"));
        ui.arm_apply().expect("arm");
        ui.apply_confirmed().expect("confirm");
        ui.apply_finished(report(false, "mig_b")).expect("applied");
        ui.arm_rollback().expect("arm");
        ui.rollback_confirmed().expect("confirm");
        ui.rollback_finished(2).expect("done");
        assert_eq!(
            ui.arm_rollback().unwrap_err(),
            "nothing to roll back — no import has been applied"
        );
    }

    #[test]
    fn every_gate_refuses_while_a_step_runs() {
        let mut ui = MigrationUi::new();
        ui.dry_run_finished(report(true, "m"));
        ui.arm_apply().expect("arm");
        ui.apply_confirmed().expect("busy");
        for refusal in [
            ui.arm_apply(),
            ui.apply_confirmed(),
            ui.arm_cutover(),
            ui.arm_rollback(),
        ] {
            assert_eq!(
                refusal.unwrap_err(),
                "a migration step is already running"
            );
        }
        ui.op_failed();
        assert!(!ui.busy);
        assert_eq!(ui.phase, MigrationPhase::DryRunReady, "backed out of the gate");
    }

    #[test]
    fn op_failed_returns_a_cutover_gate_to_the_applied_screen() {
        let mut ui = MigrationUi::new();
        ui.dry_run_finished(report(true, "m"));
        ui.arm_apply().expect("arm");
        ui.apply_confirmed().expect("confirm");
        ui.apply_finished(report(false, "mig_c")).expect("applied");
        ui.arm_cutover().expect("arm cutover");
        ui.cutover_confirmed().expect("confirmed, persistence pending");
        ui.op_failed(); // settings write failed
        assert_eq!(ui.phase, MigrationPhase::Applied);
        assert!(!ui.cutover_persisted, "the choice was never persisted");
    }
}
