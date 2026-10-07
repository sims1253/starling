//! Test doubles for the insertion crate: the scripted [`FakeBackend`]
//! and the focus world it serves. Production code never touches this
//! module (it exists behind `test-doubles`, exactly like
//! `starling-runtime`'s `testing`); it is how the bridge tests pin
//! conflict, gone, multiline refusal, evidence passthrough and
//! no-enter behavior without an X server or a Windows session.

use std::sync::Mutex;

use crate::{
    format_ref, insertion_guards, merge_excluded_pids, Availability, BackendKind, InsertError,
    InsertReceipt, InsertionBackend, SurroundingText, TargetCheck, TargetSnapshot,
    EVIDENCE_SYNTHETIC_KEYS,
};

/// One scripted focused target. The fake's refs are
/// `fake:<target-hex>:<focus-hex>:<pid>` — the same shape as the real
/// backends' refs, so [`crate::Inserter::parse_target_ref`] and the
/// runtime bridge treat fake refs exactly like production ones (that
/// is the point: the bridge must not know a backend is fake).
#[derive(Debug, Clone)]
pub struct FakeTarget {
    pub app: Option<String>,
    pub title: Option<String>,
    pub pid: Option<u32>,
}

impl FakeTarget {
    /// An ordinary other-app target named `app`/`title`.
    pub fn named(app: &str, title: &str) -> FakeTarget {
        FakeTarget {
            app: Some(app.to_string()),
            title: Some(title.to_string()),
            pid: Some(4213),
        }
    }

    /// A target whose pid is whatever the *test process* is — the
    /// Starling-owns-it refusal's setup.
    pub fn owned_by_this_process() -> FakeTarget {
        FakeTarget {
            app: Some("Starling".to_string()),
            title: Some("Starling notes".to_string()),
            pid: Some(std::process::id()),
        }
    }
}

/// What the fake's `insert` should do once the crate guards pass.
/// The guards always run first, so a `FailWith(MultilineUnsupported)`
/// script never fires on single-line text and the recorded insertions
/// stay an honest account of what reached a target.
#[derive(Debug, Clone)]
pub enum InsertBehavior {
    Type,
    FailWith(InsertError),
}

struct State {
    availability: Availability,
    focus: Option<FakeTarget>,
    /// The stable id of the *current* focus setting: minted once per
    /// [`FakeBackend::focus`] call, so two captures of the same focus
    /// produce the same ref (identity, not a counter of calls).
    current_id: u64,
    /// A ref marked destroyed by [`FakeBackend::destroy_target`]:
    /// revalidating it reports `Gone` (the window closed), the honest
    /// X11/Windows analogue.
    destroyed_ref: Option<String>,
    next_id: u64,
    insert_behavior: InsertBehavior,
    /// Scripted revalidate failures (see [`FakeBackend::fail_revalidate`]):
    /// the error to fail with and how many revalidate calls it still
    /// covers (`None` = until cleared).
    revalidate_failure: Option<(InsertError, Option<usize>)>,
    surrounding: Option<SurroundingText>,
    /// The insertions `insert` accepted, for assertions.
    insertions: Vec<(String, String)>,
    /// How many revalidations ran (guards + explicit calls).
    revalidations: usize,
}

/// A fully scripted [`InsertionBackend`]. Cheap to reason about: it
/// serves whatever focus the test last set ([`Self::focus`]), reports
/// the scripted availability, and records what insert asked of it.
pub struct FakeBackend {
    state: Mutex<State>,
    excluded_pids: Vec<u32>,
}

impl FakeBackend {
    /// A ready backend with no focus set — tests immediately
    /// [`Self::focus`] a target. Self-target protection covers exactly
    /// this process, like a production backend's default.
    pub fn new() -> FakeBackend {
        FakeBackend::with_excluded_pids(vec![std::process::id()])
    }

    /// Construct with an explicit ownership policy (the fake twin of
    /// the real backends' constructor): targets whose pid is in the
    /// list are refused with `TargetIsStarling` at capture and insert.
    pub fn with_excluded_pids(excluded_pids: Vec<u32>) -> FakeBackend {
        FakeBackend {
            state: Mutex::new(State {
                availability: Availability::Ready,
                focus: None,
                current_id: 0,
                destroyed_ref: None,
                next_id: 1,
                insert_behavior: InsertBehavior::Type,
                revalidate_failure: None,
                surrounding: None,
                insertions: Vec::new(),
                revalidations: 0,
            }),
            excluded_pids: merge_excluded_pids(excluded_pids),
        }
    }

    /// Make `target` the focused target: the next capture sees it, and
    /// revalidates of refs from an earlier focus see `Changed`.
    pub fn focus(&self, target: FakeTarget) {
        let mut state = self.state.lock().expect("fake focus lock");
        state.current_id = state.next_id;
        state.next_id += 1;
        state.focus = Some(target);
        state.destroyed_ref = None;
    }

    /// Clear focus (no target at all): capture then fails, revalidate
    /// reports the focus changed away.
    pub fn blur(&self) {
        self.state.lock().expect("fake focus lock").focus = None;
    }

    /// Destroy the current target (the window closed): revalidates of
    /// the current ref report `Gone`.
    pub fn destroy_target(&self) {
        let mut state = self.state.lock().expect("fake destroy lock");
        if let Some(reference) = state.current_ref() {
            state.destroyed_ref = Some(reference);
        }
        state.focus = None;
    }

    /// Script the backend's availability (the Linux setup-check
    /// stories).
    pub fn set_availability(&self, availability: Availability) {
        self.state
            .lock()
            .expect("fake availability lock")
            .availability = availability;
    }

    /// Script `insert`'s behavior once the crate guards pass.
    pub fn set_insert_behavior(&self, behavior: InsertBehavior) {
        self.state
            .lock()
            .expect("fake behavior lock")
            .insert_behavior = behavior;
    }

    /// Script `revalidate` to fail with `error`: the next `count`
    /// calls (or every call, when `count` is `None`, until
    /// [`Self::clear_revalidate_failures`]) return the error instead
    /// of a `TargetCheck`. Because the crate guards revalidate inside
    /// `insert`, this is how a test scripts a backend that cannot
    /// answer its last-chunk check — a dying connection, say.
    pub fn fail_revalidate(&self, error: InsertError, count: Option<usize>) {
        self.state
            .lock()
            .expect("fake revalidate script lock")
            .revalidate_failure = Some((error, count));
    }

    /// Stop scripting revalidate failures.
    pub fn clear_revalidate_failures(&self) {
        self.state
            .lock()
            .expect("fake revalidate script lock")
            .revalidate_failure = None;
    }

    /// Script what `surrounding_text` reports.
    pub fn set_surrounding(&self, surrounding: Option<SurroundingText>) {
        self.state
            .lock()
            .expect("fake surrounding lock")
            .surrounding = surrounding;
    }

    /// The insertions the fake accepted, target ref and text, in
    /// order.
    pub fn insertions(&self) -> Vec<(String, String)> {
        self.state
            .lock()
            .expect("fake insertions lock")
            .insertions
            .clone()
    }

    /// How many revalidations ran.
    pub fn revalidations(&self) -> usize {
        self.state
            .lock()
            .expect("fake revalidations lock")
            .revalidations
    }
}

impl State {
    /// The ref of the current focus setting, if one is focused.
    fn current_ref(&self) -> Option<String> {
        self.focus.as_ref().map(|target| {
            format_ref(
                BackendKind::Fake,
                self.current_id,
                self.current_id,
                target.pid,
            )
        })
    }
}

impl Default for FakeBackend {
    fn default() -> Self {
        FakeBackend::new()
    }
}

impl InsertionBackend for FakeBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Fake
    }

    fn availability(&self) -> Availability {
        self.state
            .lock()
            .expect("fake availability lock")
            .availability
            .clone()
    }

    fn capture(&self) -> Result<TargetSnapshot, InsertError> {
        let state = self.state.lock().expect("fake capture lock");
        let Some(target) = state.focus.clone() else {
            return Err(InsertError::Rejected {
                reason: "no window has input focus".to_string(),
            });
        };
        if target
            .pid
            .is_some_and(|pid| self.excluded_pids.contains(&pid))
        {
            // The fake honors the Starling-owns-it rule like a real
            // backend, so tests can pin it.
            return Err(InsertError::TargetIsStarling);
        }
        Ok(TargetSnapshot {
            backend: BackendKind::Fake,
            target_ref: state.current_ref().expect("focus just checked"),
            app: target.app,
            title: target.title,
            pid: target.pid,
            capabilities: BackendKind::Fake.capabilities(),
        })
    }

    fn revalidate(&self, target: &TargetSnapshot) -> Result<TargetCheck, InsertError> {
        let mut state = self.state.lock().expect("fake revalidate lock");
        state.revalidations += 1;
        if let Some((error, remaining)) = state.revalidate_failure.clone() {
            match remaining {
                // `Some(n)` means "fail for n more calls": this call is
                // one of them, so spend it now — `Some(1)` was the
                // last, `Some(0)` (never stored by the API) counts as
                // already spent.
                None => {}
                Some(1) | Some(0) => state.revalidate_failure = None,
                Some(count) => state.revalidate_failure = Some((error.clone(), Some(count - 1))),
            }
            return Err(error);
        }
        if state.destroyed_ref.as_deref() == Some(target.target_ref.as_str()) {
            return Ok(TargetCheck::Gone);
        }
        match state.current_ref() {
            Some(reference) if reference == target.target_ref => Ok(TargetCheck::Same),
            Some(reference) => Ok(TargetCheck::Changed {
                expected: target.target_ref.clone(),
                actual: reference,
            }),
            None => Ok(TargetCheck::Changed {
                expected: target.target_ref.clone(),
                actual: format!("{}:none", BackendKind::Fake.scheme()),
            }),
        }
    }

    fn surrounding_text(
        &self,
        _target: &TargetSnapshot,
    ) -> Result<Option<SurroundingText>, InsertError> {
        Ok(self
            .state
            .lock()
            .expect("fake surrounding lock")
            .surrounding
            .clone())
    }

    fn insert(&self, target: &TargetSnapshot, text: &str) -> Result<InsertReceipt, InsertError> {
        // The guards run like on a real backend (see `InsertBehavior`
        // for why the recording stays honest).
        insertion_guards(text, target.pid, &self.excluded_pids, || {
            self.revalidate(target)
        })?;
        let mut state = self.state.lock().expect("fake insert lock");
        // The live focus's pid is the ownership authority, like the
        // real backends' last-chunk check (the snapshot's own pid was
        // only the early refusal above).
        if state
            .focus
            .as_ref()
            .and_then(|focus| focus.pid)
            .is_some_and(|pid| self.excluded_pids.contains(&pid))
        {
            return Err(InsertError::TargetIsStarling);
        }
        match state.insert_behavior.clone() {
            InsertBehavior::Type => {
                state
                    .insertions
                    .push((target.target_ref.clone(), text.to_string()));
                Ok(InsertReceipt {
                    evidence: EVIDENCE_SYNTHETIC_KEYS,
                })
            }
            InsertBehavior::FailWith(error) => Err(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BackendKind, Inserter, TargetCheck};

    #[test]
    fn capture_reports_the_focused_target_and_round_trips_its_ref() {
        let fake = FakeBackend::new();
        fake.focus(FakeTarget::named("Notes", "Meeting notes"));
        let target = fake.capture().expect("focus is set");
        assert_eq!(target.backend, BackendKind::Fake);
        assert_eq!(target.app.as_deref(), Some("Notes"));
        assert_eq!(target.title.as_deref(), Some("Meeting notes"));
        assert_eq!(target.pid, Some(4213));
        // Refs are self-describing: the inserter parses the fake's
        // exactly like a production backend's.
        let parsed = Inserter::with_backends(vec![Box::new(FakeBackend::new())])
            .parse_target_ref(&target.target_ref)
            .expect("the ref round trips");
        assert_eq!(parsed.backend, BackendKind::Fake);
        assert_eq!(parsed.pid, Some(4213));
        // And two captures of the same focus are the same identity —
        // a capture is a fact about the world, not a counter.
        assert_eq!(fake.capture().unwrap().target_ref, target.target_ref);
    }

    #[test]
    fn capture_refuses_when_nothing_is_focused_or_starling_is() {
        let fake = FakeBackend::new();
        assert!(matches!(
            fake.capture(),
            Err(InsertError::Rejected { reason, .. }) if reason.contains("no window")
        ));
        fake.focus(FakeTarget::owned_by_this_process());
        assert_eq!(fake.capture(), Err(InsertError::TargetIsStarling));
    }

    #[test]
    fn revalidate_tracks_focus_changes_blur_and_destruction() {
        let fake = FakeBackend::new();
        fake.focus(FakeTarget::named("Notes", "Meeting notes"));
        let notes = fake.capture().expect("focus is set");

        assert_eq!(
            fake.revalidate(&notes).unwrap(),
            TargetCheck::Same,
            "unchanged focus stays Same"
        );

        fake.focus(FakeTarget::named("Browser", "A tab"));
        match fake.revalidate(&notes).unwrap() {
            TargetCheck::Changed { expected, actual } => {
                assert_eq!(expected, notes.target_ref);
                assert!(actual.starts_with("fake:"), "actual names the new target");
            }
            other => panic!("a moved focus must conflict: {other:?}"),
        }

        fake.blur();
        match fake.revalidate(&notes).unwrap() {
            TargetCheck::Changed { actual, .. } => assert_eq!(actual, "fake:none"),
            other => panic!("a cleared focus must conflict: {other:?}"),
        }

        // Destroy: the window closed, so the same ref reports Gone.
        fake.focus(FakeTarget::named("Notes", "Meeting notes"));
        let notes_again = fake.capture().expect("focus is set");
        fake.destroy_target();
        assert_eq!(fake.revalidate(&notes_again).unwrap(), TargetCheck::Gone);
    }

    #[test]
    fn insert_records_only_guard_passing_insertions() {
        let fake = FakeBackend::new();
        fake.focus(FakeTarget::named("Notes", "Meeting notes"));
        let target = fake.capture().expect("focus is set");

        let receipt = fake
            .insert(&target, "Café, bitte.")
            .expect("the fake types");
        assert_eq!(receipt.evidence, EVIDENCE_SYNTHETIC_KEYS);
        assert_eq!(
            fake.insertions(),
            vec![(target.target_ref.clone(), "Café, bitte.".to_string())]
        );
        // The guard ran inside insert (revalidate) before the record.
        assert!(fake.revalidations() >= 1);

        // Control characters never record (refused by the guards).
        assert_eq!(
            fake.insert(&target, "no\nenter").unwrap_err(),
            InsertError::MultilineUnsupported
        );
        assert_eq!(fake.insertions().len(), 1);

        // A changed focus refuses with the conflict, records nothing.
        fake.focus(FakeTarget::named("Browser", "A tab"));
        assert!(matches!(
            fake.insert(&target, "somewhere else"),
            Err(InsertError::TargetChanged { .. })
        ));
        assert_eq!(fake.insertions().len(), 1);

        // Scripted backend-level failure passes through untouched.
        fake.set_insert_behavior(InsertBehavior::FailWith(InsertError::PermissionDenied {
            reason: "elevated".into(),
            settings_hint: "same integrity level".into(),
        }));
        let browser = fake.capture().expect("focus is set");
        assert_eq!(
            fake.insert(&browser, "hello").unwrap_err().code(),
            "insertion_permission_denied"
        );
    }

    #[test]
    fn surrounding_text_serves_the_scripted_value() {
        let fake = FakeBackend::new();
        fake.focus(FakeTarget::named("Notes", "Meeting notes"));
        let target = fake.capture().expect("focus is set");
        assert_eq!(fake.surrounding_text(&target).unwrap(), None);
        fake.set_surrounding(Some(SurroundingText {
            before: "Hello ".into(),
            after: " world".into(),
            selection: None,
        }));
        assert_eq!(
            fake.surrounding_text(&target).unwrap(),
            Some(SurroundingText {
                before: "Hello ".into(),
                after: " world".into(),
                selection: None
            })
        );
    }

    #[test]
    fn the_fake_serves_through_the_shared_arc_delegation() {
        // The pattern the bridge tests use: one instance, two handles.
        let fake = std::sync::Arc::new(FakeBackend::new());
        fake.focus(FakeTarget::named("Notes", "Meeting notes"));
        let inserter = Inserter::with_backends(vec![Box::new(fake.clone())]);
        let target = inserter.capture().expect("focus is set");
        let backend = inserter.backend_for(&target).expect("routed by scheme");
        backend.insert(&target, "via the Arc").expect("delegates");
        assert_eq!(fake.insertions().len(), 1);
    }

    #[test]
    fn excluded_pids_refuse_at_capture_and_again_at_insert_time() {
        // A host-configured exclusion set: pid 555 is Starling's app
        // process (not this one — mode B is exactly that split).
        let fake = FakeBackend::with_excluded_pids(vec![555]);
        fake.focus(FakeTarget {
            app: Some("Starling".into()),
            title: Some("Starling editor".into()),
            pid: Some(555),
        });
        assert_eq!(fake.capture().unwrap_err(), InsertError::TargetIsStarling);

        // The exclusion is consulted at insert time too, not baked
        // into the snapshot: a target captured under pid 777 is
        // refused once 777 joins the excluded set.
        let other = FakeBackend::with_excluded_pids(vec![555]);
        other.focus(FakeTarget::named("Notes", "Meeting notes"));
        let target = other.capture().expect("focus is set");
        other
            .insert(&target, "hello")
            .expect("a normal target types");
        let strict = FakeBackend::with_excluded_pids(vec![555, 4213]);
        assert_eq!(
            strict.insert(&target, "hello").unwrap_err(),
            InsertError::TargetIsStarling,
            "the live pid 4213 is excluded now, whatever the snapshot said"
        );
        assert_eq!(strict.insertions().len(), 0);
    }

    #[test]
    fn scripted_revalidate_failures_clear_after_their_count() {
        let fake = FakeBackend::new();
        fake.focus(FakeTarget::named("Notes", "Meeting notes"));
        let target = fake.capture().expect("focus is set");
        fake.fail_revalidate(
            InsertError::Unavailable {
                reason: "blip".into(),
                setup_hint: None,
            },
            Some(2),
        );
        assert_eq!(
            fake.revalidate(&target).unwrap_err().code(),
            "insertion_unavailable"
        );
        assert_eq!(
            fake.revalidate(&target).unwrap_err().code(),
            "insertion_unavailable"
        );
        assert_eq!(fake.revalidate(&target).unwrap(), TargetCheck::Same);
        // The forever flavor sticks until cleared (an error, not a
        // TargetCheck — that is the scripting point).
        fake.fail_revalidate(InsertError::TargetGone, None);
        assert_eq!(
            fake.revalidate(&target).unwrap_err(),
            InsertError::TargetGone
        );
        fake.clear_revalidate_failures();
        assert_eq!(fake.revalidate(&target).unwrap(), TargetCheck::Same);
    }
}
